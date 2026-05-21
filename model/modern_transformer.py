# (Llama) The Llama 3 Herd of Models
# http://arxiv.org/abs/2407.21783

# (RoPE) RoFormer: Enhanced Transformer with Rotary Position Embedding
# http://arxiv.org/abs/2104.09864

# (GLU) GLU Variants Improve Transformer
# https://arxiv.org/abs/2002.05202

# (RMSNorm) Root Mean Square Layer Normalization
# https://arxiv.org/abs/1910.07467

# (SiLU) Swish: a Self-Gated Activation Function
# https://arxiv.org/abs/1710.05941


from typing import cast, Tuple

from torch import nn
import torch


def precompute_rope_freqs(dim: int, max_seq_length: int = 32*1024, rope_base: float = 1e6) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs, atten_factor = 1.0/ (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)), 1.0
    t = torch.arange(max_seq_length, device=freqs.device)

    # Length * dim
    freqs = torch.outer(t, freqs).float()

    freqs_cos = torch.cos(freqs.repeat(1, 2)) * atten_factor
    freqs_sin = torch.sin(freqs.repeat(1, 2)) * atten_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos_weight: torch.Tensor, sin_weight: torch.Tensor, unsqueeze_dim=1) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos_weight.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin_weight.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos_weight.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin_weight.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))


    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)


    def forward(self, input_seq_emb: torch.Tensor) -> torch.Tensor:
        return (self.weight * self.norm(input_seq_emb.float())).type_as(input_seq_emb)


class FFN(nn.Module):
    def __init__(self, dims: int, hidden_dims: int) -> None:
        super().__init__()

        self.gated = nn.Linear(dims, hidden_dims, bias=False)
        self.up = nn.Linear(dims, hidden_dims,bias=False)
        self.down = nn.Linear(hidden_dims, dims, bias=False)
        self.activateion = nn.SiLU()


    def forward(self, input_seq_embs: torch.Tensor) -> torch.Tensor:
        return self.down(self.activateion(self.gated(input_seq_embs) * self.up(input_seq_embs)))


class Attention(nn.Module):
    def __init__(self, dims: int, heads: int) -> None:
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.heads_dims = dims // heads

        assert dims == self.heads_dims * heads

        self.W_Q = nn.Linear(dims, dims, bias=False)
        self.W_K = nn.Linear(dims, dims, bias=False)
        self.W_V = nn.Linear(dims, dims, bias=False)
        self.q_norm = RMSNorm(self.heads_dims)
        self.k_norm = RMSNorm(self.heads_dims)

        self.score_dropout = nn.Dropout(p=0.1)
        self.residual_dropout = nn.Dropout(p=0.1)

        self.softmax = nn.Softmax(-1)
        # output_tran
        self.output_project = nn.Linear(dims, dims)

    def forward(self, input_seq_embs: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], is_causal: bool=False) -> torch.Tensor:
        B, L, d = input_seq_embs.shape

        Q = cast(torch.Tensor, self.W_Q(input_seq_embs)).reshape(B, L, self.heads, self.heads_dims)
        K = cast(torch.Tensor, self.W_K(input_seq_embs)).reshape(B, L, self.heads, self.heads_dims)
        V = cast(torch.Tensor, self.W_V(input_seq_embs)).reshape(B, L, self.heads, self.heads_dims)
        Q, K = self.q_norm(Q), self.k_norm(K)

        cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin, unsqueeze_dim=1)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # A: score matrix
        A = torch.matmul(Q, K.transpose(-1, -2)) / (self.heads_dims)**0.5
        if is_causal:
            mask = torch.log(torch.tril(torch.ones_like(A, dtype=torch.bool)))
            A = self.softmax(A + mask)

        else:
            A = self.softmax(A)

        # output
        output = torch.matmul(self.score_dropout(A), V)
        output = output.transpose(1, 2).reshape(B, L, d)
        output = self.residual_dropout(self.output_project(output))

        return output


class AttentionBlock(nn.Module):
    def __init__(self, dims: int, heads: int=8):
        super().__init__()

        # self.attention = SelfAttention(dims=dims)
        self.attention = Attention(dims=dims, heads=heads)
        self.LN1 = RMSNorm(dims)
        self.FFN = FFN(dims=dims, hidden_dims=2*dims)
        self.LN2 = RMSNorm(dims)


    def forward(self, input_seq_embs: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Attention
        normed = self.LN1(input_seq_embs)
        embs = self.attention(normed, position_embeddings=position_embeddings, is_causal=True)
        x = input_seq_embs + embs

        # FFN
        normed = self.LN2(x)
        embs = self.FFN(normed)
        output_embs = x + embs

        return output_embs


class ModernTransformer(nn.Module):
    def __init__(self, token_num: int, block_num: int, dims: int, heads: int=8):
        super().__init__()
        self.block_num = block_num
        self.dims = dims
        head_dims = dims // heads

        self.embeddings = nn.Embedding(token_num, dims)
        freqs_cos, freqs_sin = precompute_rope_freqs(dim=head_dims)
        self.freqs_cos = nn.Buffer(freqs_cos, persistent=False)
        self.freqs_sin = nn.Buffer(freqs_sin, persistent=False)

        # TransformerBlock
        self.layer_block = nn.ModuleList()
        for _ in range(block_num):
            transformerBlock = AttentionBlock(dims=dims, heads=heads)
            self.layer_block.append(transformerBlock)

        self.output_trans = nn.Linear(dims, token_num)


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # 编码与位置编码
        L = input_seq.shape[1]
        embeddings = self.embeddings(input_seq)
        position_embeddings = (self.freqs_cos[:L], self.freqs_sin[:L])

        # layer
        for i in range(self.block_num):
            embeddings = self.layer_block[i](embeddings, position_embeddings)

        # output
        output = self.output_trans(embeddings)

        return output

