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


from typing import cast, Any

from torch import nn
import torch

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

        self.score_dropout = nn.Dropout(p=0.1)
        self.residual_dropout = nn.Dropout(p=0.1)

        self.softmax = nn.Softmax(-1)
        # output_tran
        self.output_project = nn.Linear(dims, dims)

    def forward(self, input_seq_embs: torch.Tensor, is_causal: bool=False) -> torch.Tensor:
        B, L, d = input_seq_embs.shape

        Q = cast(torch.Tensor, self.W_Q(input_seq_embs)).reshape(B, self.heads, L, self.heads_dims)
        K = cast(torch.Tensor, self.W_K(input_seq_embs)).reshape(B, self.heads, L, self.heads_dims)
        V = cast(torch.Tensor, self.W_V(input_seq_embs)).reshape(B, self.heads, L, self.heads_dims)

        # A: score matrix
        A = torch.matmul(Q, K.transpose(-1, -2))
        if is_causal:
            mask = torch.log(torch.tril(torch.ones_like(A, dtype=torch.bool)))
            A = self.softmax(A + mask)

        else:
            A = self.softmax(A)

        # output
        output = torch.matmul(self.score_dropout(A), V).reshape(B, L, d)
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


    def forward(self, input_seq_embs: torch.Tensor) -> torch.Tensor:

        # Attention
        embs = self.attention(input_seq_embs, is_causal=True)
        embs = self.LN1(embs)
        input_seq_embs = input_seq_embs + embs

        # FFN
        embs = self.FFN(input_seq_embs)
        embs = self.LN2(embs)
        output_embs = input_seq_embs + embs

        return output_embs


class ModernTransformer(nn.Module):
    def __init__(self, token_num: int, block_num: int, dims: int, heads: int=8):
        super().__init__()
        self.block_num = block_num
        self.dims = dims

        self.embeddings = nn.Embedding(token_num, dims)

        # TransformerBlock
        self.layer_block = nn.ModuleList()
        for _ in range(block_num):
            transformerBlock = AttentionBlock(dims=dims, heads=heads)
            self.layer_block.append(transformerBlock)

        self.output_trans = nn.Linear(dims, token_num)


    def position_vector(self, seq_len: int) -> torch.Tensor:
        position_index = torch.arange(seq_len).unsqueeze(1)
        dim_index = torch.arange(self.dims)
        dim_index[::2] = dim_index[::2] / 2
        dim_index[1::2] = (dim_index[1::2] - 1) / 2
        dim_index = 10000 ** (dim_index.unsqueeze(0)/self.dims)
        position_matrix = position_index/dim_index

        # 偶数为cos，奇数为sin
        position_matrix[:, ::2] = torch.sin(position_matrix[:, ::2])
        position_matrix[:, 1::2] = torch.cos(position_matrix[:, 1::2])

        return position_matrix


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # 编码与位置编码
        L = input_seq.shape[1]
        embeddings = self.embeddings(input_seq)
        position_embeddings = self.position_vector(L).to(embeddings.device)

        embeddings = embeddings + position_embeddings

        # layer
        for i in range(self.block_num):
            embeddings = self.layer_block[i](embeddings)

        # output
        output = self.output_trans(embeddings)

        return output

