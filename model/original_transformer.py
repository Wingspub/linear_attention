from typing import cast
from torch import nn
import torch

# Tool Model
class LayerNorm(nn.Module):
    ...
    # self.training


class Dropout(nn.Module):
    ...


class FFN(nn.Module):
    def __init__(self, dims: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims)
        )


    def forward(self, input_embs: torch.Tensor) -> torch.Tensor:
        return self.layer(input_embs)


class SelfAttention(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

        self.Q_trans = nn.Linear(dims, dims, bias=False)
        self.K_trans = nn.Linear(dims, dims, bias=False)
        self.V_trans = nn.Linear(dims, dims, bias=False)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input_seq_embs: torch.Tensor, is_causal: bool=True) -> torch.Tensor:

        W_Q = cast(torch.Tensor, self.Q_trans(input_seq_embs))
        W_K = cast(torch.Tensor, self.K_trans(input_seq_embs))
        W_V = cast(torch.Tensor, self.V_trans(input_seq_embs))

        # A: score matrix
        A = torch.matmul(W_Q, W_K.transpose(-1, -2))
        if is_causal:
            mask_index = ~torch.tril(torch.ones_like(A, dtype=torch.bool))
            mask = torch.zeros_like(A)
            mask[mask_index] = torch.inf
            A = self.softmax(A - mask)

        else:
            A = self.softmax(A)

        # output
        output = torch.matmul(A, W_V)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.heads_dims = dims // heads

        assert dims == self.heads_dims * heads

        self.Q_trans = nn.Linear(dims, dims, bias=False)
        self.K_trans = nn.Linear(dims, dims, bias=False)
        self.V_trans = nn.Linear(dims, dims, bias=False)

        self.softmax = nn.Softmax(-1)


    def forward(self, input_seq_embs: torch.Tensor, is_causal: bool=True) -> torch.Tensor:
        B, L, d = input_seq_embs.shape

        W_Q = cast(torch.Tensor, self.Q_trans(input_seq_embs)).reshape(B, L, self.heads, self.heads_dims)
        W_K = cast(torch.Tensor, self.K_trans(input_seq_embs)).reshape(B, L, self.heads, self.heads_dims)
        W_V = cast(torch.Tensor, self.V_trans(input_seq_embs)).reshape(B, L, self.heads, self.heads_dims)

        # A: score matrix
        A = torch.matmul(W_Q, W_K.transpose(-1, -2))
        if is_causal:
            mask_index = ~torch.tril(torch.ones_like(A, dtype=torch.bool))
            mask = torch.zeros_like(A)
            mask[mask_index] = torch.inf
            A = self.softmax(A - mask)

        else:
            A = self.softmax(A)

        # output
        output = torch.matmul(A, W_V).reshape(B, L, d)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, dims: int, heads: int=8):
        super().__init__()

        # self.attention = SelfAttention(dims=dims)
        self.attention = MultiHeadAttention(dims=dims, heads=heads)
        self.LN1 = nn.LayerNorm(normalized_shape=dims)
        self.FFN = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims)
        )
        self.LN2 = nn.LayerNorm(normalized_shape=dims)


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


class OriginalTransformer(nn.Module):
    def __init__(self, token_num: int, block_num: int, dims: int, heads: int=8):
        super().__init__()
        self.block_num = block_num
        self.dims = dims

        self.embeddings = nn.Embedding(token_num, dims)

        # TransformerBlock
        self.layer_block = nn.ModuleList()
        for _ in range(block_num):
            transformerBlock = TransformerBlock(dims=dims, heads=heads)
            self.layer_block.append(transformerBlock)

        self.output_trans = nn.Linear(dims, token_num)


    def position_vector(self, seq_len: int) -> torch.Tensor:
        position_index = torch.arange(1, 1+seq_len).unsqueeze(1)
        dim_index = 10000 **(torch.arange(1, 1+self.dims).unsqueeze(0)/self.dims)
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
