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

        self.W_Q = nn.Linear(dims, dims, bias=False)
        self.W_K = nn.Linear(dims, dims, bias=False)
        self.W_V = nn.Linear(dims, dims, bias=False)

        self.score_dropout = nn.Dropout(p=0.1)
        self.residual_dropout = nn.Dropout(p=0.1)

        self.softmax = nn.Softmax(dim=-1)
        # output_tran
        self.output_project = nn.Linear(dims, dims)


    def forward(self, input_seq_embs: torch.Tensor, is_causal: bool=True) -> torch.Tensor:

        Q = cast(torch.Tensor, self.W_Q(input_seq_embs))
        K = cast(torch.Tensor, self.W_K(input_seq_embs))
        V = cast(torch.Tensor, self.W_V(input_seq_embs))

        # A: score matrix
        A = torch.matmul(Q, K.transpose(-1, -2))
        if is_causal:
            mask = torch.log(torch.tril(torch.ones_like(A, dtype=torch.bool)))
            A = self.softmax(A + mask)
        else:
            A = self.softmax(A)

        # output
        output = torch.matmul(self.score_dropout(A), V)
        output = self.residual_dropout(self.output_project(output))

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, dims: int, heads: int):
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


    def forward(self, input_seq_embs: torch.Tensor, is_causal: bool=True) -> torch.Tensor:
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


class TransformerBlock(nn.Module):
    def __init__(self, dims: int, heads: int=8):
        super().__init__()

        # self.attention = SelfAttention(dims=dims)
        self.attention = MultiHeadAttention(dims=dims, heads=heads)
        self.LN1 = nn.LayerNorm(dims)
        self.FFN = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims)
        )
        self.LN2 = nn.LayerNorm(dims)


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
        dim_index = 10000 ** (torch.arange(1, 1+self.dims).unsqueeze(0)/self.dims)
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
