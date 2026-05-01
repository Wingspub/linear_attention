# Efficient Attention: Attention with Linear Complexities
# https://ieeexplore.ieee.org/document/9423033/

from typing import cast
from torch import nn
import torch


class EfficientAttention(nn.Module):
    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.heads_dims = dims // heads
        assert self.dims == self.heads_dims * self.heads

        self.W_Q = nn.Linear(dims, dims, bias=False)
        self.W_K = nn.Linear(dims, dims, bias=False)
        self.W_V = nn.Linear(dims, dims, bias=False)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input_seq_embs: torch.Tensor) -> torch.Tensor:
        B, L, d = input_seq_embs.shape

        Q = cast(torch.Tensor, self.W_Q(input_seq_embs).reshape(B, L, self.heads, self.heads_dims))
        K = cast(torch.Tensor, self.W_K(input_seq_embs).reshape(B, L, self.heads, self.heads_dims))
        V = cast(torch.Tensor, self.W_V(input_seq_embs).reshape(B, L, self.heads, self.heads_dims))

        # No causality

        tran_V = self.softmax(K.transpose(-1, -2)) @ V
        output_V = self.softmax(Q) @ tran_V

        return output_V.reshape(B, L, d)


class EfficientTransformer(nn.Module):
    def __init__(self, Token_num: int, layers_num: int, dims: int):
        super().__init__()
        self.layers_num = layers_num

        self.embeddings = nn.Embedding(Token_num, dims)
        self.activate = nn.ReLU()
        self.output_trans = nn.Linear(dims, Token_num)

        self.layers = nn.ModuleList()
        for _ in range(layers_num):
            attention_layer = EfficientAttention(dims, heads=8)
            self.layers.append(attention_layer)


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(input_seq)

        for i in range(self.layers_num):
            x = self.layers[i](embeddings)
            embeddings = embeddings + self.activate(x)

        output = self.output_trans(embeddings)

        return output
