from typing import cast
import torch
from torch import nn


class SimpleParallelSequentialModel(nn.Module):
    def __init__(self,dims: int):
        super().__init__()
        self.V_trans = nn.Linear(dims, dims, bias=False)


    def forward(self, input_seq_embeddings: torch.Tensor) -> torch.Tensor:
        # input shape -> (B, L, d)
        # output shape -> (B, L, d)

        L = input_seq_embeddings.shape[1]
        V = cast(torch.Tensor, self.V_trans(input_seq_embeddings))

        # casual matrix
        A = torch.tril(torch.ones((L, L))) / torch.arange(1, L+1).unsqueeze(1)
        output = torch.matmul(A.to(V.device), V)

        return output


class SimplestTransformer(nn.Module):
    def __init__(self, Token_num: int, layers_num: int, dims: int):
        super().__init__()
        self.layers_num = layers_num

        self.embeddings = nn.Embedding(Token_num, dims)
        self.activate = nn.ReLU()
        self.output_trans = nn.Linear(dims, Token_num)

        self.layers = nn.ModuleList()

        for _ in range(layers_num):
            attention_layer = SimpleParallelSequentialModel(dims)
            self.layers.append(attention_layer)


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(input_seq)

        for i in range(self.layers_num):
            x = self.layers[i](embeddings)
            embeddings = embeddings + self.activate(x)

        output = self.output_trans(embeddings)

        return output
