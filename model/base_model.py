from typing import cast
from torch import nn
import torch

class Base(nn.Module):
    '''基础模型模板'''
    def __init__(self, TOKEN_num: int, dims: int):
        super().__init__()
        self.embeddings = nn.Embedding(TOKEN_num, dims)
        self.Linear = nn.Linear(dims, TOKEN_num)


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # input_seq -> (batch, seq)
        # output -> (batch, seq, TOEKN_num)
        embeddings = self.embeddings(input_seq)
        output = self.Linear(embeddings)
        return output


class MLP(nn.Module):
    def __init__(self, TOKEN_num: int, dims: int):
        super().__init__()
        self.embeddings = nn.Embedding(TOKEN_num, dims)
        self.MLP = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, TOKEN_num)
        )


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # input_seq -> (batch, seq)
        # output -> (batch, seq, TOEKN_num)
        embeddings = self.embeddings(input_seq)
        out = self.MLP(embeddings)
        return out


class SimpleParallelSequentialModel(nn.Module):
    def __init__(self, TOKEN_num: int, dims: int):
        super().__init__()
        self.embeddings = nn.Embedding(TOKEN_num, dims)
        self.V_trans = nn.Linear(dims, dims, bias=False)
        self.output_trans = nn.Linear(dims, TOKEN_num)


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        L = input_seq.shape[1]
        embeddings = self.embeddings(input_seq)
        V = cast(torch.Tensor, self.V_trans(embeddings))

        # casual matrix
        A = torch.tril(torch.ones(L, L)) / torch.arange(1, L+1).unsqueeze(1)
        output = self.output_trans(torch.matmul(A.to(V.device), V))

        return output


class SelfAttention(nn.Module):
    def __init__(self, TOKEN_num: int, dims: int):
        super().__init__()
        self.embeddings = nn.Embedding(TOKEN_num, dims)
        self.dims = dims

        self.Q_trans = nn.Linear(dims, dims, bias=False)
        self.K_trans = nn.Linear(dims, dims, bias=False)
        self.V_trans = nn.Linear(dims, dims, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.output_trans = nn.Linear(dims, TOKEN_num)


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(input_seq)
        Q = self.Q_trans(embeddings)
        K = self.K_trans(embeddings)
        V = self.V_trans(embeddings)
        # print(K.shape)

        A = torch.tril(self.softmax(torch.matmul(Q, K.transpose(-1, -2))/torch.sqrt(torch.tensor(self.dims).to(Q.device))))
        output = self.output_trans(torch.matmul(A, V))

        return output


class SimpleRecurrentSequentialModel(nn.Module):
    def __init__(self, TOKEN_num: int, dims: int, device: torch.device):
        super().__init__()
        self.TOKEN_num = TOKEN_num
        self.device = device
        self.dims = dims

        self.embeddings = nn.Embedding(TOKEN_num, dims)

        self.h_matrix = nn.Linear(dims, dims)
        self.x_matrix = nn.Linear(dims, dims)
        self.activate = nn.ReLU()

        self.output_trans = nn.Linear(dims, TOKEN_num)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        B, L = input_seq.shape
        embeddings = self.embeddings(input_seq)

        h0 = torch.zeros((B,self.dims)).to(self.device)
        output = torch.zeros((B, L, self.TOKEN_num)).to(self.device)
        for i in range(L):
            x = embeddings[:, i]
            h0 = self.activate(self.h_matrix(h0) + self.x_matrix(x))
            output[:, i] = self.output_trans(h0)

        return output
