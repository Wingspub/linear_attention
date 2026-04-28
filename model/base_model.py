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


class SimpleSequentialModel(nn.Module):
    def __init__(self, TOKEN_num: int, dims: int, device: torch.device):
        super().__init__()
        self.device = device
        self.embeddings = nn.Embedding(TOKEN_num, dims)
        self.V_trans = nn.Linear(dims, dims, bias=False)

        self.output_trans = nn.Linear(dims, TOKEN_num)


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        L = input_seq.shape[1]
        embeddings = self.embeddings(input_seq)
        V = self.V_trans(embeddings)

        # casual matrix
        A = torch.tril(torch.ones(L, L)) / torch.arange(1, L+1).unsqueeze(1)
        output = self.output_trans(torch.matmul(A.to(self.device), V))

        return output


class SelfAttention(nn.Module):
    def __init__(self, TOKEN_num: int, dims: int, device: torch.device):
        super().__init__()
        self.embeddings = nn.Embedding(TOKEN_num, dims)
        self.dims = dims
        self.device = device

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

        A = torch.tril(self.softmax(torch.matmul(Q, K.transpose(-1, -2))/torch.sqrt(torch.tensor(self.dims).to(self.device))))
        output = self.output_trans(torch.matmul(A, V))

        return output


class RNN_base(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def froward(self, input_seq: torch.Tensor) -> torch.Tensor:
        ...

