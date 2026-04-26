from torch import nn
import torch

class Base(nn.Module):
    def __init__(self, TOKEN_num :int, dims):
        super().__init__()
        self.embeddings = nn.Embedding(TOKEN_num, dims)
        self.Linear = nn.Linear(dims, TOKEN_num)


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # input_seq -> (batch, seq)
        # output -> (batch, seq, TOEKN_num)
        embeddings = self.embeddings(input_seq)
        out = self.Linear(embeddings)
        return out


class MLP(nn.Module):
    def __init__(self, TOKEN_num :int, dims :int):
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


class SelfAttent(nn.Module):
    def __init__(self, TOKEN_num :int, dims :int = 32):
        super().__init__()

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        return input_seq


class RNN_base(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def froward(self, input_seq: torch.Tensor) -> torch.Tensor:
        ...



