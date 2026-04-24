from torch import nn
import torch

class Base(nn.Module):
    def __init__(self, TOKEN_num :int, dims :int = 32):
        super().__init__()
        self.embeddings = nn.Embedding(TOKEN_num, dims)
        self.MLP = nn.Linear(dims, dims)

    def forward(self, input_seq: torch.Tensor):
        # input_seq -> (batch, seq)
        embeddings = self.embeddings(input_seq)
        out = self.MLP(embeddings)
        return out
