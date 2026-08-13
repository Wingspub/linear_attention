from torch import nn
import torch

class Base(nn.Module):
    """基础模型模板"""
    def __init__(self, vocab_size: int, embed_dims: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dims)
        self.Linear = nn.Linear(embed_dims, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids -> (batch, seq_len)
        # output -> (batch, seq, vocab_size)
        embeddings = self.embeddings(input_ids)
        output = self.Linear(embeddings)
        return output


class MLP(nn.Module):
    def __init__(self, vocab_size: int, dims: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dims)
        self.MLP = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, vocab_size)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids -> (batch, seq)
        # output -> (batch, seq, vocab_size)
        embeddings = self.embeddings(input_ids)
        out = self.MLP(embeddings)
        return out


class SimpleRecurrentSequentialModel(nn.Module):
    def __init__(self, vocab_size: int, dims: int, device: torch.device):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.dims = dims

        self.embeddings = nn.Embedding(vocab_size, dims)

        self.h_matrix = nn.Linear(dims, dims)
        self.x_matrix = nn.Linear(dims, dims)
        self.activate = nn.ReLU()

        self.output_trans = nn.Linear(dims, vocab_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        B, L = input_seq.shape
        embeddings = self.embeddings(input_seq)

        h0 = torch.zeros((B,self.dims)).to(self.device)
        output = torch.zeros((B, L, self.vocab_size)).to(self.device)
        for i in range(L):
            x = embeddings[:, i]
            h0 = self.activate(self.h_matrix(h0) + self.x_matrix(x))
            output[:, i] = self.output_trans(h0)

        return output
