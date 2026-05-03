# An Attention Free Transformer
# http://arxiv.org/abs/2105.14103

from torch import nn
import torch

class FullFreeAttention(nn.Module):
    # learnable KV param(w) is not suitable for variable-length sequence
    pass


class SimpleFreeAttention(nn.Module):
    def __init__(self, dims: int):
        super().__init__()

        self.W_Q = nn.Linear(dims, dims, bias=False)
        self.W_K = nn.Linear(dims, dims, bias=False)
        self.W_V = nn.Linear(dims, dims, bias=False)


    def forward(self, input_seq_emb: torch.Tensor) -> torch.Tensor:
        Q: torch.Tensor = self.W_Q(input_seq_emb)
        K: torch.Tensor = self.W_K(input_seq_emb)
        V: torch.Tensor = self.W_V(input_seq_emb)

        # Q
        Q = torch.sigmoid(Q)

        # K
        K = torch.softmax(K, dim=-2)

        # Q \odot K \odot V
        output = Q * (K * V).sum(dim=-1, keepdim=True)

        return output