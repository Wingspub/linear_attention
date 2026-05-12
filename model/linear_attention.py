# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
# http://arxiv.org/abs/2006.16236

from torch import nn
import torch


class LinearAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CausalLinearAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LinearTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
