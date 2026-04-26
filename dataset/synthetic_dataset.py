import torch
from torch.utils.data import IterableDataset

class SyntheticDataset(IterableDataset):
    '''人工合成数据'''
    def __init__(self, TOKEN_num :int, seq_len :int):
        self.TOEKN_num = TOKEN_num
        self.seq_len = seq_len

    def __iter__(self):
        while True:
            seq = torch.randint(0, self.TOEKN_num, size=(self.seq_len,))
            # seq = torch.arange(0, self.seq_len) % self.TOEKN_num
            yield seq



