import torch
from torch.utils.data import IterableDataset


class Synthetic4RepetionDataset(IterableDataset):
    '''人工合成的重复数据'''
    def __init__(self, TOKEN_num: int, src_seq_len: int):
        super().__init__()
        self.Token_num = TOKEN_num                      # 有一个额外的token作为<BOS>，默认为0
        self.src_seq_len = src_seq_len
        self.seq_len = 1 + src_seq_len + 1 + src_seq_len    # [<BOS>, src, <BOS>, src]

    def __iter__(self):
        while True:
            src_data = torch.randint(1, self.Token_num, size=(self.src_seq_len,))
            seq = torch.concat([torch.tensor([0]), src_data, torch.tensor([0]), src_data])
            yield seq
