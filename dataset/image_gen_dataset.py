from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torch

class MNISTSeqDataset(Dataset):
    def __init__(self, data: MNIST):
        super().__init__()
        self.total_num, self.height, self.wide = data.data.shape
        self.data = data.data.reshape(self.total_num, -1)


    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.cat([torch.tensor([256]), self.data[index]])


    def __len__(self) -> int:
        return self.total_num




class CIFAR10SeqDataset(Dataset):
    def __init__(self):
        super().__init__()


class ImageNetSeqDataset(Dataset):
    def __init__(self):
        super().__init__()