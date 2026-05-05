from torch.utils.data import Dataset

class MNISTSeqDataset(Dataset):
    def __init__(self):
        super().__init__()


    def __getitem__(self, index: int):
        pass

    def __len__(self):
        ...




class CIFAR10SeqDataset(Dataset):
    def __init__(self):
        super().__init__()


class ImageNetSeqDataset(Dataset):
    def __init__(self):
        super().__init__()