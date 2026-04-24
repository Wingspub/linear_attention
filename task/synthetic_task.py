import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from model.base import Base
from dataset.synthetic_dataset import SyntheticDataset

print("this is a synthetic task")

# config
TOKEN_num = 10
seq_len = 128
iter_num = 100
lr = 1e-4

# init
dataset = SyntheticDataset(TOKEN_num=TOKEN_num, seq_len=seq_len)
dataloader = DataLoader(dataset=dataset, batch_size=32, num_workers=2)
model = Base(TOKEN_num=TOKEN_num, dims=32)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_func = CrossEntropyLoss()

# Train
for seq in dataloader:
    seq: torch.Tensor
    model.train()
    print(seq)
    print(seq.shape)
    output: torch.Tensor = model(seq)
    print(output.shape)
    break

    # Eval
