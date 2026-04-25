import torch
from typing import cast
from torch import optim
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from model.base import Base
from dataset.synthetic_dataset import SyntheticDataset

print("this is a synthetic task")

# config
TOKEN_num = 10
seq_len = 1024
iter_num = 100
lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## model
dims = 32
model = Base(TOKEN_num=TOKEN_num, dims=dims).to(device)


# init
dataset = SyntheticDataset(TOKEN_num=TOKEN_num, seq_len=seq_len)
dataloader = DataLoader(dataset=dataset, batch_size=32, num_workers=2)

optimizer = optim.SGD(model.parameters(), lr=lr)
loss_func = CrossEntropyLoss()


def train(model: Module, seq_data: torch.Tensor, device: torch.cuda.device) -> float:
    '''模型训练'''
    model.train()
    # source data and target data
    X = seq_data[:-1].detach().clone().to(device)
    Y = seq_data[1:].detach().clone().to(device)

    y_pred = cast(torch.Tensor, model(X))

    loss = cast(torch.Tensor, loss_func(y_pred.reshape(-1,TOKEN_num), Y.reshape(-1)))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.cpu().item()


temp_step = 0
for seq in dataloader:
    loss = train(model=model, seq_data=seq, device=device)
    print(f"step:{temp_step+1}, loss:{loss:.6f}")

    temp_step += 1
    if temp_step >= iter_num:
        break
