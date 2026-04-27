import torch
from typing import cast
from torch import optim
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from model.base_model import Base, MLP
from dataset.synthetic_dataset import Synthetic4RepetionDataset
from time import time

print("this is a synthetic copy task")

# config
TOKEN_num = 1+10 # 需要有一个token作为<BOS>
seq_len = 511
iter_num = 100000
print_num = 100
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## model
dims = 128
# model = Base(TOKEN_num=TOKEN_num, dims=dims).to(device)
model = MLP(TOKEN_num=TOKEN_num, dims=dims).to(device)

# init
dataset = Synthetic4RepetionDataset(TOKEN_num=TOKEN_num, src_seq_len=seq_len)
dataloader = DataLoader(dataset=dataset, batch_size=32, num_workers=2)

optimizer = optim.SGD(model.parameters(), lr=lr)
loss_func = CrossEntropyLoss()


def train(model: Module, seq_data: torch.Tensor, seq_len: int, device: torch.device) -> float:
    '''模型训练'''
    model.train()
    # source data and target data
    X = seq_data[:,:-1].detach().clone().to(device)
    Y = seq_data[:,-seq_len:].detach().clone().to(device)

    y_pred = cast(torch.Tensor, model(X))[:, -seq_len:]

    loss = cast(torch.Tensor, loss_func(y_pred.reshape(-1, TOKEN_num), Y.reshape(-1)))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.cpu().item()


@torch.inference_mode()
def generate(model: Module, src_seq: torch.Tensor, seq_len: int=128) -> torch.Tensor:
    '''生成字符'''
    model.eval()
    src_len = len(src_seq)
    assert src_len <= seq_len

    response = torch.zeros(seq_len+1, dtype=torch.int32)
    response[:src_len] = src_seq

    start = time()
    for i in range(src_len, seq_len):
        input_token = response[:i].detach().clone().cuda()
        output_pred = model(input_token)
        response[i+1] = torch.argmax(output_pred[-1].cpu())

    end = time()
    rate = seq_len / (end-start)
    print(f"speed:{rate:.2f} tokens/s")
    return response


@torch.inference_mode()
def eval(model: Module, seq_data: torch.Tensor, seq_len: int, device: torch.device) -> float:
    model.eval()

    input_seq = seq_data[0, :1+seq_len].detach().clone().cuda()
    res = generate(model=model, src_seq=input_seq, seq_len=(1+seq_len*2))

    # acc
    target_seq = seq_data[0, 1+seq_len:].detach().clone()
    hat_seq = res[1+seq_len:].detach().clone()
    acc = torch.sum(target_seq == hat_seq).item()/seq_len

    return acc

temp_step = 0
for seq in dataloader:
    loss = train(model=model, seq_data=seq, seq_len=seq_len, device=device)

    if temp_step % print_num == 0: print(f"step:{temp_step+1}, loss:{loss:.6f}")

    if temp_step % 1000 == 0:
        acc = eval(model=model, seq_data=seq, seq_len=seq_len, device=device)
        print(f"acc:{acc:.6f}")

    temp_step += 1
    if temp_step >= iter_num:
        break
