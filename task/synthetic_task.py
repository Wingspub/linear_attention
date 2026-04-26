import torch
from typing import cast
from torch import optim
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from linear_attention.model.base_model import Base, MLP
from dataset.synthetic_dataset import SyntheticDataset
from time import time

print("this is a synthetic task")

# config
TOKEN_num = 10
seq_len = 1024
iter_num = 1000
print_num = 100
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## model
dims = 1024
# model = Base(TOKEN_num=TOKEN_num, dims=dims).to(device)
model = MLP(TOKEN_num=TOKEN_num, dims=dims).to(device)


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

@torch.inference_mode()
def generate(model: Module, seq_len: int=128) -> torch.Tensor:
    '''生成字符'''
    model.eval()
    start_token = torch.randint(0, TOKEN_num, size=(1,))
    response = torch.zeros(seq_len+1, dtype=torch.int32)
    response[0] = start_token

    start = time()
    for i in range(1, seq_len):
        input_token = response[:i].detach().clone().cuda()
        output_pred = model(input_token)
        response[i+1] = torch.argmax(output_pred[-1].cpu())

        # print(response)
    end = time()
    rate = seq_len / (end-start)
    print(f"speed:{rate:.2f} tokens/s")
    return response


temp_step = 0
for seq in dataloader:
    loss = train(model=model, seq_data=seq, device=device)

    if temp_step % print_num == 0: print(f"step:{temp_step+1}, loss:{loss:.6f}")

    temp_step += 1
    if temp_step >= iter_num:
        break

# 字符生成测试
for _ in range(5):
    res = generate(model, seq_len=128)
    print("res:", res)

print("over")