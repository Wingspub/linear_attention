from typing import Tuple
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from dataset.image_gen_dataset import MNISTSeqDataset
from model.original_transformer import OriginalTransformer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch

# dataset pre-process
train_data = MNIST(root="./dataset/data", train=True, download=True)
valid_data = MNIST(root="./dataset/data", train=False, download=True)

train_dataset = MNISTSeqDataset(train_data)
valid_dataset = MNISTSeqDataset(valid_data)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

# init
length = 1*28*28    # 784
token_num = 256 + 1  # <BOS> + pixel intensities
train_epoch_num = 10
loss_print_num = 100
eval_num = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## model
generator_ratio = 0.5   # mask part of image to generate the image
dims = 256
lr = 1e-4

model = OriginalTransformer(token_num=token_num, block_num=6, dims=dims, heads=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()


@torch.inference_mode()
def generate(model: nn.Module, data: torch.Tensor, mask_len: int, device: torch.device, writer: SummaryWriter | None = None, temp_step: int = 0) -> torch.Tensor:
    '''给定mask部分，生成后续部分'''
    length = data.shape[-1]
    mask_len = min(length, mask_len)

    response = torch.ones((1, length), dtype=torch.int32).to(device) * 255
    response[:, :mask_len] = data[:1, :mask_len]

    for i in range(mask_len, length):
        output_pred = model(response[:, :i])
        response[:, i] = torch.argmax(output_pred[:,-1], dim=-1)
        del output_pred

    # print the image
    if writer:
        writer.add_image("generation_image", response[:, 1:].reshape(28,28).to(torch.uint8), dataformats="HW", global_step=temp_step)
        writer.add_image("src_image", data[0, 1:].reshape(28,28).to(torch.uint8), dataformats="HW", global_step=temp_step)

    return response


@torch.inference_mode()
def eval(model: nn.Module, valid_dataloader: DataLoader, device: torch.device, writer: SummaryWriter | None = None, temp_step: int = 0) ->Tuple[float, float]:
    for valid_data in valid_dataloader:
        X = valid_data[:, :-1].detach().clone().to(device)
        Y = valid_data[:, 1:].detach().clone().to(device)

        y_pred = model(X)
        loss = loss_func(y_pred.reshape(-1, token_num), Y.reshape(-1))

        if writer:
            writer.add_scalar("Valid/loss", loss.cpu().item(), temp_step)

        # ACC
        acc = torch.mean((torch.argmax(y_pred, dim=-1) == Y).to(torch.float))
        if writer:
            writer.add_scalar("Valid/acc", acc, temp_step)

        # generate
        res = generate(model, valid_data, mask_len=392, device=device, writer=writer, temp_step=temp_step)

        break

    return loss.cpu().item(), acc

writer = SummaryWriter("logs")

# train
temp_step = 0
for _ in range(train_epoch_num):
    for image_data in train_dataloader:
        image_data: torch.Tensor

        X = image_data[:, :-1].detach().clone().to(device)
        Y = image_data[:, 1:].detach().clone().to(device)

        y_pred = model(X)

        loss = loss_func(y_pred.reshape(-1, token_num), Y.reshape(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if temp_step % loss_print_num == 0:
            print(f"{temp_step}: loss:{loss.cpu().item():.6f}")
        temp_step += 1

        writer.add_scalar("Train/loss", loss.cpu().item(), temp_step)

        # valid
        if (temp_step+1) % eval_num == 0:
            eval_loss, acc = eval(model, valid_dataloader, device, writer, temp_step)
            print(f"eval, loss:{eval_loss:.6f}, acc:{acc:.6f}")
        break

writer.close()