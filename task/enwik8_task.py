from typing import Tuple, cast
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch import optim
from time import time
from dataset.enwik8_dataset import Enwik8Dataset
from model.simplest_transformer import SimplestTransformer
from model.original_transformer import OriginalTransformer
import torch
from torch.utils.tensorboard import SummaryWriter

print("this is a text fitting task")

# Tool Class and Tool Function

class Enwik8Tokenizer(object):
    def __init__(self):
        pass

    def decode(self, text: bytes) -> str:
        return text.decode('utf-8', errors="replace")


def enwik8_read(train_spilt_rate: float) -> Tuple[torch.Tensor, torch.Tensor, int]:
    import gzip
    import numpy as np
    assert train_spilt_rate > 0 and train_spilt_rate < 1.0

    f = gzip.open("./dataset/data/enwik8.gz", 'rb')
    bytes_text = np.frombuffer(f.read(), dtype=np.uint8)
    token_num = max(np.unique(bytes_text)) + 1

    train_spilt_index = int(len(bytes_text) * train_spilt_rate)
    train_data = torch.from_numpy(bytes_text[:train_spilt_index].copy())
    valid_data = torch.from_numpy(bytes_text[train_spilt_index:].copy())

    return train_data, valid_data, token_num

# config
train_vaild_spilt_rate = 0.9
SEQ_LEN = 768
GEN_LEN = 128
iter_num = 100000
loss_print_num = 100
eval_num = 1000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

## model
dims = 512
lr = 5e-4

# init
tokenizer = Enwik8Tokenizer()
train_text, valid_text, token_num = enwik8_read(train_vaild_spilt_rate)

train_dataset = Enwik8Dataset(train_text, seq_len=SEQ_LEN)
valid_dataset = Enwik8Dataset(valid_text, seq_len=SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=2)
valid_dataset = DataLoader(valid_dataset, batch_size=8, num_workers=2)

# model
# model = SimplestTransformer(Token_num=token_num, layers_num=5, dims=dims, device=device).to(device)
model = OriginalTransformer(token_num=token_num, block_num=6, dims=dims, heads=8).to(device)
torch.set_float32_matmul_precision('high')
model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = CrossEntropyLoss()


def train(model: Module, seq_data: torch.Tensor, device: torch.device) -> float:
    '''模型训练'''
    model.train()
    # source data and target data
    X = seq_data[:,:-1].detach().clone().to(device)
    Y = seq_data[:,1:].detach().clone().to(device)

    y_pred = cast(torch.Tensor, model(X))

    loss = cast(torch.Tensor, loss_func(y_pred.reshape(-1, token_num), Y.reshape(-1)))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    optimizer.zero_grad()

    return loss.cpu().item()


@torch.inference_mode()
def generate(model: Module, src_seq: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    '''生成字符'''
    model.eval()
    batch, src_len = src_seq.shape
    assert src_len <= seq_len

    response = torch.zeros((batch, seq_len), dtype=torch.int32).to(device)
    response[:, :src_len] = src_seq


    start = time()
    for i in range(src_len, seq_len):
        output_pred = model(response[:, :i])
        response[:, i] = torch.argmax(output_pred[:,-1], dim=-1)
        del output_pred

    end = time()
    rate = seq_len / (end-start)
    print(f"speed:{rate:.2f} tokens/s")
    return response


@torch.inference_mode()
def eval(model: Module, seq_data: torch.Tensor, device: torch.device) -> Tuple[float, str]:
    model.eval()
    # CE loss
    X = seq_data[:,:-1].detach().clone().to(device)
    Y = seq_data[:,1:].detach().clone().to(device)

    y_pred = cast(torch.Tensor, model(X))
    loss = cast(torch.Tensor, loss_func(y_pred.reshape(-1, token_num), Y.reshape(-1)))

    # generate
    gen_bytes = generate(model=model, src_seq=X, seq_len=SEQ_LEN+GEN_LEN, device=device)[0]
    gen_bytes_text = bytes([c.item() for c in gen_bytes])
    gen_text = tokenizer.decode(gen_bytes_text)

    return loss, gen_text


# record
writer = SummaryWriter("logs")

temp_step = 0
for data in train_dataloader:
    if temp_step % eval_num == 0:
        for valid_data in valid_dataset:
            valid_loss, gen_text = eval(model=model, seq_data=data, device=device)
            print(f"valid_loss:{valid_loss:.6f}")
            print(f"gen_text:\n{gen_text}")
            break
        writer.add_scalar("Valid/loss", valid_loss, temp_step)

    loss = train(model=model, seq_data=data, device=device)
    writer.add_scalar("Train/loss", loss, temp_step)

    if temp_step % loss_print_num == 0: print(f"step:{temp_step+1}, loss:{loss:.6f}")

    temp_step += 1
    if temp_step >= iter_num:
        break

