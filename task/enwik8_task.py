from typing import Tuple, List, cast
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim, nn
from time import time
from dataset.enwik8_dataset import Enwik8Dataset
from model.simplest_transformer import SimplestTransformer
from model.original_transformer import OriginalTransformer
from model.morden_transformer import MordenTransformer
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

print("this is a text fitting task")

# Tool Class and Tool Function

class Enwik8Tokenizer(object):
    def __init__(self, text_data):
        pass
        self.c2id = {}
        self.id2c = {}
        data_set = list(set(text_data))
        for id, c in enumerate(data_set):
            self.c2id[c] = id
            self.id2c[id] = c
        print(f"loding tokenizer is done. vocab num: {len(data_set)}")

    @property
    def vocab_num(self) -> int:
        return len(self.c2id.items())

    def encode(self, text: np.ndarray) -> np.ndarray:
        return np.array([self.c2id[c] for c in text])


    def decode(self, text: List[int]) -> str:
        c_list = bytes([self.id2c[id] for id in text])
        return c_list.decode('utf-8', errors="replace")


def enwik8_read(train_spilt_rate: float) -> Tuple[torch.Tensor, torch.Tensor, Enwik8Tokenizer]:
    import gzip
    assert train_spilt_rate > 0 and train_spilt_rate < 1.0

    f = gzip.open("./dataset/data/enwik8.gz", 'rb')
    bytes_text = np.frombuffer(f.read(), dtype=np.uint8)
    tokenizer = Enwik8Tokenizer(text_data=bytes_text)
    text = tokenizer.encode(bytes_text)
    train_spilt_index = int(len(bytes_text) * train_spilt_rate)
    train_data = torch.from_numpy(text[:train_spilt_index].copy())
    valid_data = torch.from_numpy(text[train_spilt_index:].copy())

    return train_data, valid_data, tokenizer

# config
train_vaild_spilt_rate = 0.9
SEQ_LEN = 256
GEN_LEN = 128
iter_num = 200000
loss_print_num = 100
eval_num = 1000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

## model
dims = 512
lr = 5e-5

# init
train_text, valid_text, tokenizer = enwik8_read(train_vaild_spilt_rate)
vocab_num = tokenizer.vocab_num

train_dataset = Enwik8Dataset(train_text, seq_len=SEQ_LEN)
valid_dataset = Enwik8Dataset(valid_text, seq_len=SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=2)
valid_dataset = DataLoader(valid_dataset, batch_size=4, num_workers=2)

# model
model = SimplestTransformer(vocab_num=vocab_num, layers_num=6, dims=dims).to(device)
# model = OriginalTransformer(vocab_num=vocab_num, block_num=6, dims=dims, heads=4).to(device)
# model = MordenTransformer(token_num=vocab_num, block_num=6, dims=dims, heads=4).to(device)
torch.set_float32_matmul_precision('high')
# model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = CrossEntropyLoss()


def train(model, seq_data: torch.Tensor, device: torch.device) -> float:
    '''模型训练'''
    model.train()
    # source data and target data
    seq_data = seq_data.to(device)
    X = seq_data[:, :-1]
    Y = seq_data[:, 1:]

    y_pred = cast(torch.Tensor, model(X))   # [B, L, token_num]

    # loss = cast(torch.Tensor, loss_func(y_pred.reshape(-1, token_num), Y.reshape(-1)))
    loss = cast(torch.Tensor, loss_func(y_pred.transpose(1, 2), Y))
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    optimizer.zero_grad()

    return loss.cpu().item()


def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


@torch.inference_mode()
def generate(
    model,
    src_seq: torch.Tensor,
    seq_len: int,
    device: torch.device,
    temperature: float = 1.,
    filter_logits_fn = top_k,
    filter_thres: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''生成字符'''
    model.eval()
    batch, src_len = src_seq.shape
    assert src_len <= seq_len

    gen_len = seq_len - src_len
    sub_len = src_len - gen_len

    response = torch.zeros((batch, seq_len), dtype=torch.int32).to(device)
    response[:, :sub_len] = src_seq[:, :sub_len]
    start = time()
    for i in range(sub_len, seq_len):
        output_pred = model(response[:, :i])[:, -1, :]
        # response[:, i] = torch.argmax(output_pred, dim=-1)
        filtered_logits = filter_logits_fn(output_pred, thres = filter_thres)
        probs = nn.functional.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)
        response[:, i] = sample.squeeze(1)
        del output_pred

    end = time()
    rate = (seq_len-sub_len) / (end-start)
    print(f"speed:{rate:.2f} tokens/s")
    return src_seq, response


@torch.inference_mode()
def eval(model, seq_data: torch.Tensor, device: torch.device) -> Tuple[float, str, str]:
    model.eval()
    # CE loss
    seq_data = seq_data.to(device)
    X = seq_data[:, :-1]
    Y = seq_data[:, 1:]

    y_pred = cast(torch.Tensor, model(X))
    loss = cast(torch.Tensor, loss_func(y_pred.reshape(-1, vocab_num), Y.reshape(-1)))

    # generate
    src_bytes, gen_bytes = generate(model=model, src_seq=seq_data, seq_len=SEQ_LEN+GEN_LEN, device=device)
    src_bytes_text, gen_bytes_text = [c.item() for c in src_bytes[0]], [c.item() for c in gen_bytes[0]]
    src_text, gen_text = tokenizer.decode(src_bytes_text), tokenizer.decode(gen_bytes_text)

    return loss.item(), src_text, gen_text


# record
writer = SummaryWriter("logs")

temp_step = 0
for data in train_dataloader:
    if (temp_step+1) % eval_num == 0:
        loss = []
        flag = True
        for valid_data in valid_dataset:
            valid_loss, src_text, gen_text = eval(model=model, seq_data=valid_data, device=device)
            loss.append(valid_loss)
            if flag:
                print(f"[src_text]:\n{src_text}")
                print(f"[gen_text]:\n{gen_text}")
                flag = False
            else:
                continue
        valid_mean = np.mean(valid_loss)
        print(f"valid_loss:{valid_mean:.6f}")
        writer.add_scalar("Valid/loss", valid_mean, temp_step+1)

    loss = train(model=model, seq_data=data, device=device)
    writer.add_scalar("Train/loss", loss, temp_step+1)

    if temp_step % loss_print_num == 0: print(f"step:{temp_step}, loss:{loss:.6f}")

    temp_step += 1
    if temp_step >= iter_num:
        break

writer.close()