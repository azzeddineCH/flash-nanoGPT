import os
import numpy as np
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
args = parser.parse_args()

data_dir = os.path.join(args.data_dir, "shakespeare-char")
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "input.txt")

if not os.path.exists(file_path):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(file_path, 'w') as f:
        data = requests.get(url)
        f.write(data.text)

with open(file_path, 'r') as f:
    data = f.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

train_ids = encode(train_data)
val_ids = encode(val_data)

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")
