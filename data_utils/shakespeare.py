import os
import tiktoken
import numpy as np
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir")
args = parser.parse_args()

data_dir = os.path.join(args.data_dir, "shakespeare")
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "input.txt")

if not os.path.exists(file_path):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(file_path, 'w') as f:
        data = requests.get(url)
        f.write(data.text)

with open(file_path, 'r') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

print(f"train: {len(train_ids)} tokens")
print(f"val: {len(val_ids)} tokens")
print(f"vocab size: {enc.n_vocab}")
