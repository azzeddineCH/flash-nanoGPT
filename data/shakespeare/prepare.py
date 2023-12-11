"""
train.bin has 301,966 tokens
val.bin has 36,059 tokens
"""
import os
import tiktoken
import numpy as np
import requests

file_path = "input.txt"
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

train_ids.tofile('train.bin')
val_ids.tofile('val.bin')

print(f"train has {len(train_ids)} tokens")
print(f"val has {len(val_ids)} tokens")
print(f"vocab size {enc.n_vocab}")
