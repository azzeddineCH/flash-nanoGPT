import os
import shutil

import jax.numpy as jnp
import requests
import argparse
import tensorflow as tf

from utils import upload_directory_with_transfer_manager, make_tf_record_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data")
    parser.add_argument("--gcs_bucket", default="flash-nano-gpt-bucket")
    args = parser.parse_args()

    directory = os.path.join(args.directory, "shakespeare-char")
    os.makedirs(directory, exist_ok=True)

    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    data = requests.get(url).text

    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    dataset = dict(
        train=jnp.array(encode(train_data), dtype=jnp.uint16),
        valid=jnp.array(encode(val_data), dtype=jnp.uint16)
    )

    for name, data in dataset.items():
        file_path = os.path.join(directory, f"{name}.tfrecord")
        with tf.io.TFRecordWriter(file_path) as writer:
            example = make_tf_record_example(data)
            writer.write(example.SerializeToString())

    if args.gcs_bucket:
        upload_directory_with_transfer_manager(args.gcs_bucket, directory)
        shutil.rmtree(args.directory)

    print(f"vocab size: {vocab_size:,}")


if __name__ == '__main__':
    main()
