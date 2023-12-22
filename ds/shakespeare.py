import multiprocessing
import os
import shutil

import tiktoken
import numpy as np
import requests
import argparse
import tensorflow as tf

from ds.utils import make_tf_record_example, upload_directory_with_transfer_manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data")
    parser.add_argument("--gcs_bucket", default=None)
    args = parser.parse_args()

    directory = os.path.join(args.directory, "shakespeare")
    os.makedirs(directory, exist_ok=True)

    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    data = requests.get(url).text

    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    encoder = tiktoken.get_encoding("gpt2")
    dataset = dict(
        train=np.array(encoder.encode_ordinary(train_data), dtype=np.uint16),
        valid=np.array(encoder.encode_ordinary(val_data), dtype=np.uint16)
    )

    for name, data in dataset.items():
        file_path = os.path.join(directory, f"{name}.tfrecord")
        with tf.io.TFRecordWriter(file_path) as writer:
            example = make_tf_record_example(data)
            writer.write(example.SerializeToString())

    if args.bucket:
        upload_directory_with_transfer_manager(os.path.join(args.bucket, args.directory), directory)
        shutil.rmtree(args.directory)

    print(f"vocab size: {encoder.n_vocab}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
