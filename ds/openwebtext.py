import argparse
import shutil

import tiktoken
import datasets
import os
import tensorflow as tf
import numpy as np
from utils import upload_directory_with_transfer_manager, make_tf_record_example
import multiprocessing


def main():
    encoder = tiktoken.get_encoding("gpt2")

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data")
    parser.add_argument("--num_valid_shards")
    parser.add_argument("--num_train_shards")
    parser.add_argument("--gcs_bucket", default="flash-nano-gpt-bucket")
    args = parser.parse_args()

    directory = os.path.join(args.directory, "openwebtext")
    os.makedirs(directory, exist_ok=True)

    num_workers = multiprocessing.cpu_count() // 2
    shards = dict(train=args.num_train_shards, val=args.num_valid_shards)

    dataset = datasets.load_dataset("openwebtext", num_proc=num_workers)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def encode(example):
        ids = encoder.encode_ordinary(example['text'])
        ids.append(encoder.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    dataset = split_dataset.map(
        encode,
        remove_columns=['text'],
        num_proc=num_workers,
    )

    for split, dst in dataset.items():
        for i in range(shards[split]):
            file_path = os.path.join(directory, f"{split}_{i}.tfrecord")
            with tf.io.TFRecordWriter(file_path) as writer:
                shard_ds = dst.shard(shards[split], i)
                for example in shard_ds["ids"]:
                    example = np.asarray(example, dtype=np.uint16)
                    example = make_tf_record_example(example)
                    writer.write(example.SerializeToString())

    if args.gcs_bucket:
        upload_directory_with_transfer_manager(args.gcs_bucket, directory)
        shutil.rmtree(args.directory)

    print(f"vocab size: {encoder.n_vocab}")


if __name__ == '__main__':
    main()
