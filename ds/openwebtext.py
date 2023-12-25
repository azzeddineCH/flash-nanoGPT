import argparse
import multiprocessing
import os

import datasets
import numpy as np
import tensorflow as tf
import tiktoken

from ds.utils import make_tf_record_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="data")
    parser.add_argument("--num_valid_shards", type=int)
    parser.add_argument("--num_train_shards", type=int)
    parser.add_argument("--cache_dir", type=str)
    args = parser.parse_args()

    directory = os.path.join(args.directory, "openwebtext")
    os.makedirs(directory, exist_ok=True)

    num_workers = multiprocessing.cpu_count() // 2
    shards = dict(train=args.num_train_shards, val=args.num_valid_shards)

    print("loading ...")
    dataset = datasets.load_dataset(
        "openwebtext", num_proc=num_workers, cache_dir=args.cache_dir
    )
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    def encode(example):
        encoder = tiktoken.get_encoding("gpt2")
        ids = encoder.encode_ordinary(example["text"])
        ids.append(encoder.eot_token)
        out = {"ids": ids, "len": len(ids)}
        return out

    print("encoding ...")
    dataset = split_dataset.map(encode, remove_columns=["text"], num_proc=num_workers)

    print("saving ...")
    for split, dst in dataset.items():
        for i in range(shards[split]):
            file_path = os.path.join(directory, f"{split}_{i}.tfrecord")
            with tf.io.TFRecordWriter(file_path) as writer:
                shard_ds = dst.shard(shards[split], i)
                print(f"saving to {file_path} ...")
                for example in shard_ds["ids"]:
                    example = np.asarray(example, dtype=np.uint16)
                    example = make_tf_record_example(example)
                    writer.write(example.SerializeToString())

    encoder = tiktoken.get_encoding("gpt2")
    print(f"vocab size: {encoder.n_vocab}")


if __name__ == "__main__":
    main()
