from typing import Iterator

import tensorflow as tf

from ds.utils import decode_tf_record_example


class DataLoader:
    def __init__(
        self,
        directory: str,
        split: str,
        batch_size: int,
        block_size: int,
        num_shards: int = 1,
        shard: int = 0,
        num_workers: int = tf.data.AUTOTUNE,
        buffer_size: int = 5,
        prefetch: int = 2,
        documents_dataset: bool = False,
    ):
        self.directory = directory
        self.split = split

        self.batch_size = batch_size
        self.block_size = block_size
        self.num_shards = num_shards
        self.shard = shard
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.documents_dataset = documents_dataset
        self.dataset = self._load()

    def _load(self):
        file_ds = tf.data.Dataset.list_files(f"{self.directory}/{self.split}*.tfrecord")

        file_ds = file_ds.shard(num_shards=self.num_shards, index=self.shard)

        dataset = (
            tf.data.TFRecordDataset(
                # build a tf Dataset from data files
                filenames=file_ds,
                num_parallel_reads=self.num_workers,
                buffer_size=int(1e8),
            )
            .map(
                # decode each of the tfrecords
                decode_tf_record_example,
                num_parallel_calls=self.num_workers,
            )
            .repeat(
                # repeat the dataset inf
            )
        )

        if self.documents_dataset:
            # shuffle documents as it is the case for openwebtext dataset
            dataset = dataset.shuffle(self.buffer_size)

        dataset = (
            dataset.unbatch()
            .batch(
                # un-batch the blocks to form a single sequence then
                # batch it by block_size + 1 ( add one to consider the last prediction)
                self.block_size + 1,
                drop_remainder=True,
            )
            .shuffle(self.buffer_size)
            .batch(
                # batch the dataset to get the shape of (batch_size, block_size)
                self.batch_size,
                drop_remainder=True,
            )
            .shuffle(self.buffer_size)
            .prefetch(
                # prefetch the next N batches while training running on the accelerator
                self.prefetch
            )
        )

        return dataset

    def get_iterator(self) -> Iterator:
        import jax
        from jax import numpy as jnp

        from ds.utils import Batch

        @jax.jit
        def make_batch(item):
            x, y = item[:, :-1], item[:, 1:]
            return Batch(inputs=x, labels=y)
            return item

        def jax_iterator():
            for item in self.dataset:
                yield make_batch(jnp.array(item, dtype=jnp.int32))

        return jax_iterator()
