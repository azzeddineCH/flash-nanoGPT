from typing import Iterator


class DataLoader:
    def __init__(
        self,
        directory: str,
        split: str,
        batch_size: int,
        block_size: int,
        num_shards: int = 1,
        shard: int = 0,
        num_workers: int = 2,
        buffer_size: int = 5,
        prefetch: int = 2,
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

        self.dataset = self._load()

    def _load(self):
        import tensorflow as tf

        from ds.utils import decode_tf_record_example

        file_ds = tf.data.Dataset.list_files(f"{self.directory}/{self.split}*.tfrecord")

        file_ds = file_ds.shard(num_shards=self.num_shards, index=self.shard)

        dataset = (
            tf.data.TFRecordDataset(
                # build a tf Dataset from data files
                filenames=file_ds,
                num_parallel_reads=self.num_workers,
                buffer_size=int(5e8),  # 500 MB
            )
            .map(
                # decode each of the tfrecords
                decode_tf_record_example,
                num_parallel_calls=self.num_workers,
            )
            .repeat(
                # repeat the dataset inf
            )
            .unbatch()
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
                yield make_batch(jnp.asarray(item))

        return jax_iterator()
