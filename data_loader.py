import time
from functools import partial

import jax.random
import os
from jax import numpy as jnp
import numpy as np

from utils import Batch


class DataLoader:

    def __init__(self, dataset_dir, batch_size, block_size, split="train"):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.split = split
        self.data = self._load(self.dataset_dir, self.split)

    def _load(self, dataset_dir, split):
        file = os.path.join(dataset_dir, f"{split}.bin")
        data = np.memmap(file, dtype=np.uint16, mode='r')
        return jnp.array(data, dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def get(self, rng_key) -> Batch:
        idx = jax.random.randint(rng_key, shape=(self.batch_size,), minval=0, maxval=len(self.data) - self.block_size)

        x_idx = idx.repeat(self.block_size) + jnp.tile(jnp.arange(self.block_size), idx.size)
        y_idx = idx.repeat(self.block_size) + jnp.tile(jnp.arange(1, self.block_size + 1), idx.size)

        x = self.data[x_idx].reshape(-1, self.block_size)
        y = self.data[y_idx].reshape(-1, self.block_size)

        return Batch(inputs=x, labels=y)


if __name__ == '__main__':
    rng_key = jax.random.PRNGKey(0)
    loader = DataLoader(
        dataset_dir="data/shakespeare",
        batch_size=256,
        block_size=1024,
    )

    for i in range(100):
        t0 = time.time()
        loader.get(jax.random.fold_in(rng_key, i))
        t1 = time.time() - t0

        print(f"iter {i} | sampling time {t1}")
