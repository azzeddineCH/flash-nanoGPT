import time
from functools import partial

import jax.random
import os
from jax import numpy as jnp
import numpy as np

from utils import Batch


class DataLoader:
    VOCAB_SIZE = 50304

    def __init__(self, rng_key, dataset, batch_size, block_size, split="train"):
        self._current_batch = 0
        self.rng_key = rng_key
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_size = block_size
        self.split = split
        self.data = self._load(self.dataset, self.split)

    def _load(self, dataset, split):
        file = os.path.join("data", dataset, f"{split}.bin")
        data = np.memmap(file, dtype=np.uint16, mode='r')
        return jnp.array(data, dtype=jnp.int32)

    def __iter__(self):
        self._current_batch = 0
        return self

    @partial(jax.jit, static_argnums=(0,))
    def __next__(self) -> Batch:
        self._current_batch += 1

        def _sample_batch(i):
            key = jax.random.fold_in(self.rng_key, data=i)
            idx = jax.random.randint(key, shape=(self.batch_size,), minval=0, maxval=len(self.data) - self.block_size)

            x_idx = idx.repeat(self.block_size) + jnp.tile(jnp.arange(self.block_size), idx.size)
            y_idx = idx.repeat(self.block_size) + jnp.tile(jnp.arange(1, self.block_size + 1), idx.size)

            x = self.data[x_idx].reshape(-1, self.block_size)
            y = self.data[y_idx].reshape(-1, self.block_size)

            return Batch(inputs=x, labels=y)

        return _sample_batch(self._current_batch)


if __name__ == '__main__':
    loader = DataLoader(
        rng_key=jax.random.PRNGKey(0),
        dataset="shakespeare",
        batch_size=256,
        block_size=1024,
    )

    for i in range(100):
        t0 = time.time()
        next(loader)
        t1 = time.time() - t0

        print(f"iter {i} | sampling time {t1}")
