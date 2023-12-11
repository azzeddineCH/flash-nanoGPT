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
        return data

    def __iter__(self):
        self._current_batch = 0
        return self

    def __next__(self) -> Batch:
        self._current_batch += 1

        key = jax.random.fold_in(self.rng_key, data=self._current_batch)
        idx = jax.random.randint(key, shape=(self.batch_size,), minval=0, maxval=len(self.data) - self.block_size)

        x = jnp.stack([jnp.asarray(self.data[i:i + self.block_size], dtype=jnp.int32) for i in idx])
        y = jnp.stack([jnp.asarray(self.data[i+1:i + self.block_size + 1], dtype=jnp.int32) for i in idx])

        return Batch(inputs=x, labels=y)


if __name__ == '__main__':
    loader = DataLoader(
        rng_key=jax.random.PRNGKey(0),
        dataset="shakespeare",
        batch_size=8,
        block_size=16,
    )

    x, y = next(loader)
    print(x.shape, y.shape)
