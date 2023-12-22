from dataclasses import dataclass

import jax
from flax import struct
import jmp
from jax import numpy as jnp
from jmp._src.policy import _cast_floating_to


@struct.dataclass
class TrainMetrics:
    loss: jax.Array
    grads_gnorm: jax.Array = None
    params_gnorm: jax.Array = None


@dataclass(frozen=True)
class Policy(jmp.Policy):
    reduce_ops_dtype: jnp.dtype

    def cast_to_reduce_ops(self, x):
        return _cast_floating_to(x, self.reduce_ops_dtype)
