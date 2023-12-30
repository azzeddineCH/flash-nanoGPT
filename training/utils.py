from dataclasses import dataclass
from datetime import datetime

import jax
import jmp
from flax import struct
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


def get_time_string():
    current_time = datetime.now()
    time_string = current_time.strftime("%m_%d_%H_%M")
    return time_string
