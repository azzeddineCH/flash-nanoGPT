import jax
from flax import struct
from flax.training.train_state import TrainState as _TrainState


@struct.dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array


class TrainState(_TrainState):
    num_params: int
