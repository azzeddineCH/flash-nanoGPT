from dataclasses import dataclass

import jax
from flax import struct
from flax.training.train_state import TrainState as _TrainState
import jmp
import optax
from jax import numpy as jnp
from jmp._src.policy import _cast_floating_to


@struct.dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array


@struct.dataclass
class TrainMetrics:
    loss: jax.Array
    grads_gnorm: jax.Array = None


class TrainState(_TrainState):
    num_params: int
    loss_scale: jmp.LossScale
    skip_infinite: bool = True

    def apply_gradients(self, *, grads, skip_infinite=True, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        loss_scale = self.loss_scale
        if skip_infinite:
            # handle infinite grads:
            # 1 - check if there is no overflow in grads tree
            # 2 - adjust the loss scale based on that
            # 3 - for each weight, if the corresponding grad is inf, skip the update else return the new weight
            grads_finite = jmp.all_finite(grads)
            loss_scale = self.loss_scale.adjust(grads_finite)
            new_params, new_opt_state = jmp.select_tree(
                grads_finite,
                (new_params, new_opt_state),
                (self.params, self.opt_state)
            )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            loss_scale=loss_scale,
            **kwargs,
        )

    @property
    def lr(self):
        return self.opt_state.inner_opt_state[1].hyperparams["learning_rate"]

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        num_params = sum([a.size for a in jax.tree_util.tree_leaves(params)])
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            num_params=num_params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


@dataclass(frozen=True)
class Policy(jmp.Policy):
    reduce_ops_dtype: jnp.dtype

    def cast_to_reduce_ops(self, x):
        return _cast_floating_to(x, self.reduce_ops_dtype)
