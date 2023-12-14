from typing import Tuple

import optax
from jax.random import PRNGKeyArray

from config import Config
from model import GPTConfig, GPT
import jax

from utils import Batch, TrainState, Policy, TrainMetrics
import jmp
from jax import numpy as jnp


class Trainer:

    def __init__(self, config: Config, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size
        self.on_tpu = jax.local_devices()[0].platform == "tpu"

        self.training_step = jax.jit(self._training_step) if config.jit else self._training_step
        self.validation_step = jax.jit(self._validation_step) if config.jit else self._validation_step
        self.policy = Policy(
            param_dtype=jnp.float32,
            compute_dtype=jmp.half_dtype() if self.config.amp else jnp.float32,
            output_dtype=jmp.half_dtype() if self.config.amp else jnp.float32,
            reduce_ops_dtype=jmp.half_dtype() if (self.on_tpu and self.config.amp) else jnp.float32,
        )

    def _make_model(self):
        config = GPTConfig(
            block_size=self.config.block_size,
            vocab_size=self.vocab_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            embd_dim=self.config.embd_dim,
            dropout_rate=self.config.dropout_rate,
            use_bias=self.config.use_bias,
            reduce_ops_dtype=self.policy.reduce_ops_dtype
        )

        key = jax.random.PRNGKey(0)
        random_input = jax.random.randint(
            key,
            shape=(1, 8),
            minval=0,
            maxval=config.vocab_size,
            dtype=jnp.int16 if self.config.amp else jnp.int32
        )
        model = GPT(config)
        state = model.init(key, x=random_input, train=False)

        return model, state

    def _make_optimizer(self, params):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.config.lr_min,
            peak_value=self.config.lr,
            warmup_steps=self.config.lr_warmup_iters,
            decay_steps=self.config.lr_decay_iters,
            end_value=self.config.lr_min
        )
        optimizer = optax.adamw(
            learning_rate=schedule,
            b1=self.config.beta1,
            b2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            mask=jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)
        )

        return optimizer

    def make_train_state(self):
        model, model_state = self._make_model()
        params = model_state["params"]

        optimizer = self._make_optimizer(params)

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            loss_scale=jmp.DynamicLossScale(jnp.asarray(2.0 ** 15)) if self.config.amp else jmp.NoOpLossScale(),
            skip_infinite=self.config.skip_infinite
        )

        print(f"Train state created | model parameters : {state.num_params}")

        return state

    def _loss(self, rng_key: PRNGKeyArray, params, state: TrainState, batch: Batch, train: bool = True):
        """
        calculate the batch loss following these steps:

        1 - cast the input to half precision
        2 - forward pass
        3 - cast the output to full precision to handle softmax accumulation
        4 - cast the loss back to half precision
        5 - scale loss to avoid non representative grad values when backprop
        """

        logits = state.apply_fn({"params": params}, x=batch.inputs, train=train, rngs={"dropout": rng_key})
        logits = self.policy.cast_to_reduce_ops(logits)

        loss = optax.softmax_cross_entropy(
            logits=logits.reshape((-1, self.vocab_size)),
            labels=jax.nn.one_hot(batch.labels.reshape(-1), num_classes=self.vocab_size, dtype=logits.dtype)
        ).mean()

        if not train:
            return loss

        # scale the loss before casting to half precision to avoid losing precision
        scaled_loss = state.loss_scale.scale(loss)
        scaled_loss = self.policy.cast_to_compute(scaled_loss)
        return scaled_loss, loss

    def _training_step(self, rng_key: PRNGKeyArray, state: TrainState, batch: Batch) -> Tuple[TrainState, TrainMetrics]:
        params = self.policy.cast_to_compute(state.params)
        (_, loss), grads = jax.value_and_grad(self._loss, argnums=1, has_aux=True)(
            rng_key, params, state, batch
        )

        # before unscaling the grads, cast them into full precision to avoid non representative values post unscaling
        # this common when using GPUs float16 and less common when using TPUs bfloat16
        grads = self.policy.cast_to_param(grads)
        grads = state.loss_scale.unscale(grads)

        state = state.apply_gradients(grads=grads, skip_infinite=self.config.skip_infinite)

        metrics = TrainMetrics(
            loss=loss,
            all_finite_grads=jmp.all_finite(grads)
        )

        return state, metrics

    def _validation_step(self, rng_key: jax.Array, state: TrainState, batch: Batch):
        params = self.policy.cast_to_compute(state.params)
        loss = self._loss(rng_key, params, state, batch, train=False)
        return loss
