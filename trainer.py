from typing import Tuple

import optax

from config import Config
from model import GPTConfig, GPT
import jax

from utils import Batch, TrainState


class Trainer:

    def __init__(self, config: Config, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size
        self.training_step = jax.jit(self._training_step) if config.jit else self._training_step
        self.validation_step = jax.jit(self._validation_step) if config.jit else self._validation_step

    def _make_model(self):
        config = GPTConfig(
            block_size=self.config.block_size,
            vocab_size=self.vocab_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            embd_dim=self.config.embd_dim,
            dropout_rate=self.config.dropout_rate,
            use_bias=self.config.use_bias
        )

        key = jax.random.PRNGKey(0)
        random_input = jax.random.randint(key, (1, 8), minval=0, maxval=config.vocab_size)
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

        params = model_state["params"]
        num_params = sum([a.size for a in jax.tree_util.tree_leaves(params)])

        print(f"Train state created | model parameters : {num_params}")

        return TrainState.create(
            apply_fn=model.apply,
            params=model_state["params"],
            num_params=num_params,
            tx=optimizer
        )

    def _loss(self, rng_key, params, batch, apply_fn, train=True):
        logits = apply_fn({"params": params}, x=batch.inputs, train=train, rngs={"dropout": rng_key})
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits.reshape((-1, self.vocab_size)),
            labels=batch.labels.reshape(-1)
        ).mean()

        return loss

    def _training_step(self, rng_key: jax.Array, state: TrainState, batch: Batch) -> Tuple[TrainState, float]:
        batch_loss, grads = jax.value_and_grad(self._loss, argnums=1)(rng_key, state.params, batch, state.apply_fn)
        state = state.apply_gradients(grads=grads)

        return state, batch_loss

    def _validation_step(self, rng_key: jax.Array, state: TrainState, batch: Batch):
        batch_loss = self._loss(rng_key, state.params, batch, state.apply_fn, train=False)
        return batch_loss
