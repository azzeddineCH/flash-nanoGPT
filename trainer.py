from typing import Tuple
from jax.experimental import mesh_utils
import optax
from jax.random import PRNGKeyArray

from config import Config
from model import GPTConfig, GPT
import jax

from utils import Batch, TrainState, Policy, TrainMetrics
import jmp
from jax import numpy as jnp, tree_util as trx
from jax import sharding as shx
import os
import orbax.checkpoint as ocp


class Trainer:

    def __init__(self, config: Config, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size

        # ============= Mixed Precision Policy ============= #
        self.on_tpu = jax.local_devices()[0].platform == "tpu"

        self.policy = Policy(
            param_dtype=jnp.float32,
            compute_dtype=jmp.half_dtype() if self.config.amp else jnp.float32,
            output_dtype=jmp.half_dtype() if self.config.amp else jnp.float32,
            reduce_ops_dtype=jmp.half_dtype() if (self.on_tpu and self.config.amp) else jnp.float32,
        )

        # ============= Sharding Policy ============= #
        self.host = jax.devices("cpu")[0]

        self.device_mesh = self._make_device_mesh()

        self.train_data_sharding = shx.NamedSharding(
            self.device_mesh, spec=shx.PartitionSpec(None, "data")  # accumulation, batch, ...
        )
        self.valid_data_sharding = shx.NamedSharding(
            self.device_mesh, spec=shx.PartitionSpec("data")  # batch, ...
        )

        self.state_sharding = shx.NamedSharding(
            self.device_mesh, spec=shx.PartitionSpec()  # replicated
        )

        # ============= Jitting methods  ============= #
        self.training_step = jax.jit(self._training_step) if config.jit else self._training_step
        self.validation_step = jax.jit(self._validation_step) if config.jit else self._validation_step
        self.make_train_state = jax.jit(self._make_train_state) if config.jit else self._make_train_state

        # ============= Checkpointing ============= #
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.checkpointer = ocp.CheckpointManager(
            self.config.checkpoint_dir,
            checkpointers=dict(
                state=ocp.PyTreeCheckpointer(),
                train_metrics=ocp.PyTreeCheckpointer()
            ),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=3,
                best_fn=lambda metrics: -metrics["loss"],
            )
        )

    def _make_device_mesh(self):
        devices = jax.local_devices()
        assert len(devices) >= self.config.num_devices, \
            f"Not enough devices, requested {self.config.num_devices} found {len(devices)}"
        devices = devices[:self.config.num_devices]
        devices = mesh_utils.create_device_mesh((len(devices), 1), devices=devices)
        mesh = shx.Mesh(devices, axis_names=("data", "state"))
        return mesh

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

        return model, state["params"]

    def _make_optimizer(self, params):

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.lr,
            warmup_steps=self.config.lr_warmup_iters,
            decay_steps=self.config.lr_decay_iters,
            end_value=self.config.lr_min
        )

        # we use "optax.inject_hyperparams" in order to track the learning rate
        optimizer = optax.inject_hyperparams(optax.adamw)(
            learning_rate=schedule,
            b1=self.config.beta1,
            b2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            mask=trx.tree_map(lambda p: p.ndim >= 2, params)
        )

        optimizer = optax.chain(
            optax.clip(self.config.grad_clip),
            optimizer,
        )

        optimizer = optax.MultiSteps(
            optimizer,
            every_k_schedule=self.config.grad_accum_steps
        )

        return optimizer

    def _make_train_state(self):
        model, params = self._make_model()

        optimizer = self._make_optimizer(params)

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            loss_scale=jmp.DynamicLossScale(jnp.asarray(2.0 ** 15)) if self.config.amp else jmp.NoOpLossScale(),
            skip_infinite=self.config.skip_infinite
        )

        print(f"Train state created | model parameters : {state.num_params}")

        state = jax.device_put(state, self.state_sharding)

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

    def _update(self, rng_key: PRNGKeyArray, state: TrainState, batch: Batch) -> Tuple[TrainState, TrainMetrics]:
        params = self.policy.cast_to_compute(state.params)

        (_, loss), grads = jax.value_and_grad(self._loss, argnums=1, has_aux=True)(
            rng_key, params, state, batch
        )

        # before unscaling the grads, cast them into full precision to avoid non representative values post unscaling
        # this common when using GPUs float16 and less common when using TPUs bfloat16
        grads = self.policy.cast_to_param(grads)
        grads = state.loss_scale.unscale(grads)

        state = state.apply_gradients(grads=grads, skip_infinite=self.config.skip_infinite)

        metrics = TrainMetrics(loss=loss)

        return state, metrics

    def _training_step(self, rng_key: PRNGKeyArray, state: TrainState, batch: Batch) -> Tuple[TrainState, TrainMetrics]:
        batch = trx.tree_map(
            lambda x: x.reshape(
                self.config.grad_accum_steps,
                -1,
                self.config.block_size,
            ), batch
        )

        batch = jax.device_put(batch, self.train_data_sharding)
        rng_keys = jax.random.split(rng_key, self.config.grad_accum_steps)

        state, metrics = jax.lax.scan(
            f=lambda state, xs: self._update(
                rng_key=xs[0],
                batch=xs[1],
                state=state,
            ),
            init=state,
            xs=(rng_keys, batch)
        )

        metrics = trx.tree_map(lambda m: jnp.mean(m), metrics)
        metrics = jax.device_put(metrics, self.host)

        return state, metrics

    def _validation_step(self, rng_key: jax.Array, state: TrainState, batch: Batch):
        params = self.policy.cast_to_compute(state.params)
        batch = jax.device_put(batch, self.valid_data_sharding)

        loss = self._loss(rng_key, params, state, batch, train=False)
        loss = jax.device_put(loss, self.host)

        return loss

    def save(self, step: int, state: TrainState, metrics: TrainMetrics):
        saved = self.checkpointer.save(
            step,
            items=dict(
                state=state,
                train_metrics=metrics
            ),
            metrics=dict(loss=float(metrics.loss))
        )
        if saved:
            print(f"checkpoint saved ...{step}")

    def restore(self, step=None):
        if not step:
            step = self.checkpointer.best_step()

        ckpt = self.checkpointer.restore(
            step,
            items=dict(
                state=self.make_train_state(),
                train_metrics=TrainMetrics(loss=0)
            )
        )
        return ckpt["state"], step
