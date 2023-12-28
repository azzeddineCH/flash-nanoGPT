import os
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import jax
import jmp
import optax
import orbax.checkpoint as ocp
from flax.struct import PyTreeNode
from jax import numpy as jnp
from jax import sharding as shx
from jax import tree_util as trx
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.random import PRNGKeyArray

from config import Config, get_default_config
from ds.utils import Batch
from training.model import GPT, GPTConfig
from training.state import TrainState
from training.utils import Policy, TrainMetrics


class Trainer:
    def __init__(self, config: Config):
        self.config = config

        # ============= Mixed Precision Policy ============= #
        self.on_tpu = jax.local_devices()[0].platform == "tpu"

        self.policy = Policy(
            param_dtype=jnp.float32,
            compute_dtype=jmp.half_dtype() if self.config.amp else jnp.float32,
            output_dtype=jmp.half_dtype() if self.config.amp else jnp.float32,
            reduce_ops_dtype=jmp.half_dtype()
            if (self.on_tpu and self.config.amp)
            else jnp.float32,
        )

        # ============= Sharding Policy ============= #
        self.host = jax.devices("cpu")[0]

        self.device_mesh = self._make_device_mesh()

        self.train_data_sharding_spec = shx.PartitionSpec(
            None, "data"
        )  # accumulation, batch, ...
        self.valid_data_sharding_spec = shx.PartitionSpec("data")  # batch, ...
        self.replicated_specs = shx.PartitionSpec()  # replicated

        _sharded_update_loop = shard_map(
            f=self._update_loop,
            mesh=self.device_mesh,
            in_specs=(
                self.replicated_specs,  # rng_keys
                self.replicated_specs,  # state
                self.train_data_sharding_spec,  # batch
            ),
            out_specs=(
                self.replicated_specs,  # state
                self.replicated_specs,  # metrics
            ),
            check_rep=False,
        )

        _sharded_validation_loss = shard_map(
            f=self._validation_loss,
            mesh=self.device_mesh,
            in_specs=(
                self.replicated_specs,  # rng_keys
                self.replicated_specs,  # state
                self.valid_data_sharding_spec,  # batch
            ),
            out_specs=self.replicated_specs,
            check_rep=False,
        )

        # ============= Jitting methods  ============= #

        self.update_loop = (
            jax.jit(_sharded_update_loop) if config.jit else _sharded_update_loop
        )

        self.validation_loss = (
            jax.jit(_sharded_validation_loss)
            if config.jit
            else _sharded_validation_loss
        )

        # ============= Checkpointing ============= #
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.checkpointer = ocp.CheckpointManager(
            Path(self.config.checkpoint_dir).absolute(),
            checkpointers=dict(
                state=ocp.PyTreeCheckpointer(),
                train_metrics=ocp.PyTreeCheckpointer(),
                config=ocp.PyTreeCheckpointer(),
            ),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=3,
                best_fn=lambda metrics: metrics["loss"],
                best_mode="min",
            ),
        )

    def _make_device_mesh(self) -> shx.Mesh:
        devices = mesh_utils.create_device_mesh(
            mesh_shape=(jax.local_device_count(),), devices=jax.local_devices()
        )
        mesh = shx.Mesh(devices, axis_names=("data",))
        return mesh

    def _make_model(self) -> Tuple[GPT, PyTreeNode]:
        config = GPTConfig(
            block_size=self.config.block_size,
            vocab_size=self.config.vocab_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            embd_dim=self.config.embd_dim,
            dropout_rate=self.config.dropout_rate,
            use_bias=self.config.use_bias,
            reduce_ops_dtype=self.policy.reduce_ops_dtype,
        )

        key = jax.random.PRNGKey(0)
        random_input = jax.random.randint(
            key,
            shape=(2, 8),
            minval=0,
            maxval=config.vocab_size,
            dtype=jnp.int16 if self.config.amp else jnp.int32,
        )
        model = GPT(config)
        state = model.init(key, x=random_input, train=False)

        return model, state["params"]

    def _make_optimizer(self, params: PyTreeNode) -> optax.MultiSteps:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.lr,
            warmup_steps=self.config.lr_warmup_iters,
            decay_steps=self.config.lr_decay_iters,
            end_value=self.config.lr_min,
        )

        # we use "optax.inject_hyperparams" in order to track the learning rate
        optimizer = optax.inject_hyperparams(optax.adamw)(
            learning_rate=schedule,
            b1=self.config.beta1,
            b2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            mask=trx.tree_map(lambda p: p.ndim >= 2, params),
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optimizer,
        )

        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=self.config.grad_accum_steps
        )

        return optimizer

    def make_train_state(self) -> TrainState:
        if jax.process_index() == 0:
            print("Creating Train state ...")

        model, params = self._make_model()

        optimizer = self._make_optimizer(params)

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            loss_scale=jmp.DynamicLossScale(jnp.asarray(2.0**15, dtype=jnp.float32))
            if self.config.amp
            else jmp.NoOpLossScale(),
            skip_infinite=self.config.skip_infinite,
        )

        if jax.process_index() == 0:
            print(f"Train state created | model parameters : {state.num_params}")

        return state

    def _loss(
        self,
        rng_key: PRNGKeyArray,
        params: PyTreeNode,
        state: TrainState,
        batch: Batch,
        train: bool = True,
    ) -> Tuple[float, float]:
        """
        calculate the batch loss following these steps:

        1 - cast the input to half precision
        2 - forward pass
        3 - cast the output to full precision to handle softmax accumulation
        4 - cast the loss back to half precision
        5 - scale loss to avoid non representative grad values when backprop
        """

        logits = state.apply_fn(
            variables={"params": params},
            x=batch.inputs,
            train=train,
            rngs={"dropout": rng_key},
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=self.policy.cast_to_reduce_ops(logits), labels=batch.labels
        ).mean()

        loss = self.policy.cast_to_output(loss)
        if not train:
            return loss

        scaled_loss = state.loss_scale.scale(loss)
        return scaled_loss, loss

    def _validation_loss(self, _rng_key, _state, _batch):
        params = self.policy.cast_to_compute(_state.params)
        loss = self._loss(_rng_key, params, _state, _batch, train=False)
        return jax.lax.pmean(loss, axis_name="data")

    def _update(
        self, rng_key: PRNGKeyArray, state: TrainState, batch: Batch
    ) -> Tuple[TrainState, TrainMetrics]:
        params = self.policy.cast_to_compute(state.params)

        (_, loss), grads = jax.value_and_grad(self._loss, argnums=1, has_aux=True)(
            rng_key, params, state, batch
        )

        grads = self.policy.cast_to_param(grads)
        grads = state.loss_scale.unscale(grads)

        grads = jax.tree_util.tree_map(
            lambda g: jax.lax.pmean(g, axis_name="data"), grads
        )

        state = state.apply_gradients(
            grads=grads, skip_infinite=self.config.skip_infinite
        )

        metrics = TrainMetrics(
            loss=loss,
            grads_gnorm=optax.global_norm(grads),
            params_gnorm=optax.global_norm(params),
        )

        return state, metrics

    def _update_loop(self, rng_key: PRNGKeyArray, state: TrainState, batch: Batch):
        rng_keys = jax.random.split(rng_key, self.config.grad_accum_steps)
        state, metrics = jax.lax.scan(
            f=lambda state, xs: self._update(
                rng_key=xs[0],
                batch=xs[1],
                state=state,
            ),
            init=state,
            xs=(rng_keys, batch),
        )
        metrics = trx.tree_map(lambda m: jnp.mean(m), metrics)
        return state, metrics

    def training_step(
        self, rng_key: PRNGKeyArray, state: TrainState, batch: Batch
    ) -> Tuple[TrainState, TrainMetrics]:
        # ============= adding grad accumulation dim ============= #
        batch = trx.tree_map(
            lambda x: x.reshape(
                self.config.grad_accum_steps,
                -1,
                self.config.block_size,
            ),
            batch,
        )

        # ============= running update loop on grad_accum dim ============= #
        state, metrics = self.update_loop(rng_key, state, batch)
        metrics = jax.device_put(metrics, self.host)
        return state, metrics

    def validation_step(
        self, rng_key: PRNGKeyArray, state: TrainState, batch: Batch
    ) -> float:
        loss = self.validation_loss(rng_key, state, batch)

        loss = jax.device_put(loss, self.host)

        return loss

    def save(self, state: TrainState, metrics: TrainMetrics):
        state = jax.device_put(state, self.host)
        saved = self.checkpointer.save(
            state.step,
            items=dict(
                state=state,
                train_metrics=metrics,
                config=asdict(self.config),
            ),
            metrics=dict(loss=float(metrics.loss)),
        )
        if saved:
            print(f"checkpoint saved ...{state.step}")

    def restore(self) -> Tuple:
        ckpt = self.checkpointer.restore(
            self.checkpointer.best_step(),
            items=dict(
                state=self.make_train_state(),
                train_metrics=TrainMetrics(loss=0),
                config=get_default_config(),
            ),
        )

        state = ckpt["state"]
        loss = ckpt["train_metrics"].loss
        ckpt.pop("config")  # todo: handle different model config

        return state, loss

    def restore_openai_gpt(self):
        pass
