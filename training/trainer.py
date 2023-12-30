import logging
import os
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

from config import Config
from ds.utils import Batch
from training.model import GPT, GPTConfig
from training.state import DynamicLossScale, TrainState
from training.utils import Policy, TrainMetrics


class Trainer:
    def __init__(self, config: Config):
        self.config = config

        # ============= Mixed Precision Policy ============= #
        self.on_tpu = jax.local_devices()[0].platform == "tpu"

        if self.config.amp and self.on_tpu:
            # when training on TPUs with bflaot16 there is no need for mixed precision
            # as each multiply-accumulate operation in a matrix multiplication
            # uses bfloat16 for the multiplication and 32-bit IEEE floating point for accumulation.
            # check: https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
            self.policy = Policy(
                param_dtype=jmp.half_dtype(),
                compute_dtype=jmp.half_dtype(),
                output_dtype=jmp.half_dtype(),
                reduce_ops_dtype=jmp.half_dtype(),
            )
        else:
            self.policy = Policy(
                param_dtype=jnp.float32,
                compute_dtype=jmp.half_dtype() if self.config.amp else jnp.float32,
                output_dtype=jmp.half_dtype() if self.config.amp else jnp.float32,
                reduce_ops_dtype=jnp.float32,
            )

        # ============= Sharding Policy ============= #

        self.device_mesh = self._make_device_mesh()

        self.data_sharding_spec = shx.PartitionSpec("data")  # batch, ...
        self.replicated_specs = shx.PartitionSpec()  # replicated

        self.update = shard_map(
            f=self._update,
            mesh=self.device_mesh,
            in_specs=(
                self.data_sharding_spec,  # rng_keys
                self.data_sharding_spec,  # batch
                self.replicated_specs,  # state
            ),
            out_specs=(
                self.replicated_specs,  # state
                self.replicated_specs,  # metrics
            ),
            check_rep=False,
        )

        _sharded_eval_step = shard_map(
            f=self._eval_step,
            mesh=self.device_mesh,
            in_specs=(
                self.replicated_specs,  # rng_keys (rng keys are ignored in eval)
                self.replicated_specs,  # state
                self.data_sharding_spec,  # batch
            ),
            out_specs=self.replicated_specs,
            check_rep=False,
        )

        # ============= Jitting methods  ============= #

        self.eval_step = (
            jax.jit(_sharded_eval_step) if config.jit else _sharded_eval_step
        )

        self.training_step = (
            jax.jit(self._training_step) if config.jit else self._training_step
        )

        # ============= Checkpointing ============= #
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.checkpointer = ocp.CheckpointManager(
            Path(self.config.checkpoint_dir).absolute(),
            checkpointers=dict(
                state=ocp.PyTreeCheckpointer(),
                train_metrics=ocp.PyTreeCheckpointer(),
            ),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=2,
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

    def _make_model(self, params_key: PRNGKeyArray) -> Tuple[GPT, PyTreeNode]:
        config = GPTConfig(
            block_size=self.config.block_size,
            vocab_size=self.config.vocab_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            embd_dim=self.config.embd_dim,
            dropout_rate=self.config.dropout_rate,
            use_bias=self.config.use_bias,
            reduce_ops_dtype=self.policy.reduce_ops_dtype,
            param_dtype=self.policy.param_dtype,
        )

        random_input = jax.random.randint(
            params_key,
            shape=(2, 8),
            minval=0,
            maxval=config.vocab_size,
            dtype=jnp.int32,
        )

        model = GPT(config)
        state = model.init(params_key, x=random_input, train=False)

        return model, state["params"]

    def _make_optimizer(self, params: PyTreeNode) -> optax.MultiSteps:
        # ============= learning rate schedular ============= #
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.lr,
            warmup_steps=self.config.lr_warmup_iters,
            decay_steps=self.config.lr_decay_iters,
            end_value=self.config.lr_min,
        )

        # ============= use "inject_hyperparams" in order to track the learning rate ============= #
        optimizer = optax.inject_hyperparams(optax.adamw)(
            learning_rate=schedule,
            b1=self.config.beta1,
            b2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            mask=trx.tree_map(lambda p: p.ndim >= 2, params),
        )

        # ============= gradient global norm clipping ============= #
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optimizer,
        )

        # ============= gradient accumulation ============= #
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=self.config.grad_accum_steps
        )

        return optimizer

    def make_train_state(self, state_key: PRNGKeyArray) -> TrainState:
        model, params = self._make_model(state_key)

        optimizer = self._make_optimizer(params)

        scale = jmp.NoOpLossScale()
        if self.config.amp and not self.on_tpu:
            scale = DynamicLossScale(jnp.asarray(2.0**15, dtype=jnp.float32))

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            loss_scale=scale,
            skip_infinite=self.config.skip_infinite,
        )

        return state

    def _loss(
        self,
        dropout_key: PRNGKeyArray,
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
            rngs={"dropout": dropout_key},
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=self.policy.cast_to_reduce_ops(logits), labels=batch.labels
        ).mean()

        loss = self.policy.cast_to_output(loss)
        if not train:
            return loss

        scaled_loss = state.loss_scale.scale(loss)
        return scaled_loss, loss

    def _update(
        self, update_key: PRNGKeyArray, batch: Batch, state: TrainState
    ) -> Tuple[TrainState, TrainMetrics]:
        # ============= cast params to half precision for forward and back propagation ============= #
        params = self.policy.cast_to_compute(state.params)
        (_, loss), grads = jax.value_and_grad(self._loss, argnums=1, has_aux=True)(
            update_key, params, state, batch
        )

        # =============  cast the grads to full precision for updating the weights ============= #
        grads = self.policy.cast_to_param(grads)
        grads = state.loss_scale.unscale(grads)

        # ============= average grads and loss across replicas ============= #
        grads = jax.tree_util.tree_map(
            lambda g: jax.lax.pmean(g, axis_name="data"), grads
        )
        loss = jax.lax.pmean(loss, axis_name="data")

        # ============= apply the gradients and skip the update if inf grads are found ============= #
        state = state.apply_gradients(
            grads=grads, skip_infinite=self.config.skip_infinite
        )

        metrics = TrainMetrics(
            loss=loss,
            grads_gnorm=optax.global_norm(grads),
            params_gnorm=optax.global_norm(params),
        )

        return state, metrics

    def _training_step(
        self, step_key: PRNGKeyArray, state: TrainState, batch: Batch
    ) -> Tuple[TrainState, TrainMetrics]:
        # ============= adding grad accumulation dim ============= #
        batch = trx.tree_map(
            lambda x: x.reshape(
                self.config.grad_accum_steps,
                -1,  # micro_batch_size
                self.config.block_size,
            ),
            batch,
        )

        grad_accum_keys = jax.random.split(
            step_key, num=self.device_mesh.size * self.config.grad_accum_steps
        ).reshape(self.config.grad_accum_steps, -1)

        # ============= running update loop on grad_accum dim ============= #
        for mini_step in range(self.config.grad_accum_steps):
            state, metrics = self.update(
                grad_accum_keys[mini_step],
                jax.tree_map(lambda d: d[mini_step], batch),
                state,
            )
        # ============= average the metrics across grad accumulation steps ============= #
        metrics = trx.tree_map(lambda m: jnp.mean(m), metrics)

        return state, metrics

    def _eval_step(
        self, rng_key: PRNGKeyArray, state: TrainState, batch: Batch
    ) -> jax.Array:
        params = self.policy.cast_to_compute(state.params)
        loss = self._loss(rng_key, params, state, batch, train=False)
        return jax.lax.pmean(loss, axis_name="data")

    def save(self, state: TrainState, metrics: TrainMetrics):
        saved = self.checkpointer.save(
            state.step,
            items=dict(
                state=state,
                train_metrics=metrics,
            ),
            metrics=dict(loss=float(metrics.loss)),
        )

        if saved:
            logging.info(f"checkpoint saved ...{state.step}")

    def restore(self) -> Tuple:
        ckpt = self.checkpointer.restore(
            self.checkpointer.best_step(),
            items=dict(
                state=self.make_train_state(state_key=jax.random.PRNGKey(0)),
                train_metrics=TrainMetrics(loss=0),
            ),
        )

        state = ckpt["state"]
        loss = ckpt["train_metrics"].loss

        return state, loss

    def close(self):
        self.checkpointer.wait_until_finished()

    def restore_openai_gpt(self):
        pass
