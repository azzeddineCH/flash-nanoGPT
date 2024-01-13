import logging
from functools import partial
from pathlib import Path
from typing import Callable, Tuple

import jax
import jmp
import optax
import orbax.checkpoint as ocp
from flax.struct import PyTreeNode
from jax import numpy as jnp
from jax import sharding as shx
from jax import tree_util as trx
from jax._src.tree_util import DictKey
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.random import PRNGKeyArray
from orbax.checkpoint.checkpoint_utils import construct_restore_args
from orbax.checkpoint.utils import is_gcs_path

from config import Config
from ds.utils import Batch
from training.model import GPT, GPTConfig
from training.state import DynamicLossScale, TrainState
from training.utils import Policy, TrainMetrics

GPTS_CONFIG = {
    # 124M params
    "gpt2": dict(
        num_layers=12,
        num_heads=12,
        embd_dim=768,
        vocab_size=50257,
        block_size=1024,
        use_bias=True,
    ),
    # 350M params
    "gpt2-medium": dict(
        num_layers=24,
        num_heads=16,
        embd_dim=1024,
        vocab_size=50257,
        block_size=1024,
        use_bias=True,
    ),
    # 774M params
    "gpt2-large": dict(
        num_layers=36,
        num_heads=20,
        embd_dim=1280,
        vocab_size=50257,
        block_size=1024,
        use_bias=True,
    ),
    # 1558M params
    "gpt2-xl": dict(
        num_layers=48,
        num_heads=25,
        embd_dim=1600,
        vocab_size=50257,
        block_size=1024,
        use_bias=True,
    ),
}


class Trainer:
    def __init__(self, config: Config):
        self.config = config

        # ============= Mixed Precision Policy ============= #
        assert jax.local_devices(backend=self.config.device)

        if self.config.amp and self.config.device == "tpu":
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
        ckpt_dir = self.config.checkpoint_dir
        if not is_gcs_path(ckpt_dir):
            ckpt_dir = Path(ckpt_dir)
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            ckpt_dir = ckpt_dir.absolute()

        self.checkpointer = ocp.CheckpointManager(
            ckpt_dir,
            checkpointers=dict(
                step=ocp.Checkpointer(ocp.ArrayCheckpointHandler()),
                params=ocp.PyTreeCheckpointer(),
                opt_state=ocp.PyTreeCheckpointer(),
                valid_loss=ocp.Checkpointer(ocp.ArrayCheckpointHandler()),
            ),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=2,
                best_fn=lambda metrics: metrics["loss"],
                best_mode="min",
            ),
        )

    def _make_device_mesh(self) -> shx.Mesh:
        devices = mesh_utils.create_device_mesh(
            mesh_shape=(jax.local_device_count(self.config.device),),
            devices=jax.local_devices(backend=self.config.device),
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
        def _to_decay(p):
            is_embeddings = (
                p.shape == params["wte"]["embedding"].shape
                or p.shape == params["wpe"]["embedding"].shape
            )
            is_flat = p.ndim < 2

            return not (is_embeddings or is_flat)

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
            learning_rate=schedule if self.config.lr_decay else self.config.lr,
            b1=self.config.beta1,
            b2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            mask=trx.tree_map(_to_decay, params),
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

    def _make_loss_scale(self):
        scale = jmp.NoOpLossScale()
        if self.config.amp and not self.config.device == "tpu":
            scale = DynamicLossScale(jnp.asarray(2.0**15, dtype=jnp.float32))

        return scale

    def make_train_state(self, state_key: PRNGKeyArray) -> TrainState:
        model, params = self._make_model(state_key)
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=self._make_optimizer(params),
            loss_scale=self._make_loss_scale(),
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
        grads = jax.lax.pmean(grads, axis_name="data")
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
        state, metrics = jax.lax.scan(
            f=lambda _state, xs: self.update(xs[0], xs[1], _state),
            xs=(grad_accum_keys, batch),
            init=state,
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

    def save(self, state: TrainState, loss: jax.Array):
        saved = self.checkpointer.save(
            state.step,
            items=dict(
                step=state.step,
                params=state.params,
                opt_state=state.opt_state,
                valid_loss=loss,
            ),
            metrics=dict(loss=float(loss)),
        )

        if saved:
            logging.info(f"checkpoint saved ...{state.step}")

    def restore(self) -> Tuple:
        train_state, valid_loss = jax.device_put(
            (self.make_train_state(state_key=jax.random.PRNGKey(0)), 0),
            device=shx.NamedSharding(self.device_mesh, self.replicated_specs),
        )

        train_state_sharding, metrics_sharding = jax.tree_map(
            lambda x: x.sharding, (train_state, valid_loss)
        )
        ckpt = self.checkpointer.restore(
            self.checkpointer.best_step(),
            items=dict(
                step=train_state.step,
                params=train_state.params,
                opt_state=train_state.opt_state,
                valid_loss=valid_loss,
            ),
            restore_kwargs=dict(
                step=dict(
                    restore_args=construct_restore_args(
                        train_state.step, train_state_sharding.step
                    )
                ),
                params=dict(
                    restore_args=construct_restore_args(
                        train_state.params, train_state_sharding.params
                    )
                ),
                opt_state=dict(
                    restore_args=construct_restore_args(
                        train_state.opt_state, train_state_sharding.opt_state
                    )
                ),
                valid_loss=dict(
                    restore_args=construct_restore_args(valid_loss, metrics_sharding)
                ),
            ),
        )

        train_state = train_state.replace(
            step=ckpt["step"],
            params=ckpt["params"],
            opt_state=ckpt["opt_state"],
        )

        return train_state, ckpt["valid_loss"]

    def make_sampling_fn(
        self, train_state: TrainState, max_new_tokens: int, temperature: float
    ) -> Callable:
        fn = partial(
            train_state.apply_fn,
            variables={"params": train_state.params},
            method="generate",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return jax.jit(fn) if self.config.jit else fn

    def close(self):
        self.checkpointer.wait_until_finished()

    def restore_openai_gpt(self):
        from transformers import GPT2LMHeadModel

        def _copy(path, ph):
            hf_path = path
            if path[-1].key in {"kernel", "scale", "embedding"}:
                hf_path = path[:-1] + (DictKey("weight"),)
            name = "transformer." + ".".join([p.key for p in hf_path])

            params = jnp.asarray(
                a=hf_params[name],
                dtype=self.policy.param_dtype,
            )

            assert (
                params.shape == ph.shape
            ), f"expected {ph.shape}, got {params.shape} for {name}"
            return params

        model_hf = GPT2LMHeadModel.from_pretrained(self.config.gpt_type)
        hf_params = model_hf.state_dict()
        model, params = self._make_model(params_key=jax.random.PRNGKey(0))
        params = jax.tree_util.tree_map_with_path(_copy, params)

        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=self._make_optimizer(params),
            loss_scale=self._make_loss_scale(),
            skip_infinite=self.config.skip_infinite,
        )

    def estimate_mfu(self, train_state, samples_per_iter, time_per_iter_s):
        num_params = int(train_state.num_params)
        L, H, Q, T = (
            self.config.num_layers,
            self.config.num_heads,
            self.config.embd_dim // self.config.num_heads,
            self.config.block_size,
        )
        flops_per_token = 6 * num_params + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * samples_per_iter
        flops_achieved = flops_per_iter * (1.0 / time_per_iter_s)
        flops_promised = 180e12  # 180 tera flops for TPUv2
        return flops_achieved / flops_promised
