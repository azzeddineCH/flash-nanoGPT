import dataclasses
import logging
import sys
import time

import jax
import tyro

import wandb
from config import Config, get_default_config
from ds.loader import DataLoader
from training.trainer import Trainer
from training.utils import TrainMetrics

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# ============= Init tpu pod ============= #
# jax.distributed.initialize()

if jax.process_index() == 0:
    logging.info(
        f"TPU pod initialized, {jax.process_count()} host/s, {jax.local_device_count()} core per host, {jax.device_count()} total"
    )

# ============= Init configs ============= #
config = tyro.cli(Config, default=get_default_config())

# ============= Init Logging ============= #
if config.wandb and jax.process_index() == 0:
    wandb.init(
        project=config.wandb_project_name,
        name=config.wandb_run_id,
        config=dataclasses.asdict(config),
    )

# ============= Init Random keys loaders ============= #
key = jax.random.PRNGKey(config.seed)
state_key, training_key, key = jax.random.split(key, 3)
training_key = jax.random.fold_in(training_key, jax.process_index())

# ============= Init training state ============= #
trainer = Trainer(config=config)
start_iter = 0
best_valid_loss = 1e6

if config.restore == "scratch":
    train_state = trainer.make_train_state(state_key)
elif config.restore == "pre-trained":
    train_state, best_valid_loss = trainer.restore()
    start_iter = train_state.step + 1
elif config.restore == "openai":
    train_state = trainer.restore_openai_gpt()
    raise ValueError(f"unknown restore method {config.restore}")

if jax.process_index() == 0:
    logging.info(f"Train state created | model parameters : {train_state.num_params}")

# ============= Init ds loaders ============= #
train_data_iter = DataLoader(
    directory=config.dataset_dir,
    batch_size=config.batch_size // jax.process_count(),
    block_size=config.block_size,
    split="train",
    prefetch=config.prefetch,
    buffer_size=config.buffer_size,
    num_shards=jax.process_count(),
    shard=jax.process_index(),
).get_iterator()

validation_data_iter = DataLoader(
    directory=config.dataset_dir,
    batch_size=config.batch_size // jax.process_count(),
    block_size=config.block_size,
    split="val",
    prefetch=config.prefetch,
    buffer_size=config.buffer_size,
    num_shards=jax.process_count(),
    shard=jax.process_index(),
).get_iterator()

# ============= Training Loop ============= #
for _ in range(start_iter, config.num_iters):
    # ============= Training ============= #
    t0 = time.time()
    train_batch = next(train_data_iter)

    iter_key = jax.random.fold_in(training_key, train_state.step)
    training_step_key, iter_key = jax.random.split(iter_key, 2)

    train_state, train_metrics = trainer.training_step(
        training_step_key, train_state, train_batch
    )
    step_time_ms = (time.time() - t0) * 1000

    # ============= Evaluation ============= #
    if train_state.step == 1 or train_state.step % config.eval_freq == 0:
        valid_loss = train_loss = 0
        train_eval_key, valid_eval_key, iter_key = jax.random.split(iter_key, 3)
        for j in range(config.eval_num_steps):
            valid_batch = next(validation_data_iter)
            train_batch = next(train_data_iter)
            valid_loss += (
                trainer.eval_step(valid_eval_key, train_state, valid_batch)
                / config.eval_num_steps
            )
            train_loss += (
                trainer.eval_step(train_eval_key, train_state, train_batch)
                / config.eval_num_steps
            )

        best_valid_loss = (
            valid_loss if valid_loss < best_valid_loss else best_valid_loss
        )
        if jax.process_index() == 0:
            # ============= Checkpointing and logging ============= #
            logging.info(
                f"iter: {train_state.step} | val loss {valid_loss} | train loss {train_loss}"
            )
            if config.save_checkpoint:
                trainer.save(train_state, metrics=TrainMetrics(loss=valid_loss))

            if config.wandb:
                wandb.log(
                    {
                        "iter": train_state.step,
                        "train/loss": train_loss,
                        "val/loss": valid_loss,
                        "lr": train_state.lr,
                        "loss_scale": train_state.loss_scale.loss_scale,
                        "grads_gnorm": train_metrics.grads_gnorm,
                        "params_gnorm": train_metrics.params_gnorm,
                        "time_ms": step_time_ms if train_state.step > 1 else 0,
                    },
                    step=train_state.step,
                    commit=True,
                )

    # ============= Terminal logging ============= #
    if (
        train_state.step == 1 or train_state.step % config.log_freq == 0
    ) and jax.process_index() == 0:
        logging.info(
            f"iter: {train_state.step} | loss: {train_metrics.loss} | time_ms: {step_time_ms}"
        )

if jax.process_index() == 0:
    trainer.close()
    if config.wandb:
        wandb.finish()

jax.distributed.shutdown()
