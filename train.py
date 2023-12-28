import dataclasses
import multiprocessing
import time

import jax
import tyro

import wandb
from config import Config, get_default_config
from ds.loader import DataLoader
from training.trainer import Trainer
from training.utils import TrainMetrics

if jax.process_index() == 0:
    print(
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

key = jax.random.PRNGKey(0)
data_rng_key, training_key, key = jax.random.split(key, 3)

# ============= Init ds loaders ============= #
if jax.process_index() == 0:
    print("Loading dataset ...")

train_data_iter = DataLoader(
    directory=config.dataset_dir,
    batch_size=config.batch_size // jax.process_count(),
    block_size=config.block_size,
    split="train",
    prefetch=config.prefetch,
    buffer_size=config.buffer_size,
    num_shards=jax.process_count(),
    shard=jax.process_index(),
    num_workers=multiprocessing.cpu_count() // 2,
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
    num_workers=multiprocessing.cpu_count() // 4,
).get_iterator()

# ============= Init training state ============= #

trainer = Trainer(config=config)

start_iter = 0
best_valid_loss = 1e6
if config.restore == "scratch":
    train_state = trainer.make_train_state()
elif config.restore == "pre-trained":
    train_state, best_valid_loss = trainer.restore()
    start_iter = train_state.step + 1
elif config.restore == "gpt-2":
    train_state = trainer.restore_openai_gpt()
    raise ValueError(f"unknown restore method {config.restore}")

# ============= Training Loop ============= #

for _ in range(start_iter, config.num_iters):
    # ============= Training ============= #

    t0 = time.time()
    train_batch_key, train_step_key, training_key = jax.random.split(training_key, 3)
    train_state, train_metrics = trainer.training_step(
        train_step_key, train_state, batch=next(train_data_iter)
    )
    step_time_s = time.time() - t0

    # ============= Evaluation ============= #

    if train_state.step == 1 or train_state.step % config.eval_freq == 0:
        valid_loss = train_loss = 0
        train_eval_key, valid_eval_key, training_key = jax.random.split(training_key, 3)
        for j in range(config.eval_num_steps):
            valid_loss += (
                trainer.validation_step(
                    train_step_key, train_state, batch=next(validation_data_iter)
                )
                / config.eval_num_steps
            )

            train_loss += (
                trainer.validation_step(
                    train_step_key, train_state, batch=next(train_data_iter)
                )
                / config.eval_num_steps
            )

        if config.wandb and jax.process_index() == 0:
            wandb.log(
                {
                    "iter": train_state.step,
                    "train/loss": train_loss,
                    "val/loss": valid_loss,
                    "lr": train_state.lr,
                    "loss_scale": train_state.loss_scale.loss_scale,
                    "grads_gnorm": train_metrics.grads_gnorm,
                    "params_gnorm": train_metrics.params_gnorm,
                    "time_ms": step_time_s * 1000,
                }
            )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if jax.process_index() == 0:
                print(
                    f"iter: {train_state.step} |  val loss {valid_loss} | train loss {train_loss}"
                )
                if config.save_checkpoint:
                    trainer.save(train_state, metrics=TrainMetrics(loss=valid_loss))

    # ============= Logging ============= #

    if train_state.step % config.log_freq == 0 and jax.process_index() == 0:
        print(
            f"iter: {train_state.step} | loss: {train_metrics.loss} | time_ms: {step_time_s * 1000}"
        )

if config.wandb and jax.process_index() == 0:
    wandb.finish()
