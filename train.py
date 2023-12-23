import dataclasses
import time

from config import Config
from ds.loader import DataLoader
from trainaing.trainer import Trainer
import jax
import wandb
from trainaing.utils import TrainMetrics
import tyro
import os

# ============= Init configs ============= #

config_path = os.environ.get("GPT_CONFIG_PATH", "yaml/test-gpt.yaml")
assert os.path.exists(config_path), f"Can't find env variable 'gpt-config-path', f{config_path}"

with open(config_path, "r") as f:
    default_config = tyro.from_yaml(Config, f)
config = tyro.cli(Config, default=default_config)

# ============= Init Logging ============= #

if config.wandb:
    wandb.init(
        project=config.wandb_project_name,
        name=config.wandb_run_id,
        config=dataclasses.asdict(config)
    )

# ============= Init Random keys loaders ============= #

key = jax.random.PRNGKey(0)
data_rng_key, training_key, key = jax.random.split(key, 3)

# ============= Init ds loaders ============= #

train_data_iter = DataLoader(
    directory=config.dataset_dir,
    batch_size=config.batch_size,
    block_size=config.block_size,
    split="train",
    prefetch=config.prefetch,
    buffer_size=config.buffer_size,
    num_workers=4
).get_iterator()

validation_data_iter = DataLoader(
    directory=config.dataset_dir,
    batch_size=config.batch_size,
    block_size=config.block_size,
    split="val",
    prefetch=config.prefetch,
    buffer_size=config.buffer_size,
    num_workers=4
).get_iterator()

# ============= Init training state ============= #

trainer = Trainer(config=config)

start_iter = 0
if config.restore == "scratch":
    train_state = trainer.make_train_state()
elif config.restore == "pre-trained":
    train_state, start_iter = trainer.restore()
else:
    raise ValueError(f"Unknown restore method {config.restore}")

# ============= Training Loop ============= #

for i in range(start_iter, config.num_iters):

    # ============= Training ============= #

    t0 = time.time()
    train_batch_key, train_step_key, training_key = jax.random.split(training_key, 3)
    train_state, train_metrics = trainer.training_step(
        train_step_key, train_state,
        batch=next(train_data_iter)
    )
    step_time_s = time.time() - t0

    # ============= Evaluation ============= #

    if i % config.eval_freq == 0:
        valid_loss = 0
        train_loss = 0
        train_eval_key, valid_eval_key, training_key = jax.random.split(training_key, 3)
        for j in range(config.eval_num_steps):
            valid_loss += trainer.validation_step(
                train_step_key,
                train_state,
                batch=next(validation_data_iter)
            ) / config.eval_num_steps

            train_loss += trainer.validation_step(
                train_step_key,
                train_state,
                batch=next(train_data_iter)
            ) / config.eval_num_steps

        print(f"Iter {i + 1} |  Val loss {valid_loss} | Train loss {train_loss}")

        if config.wandb:
            wandb.log({
                "iter": i + 1,
                "train/loss": train_loss,
                "val/loss": valid_loss,
                "lr": train_state.lr,
                "loss_scale": train_state.loss_scale.loss_scale,
                "grads_gnorm": train_metrics.grads_gnorm,
                "params_gnorm": train_metrics.params_gnorm,
                "time_ms": step_time_s * 1000
            })

        if config.save_checkpoint:
            trainer.save(i, train_state, metrics=TrainMetrics(loss=valid_loss))

    # ============= Logging ============= #

    if i % config.log_freq == 0:
        print(
            f"Iter: {i + 1} | "
            f"loss: {train_metrics.loss} | ",
            f"time_ms: {step_time_s * 1000} | "
        )

if config.wandb:
    wandb.finish()
