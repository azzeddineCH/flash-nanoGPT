import dataclasses
import time

from config import Config
from data_loader import DataLoader
from trainer import Trainer
import jax
import wandb
from utils import TrainMetrics
import tyro
import os

# ============= Init configs ============= #

config_path = os.environ.get("GPT_CONFIG_PATH", None)
assert os.path.exists(config_path), f"Can't find env variable 'gpt-config-path', f{config_path}"

with open(config_path, "r") as f:
    default_config = tyro.from_yaml(Config, f)
config = tyro.cli(Config, default=default_config)

# ============= Init Random keys loaders ============= #

key = jax.random.PRNGKey(0)
data_rng_key, training_key, key = jax.random.split(key, 3)

# ============= Init dataset loaders ============= #

train_data_loader = DataLoader(
    dataset_dir=config.dataset_dir,
    batch_size=config.batch_size,
    block_size=config.block_size,
    split="train"
)

validation_data_loader = DataLoader(
    dataset_dir=config.dataset_dir,
    batch_size=config.batch_size,
    block_size=config.block_size,
    split="val"
)

# ============= Init training state ============= #

trainer = Trainer(config=config)

start_iter = 0
if config.restore == "scratch":
    train_state = trainer.make_train_state()
elif config.restore == "pre-trained":
    train_state, start_iter = trainer.restore()
else:
    raise ValueError(f"Unknown restore method {config.restore}")

# ============= Init Logging ============= #

if config.wandb:
    wandb.init(
        project=config.wandb_project_name,
        name=config.wandb_run_id,
        config=dataclasses.asdict(config)
    )

# ============= Training Loop ============= #

for i in range(start_iter, config.num_iters):

    # ============= Training ============= #

    t0 = time.time()
    train_batch_key, train_step_key, training_key = jax.random.split(training_key, 3)
    batch = train_data_loader.get(train_batch_key)
    train_state, train_metrics = trainer.training_step(train_step_key, train_state, batch)
    step_time_s = time.time() - t0

    # ============= Evaluation ============= #

    if i % config.eval_freq == 0:
        valid_loss = 0
        train_loss = 0
        train_eval_key, valid_eval_key, training_key = jax.random.split(training_key, 3)
        for j in range(config.eval_num_steps):
            valid_batch = validation_data_loader.get(jax.random.fold_in(valid_eval_key, j))
            train_batch = train_data_loader.get(jax.random.fold_in(train_eval_key, j))

            valid_loss += trainer.validation_step(
                train_step_key,
                train_state,
                valid_batch
            ) / config.eval_num_steps

            train_loss += trainer.validation_step(
                train_step_key,
                train_state,
                train_batch
            ) / config.eval_num_steps

        print(f"Iter {i + 1} |  Val loss {valid_loss} | Train loss {train_loss}")

        if config.wandb:
            wandb.log({
                "iter": i + 1,
                "train/loss": train_loss,
                "val/loss": valid_loss,
                "lr": train_state.lr,
                "loss_scale": train_state.loss_scale.loss_scale,
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
