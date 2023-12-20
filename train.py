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
    rng_key=data_rng_key,
    dataset_dir=config.dataset_dir,
    batch_size=config.batch_size,
    block_size=config.block_size,
    split="train"
)

validation_data_loader = DataLoader(
    rng_key=data_rng_key,
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
    batch = next(train_data_loader)
    step_rng_key = jax.jit(jax.random.fold_in)(training_key, i)
    train_state, train_metrics = trainer.training_step(step_rng_key, train_state, batch)
    step_time_s = time.time() - t0

    # ============= Evaluation ============= #

    if i % config.eval_freq == 0:
        valid_loss = 0
        for j in range(config.eval_num_steps):
            valid_batch = next(validation_data_loader)
            valid_loss += trainer.validation_step(
                step_rng_key,
                train_state,
                valid_batch
            ) / config.eval_num_steps

        print(f"Iter {i + 1} |  Validation loss {valid_loss}")
        if config.save_checkpoint:
            trainer.save(i, train_state, metrics=TrainMetrics(loss=valid_loss))

        if config.wandb:
            wandb.log({"valid/loss": valid_loss})

    # ============= Logging ============= #

    if i % config.log_freq == 0:
        print(
            f"Iter: {i + 1} | "
            f"loss: {train_metrics.loss} | ",
            f"lr: {train_state.lr} | ",
            f"loss scale: {train_state.loss_scale.loss_scale} | "
            f"train time ms: {step_time_s * 1000} | "
        )

        if config.wandb:
            wandb.log({
                "iter": i + 1,
                "iter/loss": train_metrics.loss,
                "iter/lr": train_state.lr,
                "iter/loss_scale": train_state.loss_scale.loss_scale,
                "iter/train_time_ms": step_time_s * 1000
            })

if config.wandb:
    wandb.finish()
