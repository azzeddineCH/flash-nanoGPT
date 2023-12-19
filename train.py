import dataclasses
import time

from config import Config
from data_loader import DataLoader
from trainer import Trainer
import jax
import wandb
from utils import TrainMetrics

key = jax.random.PRNGKey(0)

config = Config(
    eval_freq=25,
    save_checkpoint=True,
    eval_num_steps=5,
    log_freq=1,
    batch_size=32,
    grad_accum_steps=1,
    block_size=32,
    num_layers=8,
    num_heads=4,
    embd_dim=128,
    dropout_rate=0.0,
    jit=True,
    wandb=False,
    amp=False,
    skip_infinite=True,
    num_devices=1,
    restore="pre-trained"
)

data_rng_key, training_key, key = jax.random.split(key, 3)

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

trainer = Trainer(
    config=config,
    vocab_size=train_data_loader.VOCAB_SIZE
)

start_iter = 0
if config.restore == "scratch":
    train_state = trainer.make_train_state()
elif config.restore == "pre-trained":
    train_state, start_iter = trainer.restore()

if config.wandb:
    wandb.init(
        project=config.wandb_project_name,
        name=config.wandb_run_id,
        config=dataclasses.asdict(config)
    )

for i in range(start_iter, config.num_iters):

    t0 = time.time()
    batch = next(train_data_loader)
    step_rng_key = jax.jit(jax.random.fold_in)(training_key, i)
    train_state, train_metrics = trainer.training_step(step_rng_key, train_state, batch)
    step_time_s = time.time() - t0

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

    if i % config.log_freq == 0:
        print(
            f"Iter: {i + 1} | "
            f"loss: {train_metrics.loss} | "
            f"loss scale: {train_state.loss_scale.loss_scale} | "
            f"train time ms: {step_time_s * 1000} | "
        )

        if config.wandb:
            wandb.log({"iter/loss": train_metrics.loss})

if config.wandb:
    wandb.finish()
