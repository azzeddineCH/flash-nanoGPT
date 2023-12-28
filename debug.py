import dataclasses

import jax
import tyro

import wandb
from config import Config, get_default_config
from training.trainer import Trainer

# import multiprocessing


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

# ============= Init ds loaders ============= #

# train_data_iter = DataLoader(
#     directory=config.dataset_dir,
#     batch_size=config.batch_size // jax.process_count(),
#     block_size=config.block_size,
#     split="train",
#     prefetch=config.prefetch,
#     buffer_size=config.buffer_size,
#     num_shards=jax.process_count(),
#     shard=jax.process_index(),
#     num_workers=multiprocessing.cpu_count() // 2,
# ).get_iterator()

# validation_data_iter = DataLoader(
#     directory=config.dataset_dir,
#     batch_size=config.batch_size // jax.process_count(),
#     block_size=config.block_size,
#     split="val",
#     prefetch=config.prefetch,
#     buffer_size=config.buffer_size,
#     num_shards=jax.process_count(),
#     shard=jax.process_index(),
#     num_workers=multiprocessing.cpu_count() // 4,
# ).get_iterator()
