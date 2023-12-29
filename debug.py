import dataclasses

import jax
import tyro

import wandb
from config import Config, get_default_config
from ds.loader import DataLoader


def log(tree):
    host = jax.devices("cpu")[0]
    wandb.log(jax.device_put(tree, host))


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

key = jax.random.PRNGKey(config.seed)
state_key, training_key, key = jax.random.split(key, 3)
training_key = jax.random.fold_in(training_key, jax.process_index())

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
).get_iterator()

# ============= Training Loop ============= #

for i in range(0, config.num_iters):
    # ============= Training ============= #

    train_batch = next(train_data_iter)
    wandb.log({"iter": i})
