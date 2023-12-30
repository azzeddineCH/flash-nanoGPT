import logging
import sys

import jax
import tyro

from config import Config, get_default_config
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
elif config.restore == "gpt-2":
    train_state = trainer.restore_openai_gpt()
    raise ValueError(f"unknown restore method {config.restore}")

if jax.process_index() == 0:
    logging.info(f"Train state created | model parameters : {train_state.num_params}")
    trainer.save(train_state, metrics=TrainMetrics(loss=0))
