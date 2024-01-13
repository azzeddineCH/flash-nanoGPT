import dataclasses
import logging
import sys
import time

import jax
import tyro

from config import Config, get_default_config
from ds.utils import Batch
from training.trainer import GPTS_CONFIG, Trainer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# ============= Init configs ============= #
config = tyro.cli(Config, default=get_default_config())
if config.restore == "openai":
    assert config.gpt_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    config = Config(
        **{
            **dataclasses.asdict(config),
            **GPTS_CONFIG[config.gpt_type],
        }
    )

# ============= Init Random keys loaders ============= #
key = jax.random.PRNGKey(config.seed)
state_key, data_key, training_key, key = jax.random.split(key, 4)
training_key = jax.random.fold_in(training_key, jax.process_index())

# ============= Init training state ============= #
trainer = Trainer(config=config)

train_state = trainer.make_train_state(state_key)
logging.info(f"Train state created | model parameters : {train_state.num_params}")

batch = jax.random.randint(
    data_key,
    shape=(config.batch_size, config.block_size),
    minval=0,
    maxval=config.vocab_size,
    dtype=jax.numpy.int32,
)

iter_key = jax.random.fold_in(training_key, train_state.step)
training_step_key, iter_key = jax.random.split(iter_key, 2)

logging.info("Compiling...")
training_step_fn = trainer.training_step.lower(
    training_step_key, train_state, Batch(inputs=batch, labels=batch)
).compile()

logging.info("Running iteration...")
with jax.profiler.trace("./tensorboard"):
    for i in range(10):
        t0 = time.time()
        training_step_fn(
            training_step_key, train_state, Batch(inputs=batch, labels=batch)
        )
        step_time_ms = (time.time() - t0) * 1000
        mfu = trainer.estimate_mfu(
            train_state,
            samples_per_iter=config.batch_size,
            time_per_iter_s=step_time_ms / 1000,
        )
        logging.info(f"iter {i} | time_ms {step_time_ms} | mfu {mfu}")

trainer.close()
