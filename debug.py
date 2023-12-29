import jax
import tyro

from config import Config, get_default_config
from ds.utils import Batch
from training.trainer import Trainer

# ============= Init configs ============= #

config = tyro.cli(Config, default=get_default_config())

# ============= Init Random keys loaders ============= #

key = jax.random.PRNGKey(0)
data_rng_key, training_key, key = jax.random.split(key, 3)

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

random_input = jax.random.randint(
    key, (config.batch_size, config.block_size), minval=0, maxval=config.vocab_size
)
dummy_batch = Batch(
    inputs=random_input,
    labels=random_input,
)

train_batch_key, train_step_key, training_key = jax.random.split(training_key, 3)
train_state, train_metrics = trainer.training_step(
    train_step_key, train_state, batch=dummy_batch
)
