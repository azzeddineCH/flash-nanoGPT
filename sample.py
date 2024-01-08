import dataclasses
import logging
import os
import sys

import jax.random
import tiktoken
import tyro
from jax import numpy as jnp

from config import Config, get_default_config
from training.trainer import GPTS_CONFIG, Trainer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# ============= Encoding ============= #
enc = tiktoken.get_encoding("gpt2")


def encode(s):
    return jnp.asarray(
        enc.encode(s, allowed_special={"<|endoftext|>"}), dtype=jnp.int32
    )


def decode(tokens):
    return enc.decode(tokens)


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

# ============= Init state ============= #
trainer = Trainer(config=config)
if config.restore == "pre-trained":
    train_state, _ = trainer.restore()
elif config.restore == "openai":
    logging.info(f"loading weights from pretrained gpt: {config.gpt_type}")
    logging.info("forcing vocab_size=50257, block_size=1024, use_bias=True")
    train_state = trainer.restore_openai_gpt()
else:
    raise ValueError()

sampling_fn = trainer.make_sampling_fn(
    train_state, max_new_tokens=config.max_new_tokens, temperature=config.temperature
)

# ============= Sampling ============= #
key = jax.random.PRNGKey(config.seed)
for file in os.listdir(config.samples_dir):
    with open(os.path.join(config.samples_dir, file), "r") as f:
        context = encode(f.read())

    for i in range(config.num_samples):
        sampling_key = jax.random.fold_in(key, i)
        [output] = sampling_fn(
            rng_key=sampling_key, context=context[None, ...]
        ).tolist()
        print(f"- {decode(output)} | length {len(output)}")
