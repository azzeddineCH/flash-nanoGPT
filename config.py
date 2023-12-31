import os
from dataclasses import dataclass

import tyro


@dataclass(frozen=True)
class Config:
    # checkpoint and eval
    eval_freq: int
    eval_num_steps: int
    save_checkpoint: bool
    restore: str  # scratch | pre-trained | gpt
    gpt_type: str  # gpt2 | gpt2-medium | gpt2-large | gpt2-xl
    checkpoint_dir: str
    log_freq: int
    wandb: bool
    wandb_project_name: str
    wandb_run_id: str
    # data
    dataset_dir: str
    document_dataset: bool
    vocab_size: int
    batch_size: int
    block_size: int
    buffer_size: int
    prefetch: int
    # model
    num_layers: int
    num_heads: int
    embd_dim: int
    dropout_rate: float
    use_bias: bool
    # Optimizer
    num_iters: int
    lr: float
    lr_decay: bool
    lr_warmup_iters: int
    lr_decay_iters: int
    lr_min: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    # training
    device: str
    grad_accum_steps: int
    amp: bool
    skip_infinite: bool
    jit: bool
    seed: int
    # sampling
    samples_dir: str
    num_samples: int
    max_new_tokens: int
    temperature: int


def get_default_config():
    config_path = os.environ.get("GPT_CONFIG_PATH", "yaml/train-test-gpt.yaml")
    assert os.path.exists(
        config_path
    ), f"Can't find env variable 'gpt-config-path', f{config_path}"

    with open(config_path, "r") as f:
        default_config = tyro.from_yaml(Config, f)

    return default_config
