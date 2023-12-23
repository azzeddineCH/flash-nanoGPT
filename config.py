from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # checkpoint and eval
    eval_freq: int
    eval_num_steps: int
    save_checkpoint: int
    restore: str  # scratch | pre-trained | gpt
    checkpoint_dir: str
    log_freq: int
    wandb: bool
    wandb_project_name: str
    wandb_run_id: str
    # data
    dataset_dir: str
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
    grad_accum_steps: int
    amp: bool
    skip_infinite: bool
    device: str
    jit: bool
    num_devices: int
