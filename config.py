from dataclasses import dataclass


@dataclass
class Config:
    # checkpoint and eval
    eval_freq: int = 2000
    eval_num_steps: int = 10
    save_checkpoint: bool = False
    log_freq: int = 1
    wandb: bool = True
    wandb_project_name: str = "flash-nano-gpt"
    wandb_run_id: str = "flash-nanoGPT-1"
    # data
    dataset: str = "shakespeare"
    grad_accum_steps: int = 40
    batch_size: int = 12
    block_size: int = 1024
    # model
    num_layers: int = 12
    num_heads: int = 12
    embd_dim: int = 768
    dropout_rate: float = 0.2
    use_bias: bool = False
    # Optimizer
    num_iters: int = int(6e5)
    lr: float = 6e-4
    lr_decay: bool = True
    lr_warmup_iters: int = 2000
    lr_decay_iters: int = int(6e5)
    lr_min: float = 6e-5
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # training
    amp: bool = True
    skip_infinite : bool = True
    device: str = "cpu"
    jit: bool = True
