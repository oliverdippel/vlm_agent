from dataclasses import dataclass, field


@dataclass
class FlowConfig:
    action_dim: int = 7
    hidden_dim: int = 512
    n_layers: int = 4
    conditioning_dim: int = 4096
    dropout: float = 0.1
    n_euler_steps: int = 10
    sigma_min: float = 1e-4


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 50
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999


@dataclass
class LoggingConfig:
    eval_every_n_epochs: int = 5
    checkpoint_dir: str = "models/"


@dataclass
class DataConfig:
    data_dir: str = "data/libero"
    num_workers: int = 0


@dataclass
class WandbConfig:
    project: str = "mini-pi0"
    mode: str = "disabled"


@dataclass
class Config:
    flow: FlowConfig = field(default_factory=FlowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)