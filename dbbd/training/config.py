"""Training Configuration"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class TrainingConfig:
    small_mode: bool = False

    input_feat_dim: int = 3
    hidden_dim: int = 96
    output_dim: int = 96
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [64, 128])
    encoder_pooling: str = "max"

    propagator_hidden_dim: int = 128
    propagator_num_layers: int = 2

    aggregator_mode: str = "max"
    aggregator_use_spatial: bool = True

    projection_dim: int = 128
    temperature: float = 0.1
    alpha: float = 1.0
    beta: float = 0.5
    point_num_samples: int = 4096

    learning_rate: float = 1e-3
    weight_decay: float = 0.01

    scheduler_type: str = "cosine"
    T_max: int = 100
    eta_min: float = 1e-5

    num_epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    save_interval: int = 10

    log_interval: int = 100
    eval_interval: int = 5
    verbose: bool = True

    data_root: str = "datasets/scannet"
    max_train_scenes: Optional[int] = None
    max_val_scenes: Optional[int] = None

    device: str = "cuda"

    seed: int = 42

    def __post_init__(self):
        if self.small_mode:
            if self.encoder_hidden_dims == [64, 128]:
                self.encoder_hidden_dims = [32, 64]
            if self.hidden_dim == 96:
                self.hidden_dim = 64
            if self.output_dim == 96:
                self.output_dim = 64
            if self.projection_dim == 128:
                self.projection_dim = 64
            if self.propagator_hidden_dim == 128:
                self.propagator_hidden_dim = 64
            if self.batch_size == 8:
                self.batch_size = 2
            if self.num_workers == 4:
                self.num_workers = 0
            if self.point_num_samples == 4096:
                self.point_num_samples = 512
            if self.max_train_scenes is None:
                self.max_train_scenes = 4
            if self.max_val_scenes is None:
                self.max_val_scenes = 2

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str):
        import yaml
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
