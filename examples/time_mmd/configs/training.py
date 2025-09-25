"""Training configuration dataclasses for multimodal TimesFM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.yaml import load_yaml


@dataclass
class RunnerConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    wandb_run_name: str = "multimodal-timesfm-time-mmd"


@dataclass
class HardwareConfig:
    device: str | None = None


@dataclass
class DataConfig:
    data_path: str = "data/Time-MMD"
    domains: list[str] = field(default_factory=lambda: ["Environment"])
    split_ratio: float = 0.8
    patch_len: int = 32
    context_len: int = 128
    horizon_len: int = 128


@dataclass
class LogConfig:
    save_dir: str = "logs"
    experiment_name: str = "multimodal_timesfm_time_mmd"


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_frequency: int = 5


@dataclass
class TrainingConfig:
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log: LogConfig = field(default_factory=LogConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> TrainingConfig:
        config_dict = load_yaml(yaml_path)
        return cls(
            runner=RunnerConfig(**config_dict.get("runner", {})),
            hardware=HardwareConfig(**config_dict.get("hardware", {})),
            data=DataConfig(**config_dict.get("data", {})),
            log=LogConfig(**config_dict.get("log", {})),
            checkpoint=CheckpointConfig(**config_dict.get("checkpoint", {})),
        )
