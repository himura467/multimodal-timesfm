"""Training configuration dataclasses for multimodal TimesFM on Time-MMD dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from examples.time_mmd.configs.domain_columns import DomainColumnsConfig
from multimodal_timesfm.cross_validation import CrossValidationConfig
from multimodal_timesfm.utils.yaml import load_yaml


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
    column_config: DomainColumnsConfig | None = None
    split_ratio: float = 0.8
    patch_len: int = 32
    context_len: int = 32
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
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> TrainingConfig:
        config_dict = load_yaml(yaml_path)

        # Handle column_config separately
        data_config = config_dict.get("data", {})
        column_config_dict = data_config.pop("column_config", None)
        column_config = DomainColumnsConfig.from_dict(column_config_dict) if column_config_dict else None

        return cls(
            runner=RunnerConfig(**config_dict.get("runner", {})),
            hardware=HardwareConfig(**config_dict.get("hardware", {})),
            data=DataConfig(**data_config, column_config=column_config),
            log=LogConfig(**config_dict.get("log", {})),
            checkpoint=CheckpointConfig(**config_dict.get("checkpoint", {})),
            cross_validation=CrossValidationConfig(**config_dict.get("cross_validation", {})),
        )
