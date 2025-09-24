"""Evaluation configuration dataclasses for multimodal TimesFM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.yaml import load_yaml


@dataclass
class RunnerConfig:
    batch_size: int = 8


@dataclass
class DataConfig:
    data_path: str = "data/Time-MMD"
    domain: str = "Environment"


@dataclass
class EvaluationConfig:
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> EvaluationConfig:
        config_dict = load_yaml(yaml_path)
        return cls(
            runner=RunnerConfig(**config_dict.get("runner", {})),
            data=DataConfig(**config_dict.get("data", {})),
        )
