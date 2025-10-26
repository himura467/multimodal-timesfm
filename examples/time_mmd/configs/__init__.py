"""Configuration dataclasses for multimodal TimesFM on Time-MMD dataset."""

from .model import ModelConfig
from .training import TrainingConfig

__all__ = [
    "ModelConfig",
    "TrainingConfig",
]
