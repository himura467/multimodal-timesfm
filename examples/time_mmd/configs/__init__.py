"""Configuration dataclasses for multimodal TimesFM."""

from .evaluation import EvaluationConfig
from .model import ModelConfig
from .training import TrainingConfig

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
]
