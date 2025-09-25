"""Model configuration dataclasses for multimodal TimesFM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.utils.yaml import load_yaml


@dataclass
class TimesFMConfig:
    # TimesFmHparams
    context_len: int = 128
    horizon_len: int = 128
    input_patch_len: int = 32
    output_patch_len: int = 128
    num_layers: int = 50
    num_heads: int = 16
    model_dims: int = 1280
    per_core_batch_size: int = 32
    backend: Literal["cpu", "gpu", "tpu"] = "gpu"
    quantiles: list[float] | None = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    use_positional_embedding: bool = True
    point_forecast_mode: Literal["mean", "median"] = "median"

    # TimesFMConfig
    num_kv_heads: int = 16
    rms_norm_eps: float = 1e-6
    pad_val: float = 1123581321.0
    tolerance: float = 1e-6
    dtype: str = "bfloat32"


@dataclass
class TextEncoderConfig:
    text_encoder_type: Literal["english", "japanese"] = "english"


@dataclass
class ModelConfig:
    timesfm: TimesFMConfig = field(default_factory=TimesFMConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> ModelConfig:
        config_dict = load_yaml(yaml_path)
        return cls(
            timesfm=TimesFMConfig(**config_dict.get("timesfm", {})),
            text_encoder=TextEncoderConfig(**config_dict.get("text_encoder", {})),
        )
