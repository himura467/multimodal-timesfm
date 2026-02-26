"""Forecast configuration for Time-MMD dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multimodal_timesfm.utils.yaml import parse_yaml


@dataclass
class ForecastConfig:
    """Configuration for forecasting parameters."""

    context_len: int = 32
    horizon_len: int = 32

    @classmethod
    def from_yaml(cls, path: Path) -> ForecastConfig:
        return parse_yaml(path, cls)
