"""MultimodalTimesFM wrapper class that extends TimesFM to support multimodal inputs."""

from dataclasses import dataclass
from typing import Any, Literal

import timesfm
import torch

from src.models.multimodal_fusion import MultimodalFusion
from src.models.text_encoder import EnglishTextEncoder, JapaneseTextEncoder, TextEncoderBase


@dataclass(kw_only=True)
class MultimodalTimesFmConfig:
    """Configuration for MultimodalTimesFM.

    Attributes:
        text_encoder_type: Type of text encoder to use ('english' or 'japanese').
        forecast_config: Configuration for forecasting.
    """

    text_encoder_type: Literal["english", "japanese"] = "english"
    forecast_config: timesfm.ForecastConfig | None = None


class MultimodalTimesFM:
    """Wrapper class for TimesFM that supports multimodal inputs including text.

    This class extends TimesFM to handle both time series data and text descriptions,
    using a text encoder and fusion mechanism to combine multimodal information
    for improved time series forecasting.
    """

    def __init__(
        self,
        config: MultimodalTimesFmConfig,
        device: torch.device | str | None = None,
    ) -> None:
        """Initializes MultimodalTimesFM wrapper.

        Args:
            config: Multimodal configuration for the model.
            device: Device to use for the model. If None, will auto-resolve the best available device.
        """
        # Initialize the TimesFM 2.5 model
        self.timesfm_model = timesfm.TimesFM_2p5_200M_torch()

        # Store config
        self.config = config

        # Initialize text encoder based on type
        self.text_encoder: TextEncoderBase
        if config.text_encoder_type == "english":
            self.text_encoder = EnglishTextEncoder(device=device)
        elif config.text_encoder_type == "japanese":
            self.text_encoder = JapaneseTextEncoder(device=device)
        else:
            raise ValueError(
                f"Unsupported text encoder type: {config.text_encoder_type}. Must be 'english' or 'japanese'."
            )

        # Initialize fusion mechanism
        # Note: Using model_dims from TimesFM 2.5 configuration (1280)
        self.fusion = MultimodalFusion(
            ts_feature_dim=1280,  # TimesFM 2.5 uses 1280 model dimensions
            text_feature_dim=self.text_encoder.embedding_dim,
        )

    def load_checkpoint(
        self,
        *,
        path: str | None = None,
        hf_repo_id: str | None = "google/timesfm-2.5-200m-pytorch",
    ) -> None:
        """Loads a TimesFM checkpoint.

        Args:
            path: Path to a local checkpoint. If not provided, will try to download
              from the default Hugging Face repo.
            hf_repo_id: If provided, will download from the specified Hugging Face
              repo instead.
        """
        self.timesfm_model.load_checkpoint(path=path, hf_repo_id=hf_repo_id)

    def compile(self, forecast_config: timesfm.ForecastConfig | None = None, **kwargs: Any) -> None:
        """Compiles the TimesFM model for fast decoding.

        Args:
            forecast_config: Configuration for forecasting flags.
            **kwargs: Additional keyword arguments to pass to model.compile().
        """
        if forecast_config is None:
            forecast_config = self.config.forecast_config
        if forecast_config is None:
            raise ValueError("forecast_config must be provided either in config or as argument")

        self.timesfm_model.compile(forecast_config, **kwargs)

    def forecast(self, horizon: int, inputs: list[Any]) -> tuple[Any, Any]:
        """Forecasts the time series.

        Args:
            horizon: The number of time points to forecast.
            inputs: A list of numpy arrays, each representing a time series to query forecast for.

        Returns:
            A tuple of (point_forecast, quantile_forecast).
        """
        return self.timesfm_model.forecast(horizon, inputs)  # type: ignore[no-any-return]
