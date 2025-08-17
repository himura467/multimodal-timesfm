"""MultimodalTimesFM wrapper class that extends TimesFM to support multimodal inputs."""

import dataclasses
from typing import Any, Sequence

import numpy as np
import torch
from timesfm import TimesFmCheckpoint, TimesFmHparams
from timesfm.timesfm_torch import TimesFmTorch as TimesFm

from src.models.text_encoder import MultimodalFusion, TextEncoder


@dataclasses.dataclass(kw_only=True)
class MultimodalTimesFmHparams(TimesFmHparams):  # type: ignore[misc]
    """Hyperparameters for MultimodalTimesFM that extend TimesFmHparams.

    Attributes:
        text_encoder_model: Name of the sentence transformer model for text encoding.
        text_embedding_dim: Dimension of text embeddings.
        enable_multimodal: Whether to enable multimodal functionality.
    """

    text_encoder_model: str = "all-MiniLM-L6-v2"
    text_embedding_dim: int = 384
    enable_multimodal: bool = True


class MultimodalTimesFM(TimesFm):  # type: ignore[misc]
    """Wrapper class for TimesFM that supports multimodal inputs including text.

    This class extends TimesFM to handle both time series data and text descriptions,
    using a text encoder and fusion mechanism to combine multimodal information
    for improved time series forecasting.
    """

    def __init__(
        self,
        hparams: MultimodalTimesFmHparams,
        checkpoint: TimesFmCheckpoint,
    ) -> None:
        """Initializes MultimodalTimesFM wrapper.

        Args:
            hparams: Multimodal hyperparameters of the model.
            checkpoint: Checkpoint to load. checkpoint.version decides which TimesFM version to use.
        """
        # Initialize the parent TimesFM model
        super().__init__(hparams, checkpoint)

        self.enable_multimodal = hparams.enable_multimodal

        if self.enable_multimodal:
            # Initialize text encoder
            self.text_encoder: TextEncoder | None = TextEncoder(
                model_name=hparams.text_encoder_model, embedding_dim=hparams.text_embedding_dim
            )

            # Initialize fusion mechanism using addition-based fusion
            # Note: We'll need to determine the actual TimesFM feature dimensions
            # For now, using placeholder values that will need to be adjusted
            self.fusion: MultimodalFusion | None = MultimodalFusion(
                ts_feature_dim=512,  # Placeholder - needs actual TimesFM feature dim
                text_feature_dim=hparams.text_embedding_dim,
            )
        else:
            self.text_encoder = None
            self.fusion = None

    def forecast(
        self,
        inputs: Sequence[Any],
        freq: Sequence[int] | None = None,
        text_inputs: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecasts time series using multimodal inputs.

        Args:
            inputs: Input time series data sequences.
            freq: Optional frequency information for each input sequence.
            text_inputs: Optional text descriptions for each time series.
            **kwargs: Additional keyword arguments passed to `TimesFM.forecast()`.

        Returns:
        A tuple for np.array:
        - The mean forecast of size (# inputs, # forecast horizon).
        - The full forecast (mean + quantiles) of size (# inputs, # forecast horizon, 1 + # quantiles).
        """
        # If multimodal is disabled or no text inputs provided, use original TimesFM
        if not self.enable_multimodal or text_inputs is None:
            return super().forecast(inputs, freq=freq, **kwargs)  # type: ignore[no-any-return]

        # For now, use original TimesFM as the underlying model doesn't support our fusion yet
        # TODO: Implement actual multimodal forecasting when we can access TimesFM internals
        return super().forecast(inputs, freq=freq, **kwargs)  # type: ignore[no-any-return]

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text inputs into embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            Text embeddings tensor.

        Raises:
            RuntimeError: If multimodal functionality is disabled.
        """
        if not self.enable_multimodal or self.text_encoder is None:
            raise RuntimeError("Multimodal functionality is disabled")

        result = self.text_encoder(texts)
        return torch.as_tensor(result)

    def get_text_encoder(self) -> TextEncoder | None:
        """Get the text encoder component.

        Returns:
            Text encoder instance or None if multimodal is disabled.
        """
        return self.text_encoder

    def get_fusion_module(self) -> MultimodalFusion | None:
        """Get the fusion module.

        Returns:
            Fusion module instance or None if multimodal is disabled.
        """
        return self.fusion

    def is_multimodal_enabled(self) -> bool:
        """Check if multimodal functionality is enabled.

        Returns:
            True if multimodal is enabled, False otherwise.
        """
        return self.enable_multimodal
