"""MultimodalTimesFM wrapper class that extends TimesFM to support multimodal inputs."""

from dataclasses import dataclass
from typing import Literal

import torch
from timesfm import TimesFmCheckpoint, TimesFmHparams
from timesfm.timesfm_torch import TimesFmTorch as TimesFm

from src.models.multimodal_fusion import MultimodalFusion
from src.models.text_encoder import EnglishTextEncoder, JapaneseTextEncoder, TextEncoderBase


@dataclass(kw_only=True)
class MultimodalTimesFmHparams(TimesFmHparams):  # type: ignore[misc]
    """Hyperparameters for MultimodalTimesFM that extend TimesFmHparams.

    Attributes:
        text_encoder_type: Type of text encoder to use ('english' or 'japanese').
    """

    text_encoder_type: Literal["english", "japanese"] = "english"


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
        device: torch.device | str | None = None,
    ) -> None:
        """Initializes MultimodalTimesFM wrapper.

        Args:
            hparams: Multimodal hyperparameters of the model.
            checkpoint: Checkpoint to load. checkpoint.version decides which TimesFM version to use.
            device: Device to use for the model. If None, will auto-resolve the best available device.
        """
        # Initialize the parent TimesFM model
        super().__init__(hparams, checkpoint)

        # Initialize text encoder based on type
        self.text_encoder: TextEncoderBase
        if hparams.text_encoder_type == "english":
            self.text_encoder = EnglishTextEncoder(device=device)
        elif hparams.text_encoder_type == "japanese":
            self.text_encoder = JapaneseTextEncoder(device=device)
        else:
            raise ValueError(
                f"Unsupported text encoder type: {hparams.text_encoder_type}. Must be 'english' or 'japanese'."
            )

        # Initialize fusion mechanism
        self.fusion = MultimodalFusion(
            ts_feature_dim=hparams.model_dims,
            text_feature_dim=self.text_encoder.embedding_dim,
        )
