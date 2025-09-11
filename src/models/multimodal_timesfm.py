"""MultimodalTimesFM wrapper class that extends TimesFM to support multimodal inputs."""

from dataclasses import dataclass

import torch
from timesfm import TimesFmCheckpoint, TimesFmHparams
from timesfm.timesfm_torch import TimesFmTorch as TimesFm

from src.models.multimodal_fusion import MultimodalFusion
from src.models.text_encoder import TextEncoder


@dataclass(kw_only=True)
class MultimodalTimesFmHparams(TimesFmHparams):  # type: ignore[misc]
    """Hyperparameters for MultimodalTimesFM that extend TimesFmHparams.

    Attributes:
        text_encoder_model: Name of the sentence transformer model for text encoding.
        text_embedding_dim: Dimension of text embeddings.
    """

    text_encoder_model: str = "all-MiniLM-L6-v2"
    text_embedding_dim: int = 384


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

        # Initialize text encoder
        self.text_encoder = TextEncoder(
            model_name=hparams.text_encoder_model, embedding_dim=hparams.text_embedding_dim, device=device
        )

        # Initialize fusion mechanism
        self.fusion = MultimodalFusion(
            ts_feature_dim=hparams.model_dims,
            text_feature_dim=hparams.text_embedding_dim,
        )
