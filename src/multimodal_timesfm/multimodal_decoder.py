"""Multimodal decoder for time series forecasting with text fusion."""

from dataclasses import dataclass, field

import torch

from multimodal_timesfm.tsfm.base import TsfmAdapter
from multimodal_timesfm.multimodal_fusion import MultimodalFusion


@dataclass
class MultimodalDecoderConfig:
    """Configuration for MultimodalDecoder."""

    ts_embedding_dims: int = 1280
    text_embedding_dims: int = 384
    num_fusion_layers: int = 1
    fusion_hidden_dims: list[int | None] = field(default_factory=list)


class MultimodalDecoder:
    """Decoder for multimodal time series forecasting.

    Pipeline: adapter.preprocess -> fusion -> adapter.decode -> adapter.postprocess
    """

    def __init__(self, adapter: TsfmAdapter, config: MultimodalDecoderConfig) -> None:
        self.adapter = adapter
        self.config = config
        self.fusion = MultimodalFusion(
            ts_embedding_dims=config.ts_embedding_dims,
            text_embedding_dims=config.text_embedding_dims,
            num_layers=config.num_fusion_layers,
            hidden_dims=config.fusion_hidden_dims,
        )

    def forecast(
        self,
        horizon: int,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the forecasting pipeline.

        When text_embeddings is provided, fusion is applied before decoding.
        When None, the pipeline runs without fusion.

        Args:
            horizon: Number of time steps to forecast.
            inputs: Input time series (batch_size, context_len).
            masks: Boolean masks (batch_size, context_len). True = padded, False = valid.
            text_embeddings: Pre-computed text embeddings (batch_size, num_patches, text_dims).

        Returns:
            Predictions (batch_size, horizon, num_outputs).
        """
        preprocessed = self.adapter.preprocess(inputs, masks)
        embeddings = (
            self.fusion(preprocessed.input_embeddings, text_embeddings)
            if text_embeddings is not None
            else preprocessed.input_embeddings
        )
        output_embeddings = self.adapter.decode(embeddings, preprocessed.masks)
        return self.adapter.postprocess(horizon, output_embeddings, preprocessed.normalization_stats)
