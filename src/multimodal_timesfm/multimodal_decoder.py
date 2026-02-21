"""Multimodal decoder for time series forecasting with text fusion."""

from dataclasses import dataclass, field

import torch
from torch import nn

from multimodal_timesfm.multimodal_fusion import MultimodalFusion
from multimodal_timesfm.tsfm.base import TsfmAdapter


@dataclass
class MultimodalDecoderConfig:
    """Configuration for MultimodalDecoder."""

    ts_embedding_dims: int = 1280
    text_embedding_dims: int = 384
    num_fusion_layers: int = 1
    fusion_hidden_dims: list[int | None] = field(default_factory=list)


class MultimodalDecoder(nn.Module):
    """Decoder for multimodal time series forecasting.

    Pipeline: adapter.preprocess -> fusion -> adapter.forward -> adapter.postprocess
    """

    def __init__(self, adapter: TsfmAdapter, config: MultimodalDecoderConfig) -> None:
        super().__init__()
        self.adapter = adapter
        self.config = config
        self.fusion = MultimodalFusion(
            ts_embedding_dims=config.ts_embedding_dims,
            text_embedding_dims=config.text_embedding_dims,
            num_layers=config.num_fusion_layers,
            hidden_dims=config.fusion_hidden_dims,
        )

    def forward_full(
        self,
        horizon: int,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the forecasting pipeline, returning all output channels.

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
        output_embeddings = self.adapter(embeddings, preprocessed.masks)
        return self.adapter.postprocess(horizon, output_embeddings, preprocessed.normalization_stats)

    def forward(
        self,
        horizon: int,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the forecasting pipeline, returning the point forecast.

        Args:
            horizon: Number of time steps to forecast.
            inputs: Input time series (batch_size, context_len).
            masks: Boolean masks (batch_size, context_len). True = padded, False = valid.
            text_embeddings: Pre-computed text embeddings (batch_size, num_patches, text_dims).

        Returns:
            Point forecast (batch_size, horizon).
        """
        return self.forward_full(horizon, inputs, masks, text_embeddings)[..., self.adapter.point_forecast_index]
