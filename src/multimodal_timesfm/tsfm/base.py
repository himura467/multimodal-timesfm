"""Abstract adapter interface for time series foundation models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PreprocessResult:
    """Result of the preprocessing.

    Attributes:
        input_embeddings: Embeddings produced by the adapter's tokenizer.
        masks: Boolean masks. True = padded, False = valid.
        normalization_stats: Adapter-specific normalization statistics.
    """

    input_embeddings: torch.Tensor
    masks: torch.Tensor
    normalization_stats: dict[str, torch.Tensor]


class TsfmAdapter(nn.Module, ABC):
    """Base interface for time series foundation model adapters.

    Pipeline: preprocess -> [fusion injection point] -> decode -> postprocess
    """

    @abstractmethod
    def preprocess(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
    ) -> PreprocessResult: ...

    @abstractmethod
    def forward(
        self,
        input_embeddings: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor: ...

    @abstractmethod
    def postprocess(
        self,
        horizon: int,
        output_embeddings: torch.Tensor,
        normalization_stats: dict[str, torch.Tensor],
    ) -> torch.Tensor: ...

    @abstractmethod
    def freeze_parameters(self) -> None: ...

    @abstractmethod
    def unfreeze_parameters(self) -> None: ...
