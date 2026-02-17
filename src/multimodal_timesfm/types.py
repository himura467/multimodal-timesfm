"""Shared types for the multimodal_timesfm package."""

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

TrainingMode = Literal["multimodal", "baseline"]


class RawSample(TypedDict):
    """A single raw dataset sample before preprocessing."""

    context: npt.NDArray[np.float32]
    horizon: npt.NDArray[np.float32]
    patched_texts: list[list[str]]
    metadata: dict[str, Any]


class PreprocessedSample(TypedDict):
    """A single dataset sample after preprocessing."""

    context: npt.NDArray[np.float32]
    horizon: npt.NDArray[np.float32]
    text_embeddings: NotRequired[npt.NDArray[np.float32]]
    metadata: dict[str, Any]


class Batch(TypedDict):
    """A collated batch of samples."""

    context: torch.Tensor
    horizon: torch.Tensor
    text_embeddings: NotRequired[torch.Tensor]
    metadata: list[dict[str, Any]]
