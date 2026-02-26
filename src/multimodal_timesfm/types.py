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


class CheckpointBase(TypedDict):
    """Base fields shared by all checkpoint types."""

    epoch: int
    global_step: int
    optimizer_state_dict: dict[str, Any]
    best_val_loss: float


class MultimodalCheckpoint(CheckpointBase):
    """Checkpoint for multimodal mode."""

    fusion_state_dict: dict[str, Any]


class BaselineCheckpoint(CheckpointBase):
    """Checkpoint for baseline mode."""

    adapter_state_dict: dict[str, Any]


class EvaluationMetrics(TypedDict):
    """Evaluation metrics."""

    mse: float
    mae: float
