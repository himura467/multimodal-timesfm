"""Dataset classes for multimodal time series."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from torch.utils.data import Dataset


class MultimodalDatasetBase(Dataset[dict[str, Any]], ABC):
    """Abstract base class for multimodal time series datasets.

    Each sample must contain:
    - context: np.ndarray - historical time series values
    - horizon: np.ndarray - target horizon values
    - patched_texts: list[list[str]] - text patches aligned to context patches
    - metadata: dict - additional sample information
    """

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class PreprocessedDataset(Dataset[dict[str, Any]]):
    """Dataset wrapping pre-processed samples with pre-computed text embeddings.

    Args:
        data: List of preprocessed samples.
            Each sample must contain context, horizon, text_embeddings, and metadata.
        mode: 'multimodal' requires text_embeddings in every sample;
            'baseline' drops text_embeddings if present.
    """

    def __init__(self, data: list[dict[str, Any]], mode: Literal["multimodal", "baseline"]) -> None:
        self.data = data
        self.mode = mode

        self._validate()

    def _validate(self) -> None:
        if self.mode == "multimodal" and not all("text_embeddings" in s for s in self.data):
            raise ValueError("All samples must contain 'text_embeddings' for multimodal mode")
        if self.mode == "baseline":
            for s in self.data:
                s.pop("text_embeddings", None)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
