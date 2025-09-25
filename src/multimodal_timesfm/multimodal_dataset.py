"""Base dataset class for multimodal time series forecasting."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Dataset


class MultimodalDatasetBase(Dataset[dict[str, Any]], ABC):
    """Abstract base class for multimodal time series datasets.

    This class provides the core interface and validation logic that all
    multimodal time series datasets should implement. Subclasses must
    implement the data loading and processing methods specific to their
    data format.

    The dataset produces samples with the following structure:
    {
        "context": numpy array of shape (context_len, n_features) - historical time series data
        "future": numpy array of shape (horizon_len, n_features) - target future values
        "freq": int - frequency indicator (0=daily, 1=weekly/monthly, 2=quarterly+)
        "patched_texts": list of lists - text data organized by temporal patches
        "metadata": dict - additional information about the sample
    }
    """

    def __init__(
        self,
        data_dir: Path,
        split_ratio: float = 0.8,
        split: Literal["train", "test"] = "train",
        patch_len: int = 32,
        context_len: int = 128,
        horizon_len: int = 32,
    ) -> None:
        """Initialize the base dataset.

        Args:
            data_dir: Root directory containing the dataset.
            split_ratio: Train/test split ratio (default 0.8 for 80% train).
            split: Dataset split ('train' or 'test').
            patch_len: Length of input patches for temporal alignment.
            context_len: Length of context window for input sequences.
                        Must be an integer multiple of patch_len.
            horizon_len: Length of forecasting horizon.
                        Must be an integer multiple of patch_len.

        Raises:
            ValueError: If context_len or horizon_len are not multiples of patch_len.
        """
        # Validate that context_len is an integer multiple of patch_len
        if context_len % patch_len != 0:
            raise ValueError(f"context_len ({context_len}) must be an integer multiple of patch_len ({patch_len})")

        # Validate that horizon_len is an integer multiple of patch_len
        if horizon_len % patch_len != 0:
            raise ValueError(f"horizon_len ({horizon_len}) must be an integer multiple of patch_len ({patch_len})")

        # Validate split_ratio
        if not 0.0 < split_ratio < 1.0:
            raise ValueError(f"split_ratio ({split_ratio}) must be between 0 and 1")

        self.data_dir = Path(data_dir)
        self.split_ratio = split_ratio
        self.split = split
        self.patch_len = patch_len
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.data: list[dict[str, Any]] = []

        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self._load_data()

    @abstractmethod
    def _load_data(self) -> None:
        """Load raw data from files.

        This method should populate self.data with processed samples.
        Each sample should be a dictionary with keys:
        - context: np.ndarray of shape (context_len, n_features)
        - future: np.ndarray of shape (horizon_len, n_features)
        - freq: int frequency indicator
        - patched_texts: list of text patches
        - metadata: dict with additional info
        """
        pass

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get dataset item by index.

        Args:
            idx: Item index.

        Returns:
            Dictionary containing context, future, freq, patched_texts, and metadata.
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return self.data[idx]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
