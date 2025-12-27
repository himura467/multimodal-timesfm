"""Cached dataset implementation for pre-processed data.

This module provides a unified dataset wrapper for both multimodal and baseline cached data.
"""

from typing import Any

from torch.utils.data import Dataset


class CachedDataset(Dataset[dict[str, Any]]):
    """Dataset that wraps pre-processed cached data with optional text embeddings.

    This dataset provides a simple wrapper around pre-processed data loaded from cache,
    avoiding redundant data loading and preprocessing operations during training.

    Supports two modes:
    1. Multimodal: Cached data includes pre-computed text embeddings
    2. Baseline: Cached data includes only preprocessed time series (no text)

    The mode is determined automatically based on the presence of 'text_embeddings' field
    in the cached data.
    """

    def __init__(self, cached_data: list[dict[str, Any]]) -> None:
        """Initialize cached dataset.

        Args:
            cached_data: List of pre-processed samples.

                        For multimodal cache, each sample contains:
                        - context: Time series context (numpy array)
                        - future: Time series future values (numpy array)
                        - freq: Frequency indicator (int)
                        - text_embeddings: Pre-computed text embeddings (numpy array)
                        - metadata: Sample metadata (dict)

                        For baseline cache, each sample contains:
                        - context: Time series context (numpy array)
                        - future: Time series future values (numpy array)
                        - freq: Frequency indicator (int)
                        - metadata: Sample metadata (dict)
        """
        self.data = cached_data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing sample data. The structure depends on the cache mode:
            - Multimodal: includes 'text_embeddings' field with pre-computed embeddings
            - Baseline: no 'text_embeddings' field
        """
        return self.data[idx]

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.data)
