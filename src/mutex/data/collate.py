"""Collate functions for DataLoader batching."""

import numpy as np
import torch

from multimodal_timesfm.types import Batch, PreprocessedSample


def _build_batch(batch: list[PreprocessedSample]) -> Batch:
    context = torch.from_numpy(np.stack([s["context"] for s in batch]))
    horizon = torch.from_numpy(np.stack([s["horizon"] for s in batch]))
    metadata = [s["metadata"] for s in batch]
    return Batch(
        context=context,
        horizon=horizon,
        metadata=metadata,
    )


def multimodal_collate_fn(batch: list[PreprocessedSample]) -> Batch:
    """Collate function for multimodal batches with pre-computed text embeddings."""
    result = _build_batch(batch)
    result["text_embeddings"] = torch.from_numpy(np.stack([s["text_embeddings"] for s in batch]))
    return result


def baseline_collate_fn(batch: list[PreprocessedSample]) -> Batch:
    """Collate function for baseline batches (no text)."""
    return _build_batch(batch)
