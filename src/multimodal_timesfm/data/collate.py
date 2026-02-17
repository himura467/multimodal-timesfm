"""Collate functions for DataLoader batching."""

from typing import Any

import numpy as np
import torch


def _build_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    context = torch.from_numpy(np.stack([s["context"] for s in batch]))
    horizon = torch.from_numpy(np.stack([s["horizon"] for s in batch]))
    metadata = [s["metadata"] for s in batch]
    return {
        "context": context,
        "horizon": horizon,
        "metadata": metadata,
    }


def multimodal_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for multimodal batches with pre-computed text embeddings."""
    result = _build_batch(batch)
    result["text_embeddings"] = torch.from_numpy(np.stack([s["text_embeddings"] for s in batch]))
    return result


def baseline_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for baseline batches (no text)."""
    return _build_batch(batch)
