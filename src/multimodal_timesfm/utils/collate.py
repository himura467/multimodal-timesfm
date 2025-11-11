"""Collate functions for multimodal data batching."""

from typing import Any

import torch


def multimodal_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for multimodal data.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Batched data dictionary with stacked tensors and collected text data.
    """
    # Stack time series data
    context = torch.stack([torch.from_numpy(sample["context"]) for sample in batch])
    future = torch.stack([torch.from_numpy(sample["future"]) for sample in batch])
    freq = torch.stack([torch.tensor(sample["freq"]) for sample in batch])

    # Collect patched texts for each batch item
    patched_texts = []
    for sample in batch:
        patched_texts.append(sample["patched_texts"])

    # Collect metadata for each batch item
    metadata = []
    for sample in batch:
        if "metadata" in sample:
            metadata.append(sample["metadata"])

    return {
        "context": context.squeeze(-1),
        "future": future.squeeze(-1),
        "freq": freq.unsqueeze(-1),
        "patched_texts": patched_texts,
        "metadata": metadata,
    }


def baseline_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for baseline (non-multimodal) data.

    This function is used for the baseline TimesFM model that doesn't use text inputs.
    It only stacks the time series data and ignores any text descriptions.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Batched data dictionary with stacked tensors (no text data).
    """
    # Stack time series data
    context = torch.stack([torch.from_numpy(sample["context"]) for sample in batch])
    future = torch.stack([torch.from_numpy(sample["future"]) for sample in batch])
    freq = torch.stack([torch.tensor(sample["freq"]) for sample in batch])

    # Collect metadata for each batch item
    metadata = []
    for sample in batch:
        if "metadata" in sample:
            metadata.append(sample["metadata"])

    return {
        "context": context.squeeze(-1),
        "future": future.squeeze(-1),
        "freq": freq.unsqueeze(-1),
        "metadata": metadata,
    }
