"""Evaluation utilities for multimodal TimesFM."""

from typing import Any

import torch
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal_timesfm.multimodal_patched_decoder import MultimodalPatchedDecoder
from multimodal_timesfm.utils.device import move_to_device


def evaluate_multimodal_model(
    model: MultimodalPatchedDecoder,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate multimodal TimesFM model (with text).

    Args:
        model: Multimodal TimesFM model instance.
        dataloader: DataLoader providing batches of samples.
        device: Device to use for evaluation.

    Returns:
        Dictionary containing evaluation metrics (mse, mae).
    """
    model.eval()
    model.to(device)

    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_tensors = move_to_device(
                {"context": batch["context"], "future": batch["future"], "freq": batch["freq"]}, device
            )
            context = batch_tensors["context"]
            future = batch_tensors["future"]
            freq = batch_tensors["freq"]

            # Create input_padding tensor (zeros for now)
            input_padding = torch.zeros_like(context)

            # Handle both cached (text_embeddings) and regular (patched_texts) batches
            if "text_embeddings" in batch:
                # Cached dataset with pre-computed embeddings
                text_embeddings = move_to_device({"text_embeddings": batch["text_embeddings"]}, device)[
                    "text_embeddings"
                ]
                predictions = model(context, input_padding.float(), freq, text_embeddings=text_embeddings)
            else:
                # Regular dataset with text descriptions
                patched_texts = batch["patched_texts"]
                predictions = model(context, input_padding.float(), freq, patched_texts)

            predictions_mean = predictions[..., 0]  # [B, patches, horizon_len]
            last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

            # Compute metrics
            mse = torch.mean((last_patch_pred - future) ** 2)
            mae = torch.mean(torch.abs(last_patch_pred - future))

            total_mse += mse.item() * context.size(0)
            total_mae += mae.item() * context.size(0)
            num_samples += context.size(0)

    # Avoid division by zero if no samples were processed
    if num_samples == 0:
        return {
            "mse": 0.0,
            "mae": 0.0,
        }

    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples

    return {
        "mse": avg_mse,
        "mae": avg_mae,
    }


def evaluate_baseline_model(
    model: PatchedTimeSeriesDecoder,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate original TimesFM model (without text).

    Args:
        model: Original TimesFM model instance.
        dataloader: DataLoader providing batches of samples.
        device: Device to use for evaluation.

    Returns:
        Dictionary containing evaluation metrics (mse, mae).
    """
    model.eval()
    model.to(device)

    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating baseline"):
            batch_tensors = move_to_device(
                {"context": batch["context"], "future": batch["future"], "freq": batch["freq"]}, device
            )
            context = batch_tensors["context"]
            future = batch_tensors["future"]
            freq = batch_tensors["freq"]

            # Create input_padding tensor (zeros for now)
            input_padding = torch.zeros_like(context)

            # Forward pass without text
            predictions = model(context, input_padding.float(), freq)
            predictions_mean = predictions[..., 0]  # [B, patches, horizon_len]
            last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

            # Compute metrics
            mse = torch.mean((last_patch_pred - future) ** 2)
            mae = torch.mean(torch.abs(last_patch_pred - future))

            total_mse += mse.item() * context.size(0)
            total_mae += mae.item() * context.size(0)
            num_samples += context.size(0)

    # Avoid division by zero if no samples were processed
    if num_samples == 0:
        return {
            "mse": 0.0,
            "mae": 0.0,
        }

    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples

    return {
        "mse": avg_mse,
        "mae": avg_mae,
    }
