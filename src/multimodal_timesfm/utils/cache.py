"""Dataset caching utilities for pre-processed data.

This module provides functionality to cache pre-processed datasets with optional text embeddings
to avoid redundant data loading, preprocessing, and text encoding during training iterations.

Supports two modes:
1. Multimodal: Caches preprocessed time series data + text embeddings
2. Baseline: Caches preprocessed time series data only (no text)
"""

import contextlib
import pickle
from pathlib import Path
from typing import Any

import torch

from multimodal_timesfm.text_encoder import TextEncoderBase
from multimodal_timesfm.utils.logging import get_logger


class DatasetCache:
    """Cache manager for pre-processed datasets with optional text embeddings.

    Supports both multimodal (with text embeddings) and baseline (without text) caching modes.
    The mode is determined by the text_encoder_type parameter passed to get_cache_path().
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize dataset cache manager.

        Args:
            cache_dir: Directory to store cached datasets.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def get_cache_path(
        self,
        dataset_name: str,
        domain: str,
        split_ratio: float,
        split: str,
        patch_len: int,
        context_len: int,
        horizon_len: int,
        text_encoder_type: str | None = None,
    ) -> Path:
        """Generate cache file path based on dataset configuration.

        Args:
            dataset_name: Name of the dataset (e.g., 'time_mmd').
            domain: Domain name.
            split_ratio: Train/test split ratio.
            split: Dataset split ('train' or 'test').
            patch_len: Patch length used for processing.
            context_len: Context length used for processing.
            horizon_len: Horizon length used for processing.
            text_encoder_type: Type of text encoder used. If None, creates baseline cache path.

        Returns:
            Path to cache file.
        """
        # Build cache filename from components
        base_parts = [
            dataset_name,
            domain,
            f"r{split_ratio:.2f}",
            split,
            f"p{patch_len}",
            f"c{context_len}",
            f"h{horizon_len}",
            text_encoder_type or "baseline",  # Use baseline suffix if no encoder type
        ]
        cache_filename = "_".join(base_parts) + ".pkl"
        return self.cache_dir / cache_filename

    def exists(self, cache_path: Path) -> bool:
        """Check if cache file exists.

        Args:
            cache_path: Path to cache file.

        Returns:
            True if cache exists, False otherwise.
        """
        return cache_path.exists()

    def save(self, cache_path: Path, data: list[dict[str, Any]]) -> None:
        """Save processed dataset to cache.

        Args:
            cache_path: Path to cache file.
            data: List of processed samples (with or without text embeddings).
        """
        self.logger.info(f"Saving {len(data)} samples to cache: {cache_path}")

        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Cache saved successfully: {cache_path} ({cache_size_mb:.2f} MB)")

    def load(self, cache_path: Path) -> list[dict[str, Any]]:
        """Load processed dataset from cache.

        Args:
            cache_path: Path to cache file.

        Returns:
            List of processed samples (with or without text embeddings depending on cache mode).
        """
        self.logger.info(f"Loading dataset from cache: {cache_path}")

        with open(cache_path, "rb") as f:
            data: list[dict[str, Any]] = pickle.load(f)

        self.logger.info(f"Loaded {len(data)} samples from cache")
        return data


def encode_patched_texts(
    patched_texts: list[list[str]], text_encoder: TextEncoderBase, device: torch.device
) -> torch.Tensor:
    """Encode patched texts into embeddings using text encoder.

    This function follows the same logic as MultimodalPatchedDecoder._encode_patch_text_features
    but operates on a single sample's patched texts.

    Args:
        patched_texts: List of text patches, where each patch contains multiple text strings.
                      Shape: [num_patches, variable_texts_per_patch]
        text_encoder: Text encoder model to use for encoding.
        device: Device to use for encoding.

    Returns:
        Tensor of text embeddings with shape [num_patches, embedding_dim].
        For each patch, all texts are joined and encoded as a single embedding.
    """
    num_patches = len(patched_texts)

    # Join multiple texts for each patch with space, similar to _encode_patch_text_features
    all_texts: list[str] = []
    for patch_texts in patched_texts:
        if patch_texts:
            text = " ".join(patch_texts)
        else:
            text = ""  # Empty text for patches without descriptions
        all_texts.append(text)

    # Encode all texts at once for efficiency
    all_embeddings: torch.Tensor
    if all_texts:
        all_embeddings = text_encoder(all_texts)  # Shape: (num_patches, text_embedding_dim)
    else:
        # Handle empty case
        all_embeddings = torch.zeros((num_patches, text_encoder.embedding_dim), device=device)

    return all_embeddings


def preprocess_dataset(
    raw_dataset: Any,
    text_encoder: TextEncoderBase | None = None,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Pre-process raw dataset with optional text encoding.

    Args:
        raw_dataset: Raw dataset object (e.g., TimeMmdDataset).
        text_encoder: Text encoder to use for encoding. If None, creates baseline cache (no text).
        device: Device to use for encoding. Required if text_encoder is provided.

    Returns:
        List of processed samples with or without text embeddings.
        Each sample contains:
        - context: Time series context (numpy array)
        - future: Time series future values (numpy array)
        - freq: Frequency indicator (int)
        - text_embeddings: Pre-encoded text embeddings (numpy array) - only if text_encoder provided
        - metadata: Original metadata
    """
    logger = get_logger()
    mode = "with text embeddings" if text_encoder is not None else "for baseline (no text)"
    logger.info(f"Pre-processing {len(raw_dataset)} samples {mode}")

    if text_encoder is not None:
        text_encoder.eval()

    processed_data = []

    context_manager = torch.no_grad() if text_encoder is not None else contextlib.nullcontext()
    with context_manager:
        for idx, sample in enumerate(raw_dataset):
            # Create base sample
            processed_sample: dict[str, Any] = {
                "context": sample["context"],
                "future": sample["future"],
                "freq": sample["freq"],
                "metadata": sample["metadata"],
            }

            # Add text embeddings if encoder is provided
            if text_encoder is not None:
                if device is None:
                    raise ValueError("device must be provided when text_encoder is specified")
                text_embeddings = encode_patched_texts(sample["patched_texts"], text_encoder, device)
                processed_sample["text_embeddings"] = text_embeddings.cpu().numpy()

            processed_data.append(processed_sample)

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(raw_dataset)} samples")

    logger.info(f"Completed pre-processing {len(processed_data)} samples")
    return processed_data


def create_or_load_cached_dataset(
    cache: DatasetCache,
    cache_path: Path,
    raw_dataset_factory: Any,
    text_encoder: TextEncoderBase | None = None,
    device: torch.device | None = None,
    force_rebuild: bool = False,
) -> list[dict[str, Any]]:
    """Create or load cached dataset with optional text embeddings.

    Args:
        cache: DatasetCache instance for managing cache files.
        cache_path: Path to cache file.
        raw_dataset_factory: Callable that creates the raw dataset when invoked.
        text_encoder: Text encoder to use for encoding. If None, creates baseline cache.
        device: Device to use for encoding. Required if text_encoder is provided.
        force_rebuild: If True, rebuild cache even if it exists.

    Returns:
        List of processed samples with or without text embeddings.
    """
    logger = get_logger()

    if not force_rebuild and cache.exists(cache_path):
        logger.info("Cache found - loading from cache")
        return cache.load(cache_path)

    logger.info("Cache not found - creating new cache")

    # Create raw dataset
    raw_dataset = raw_dataset_factory()

    # Pre-process with or without embeddings
    processed_data = preprocess_dataset(raw_dataset, text_encoder, device)

    # Save to cache
    cache.save(cache_path, processed_data)

    return processed_data
