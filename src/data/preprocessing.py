"""Data preprocessing utilities for multimodal time series data."""

import re
from typing import Any

import numpy as np


def clean_text(text: Any) -> str:
    """Clean and normalize text input.

    Args:
        text: Raw text string to clean.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s\.\,\!\?\-]", "", text)

    return str(text)


def validate_text_inputs(text_inputs: list[str]) -> list[str]:
    """Validate and clean a list of text inputs.

    Args:
        text_inputs: List of text strings to validate.

    Returns:
        List of cleaned and validated text strings.

    Raises:
        ValueError: If any text input is empty after cleaning.
    """
    cleaned_texts = []

    for i, text in enumerate(text_inputs):
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError(f"Text input at index {i} is empty after cleaning: '{text}'")
        cleaned_texts.append(cleaned)

    return cleaned_texts


def align_text_and_timeseries(timeseries_data: list[Any], text_data: list[str]) -> tuple[list[Any], list[str]]:
    """Align text descriptions with time series data.

    Args:
        timeseries_data: List of time series data.
        text_data: List of text descriptions.

    Returns:
        Tuple of (aligned_timeseries, aligned_text).

    Raises:
        ValueError: If lengths don't match and cannot be aligned.
    """
    ts_len = len(timeseries_data)
    text_len = len(text_data)

    if ts_len == text_len:
        return timeseries_data, validate_text_inputs(text_data)

    if text_len == 1:
        # Repeat single text for all time series
        aligned_text = validate_text_inputs(text_data * ts_len)
        return timeseries_data, aligned_text

    if ts_len == 1:
        # Use first time series for all texts
        aligned_ts = timeseries_data * text_len
        aligned_text = validate_text_inputs(text_data)
        return aligned_ts, aligned_text

    raise ValueError(
        f"Cannot align time series data (length {ts_len}) with text data (length {text_len}). "
        "Lengths must match, or one of them must be length 1."
    )


def standardize_timeseries(data: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """Standardize time series data (zero mean, unit variance).

    Args:
        data: Time series data array.
        epsilon: Small value to avoid division by zero.

    Returns:
        Tuple of (standardized_data, mean, std).
    """
    mean = np.mean(data)
    std = np.std(data)

    # Avoid division by zero
    if std < epsilon:
        std = 1.0

    standardized = (data - mean) / std
    return standardized, mean, std


def denormalize_timeseries(standardized_data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Denormalize standardized time series data.

    Args:
        standardized_data: Standardized time series data.
        mean: Original mean value.
        std: Original standard deviation.

    Returns:
        Denormalized time series data.
    """
    return standardized_data * std + mean


def prepare_multimodal_batch(
    timeseries_batch: list[Any], text_batch: list[str], standardize: bool = True
) -> tuple[list[Any], list[str], dict[str, Any]]:
    """Prepare a batch of multimodal data for training/inference.

    Args:
        timeseries_batch: Batch of time series data.
        text_batch: Batch of text descriptions.
        standardize: Whether to standardize time series data.

    Returns:
        Tuple of (processed_timeseries, processed_text, metadata).
        metadata contains normalization parameters if standardize=True.
    """
    # Align text and time series data
    aligned_ts, aligned_text = align_text_and_timeseries(timeseries_batch, text_batch)

    metadata: dict[str, Any] = {}

    if standardize:
        # Standardize each time series in the batch
        standardized_ts = []
        normalization_params = []

        for ts in aligned_ts:
            if isinstance(ts, np.ndarray):
                std_ts, mean, std = standardize_timeseries(ts)
                standardized_ts.append(std_ts)
                normalization_params.append({"mean": mean, "std": std})
            else:
                # If not numpy array, keep as is
                standardized_ts.append(ts)
                normalization_params.append({"mean": 0.0, "std": 1.0})

        metadata["normalization_params"] = normalization_params
        aligned_ts = standardized_ts

    return aligned_ts, aligned_text, metadata


def extract_text_features(text: str) -> dict[str, Any]:
    """Extract basic features from text for analysis.

    Args:
        text: Input text string.

    Returns:
        Dictionary containing text features.
    """
    cleaned = clean_text(text)

    features = {
        "length": len(cleaned),
        "word_count": len(cleaned.split()),
        "has_numbers": bool(re.search(r"\d", cleaned)),
        "has_punctuation": bool(re.search(r"[.,!?]", cleaned)),
        "avg_word_length": np.mean([len(word) for word in cleaned.split()]) if cleaned.split() else 0.0,
    }

    return features
