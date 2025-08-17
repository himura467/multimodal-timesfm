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
    # Validate text inputs
    validated_text = validate_text_inputs(text_batch)

    metadata: dict[str, Any] = {}

    if standardize:
        # Standardize each time series in the batch
        standardized_ts = []
        normalization_params = []

        for ts in timeseries_batch:
            if isinstance(ts, np.ndarray):
                std_ts, mean, std = standardize_timeseries(ts)
                standardized_ts.append(std_ts)
                normalization_params.append({"mean": mean, "std": std})
            else:
                # If not numpy array, keep as is
                standardized_ts.append(ts)
                normalization_params.append({"mean": 0.0, "std": 1.0})

        metadata["normalization_params"] = normalization_params
        return standardized_ts, validated_text, metadata

    return timeseries_batch, validated_text, metadata
