"""Cross-validation utilities for Time-MMD dataset.

This module provides Time-MMD-specific wrappers around the core cross-validation utilities.
"""

from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, Dataset

from examples.time_mmd.data.time_mmd_dataset import TimeMmdDataset
from multimodal_timesfm.cross_validation import (
    create_fold_datasets as create_fold_datasets_core,
)
from multimodal_timesfm.cross_validation import (
    get_cross_validation_splits as get_cross_validation_splits_core,
)


def _time_mmd_dataset_factory(
    data_path: Path,
    domain: str,
    patch_len: int,
    context_len: int,
    horizon_len: int,
    **kwargs: Any,
) -> Dataset[dict[str, Any]]:
    """Factory function to create Time-MMD dataset for a single domain.

    Args:
        data_path: Root directory containing Time-MMD dataset.
        domain: Domain name.
        patch_len: Length of input patches.
        context_len: Length of context window.
        horizon_len: Length of forecasting horizon.
        **kwargs: Additional keyword arguments (unused for Time-MMD).

    Returns:
        Time-MMD dataset instance for the specified domain.
    """
    return TimeMmdDataset(
        data_dir=data_path,
        domain=domain,
        split_ratio=1.0,  # Use all data from each domain in CV mode
        split="train",
        patch_len=patch_len,
        context_len=context_len,
        horizon_len=horizon_len,
    )


def get_all_domains(data_path: Path) -> list[str]:
    """Get all available domains from the Time-MMD dataset.

    Args:
        data_path: Root directory containing Time-MMD dataset.

    Returns:
        List of domain names.
    """
    numerical_dir = data_path / "numerical"
    if not numerical_dir.exists():
        raise FileNotFoundError(f"Numerical data directory not found: {numerical_dir}")

    domains = []
    for domain_dir in numerical_dir.iterdir():
        if domain_dir.is_dir():
            domains.append(domain_dir.name)

    domains.sort()  # Sort for consistency
    return domains


def get_cross_validation_splits(
    all_domains: list[str],
    n_folds: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int | None = None,
) -> list[tuple[list[str], list[str], list[str]]]:
    """Generate cross-validation splits for Time-MMD dataset.

    Args:
        all_domains: List of all domain names.
        n_folds: Number of folds for cross-validation.
        train_ratio: Proportion of domains for training.
        val_ratio: Proportion of domains for validation.
        test_ratio: Proportion of domains for testing.
        seed: Random seed for reproducibility.

    Returns:
        List of tuples, each containing (train_domains, val_domains, test_domains) for a fold.
    """
    return get_cross_validation_splits_core(
        all_entities=all_domains,
        n_folds=n_folds,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def create_fold_datasets(
    data_path: Path,
    train_domains: list[str],
    val_domains: list[str],
    test_domains: list[str],
    patch_len: int,
    context_len: int,
    horizon_len: int,
) -> tuple[ConcatDataset[dict[str, Any]], ConcatDataset[dict[str, Any]], ConcatDataset[dict[str, Any]]]:
    """Create datasets for a single fold.

    Args:
        data_path: Root directory containing Time-MMD dataset.
        train_domains: List of domain names for training.
        val_domains: List of domain names for validation.
        test_domains: List of domain names for testing.
        patch_len: Length of input patches.
        context_len: Length of context window.
        horizon_len: Length of forecasting horizon.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    return create_fold_datasets_core(
        data_path=data_path,
        train_entities=train_domains,
        val_entities=val_domains,
        test_entities=test_domains,
        dataset_factory=_time_mmd_dataset_factory,
        patch_len=patch_len,
        context_len=context_len,
        horizon_len=horizon_len,
    )
