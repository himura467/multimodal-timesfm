"""Cross-validation utilities for Time-MMD dataset.

This module provides Time-MMD-specific wrappers around the core cross-validation utilities.
"""

from pathlib import Path
from typing import Any, Literal

from torch.utils.data import ConcatDataset, Dataset

from examples.time_mmd.data.time_mmd_dataset import TimeMmdDataset
from multimodal_timesfm.cross_validation import (
    create_fold_datasets as create_fold_datasets_core,
)
from multimodal_timesfm.cross_validation import (
    get_cross_validation_splits as get_cross_validation_splits_core,
)
from multimodal_timesfm.utils.cache import DatasetCache
from multimodal_timesfm.utils.cached_dataset import CachedDataset


def _time_mmd_dataset_factory(
    data_path: Path,
    entity: str,
    patch_len: int,
    context_len: int,
    horizon_len: int,
    **kwargs: Any,
) -> Dataset[dict[str, Any]]:
    """Factory function to create Time-MMD dataset for a single domain.

    Args:
        data_path: Root directory containing Time-MMD dataset.
        entity: Entity name (domain name for Time-MMD).
        patch_len: Length of input patches.
        context_len: Length of context window.
        horizon_len: Length of forecasting horizon.
        **kwargs: Additional keyword arguments (unused for Time-MMD).

    Returns:
        Time-MMD dataset instance for the specified domain.
    """
    return TimeMmdDataset(
        data_dir=data_path,
        domain=entity,
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
    split_ratio: float,
    patch_len: int,
    context_len: int,
    horizon_len: int,
    text_encoder_type: Literal["english", "japanese"] | None,
    cache_dir: Path | None = None,
) -> tuple[ConcatDataset[dict[str, Any]], ConcatDataset[dict[str, Any]], ConcatDataset[dict[str, Any]]]:
    """Create datasets for a single fold.

    Args:
        data_path: Root directory containing Time-MMD dataset.
        train_domains: List of domain names for training.
        val_domains: List of domain names for validation.
        test_domains: List of domain names for testing.
        split_ratio: Train/test split ratio used when generating cache.
        patch_len: Length of input patches.
        context_len: Length of context window.
        horizon_len: Length of forecasting horizon.
        text_encoder_type: Type of text encoder used for caching. If None, uses baseline cache.
        cache_dir: Optional directory containing cached datasets. If provided, loads from cache.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    if cache_dir is not None:
        cache = DatasetCache(cache_dir)

        def load_cached_domains(domains: list[str]) -> list[Dataset[dict[str, Any]]]:
            datasets = []
            for domain in domains:
                cache_path = cache.get_cache_path(
                    dataset_name="time_mmd",
                    domain=domain,
                    split_ratio=split_ratio,
                    split="train",  # CV mode uses split="train" with split_ratio=1.0
                    patch_len=patch_len,
                    context_len=context_len,
                    horizon_len=horizon_len,
                    text_encoder_type=text_encoder_type,
                )
                cached_data = cache.load(cache_path)
                datasets.append(CachedDataset(cached_data))
            return datasets

        train_datasets = load_cached_domains(train_domains)
        val_datasets = load_cached_domains(val_domains)
        test_datasets = load_cached_domains(test_domains)

        return (
            ConcatDataset(train_datasets),
            ConcatDataset(val_datasets),
            ConcatDataset(test_datasets),
        )
    else:
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
