"""Generic cross-validation utilities for multimodal TimesFM."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from torch.utils.data import ConcatDataset, Dataset


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation training.

    Attributes:
        n_folds: Number of folds for cross-validation (default: 5).
        train_ratio: Proportion of entities for training (default: 0.8).
        val_ratio: Proportion of entities for validation (default: 0.1).
        test_ratio: Proportion of entities for testing (default: 0.1).
    """

    n_folds: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def __post_init__(self) -> None:
        """Validate cross-validation configuration."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Train, validation, and test ratios must sum to 1.0, got {total_ratio}")
        if self.n_folds < 2:
            raise ValueError(f"Number of folds must be at least 2, got {self.n_folds}")


class DatasetFactory(Protocol):
    """Protocol for dataset factory functions.

    A dataset factory is a callable that takes an entity name and dataset
    parameters, and returns a Dataset instance.
    """

    def __call__(
        self,
        data_path: Path,
        entity: str,
        patch_len: int,
        context_len: int,
        horizon_len: int,
        **kwargs: Any,
    ) -> Dataset[dict[str, Any]]: ...


def get_cross_validation_splits(
    all_entities: list[str],
    n_folds: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int | None = None,
) -> list[tuple[list[str], list[str], list[str]]]:
    """Generate cross-validation splits by rotating entities across folds.

    Args:
        all_entities: List of all entity names.
        n_folds: Number of folds for cross-validation.
        train_ratio: Proportion of entities for training.
        val_ratio: Proportion of entities for validation.
        test_ratio: Proportion of entities for testing.
        seed: Random seed for reproducibility.

    Returns:
        List of tuples, each containing (train_entities, val_entities, test_entities) for a fold.

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train, validation, and test ratios must sum to 1.0, got {total_ratio}")

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Shuffle entities once
    shuffled_entities = all_entities.copy()
    random.shuffle(shuffled_entities)

    # Calculate sizes for each split
    n_entities = len(shuffled_entities)
    n_train = int(n_entities * train_ratio)
    n_val = int(n_entities * val_ratio)
    n_test = n_entities - n_train - n_val  # Remainder goes to test

    # Calculate fold size (entities per fold)
    fold_size = n_entities // n_folds

    splits = []
    for fold_idx in range(n_folds):
        # Rotate entities for this fold
        offset = fold_idx * fold_size
        rotated_entities = shuffled_entities[offset:] + shuffled_entities[:offset]

        # Split rotated entities
        train_entities = rotated_entities[:n_train]
        val_entities = rotated_entities[n_train : n_train + n_val]
        test_entities = rotated_entities[n_train + n_val : n_train + n_val + n_test]

        splits.append((train_entities, val_entities, test_entities))

    return splits


def create_fold_datasets(
    data_path: Path,
    train_entities: list[str],
    val_entities: list[str],
    test_entities: list[str],
    dataset_factory: DatasetFactory,
    patch_len: int,
    context_len: int,
    horizon_len: int,
    **dataset_kwargs: Any,
) -> tuple[ConcatDataset[dict[str, Any]], ConcatDataset[dict[str, Any]], ConcatDataset[dict[str, Any]]]:
    """Create datasets for a single fold using a dataset factory.

    Args:
        data_path: Root directory containing dataset.
        train_entities: List of entity names for training.
        val_entities: List of entity names for validation.
        test_entities: List of entity names for testing.
        dataset_factory: Factory function that creates a dataset for a single entity.
        patch_len: Length of input patches.
        context_len: Length of context window.
        horizon_len: Length of forecasting horizon.
        **dataset_kwargs: Additional keyword arguments to pass to dataset_factory.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_datasets = []
    val_datasets = []
    test_datasets = []

    # Load training entities
    for entity in train_entities:
        dataset = dataset_factory(
            data_path=data_path,
            entity=entity,
            patch_len=patch_len,
            context_len=context_len,
            horizon_len=horizon_len,
            **dataset_kwargs,
        )
        train_datasets.append(dataset)

    # Load validation entities
    for entity in val_entities:
        dataset = dataset_factory(
            data_path=data_path,
            entity=entity,
            patch_len=patch_len,
            context_len=context_len,
            horizon_len=horizon_len,
            **dataset_kwargs,
        )
        val_datasets.append(dataset)

    # Load test entities
    for entity in test_entities:
        dataset = dataset_factory(
            data_path=data_path,
            entity=entity,
            patch_len=patch_len,
            context_len=context_len,
            horizon_len=horizon_len,
            **dataset_kwargs,
        )
        test_datasets.append(dataset)

    # Concatenate datasets
    combined_train: ConcatDataset[dict[str, Any]] = ConcatDataset(train_datasets)
    combined_val: ConcatDataset[dict[str, Any]] = ConcatDataset(val_datasets)
    combined_test: ConcatDataset[dict[str, Any]] = ConcatDataset(test_datasets)

    return combined_train, combined_val, combined_test
