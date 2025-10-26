"""Tests for cross-validation utilities."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
from torch.utils.data import ConcatDataset, Dataset

from multimodal_timesfm.cross_validation import (
    CrossValidationConfig,
    create_fold_datasets,
    get_cross_validation_splits,
)


class TestCrossValidationConfig:
    """Test cases for CrossValidationConfig class."""

    def test_init_default_parameters(self) -> None:
        """Tests initialization with default parameters."""
        config = CrossValidationConfig()

        assert config.n_folds == 5
        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1
        assert config.test_ratio == 0.1

    def test_init_custom_parameters(self) -> None:
        """Tests initialization with custom parameters."""
        config = CrossValidationConfig(
            n_folds=10,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        assert config.n_folds == 10
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15

    def test_validation_ratios_sum_to_one(self) -> None:
        """Tests that validation fails if ratios don't sum to 1.0."""
        with pytest.raises(ValueError, match="Train, validation, and test ratios must sum to 1.0"):
            CrossValidationConfig(
                n_folds=5,
                train_ratio=0.7,
                val_ratio=0.1,
                test_ratio=0.1,  # Sum = 0.9
            )

    def test_validation_ratios_sum_exceeds_one(self) -> None:
        """Tests that validation fails if ratios sum exceeds 1.0."""
        with pytest.raises(ValueError, match="Train, validation, and test ratios must sum to 1.0"):
            CrossValidationConfig(
                n_folds=5,
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.2,  # Sum = 1.2
            )

    def test_validation_minimum_folds(self) -> None:
        """Tests that validation fails if n_folds < 2."""
        with pytest.raises(ValueError, match="Number of folds must be at least 2"):
            CrossValidationConfig(
                n_folds=1,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
            )

    def test_valid_custom_ratios(self) -> None:
        """Tests that custom ratios that sum to 1.0 are accepted."""
        config = CrossValidationConfig(
            n_folds=3,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
        )

        assert config.train_ratio == 0.5
        assert config.val_ratio == 0.25
        assert config.test_ratio == 0.25

    def test_floating_point_tolerance(self) -> None:
        """Tests that floating point errors are tolerated."""
        # 0.6 + 0.2 + 0.2 might not exactly equal 1.0 due to floating point
        config = CrossValidationConfig(
            n_folds=5,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        assert config.train_ratio == 0.6
        assert config.val_ratio == 0.2
        assert config.test_ratio == 0.2


class TestGetCrossValidationSplits:
    """Test cases for get_cross_validation_splits function."""

    def test_basic_splits(self) -> None:
        """Tests basic cross-validation splits."""
        entities = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        n_folds = 5
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2

        splits = get_cross_validation_splits(
            all_entities=entities,
            n_folds=n_folds,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=42,
        )

        # Should have n_folds splits
        assert len(splits) == n_folds

        # Each split should have train, val, test entities
        for train_entities, val_entities, test_entities in splits:
            assert len(train_entities) == 6  # 60% of 10
            assert len(val_entities) == 2  # 20% of 10
            assert len(test_entities) == 2  # 20% of 10

            # No overlap between splits
            assert len(set(train_entities) & set(val_entities)) == 0
            assert len(set(train_entities) & set(test_entities)) == 0
            assert len(set(val_entities) & set(test_entities)) == 0

            # All entities accounted for
            all_fold_entities = set(train_entities + val_entities + test_entities)
            assert len(all_fold_entities) == len(entities)

    def test_reproducibility_with_seed(self) -> None:
        """Tests that splits are reproducible with the same seed."""
        entities = ["A", "B", "C", "D", "E"]

        splits1 = get_cross_validation_splits(
            all_entities=entities,
            n_folds=3,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        splits2 = get_cross_validation_splits(
            all_entities=entities,
            n_folds=3,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        assert splits1 == splits2

    def test_different_seeds_produce_different_splits(self) -> None:
        """Tests that different seeds produce different splits."""
        entities = ["A", "B", "C", "D", "E", "F", "G", "H"]

        splits1 = get_cross_validation_splits(
            all_entities=entities,
            n_folds=4,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        splits2 = get_cross_validation_splits(
            all_entities=entities,
            n_folds=4,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=123,
        )

        assert splits1 != splits2

    def test_rotation_across_folds(self) -> None:
        """Tests that entities rotate across folds."""
        entities = ["A", "B", "C", "D", "E", "F"]

        splits = get_cross_validation_splits(
            all_entities=entities,
            n_folds=3,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )

        # Collect all entities that appeared in test sets
        all_test_entities = set()
        for _, _, test_entities in splits:
            all_test_entities.update(test_entities)

        # All entities should appear in at least one test set across folds
        assert len(all_test_entities) == len(entities)


class TestCreateFoldDatasets:
    """Test cases for create_fold_datasets function."""

    def test_create_fold_datasets_with_mock_factory(self) -> None:
        """Tests create_fold_datasets with a mock dataset factory."""

        class MockDataset(Dataset[dict[str, Any]]):
            """Mock dataset for testing."""

            def __init__(self, entity: str, size: int = 10) -> None:
                self.entity = entity
                self.size = size

            def __len__(self) -> int:
                return self.size

            def __getitem__(self, idx: int) -> dict[str, Any]:
                if idx < 0 or idx >= self.size:
                    raise IndexError(f"Index {idx} out of range")
                return {"entity": self.entity, "index": idx}

        def mock_factory(
            data_path: Path,
            entity: str,
            patch_len: int,
            context_len: int,
            horizon_len: int,
            **kwargs: Any,
        ) -> Dataset[dict[str, Any]]:
            """Mock factory function."""
            return MockDataset(entity=entity, size=5)

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            train_entities = ["A", "B"]
            val_entities = ["C"]
            test_entities = ["D", "E"]

            train_dataset, val_dataset, test_dataset = create_fold_datasets(
                data_path=data_path,
                train_entities=train_entities,
                val_entities=val_entities,
                test_entities=test_entities,
                dataset_factory=mock_factory,
                patch_len=8,
                context_len=32,
                horizon_len=16,
            )

            # Check dataset types
            assert isinstance(train_dataset, ConcatDataset)
            assert isinstance(val_dataset, ConcatDataset)
            assert isinstance(test_dataset, ConcatDataset)

            # Check dataset sizes
            assert len(train_dataset) == 10  # 2 entities * 5 samples each
            assert len(val_dataset) == 5  # 1 entity * 5 samples
            assert len(test_dataset) == 10  # 2 entity * 5 samples

            # Check samples
            sample = train_dataset[0]
            assert "entity" in sample
            assert sample["entity"] in train_entities

    def test_dataset_factory_receives_correct_parameters(self) -> None:
        """Tests that dataset_factory receives correct standard parameters."""
        received_calls: list[dict[str, Any]] = []

        def mock_factory(
            data_path: Path,
            entity: str,
            patch_len: int,
            context_len: int,
            horizon_len: int,
            **kwargs: Any,
        ) -> Dataset[dict[str, Any]]:
            """Mock factory function that captures parameters."""
            received_calls.append(
                {
                    "data_path": data_path,
                    "entity": entity,
                    "patch_len": patch_len,
                    "context_len": context_len,
                    "horizon_len": horizon_len,
                }
            )

            class MockDataset(Dataset[dict[str, Any]]):
                def __len__(self) -> int:
                    return 1

                def __getitem__(self, idx: int) -> dict[str, Any]:
                    return {"data": idx}

            return MockDataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            create_fold_datasets(
                data_path=data_path,
                train_entities=["TestEntity"],
                val_entities=["ValEntity"],
                test_entities=["TestEntity2"],
                dataset_factory=mock_factory,
                patch_len=16,
                context_len=64,
                horizon_len=32,
            )

            # Check that factory was called for all entities
            assert len(received_calls) == 3

            # Check that standard parameters were received correctly for first call
            first_call = received_calls[0]
            assert first_call["data_path"] == data_path
            assert first_call["entity"] == "TestEntity"
            assert first_call["patch_len"] == 16
            assert first_call["context_len"] == 64
            assert first_call["horizon_len"] == 32

            # Check that all entities were called
            entities_called = {call["entity"] for call in received_calls}
            assert entities_called == {"TestEntity", "ValEntity", "TestEntity2"}

    def test_multiple_entities_concatenation(self) -> None:
        """Tests that datasets from multiple entities are correctly concatenated."""

        class MockDataset(Dataset[dict[str, Any]]):
            """Mock dataset with entity-specific data."""

            def __init__(self, entity: str, size: int) -> None:
                self.entity = entity
                self.size = size

            def __len__(self) -> int:
                return self.size

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return {"entity": self.entity, "index": idx, "value": idx * 10}

        def mock_factory(
            data_path: Path,
            entity: str,
            patch_len: int,
            context_len: int,
            horizon_len: int,
            **kwargs: Any,
        ) -> Dataset[dict[str, Any]]:
            """Mock factory that creates datasets of different sizes."""
            sizes = {"A": 2, "B": 3, "C": 4, "D": 1}
            return MockDataset(entity=entity, size=sizes.get(entity, 1))

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            train_dataset, val_dataset, test_dataset = create_fold_datasets(
                data_path=data_path,
                train_entities=["A", "B"],
                val_entities=["C"],
                test_entities=["D"],
                dataset_factory=mock_factory,
                patch_len=8,
                context_len=32,
                horizon_len=16,
            )

            # Check concatenated sizes
            assert len(train_dataset) == 5  # A(2) + B(3)
            assert len(val_dataset) == 4  # C(4)
            assert len(test_dataset) == 1  # D(1)

            # Check that samples from different entities are present
            entities_in_train = {train_dataset[i]["entity"] for i in range(len(train_dataset))}
            assert entities_in_train == {"A", "B"}
