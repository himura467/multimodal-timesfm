"""Tests for Time-MMD dataset loader."""

import pytest

from src.data.time_mmd_dataset import TimeMmdDataset


class TestTimeMmdDataset:
    """Test cases for TimeMmdDataset class."""

    def test_agriculture_dataset_loading(self) -> None:
        """Tests loading Agriculture domain from Time-MMD dataset."""
        dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            context_len=50,  # Smaller for testing
            horizon_len=12,  # Smaller for testing
        )

        # Dataset should load successfully
        assert len(dataset) > 0, "Dataset should contain samples"

        # Check first sample structure
        sample = dataset[0]
        assert "time_series" in sample
        assert "target" in sample
        assert "text" in sample
        assert "metadata" in sample

        # Check shapes
        assert sample["time_series"].shape == (50, 1), f"Expected (50, 1), got {sample['time_series'].shape}"
        assert sample["target"].shape == (12, 1), f"Expected (12, 1), got {sample['target'].shape}"

        # Check data types
        assert sample["time_series"].dtype.name == "float32"
        assert sample["target"].dtype.name == "float32"

        # Check text is not empty
        assert len(sample["text"]) > 0, "Text should not be empty"

        # Check metadata structure
        metadata = sample["metadata"]
        assert "series_id" in metadata
        assert "domain" in metadata
        assert "column" in metadata
        assert metadata["domain"] == "Agriculture"

    def test_split_functionality(self) -> None:
        """Tests train/test split functionality."""
        train_dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            split_ratio=0.7,
            context_len=50,
            horizon_len=12,
        )

        test_dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="test",
            split_ratio=0.7,
            context_len=50,
            horizon_len=12,
        )

        # Both splits should have data
        assert len(train_dataset) > 0, "Train split should have samples"
        assert len(test_dataset) > 0, "Test split should have samples"

        # Train should typically have more samples than test
        assert len(train_dataset) >= len(test_dataset), "Train split should have >= test split samples"

    def test_configurable_window_sizes(self) -> None:
        """Tests that window sizes are configurable."""
        dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            context_len=100,
            horizon_len=24,
        )

        if len(dataset) > 0:
            sample = dataset[0]
            assert sample["time_series"].shape == (100, 1)
            assert sample["target"].shape == (24, 1)

    def test_get_split_info(self) -> None:
        """Tests dataset split information."""
        dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            context_len=50,
            horizon_len=12,
        )

        info = dataset.get_split_info()
        assert info["domain"] == "Agriculture"
        assert info["split"] == "train"
        assert info["split_ratio"] == 0.7
        assert info["size"] == len(dataset)
        assert info["has_text"] is True
        assert info["time_series_shape"] == (50, 1)
        assert info["target_shape"] == (12, 1)

    def test_missing_data_file(self) -> None:
        """Tests error handling for missing data files."""
        with pytest.raises(FileNotFoundError):
            TimeMmdDataset(data_dir="/nonexistent/path", domain="NonexistentDomain", split="train")

    def test_text_extraction(self) -> None:
        """Tests that textual data is properly extracted and formatted."""
        dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            context_len=50,
            horizon_len=12,
        )

        if len(dataset) > 0:
            sample = dataset[0]
            text = sample["text"]

            # Text should contain meaningful content
            assert isinstance(text, str)
            assert len(text) > 0

            # Should contain either domain info or actual textual data
            assert "Agriculture" in text or "Report:" in text or "broiler" in text.lower()
