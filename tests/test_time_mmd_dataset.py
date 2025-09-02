"""Tests for Time-MMD dataset loader."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.time_mmd_dataset import TimeMmdDataset


class TestTimeMmdDataset:
    """Test cases for TimeMmdDataset class."""

    def test_init_valid_parameters(self, sample_data_dir: Path) -> None:
        """Tests initialization with valid parameters."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        assert dataset.domain == "TestDomain"
        assert dataset.split_ratio == 0.8
        assert dataset.split == "train"
        assert dataset.patch_len == 8
        assert dataset.context_len == 32
        assert dataset.horizon_len == 16

    def test_init_missing_numerical_file(self, sample_data_dir: Path) -> None:
        """Tests initialization fails when numerical file is missing."""
        with pytest.raises(FileNotFoundError, match="Numerical data file not found"):
            TimeMmdDataset(
                data_dir=sample_data_dir,
                domain="NonexistentDomain",
                patch_len=8,
                context_len=32,
                horizon_len=16,
            )

    def test_missing_text_data_handling(self) -> None:
        """Tests handling when textual data files are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            numerical_dir = data_dir / "numerical" / "NoTextDomain"
            numerical_dir.mkdir(parents=True)

            # Create numerical data without corresponding textual data
            dates = pd.date_range("2020-01-01", periods=100, freq="D")
            numerical_data = pd.DataFrame(
                {
                    "start_date": dates.astype(str),
                    "end_date": dates.astype(str),
                    "value": np.random.randn(100),
                }
            )
            numerical_data.to_csv(numerical_dir / "NoTextDomain.csv", index=False)

            # Should initialize without errors
            dataset = TimeMmdDataset(
                data_dir=data_dir,
                domain="NoTextDomain",
                patch_len=8,
                context_len=32,
                horizon_len=16,
            )

            assert len(dataset) > 0

            # Samples should have empty text patches
            sample = dataset[0]
            assert all(len(patch) == 0 for patch in sample["patched_texts"])

    def test_missing_date_columns(self) -> None:
        """Tests handling of missing date columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            numerical_dir = data_dir / "numerical" / "BadDomain"
            numerical_dir.mkdir(parents=True)

            # Create data without proper date columns
            bad_data = pd.DataFrame(
                {
                    "value": [1, 2, 3, 4, 5],
                    "other_col": ["a", "b", "c", "d", "e"],
                }
            )
            bad_data.to_csv(numerical_dir / "BadDomain.csv", index=False)

            with pytest.raises(ValueError, match="No start_date column found"):
                TimeMmdDataset(
                    data_dir=data_dir,
                    domain="BadDomain",
                    patch_len=1,
                    context_len=4,
                    horizon_len=2,
                )

    def test_missing_end_date_column(self) -> None:
        """Tests handling of missing end_date column."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            numerical_dir = data_dir / "numerical" / "BadDomain"
            numerical_dir.mkdir(parents=True)

            # Create data with start_date but no end_date
            bad_data = pd.DataFrame(
                {
                    "start_date": ["2020-01-01", "2020-01-02"],
                    "value": [1.0, 2.0],
                }
            )
            bad_data.to_csv(numerical_dir / "BadDomain.csv", index=False)

            with pytest.raises(ValueError, match="No end_date column found"):
                TimeMmdDataset(
                    data_dir=data_dir,
                    domain="BadDomain",
                    patch_len=1,
                    context_len=4,
                    horizon_len=2,
                )

    def test_no_numeric_columns(self) -> None:
        """Tests handling when no numeric columns are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            numerical_dir = data_dir / "numerical" / "TextOnlyDomain"
            numerical_dir.mkdir(parents=True)

            # Create data with only text columns
            text_only_data = pd.DataFrame(
                {
                    "start_date": ["2020-01-01", "2020-01-02"],
                    "end_date": ["2020-01-01", "2020-01-02"],
                    "category": ["A", "B"],
                    "description": ["Text 1", "Text 2"],
                }
            )
            text_only_data.to_csv(numerical_dir / "TextOnlyDomain.csv", index=False)

            dataset = TimeMmdDataset(
                data_dir=data_dir,
                domain="TextOnlyDomain",
                patch_len=1,
                context_len=4,
                horizon_len=2,
            )

            # Should result in empty dataset
            assert len(dataset) == 0

    def test_numeric_column_filtering(self, sample_data_dir: Path) -> None:
        """Tests that only numeric columns are processed as time series."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=4,
            context_len=16,
            horizon_len=8,
        )

        # Should only have samples from value1 and value2 columns, not category
        columns_found = set()
        for i in range(len(dataset)):
            sample = dataset[i]
            columns_found.add(sample["metadata"]["column"])

        assert "value1" in columns_found
        assert "value2" in columns_found
        assert "category" not in columns_found

    def test_train_test_split(self, sample_data_dir: Path) -> None:
        """Tests train/test split functionality."""
        train_dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            split="train",
            patch_len=4,
            context_len=8,
            horizon_len=4,
        )

        test_dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            split="test",
            patch_len=4,
            context_len=8,
            horizon_len=4,
        )

        # Both splits should have data
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0

        # Verify 80/20 split ratio is approximately correct
        total_samples = len(train_dataset) + len(test_dataset)
        train_ratio = len(train_dataset) / total_samples

        # Allow tolerance due to windowing effects and discrete sample counts
        assert 0.8 <= train_ratio < 1

    def test_different_split_ratios(self, sample_data_dir: Path) -> None:
        """Tests different split ratios."""
        # Use smaller context/horizon to ensure we have enough data for both splits
        # 60/40 split
        train_dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            split="train",
            split_ratio=0.6,
            patch_len=4,
            context_len=8,
            horizon_len=4,
        )

        test_dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            split="test",
            split_ratio=0.6,
            patch_len=4,
            context_len=8,
            horizon_len=4,
        )

        # Both splits should have data
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0

        # Verify 60/40 split ratio is approximately correct
        total_samples = len(train_dataset) + len(test_dataset)
        train_ratio = len(train_dataset) / total_samples

        # Allow tolerance due to windowing effects and discrete sample counts
        assert 0.6 <= train_ratio < 0.7

    def test_init_invalid_context_patch_ratio(self, sample_data_dir: Path) -> None:
        """Tests initialization fails when context_len is not multiple of patch_len."""
        with pytest.raises(ValueError, match="context_len \\(31\\) must be an integer multiple of patch_len \\(8\\)"):
            TimeMmdDataset(
                data_dir=sample_data_dir,
                domain="TestDomain",
                patch_len=8,
                context_len=31,  # Not multiple of patch_len=8
                horizon_len=16,
            )

    def test_init_invalid_horizon_patch_ratio(self, sample_data_dir: Path) -> None:
        """Tests initialization fails when horizon_len is not multiple of patch_len."""
        with pytest.raises(ValueError, match="horizon_len \\(15\\) must be an integer multiple of patch_len \\(8\\)"):
            TimeMmdDataset(
                data_dir=sample_data_dir,
                domain="TestDomain",
                patch_len=8,
                context_len=32,
                horizon_len=15,  # Not multiple of patch_len=8
            )

    def test_insufficient_data_handling(self) -> None:
        """Tests handling when data is too short for context + horizon."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            numerical_dir = data_dir / "numerical" / "ShortDomain"
            numerical_dir.mkdir(parents=True)

            # Create very short time series (shorter than context + horizon)
            short_data = pd.DataFrame(
                {
                    "start_date": ["2020-01-01", "2020-01-02"],
                    "end_date": ["2020-01-01", "2020-01-02"],
                    "value": [1.0, 2.0],
                }
            )
            short_data.to_csv(numerical_dir / "ShortDomain.csv", index=False)

            dataset = TimeMmdDataset(
                data_dir=data_dir,
                domain="ShortDomain",
                patch_len=8,
                context_len=32,
                horizon_len=16,
            )

            # Should handle gracefully with empty dataset
            assert len(dataset) == 0

    def test_calculate_frequency_daily_data(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation for daily data."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        # Create daily dates
        daily_dates = pd.date_range("2020-01-01", periods=50, freq="D").astype(str)
        dates_series = pd.Series(daily_dates)

        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 30)
        assert freq == 0  # Daily frequency

    def test_calculate_frequency_weekly_data(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation for weekly data."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        # Create weekly dates
        weekly_dates = pd.date_range("2020-01-01", periods=20, freq="W").astype(str)
        dates_series = pd.Series(weekly_dates)

        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 15)
        assert freq == 1  # Weekly frequency

    def test_calculate_frequency_monthly_data(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation for monthly data."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        # Create monthly dates
        monthly_dates = pd.date_range("2020-01-01", periods=15, freq="MS").astype(str)
        dates_series = pd.Series(monthly_dates)

        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 12)
        assert freq == 1  # Monthly frequency

    def test_calculate_frequency_quarterly_data(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation for quarterly data."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        # Create quarterly dates
        quarterly_dates = pd.date_range("2020-01-01", periods=12, freq="QS").astype(str)
        dates_series = pd.Series(quarterly_dates)

        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 10)
        assert freq == 2  # Quarterly frequency

    def test_calculate_frequency_yearly_data(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation for yearly data."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        # Create yearly dates
        yearly_dates = pd.date_range("2020-01-01", periods=8, freq="YS").astype(str)
        dates_series = pd.Series(yearly_dates)

        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 6)
        assert freq == 2  # Yearly frequency

    def test_calculate_frequency_insufficient_data(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation with insufficient data points."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        dates_series = pd.Series(["2020-01-01"])

        # Same start and end index
        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 0)
        assert freq == 0

        # End index - start index < 1
        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 1)
        assert freq == 0

    def test_calculate_frequency_single_date_range(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation with only one date difference."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        dates_series = pd.Series(["2020-01-01", "2020-01-02"])

        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 2)
        assert freq == 0  # Single day interval

    def test_calculate_frequency_mixed_intervals(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation with mixed intervals averaging to weekly."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        # Mix of 1-day and 25-day intervals (average ~7 days)
        mixed_dates = [
            "2020-01-01",
            "2020-01-02",  # 1 day
            "2020-01-03",  # 1 day
            "2020-01-04",  # 1 day
            "2020-01-29",  # 25 days
        ]
        dates_series = pd.Series(mixed_dates)

        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 5)
        assert freq == 1  # Should classify as weekly (average ~7 days)

    def test_calculate_frequency_subsample_range(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation on a subsample of the date series."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        # Create mixed frequency data
        mixed_dates = (
            pd.date_range("2020-01-01", periods=10, freq="D").astype(str).tolist()  # Daily
            + pd.date_range("2020-01-11", periods=10, freq="W").astype(str).tolist()  # Weekly
        )
        dates_series = pd.Series(mixed_dates)

        # Test daily portion
        freq_daily = dataset._calculate_frequency_for_sample(dates_series, 0, 8)
        assert freq_daily == 0

        # Test weekly portion
        freq_weekly = dataset._calculate_frequency_for_sample(dates_series, 10, 18)
        assert freq_weekly == 1

    def test_calculate_frequency_boundary_cases(self, sample_data_dir: Path) -> None:
        """Tests frequency calculation at boundary values."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        # Test 3-day boundary (should be weekly/monthly category)
        three_day_dates = pd.date_range("2020-01-01", periods=10, freq="3D").astype(str)
        dates_series = pd.Series(three_day_dates)
        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 8)
        assert freq == 1

        # Test 35-day boundary (should be quarterly/yearly category)
        thirty_five_day_dates = pd.date_range("2020-01-01", periods=10, freq="35D").astype(str)
        dates_series = pd.Series(thirty_five_day_dates)
        freq = dataset._calculate_frequency_for_sample(dates_series, 0, 8)
        assert freq == 2

    def test_text_patching_logic(self, sample_data_dir: Path) -> None:
        """Tests text patching logic."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=4,
            context_len=16,
            horizon_len=8,
        )

        sample = dataset[0]
        patched_texts = sample["patched_texts"]

        # Should have correct number of patches
        expected_patches = dataset.context_len // dataset.patch_len
        assert len(patched_texts) == expected_patches

        # Each patch should be a list of strings
        for patch in patched_texts:
            assert isinstance(patch, list)
            for text in patch:
                assert isinstance(text, str)

    def test_text_data_with_missing_values(self) -> None:
        """Tests handling of text data with missing values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            numerical_dir = data_dir / "numerical" / "MissingTextDomain"
            textual_dir = data_dir / "textual" / "MissingTextDomain"
            numerical_dir.mkdir(parents=True)
            textual_dir.mkdir(parents=True)

            # Create numerical data
            dates = pd.date_range("2020-01-01", periods=50, freq="D")
            numerical_data = pd.DataFrame(
                {
                    "start_date": dates.astype(str),
                    "end_date": dates.astype(str),
                    "value": np.random.randn(50),
                }
            )
            numerical_data.to_csv(numerical_dir / "MissingTextDomain.csv", index=False)

            # Create textual data with NaN values
            report_data = pd.DataFrame(
                {
                    "start_date": ["2020-01-01", "2020-01-15"],
                    "end_date": ["2020-01-14", "2020-01-31"],
                    "fact": ["Valid fact", np.nan],
                    "preds": [np.nan, "Valid prediction"],
                }
            )
            report_data.to_csv(textual_dir / "MissingTextDomain_report.csv", index=False)

            dataset = TimeMmdDataset(
                data_dir=data_dir,
                domain="MissingTextDomain",
                patch_len=4,
                context_len=16,
                horizon_len=8,
            )

            # Should handle gracefully without errors
            assert len(dataset) == 3

    def test_text_filtering_na_values(self, sample_data_dir: Path) -> None:
        """Tests that NA values are filtered out from text data."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=4,
            context_len=16,
            horizon_len=8,
        )

        # Check that no "NA" strings appear in text patches
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            for patch in sample["patched_texts"]:
                for text in patch:
                    assert "NA" not in text

    def test_len_method(self, sample_data_dir: Path) -> None:
        """Tests __len__ method."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=4,
            context_len=16,
            horizon_len=8,
        )

        assert len(dataset) == 16

    def test_getitem_bounds_checking(self, sample_data_dir: Path) -> None:
        """Tests __getitem__ with invalid indices."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=4,
            context_len=16,
            horizon_len=8,
        )

        # Valid index should work
        assert isinstance(dataset[0], dict)

        # Invalid indices should raise IndexError
        with pytest.raises(IndexError):
            dataset[len(dataset)]

        with pytest.raises(IndexError):
            dataset[-len(dataset) - 1]

    def test_data_loading_and_structure(self, sample_data_dir: Path) -> None:
        """Tests that data is loaded correctly and has expected structure."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        assert len(dataset) > 0

        # Test first sample
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"context", "future", "freq", "patched_texts", "metadata"}

        # Test context
        assert sample["context"].shape == (32, 1)
        assert sample["context"].dtype == np.float32

        # Test future
        assert sample["future"].shape == (16, 1)
        assert sample["future"].dtype == np.float32

        # Test freq
        assert sample["freq"] in [0, 1, 2]  # Valid frequency values

        # Test patched texts
        assert isinstance(sample["patched_texts"], list)
        assert len(sample["patched_texts"]) == 4  # context_len // patch_len = 32 // 8 = 4
        assert all(isinstance(patch, list) for patch in sample["patched_texts"])

        # Test metadata
        assert isinstance(sample["metadata"], dict)
        assert "domain" in sample["metadata"]
        assert "column" in sample["metadata"]
        assert "start_index" in sample["metadata"]

    def test_data_consistency_across_samples(self, sample_data_dir: Path) -> None:
        """Tests that all samples have consistent structure."""
        dataset = TimeMmdDataset(
            data_dir=sample_data_dir,
            domain="TestDomain",
            patch_len=4,
            context_len=16,
            horizon_len=8,
        )

        first_sample = dataset[0]

        # Check all samples have same structure
        for i in range(min(5, len(dataset))):
            sample = dataset[i]

            assert set(sample.keys()) == set(first_sample.keys())
            assert sample["context"].shape == first_sample["context"].shape
            assert sample["future"].shape == first_sample["future"].shape
            assert len(sample["patched_texts"]) == len(first_sample["patched_texts"])

    def test_real_time_mmd_domain(self) -> None:
        """Tests with actual Time-MMD dataset if available."""
        real_data_dir = Path("data/Time-MMD")

        if not real_data_dir.exists():
            pytest.skip("Real Time-MMD dataset not available")

        # Test with Agriculture domain
        dataset = TimeMmdDataset(
            data_dir=real_data_dir,
            domain="Agriculture",
            patch_len=8,
            context_len=32,
            horizon_len=16,
        )

        assert len(dataset) > 0

        sample = dataset[0]
        assert sample["context"].shape == (32, 1)
        assert sample["future"].shape == (16, 1)
        assert sample["freq"] in [0, 1, 2]  # Valid frequency values
        assert len(sample["patched_texts"]) == 4  # context_len // patch_len
        assert sample["metadata"]["domain"] == "Agriculture"

        assert (
            "Wholesale broiler composite" in sample["metadata"]["column"]
            or "OT" in sample["metadata"]["column"]
            or "Retail-wholesale spread for broiler composite" in sample["metadata"]["column"]
        )

    def test_multiple_domains_consistency(self) -> None:
        """Tests that multiple domains can be loaded consistently."""
        real_data_dir = Path("data/Time-MMD")

        if not real_data_dir.exists():
            pytest.skip("Real Time-MMD dataset not available")

        domains_to_test = ["Agriculture", "Climate", "Economy"]
        datasets = {}

        for domain in domains_to_test:
            datasets[domain] = TimeMmdDataset(
                data_dir=real_data_dir,
                domain=domain,
                patch_len=8,
                context_len=32,
                horizon_len=16,
            )

        assert len(datasets) == 3

        # All datasets should have consistent sample structure
        sample_structures = []
        for domain, dataset in datasets.items():
            sample = dataset[0]
            structure = {
                "context_shape": sample["context"].shape,
                "future_shape": sample["future"].shape,
                "num_patches": len(sample["patched_texts"]),
                "metadata_keys": set(sample["metadata"].keys()),
            }
            sample_structures.append(structure)

        # All structures should be identical except for metadata content
        first_structure = sample_structures[0]
        for structure in sample_structures[1:]:
            assert structure["context_shape"] == first_structure["context_shape"]
            assert structure["future_shape"] == first_structure["future_shape"]
            assert structure["num_patches"] == first_structure["num_patches"]
            assert structure["metadata_keys"] == first_structure["metadata_keys"]
