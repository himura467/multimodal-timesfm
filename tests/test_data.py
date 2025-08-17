"""Comprehensive tests for Time-MMD dataset loader using actual data."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.time_mmd_dataset import TimeMmdDataset


class TestTimeMmdDataset:
    """Test cases for TimeMmdDataset class using actual Agriculture data."""

    @pytest.fixture
    def agriculture_data(self) -> dict[str, Any]:
        """Load actual Agriculture numerical and textual data."""
        # Load numerical data
        numerical_file = "data/Time-MMD/numerical/Agriculture/Agriculture.csv"
        numerical_df = pd.read_csv(numerical_file)

        # Load textual data
        report_file = "data/Time-MMD/textual/Agriculture/Agriculture_report.csv"
        search_file = "data/Time-MMD/textual/Agriculture/Agriculture_search.csv"
        reports_df = pd.read_csv(report_file)
        search_df = pd.read_csv(search_file)

        # Find numeric columns (excluding date columns)
        date_cols = ["Date", "date", "start_date", "end_date"]
        numeric_cols = [
            col
            for col in numerical_df.columns
            if col not in date_cols and pd.api.types.is_numeric_dtype(numerical_df[col])
        ]

        return {
            "numerical_df": numerical_df,
            "reports_df": reports_df,
            "search_df": search_df,
            "numeric_cols": numeric_cols,
            "total_numerical_rows": len(numerical_df),
            "total_reports": len(reports_df),
            "total_search": len(search_df),
        }

    def test_dataset_initialization_and_structure(self, agriculture_data: dict[str, Any]) -> None:
        """Test dataset initialization and basic structure validation."""
        context_len = 100
        horizon_len = 25
        split_ratio = 0.7

        dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            split_ratio=split_ratio,
            context_len=context_len,
            horizon_len=horizon_len,
        )

        # Dataset should have samples
        assert len(dataset) > 0, "Dataset should contain samples"

        # Test first sample structure
        sample = dataset[0]
        required_keys = ["time_series", "text", "target", "metadata"]
        for key in required_keys:
            assert key in sample, f"Sample missing required key: {key}"

        # Validate shapes
        assert sample["time_series"].shape == (context_len, 1)
        assert sample["target"].shape == (horizon_len, 1)

        # Validate data types
        assert sample["time_series"].dtype.name == "float32"
        assert sample["target"].dtype.name == "float32"

        # Validate text (can be None or string)
        text = sample["text"]
        assert text is None or isinstance(text, str)

        # Validate metadata
        metadata = sample["metadata"]
        assert metadata["domain"] == "Agriculture"
        assert metadata["column"] in agriculture_data["numeric_cols"]
        assert isinstance(metadata["start_index"], int)
        assert metadata["start_index"] >= 0

    def test_train_test_split_accuracy(self, agriculture_data: dict[str, Any]) -> None:
        """Test that train/test splits produce exact expected sample counts."""
        context_len = 80
        horizon_len = 20
        split_ratio = 0.6

        # Calculate expected samples manually
        numerical_df = agriculture_data["numerical_df"]
        numeric_cols = agriculture_data["numeric_cols"]
        min_length = context_len + horizon_len
        step_size = horizon_len

        expected_train_total = 0
        expected_test_total = 0

        for column in numeric_cols:
            time_series_values = numerical_df[column].dropna().values
            if len(time_series_values) < min_length:
                continue

            split_idx = int(len(time_series_values) * split_ratio)
            train_length = split_idx
            test_length = len(time_series_values) - split_idx

            # Count train samples
            if train_length >= min_length:
                train_samples = len(range(0, train_length - min_length + 1, step_size))
                expected_train_total += train_samples

            # Count test samples
            if test_length >= min_length:
                test_samples = len(range(0, test_length - min_length + 1, step_size))
                expected_test_total += test_samples

        # Create actual datasets
        train_dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            split_ratio=split_ratio,
            context_len=context_len,
            horizon_len=horizon_len,
        )

        test_dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="test",
            split_ratio=split_ratio,
            context_len=context_len,
            horizon_len=horizon_len,
        )

        # Verify exact counts
        assert len(train_dataset) == expected_train_total, (
            f"Train dataset: expected {expected_train_total}, got {len(train_dataset)}"
        )
        assert len(test_dataset) == expected_test_total, (
            f"Test dataset: expected {expected_test_total}, got {len(test_dataset)}"
        )

        # Both should have data (given Agriculture dataset size)
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0

    def test_window_size_configurations(self) -> None:
        """Test various window size configurations produce correct shapes."""
        test_configs = [
            (64, 16),  # Small windows
            (128, 32),  # Medium windows
            (256, 64),  # Large windows
        ]

        for context_len, horizon_len in test_configs:
            dataset = TimeMmdDataset(
                data_dir="data/Time-MMD",
                domain="Agriculture",
                split="train",
                context_len=context_len,
                horizon_len=horizon_len,
            )

            if len(dataset) > 0:
                sample = dataset[0]
                assert sample["time_series"].shape == (context_len, 1), (
                    f"Context length {context_len}: expected shape ({context_len}, 1), "
                    f"got {sample['time_series'].shape}"
                )
                assert sample["target"].shape == (horizon_len, 1), (
                    f"Horizon length {horizon_len}: expected shape ({horizon_len}, 1), got {sample['target'].shape}"
                )

    def test_date_based_text_matching_accuracy(self, agriculture_data: dict[str, Any]) -> None:
        """Test that text data is correctly matched based on date overlaps."""
        context_len = 50
        horizon_len = 10

        dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            context_len=context_len,
            horizon_len=horizon_len,
        )

        if len(dataset) == 0:
            pytest.skip("No samples generated for this configuration")

        numerical_df = agriculture_data["numerical_df"]
        reports_df = agriculture_data["reports_df"]
        search_df = agriculture_data["search_df"]

        # Test several samples to verify date matching logic
        samples_to_test = min(10, len(dataset))

        for i in range(samples_to_test):
            sample = dataset[i]
            text = sample["text"]
            metadata = sample["metadata"]

            # Calculate the date range for this sample
            start_idx = metadata["start_index"]
            end_idx = start_idx + context_len + horizon_len - 1

            # Get date range from numerical data
            sample_start_date = pd.to_datetime(numerical_df.iloc[start_idx]["start_date"])
            sample_end_date = pd.to_datetime(numerical_df.iloc[end_idx]["end_date"])

            # Check if there should be matching text data
            expected_text_exists = False
            matching_text_content = []

            # Check reports
            if len(reports_df) > 0:
                reports_start = pd.to_datetime(reports_df["start_date"])
                reports_end = pd.to_datetime(reports_df["end_date"])

                overlapping_reports = reports_df[
                    (reports_start <= sample_end_date) & (reports_end >= sample_start_date)
                ]

                for _, report_row in overlapping_reports.iterrows():
                    if pd.notna(report_row.get("fact")):
                        expected_text_exists = True
                        matching_text_content.append(f"Report: {report_row['fact']}")
                    if pd.notna(report_row.get("preds")):
                        expected_text_exists = True
                        matching_text_content.append(f"Prediction: {report_row['preds']}")

            # Check search data
            if len(search_df) > 0:
                search_start = pd.to_datetime(search_df["start_date"])
                search_end = pd.to_datetime(search_df["end_date"])

                overlapping_search = search_df[(search_start <= sample_end_date) & (search_end >= sample_start_date)]

                for _, search_row in overlapping_search.iterrows():
                    if pd.notna(search_row.get("fact")):
                        expected_text_exists = True
                        matching_text_content.append(f"Search: {search_row['fact']}")
                    if pd.notna(search_row.get("preds")):
                        expected_text_exists = True
                        matching_text_content.append(f"Search prediction: {search_row['preds']}")

            # Verify the text matches expectations
            if expected_text_exists:
                assert text is not None, (
                    f"Sample {i}: Expected text for period {sample_start_date.date()} to "
                    f"{sample_end_date.date()}, but got None"
                )
                assert isinstance(text, str)
                assert len(text) > 0

                # Verify the text content matches exactly what we expect
                expected_combined_text = " ".join(matching_text_content)
                assert text == expected_combined_text, (
                    f"Sample {i}: Text content mismatch.\nExpected: {expected_combined_text}\nGot: {text}"
                )

            else:
                assert text is None, (
                    f"Sample {i}: Expected None for period {sample_start_date.date()} to "
                    f"{sample_end_date.date()}, but got: {text}"
                )

    def test_numeric_column_coverage(self, agriculture_data: dict[str, Any]) -> None:
        """Test that all numeric columns from the data are represented in samples."""
        context_len = 60
        horizon_len = 15

        dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            context_len=context_len,
            horizon_len=horizon_len,
        )

        if len(dataset) == 0:
            pytest.skip("No samples generated for this configuration")

        # Collect all columns represented in the dataset
        represented_columns = set()
        for i in range(len(dataset)):
            sample = dataset[i]
            represented_columns.add(sample["metadata"]["column"])

        expected_columns = set(agriculture_data["numeric_cols"])

        # Verify that all expected columns are represented in the dataset
        assert represented_columns == expected_columns, (
            f"Missing columns: {expected_columns - represented_columns}, "
            f"Unexpected columns: {represented_columns - expected_columns}"
        )

    def test_time_series_data_validity(self, agriculture_data: dict[str, Any]) -> None:
        """Test that time series data values are valid and properly extracted."""
        context_len = 40
        horizon_len = 10

        dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            context_len=context_len,
            horizon_len=horizon_len,
        )

        if len(dataset) == 0:
            pytest.skip("No samples generated for this configuration")

        numerical_df = agriculture_data["numerical_df"]

        # Test several samples
        samples_to_test = min(5, len(dataset))

        for i in range(samples_to_test):
            sample = dataset[i]
            time_series = sample["time_series"]
            target = sample["target"]
            metadata = sample["metadata"]

            # Verify no NaN values
            assert not pd.isna(time_series).any(), f"Sample {i}: time_series contains NaN values"
            assert not pd.isna(target).any(), f"Sample {i}: target contains NaN values"

            # Verify data comes from the correct column
            column_name = metadata["column"]
            start_idx = metadata["start_index"]

            # Extract expected values from original data
            original_column_data = numerical_df[column_name].dropna().values
            expected_context = original_column_data[start_idx : start_idx + context_len]
            expected_target = original_column_data[start_idx + context_len : start_idx + context_len + horizon_len]

            # Compare with sample data (reshape for comparison)
            assert np.array_equal(time_series.flatten(), expected_context.astype("float32")), (
                f"Sample {i}: time_series data doesn't match expected values"
            )
            assert np.array_equal(target.flatten(), expected_target.astype("float32")), (
                f"Sample {i}: target data doesn't match expected values"
            )

    def test_error_handling(self) -> None:
        """Test proper error handling for invalid inputs."""
        # Test missing data directory
        with pytest.raises(FileNotFoundError):
            TimeMmdDataset(
                data_dir="/nonexistent/path",
                domain="Agriculture",
                split="train",
                context_len=50,
                horizon_len=10,
            )

        # Test missing domain
        with pytest.raises(FileNotFoundError):
            TimeMmdDataset(
                data_dir="data/Time-MMD",
                domain="NonexistentDomain",
                split="train",
                context_len=50,
                horizon_len=10,
            )

        # Test missing date columns (this should raise an error during processing)
        # Note: This would require a modified CSV without proper date columns
        # For now, we test that the normal case doesn't raise errors
        dataset = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            context_len=50,
            horizon_len=10,
        )

        # Should not raise an error for valid Agriculture data
        assert isinstance(dataset, TimeMmdDataset)

    def test_dataset_consistency_across_multiple_instantiations(self, agriculture_data: dict[str, Any]) -> None:
        """Test that multiple dataset instantiations with same parameters are consistent."""
        context_len = 70
        horizon_len = 18
        split_ratio = 0.8

        # Create two datasets with identical parameters
        dataset1 = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            split_ratio=split_ratio,
            context_len=context_len,
            horizon_len=horizon_len,
        )

        dataset2 = TimeMmdDataset(
            data_dir="data/Time-MMD",
            domain="Agriculture",
            split="train",
            split_ratio=split_ratio,
            context_len=context_len,
            horizon_len=horizon_len,
        )

        # Should have same length
        assert len(dataset1) == len(dataset2)

        if len(dataset1) > 0:
            # Should have same first sample
            sample1 = dataset1[0]
            sample2 = dataset2[0]

            assert np.array_equal(sample1["time_series"], sample2["time_series"])
            assert np.array_equal(sample1["target"], sample2["target"])
            assert sample1["text"] == sample2["text"]
            assert sample1["metadata"] == sample2["metadata"]
