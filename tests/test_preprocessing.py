"""Tests for data preprocessing utilities."""

import numpy as np
import pytest

from src.data.preprocessing import (
    clean_text,
    denormalize_timeseries,
    prepare_multimodal_batch,
    standardize_timeseries,
    validate_text_inputs,
)


class TestTextPreprocessing:
    """Test cases for text preprocessing functions."""

    def test_clean_text_basic(self) -> None:
        """Tests basic text cleaning functionality."""
        dirty_text = "  Hello    world!  @#$%  "
        cleaned = clean_text(dirty_text)

        assert cleaned == "Hello world! "
        assert not cleaned.startswith(" ")
        # Note: clean_text preserves some trailing spaces after punctuation

    def test_clean_text_special_characters(self) -> None:
        """Tests cleaning of special characters."""
        text_with_special = "Climate data: temp=25°C, humidity~60%"
        cleaned = clean_text(text_with_special)

        # Should keep basic punctuation but remove special symbols
        assert "Climate data" in cleaned
        assert "temp25C" in cleaned or "temp=25C" in cleaned
        assert "°" not in cleaned
        assert "~" not in cleaned

    def test_clean_text_numbers(self) -> None:
        """Tests that numbers are preserved."""
        text_with_numbers = "Temperature increased by 2.5 degrees in 2023"
        cleaned = clean_text(text_with_numbers)

        assert "2.5" in cleaned
        assert "2023" in cleaned

    def test_clean_text_empty_input(self) -> None:
        """Tests handling of empty input."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_clean_text_non_string_input(self) -> None:
        """Tests conversion of non-string input."""
        assert clean_text(123) == "123"
        assert clean_text(None) == "None"

    def test_validate_text_inputs_valid(self) -> None:
        """Tests validation of valid text inputs."""
        texts = ["Climate data shows trends", "Weather patterns vary", "Temperature rising"]
        validated = validate_text_inputs(texts)

        assert len(validated) == 3
        assert all(isinstance(text, str) for text in validated)
        assert all(len(text) > 0 for text in validated)

    def test_validate_text_inputs_empty(self) -> None:
        """Tests validation fails for empty texts."""
        texts = ["Valid text", "", "Another valid text"]

        with pytest.raises(ValueError, match="empty after cleaning"):
            validate_text_inputs(texts)

    def test_validate_text_inputs_whitespace_only(self) -> None:
        """Tests validation fails for whitespace-only texts."""
        texts = ["Valid text", "   ", "Another valid text"]

        with pytest.raises(ValueError, match="empty after cleaning"):
            validate_text_inputs(texts)


class TestTimeseriesNormalization:
    """Test cases for time series normalization functions."""

    def test_standardize_timeseries_basic(self) -> None:
        """Tests basic standardization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        standardized, mean, std = standardize_timeseries(data)

        assert abs(np.mean(standardized)) < 1e-10  # Should be ~0
        assert abs(np.std(standardized) - 1.0) < 1e-10  # Should be ~1
        assert mean == 3.0
        assert abs(std - np.sqrt(2.0)) < 1e-10

    def test_standardize_constant_series(self) -> None:
        """Tests standardization of constant series."""
        data = np.array([5.0, 5.0, 5.0, 5.0])

        standardized, mean, std = standardize_timeseries(data)

        assert mean == 5.0
        assert std == 1.0  # Should default to 1.0 for constant series
        assert np.allclose(standardized, 0.0)

    def test_denormalize_timeseries(self) -> None:
        """Tests denormalization of standardized data."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        standardized, mean, std = standardize_timeseries(original)

        denormalized = denormalize_timeseries(standardized, mean, std)

        assert np.allclose(denormalized, original)

    def test_standardization_round_trip(self) -> None:
        """Tests that standardization and denormalization are inverse operations."""
        original = np.random.randn(100) * 10 + 5  # Random data with mean 5, std 10

        standardized, mean, std = standardize_timeseries(original)
        recovered = denormalize_timeseries(standardized, mean, std)

        assert np.allclose(recovered, original, rtol=1e-10)


class TestBatchPreparation:
    """Test cases for multimodal batch preparation."""

    def test_prepare_batch_no_standardization(self) -> None:
        """Tests batch preparation without standardization."""
        ts_batch = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        text_batch = ["First series", "Second series"]

        processed_ts, processed_text, metadata = prepare_multimodal_batch(ts_batch, text_batch, standardize=False)

        assert len(processed_ts) == len(processed_text) == 2
        assert "normalization_params" not in metadata

    def test_prepare_batch_with_standardization(self) -> None:
        """Tests batch preparation with standardization."""
        ts_batch = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])]
        text_batch = ["First series", "Second series"]

        processed_ts, processed_text, metadata = prepare_multimodal_batch(ts_batch, text_batch, standardize=True)

        assert len(processed_ts) == len(processed_text) == 2
        assert "normalization_params" in metadata
        assert len(metadata["normalization_params"]) == 2

        # Check that standardization was applied
        for ts in processed_ts:
            if isinstance(ts, np.ndarray):
                assert abs(np.mean(ts)) < 1e-10  # Should be ~0
                assert abs(np.std(ts) - 1.0) < 1e-10  # Should be ~1
