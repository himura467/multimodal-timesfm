"""Tests for MultimodalTimesFM wrapper class and model functionality."""

import numpy as np
import pytest
from timesfm import TimesFmCheckpoint, TimesFmHparams

from src.data.time_mmd_dataset import TimeMmdDataset
from src.models.multimodal_timesfm import MultimodalTimesFM


class TestMultimodalTimesFM:
    """Test cases for MultimodalTimesFM wrapper class."""

    @pytest.fixture
    def hparams(self) -> TimesFmHparams:
        """Creates TimesFM hyperparameters for testing."""
        return TimesFmHparams(
            backend="cpu",
            context_len=512,
            horizon_len=128,
            num_layers=50,  # TimesFM 2.0
            model_dims=1280,
        )

    @pytest.fixture
    def checkpoint(self) -> TimesFmCheckpoint:
        """Creates TimesFM checkpoint for testing."""
        return TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")

    @pytest.fixture
    def wrapper(self, hparams: TimesFmHparams, checkpoint: TimesFmCheckpoint) -> MultimodalTimesFM:
        """Creates MultimodalTimesFM wrapper instance."""
        return MultimodalTimesFM(hparams, checkpoint)

    @pytest.fixture
    def dataset(self) -> TimeMmdDataset:
        """Creates Time-MMD dataset for testing."""
        return TimeMmdDataset(
            data_dir="/Users/himura/Desktop/Time-MMD", domain="Climate", split="train", context_len=512, horizon_len=128
        )

    def test_wrapper_initialization(self, hparams: TimesFmHparams, checkpoint: TimesFmCheckpoint) -> None:
        """Tests that wrapper initializes correctly with proper structure."""
        wrapper = MultimodalTimesFM(hparams, checkpoint)

        # Verify wrapper has required attributes
        assert hasattr(wrapper, "timesfm")
        assert hasattr(wrapper, "forecast")
        assert wrapper.timesfm is not None
        assert callable(wrapper.forecast)

    def test_basic_forecasting(self, wrapper: MultimodalTimesFM, dataset: TimeMmdDataset) -> None:
        """Tests basic forecasting functionality with single sample."""
        if len(dataset) == 0:
            pytest.skip("No dataset samples available")

        sample = dataset[0]
        time_series = sample["time_series"]

        # Prepare input (remove channel dimension for TimesFM)
        inputs = [time_series.squeeze()]

        # Generate forecast
        mean_forecast, quantile_forecast = wrapper.forecast(inputs)

        # Verify output structure
        assert isinstance(mean_forecast, np.ndarray)
        assert isinstance(quantile_forecast, np.ndarray)
        assert mean_forecast.shape == (1, 128)  # 1 series, 128 horizon
        assert quantile_forecast.shape[0] == 1
        assert quantile_forecast.shape[1] == 128

        # Verify no invalid values
        assert not np.isnan(mean_forecast).any()
        assert not np.isinf(mean_forecast).any()
        assert not np.isnan(quantile_forecast).any()
        assert not np.isinf(quantile_forecast).any()

    def test_batch_forecasting(self, wrapper: MultimodalTimesFM, dataset: TimeMmdDataset) -> None:
        """Tests batch forecasting with multiple samples."""
        if len(dataset) < 2:
            pytest.skip("Need at least 2 samples for batch test")

        # Prepare batch
        batch_size = min(3, len(dataset))
        samples = [dataset[i] for i in range(batch_size)]
        inputs = [sample["time_series"].squeeze() for sample in samples]

        # Process batch
        mean_forecast, quantile_forecast = wrapper.forecast(inputs)

        # Verify batch results
        assert mean_forecast.shape == (batch_size, 128)
        assert quantile_forecast.shape[0] == batch_size
        assert quantile_forecast.shape[1] == 128
        assert not np.isnan(mean_forecast).any()
        assert not np.isnan(quantile_forecast).any()

    def test_input_output_shapes(self, wrapper: MultimodalTimesFM, dataset: TimeMmdDataset) -> None:
        """Tests that input/output shapes are correct and consistent."""
        if len(dataset) == 0:
            pytest.skip("No dataset samples available")

        sample = dataset[0]
        time_series = sample["time_series"]
        target = sample["target"]

        # Verify dataset shapes match expected wrapper input/output
        assert time_series.shape == (512, 1)  # context_len, features
        assert target.shape == (128, 1)  # horizon_len, features

        # Test forecasting shapes
        inputs = [time_series.squeeze()]
        mean_forecast, quantile_forecast = wrapper.forecast(inputs)

        assert mean_forecast.shape == (1, 128)
        assert target.squeeze().shape == (128,)  # Compatible for comparison

    def test_wrapper_delegation(self, wrapper: MultimodalTimesFM, dataset: TimeMmdDataset) -> None:
        """Tests that wrapper properly delegates to underlying TimesFM."""
        if len(dataset) == 0:
            pytest.skip("No dataset samples available")

        sample = dataset[0]
        inputs = [sample["time_series"].squeeze()]

        # Call through wrapper
        wrapper_result = wrapper.forecast(inputs)

        # Call underlying TimesFM directly
        direct_result = wrapper.timesfm.forecast(inputs)

        # Results should be identical (same underlying method)
        np.testing.assert_array_equal(wrapper_result[0], direct_result[0])
        np.testing.assert_array_equal(wrapper_result[1], direct_result[1])

    def test_data_types_and_ranges(self, wrapper: MultimodalTimesFM, dataset: TimeMmdDataset) -> None:
        """Tests that forecast outputs have correct data types and reasonable ranges."""
        if len(dataset) == 0:
            pytest.skip("No dataset samples available")

        sample = dataset[0]
        inputs = [sample["time_series"].squeeze()]

        mean_forecast, quantile_forecast = wrapper.forecast(inputs)

        # Check data types
        assert mean_forecast.dtype == np.float32
        assert quantile_forecast.dtype == np.float32

        # Check for reasonable value ranges (Climate data can have large values)
        assert np.abs(mean_forecast).max() < 1e8  # No extremely large values
        assert np.abs(quantile_forecast).max() < 1e8

    def test_multiple_samples_consistency(self, wrapper: MultimodalTimesFM, dataset: TimeMmdDataset) -> None:
        """Tests consistency across multiple dataset samples."""
        if len(dataset) < 2:
            pytest.skip("Need at least 2 samples for consistency test")

        # Test multiple samples individually
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            inputs = [sample["time_series"].squeeze()]

            mean_forecast, quantile_forecast = wrapper.forecast(inputs)

            # Each sample should produce consistent output shapes
            assert mean_forecast.shape == (1, 128)
            assert quantile_forecast.shape[0] == 1
            assert quantile_forecast.shape[1] == 128
            assert not np.isnan(mean_forecast).any()
            assert not np.isnan(quantile_forecast).any()

    def test_error_handling(self, wrapper: MultimodalTimesFM) -> None:
        """Tests error handling for invalid inputs."""
        # Test empty input
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            wrapper.forecast([])

        # Test wrong data type
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            wrapper.forecast(["invalid"])

    def test_dataset_compatibility(self, dataset: TimeMmdDataset) -> None:
        """Tests that dataset produces data compatible with wrapper requirements."""
        if len(dataset) == 0:
            pytest.skip("No dataset samples available")

        sample = dataset[0]

        # Verify sample structure
        assert "time_series" in sample
        assert "target" in sample
        assert "text" in sample
        assert "metadata" in sample

        # Verify data shapes and types
        time_series = sample["time_series"]
        target = sample["target"]

        assert time_series.shape == (512, 1)
        assert target.shape == (128, 1)
        assert time_series.dtype == np.float32
        assert target.dtype == np.float32

        # Verify no invalid values
        assert not np.isnan(time_series).any()
        assert not np.isinf(time_series).any()
        assert not np.isnan(target).any()
        assert not np.isinf(target).any()

    def test_text_extraction_integration(self, dataset: TimeMmdDataset) -> None:
        """Tests that textual data is properly extracted and formatted."""
        if len(dataset) == 0:
            pytest.skip("No dataset samples available")

        sample = dataset[0]
        text = sample["text"]

        # Verify text structure
        assert isinstance(text, str)
        assert len(text) > 0

        # Should contain domain-relevant content
        assert "Climate" in text or "Report:" in text or len(text) > 50

    def test_metadata_structure(self, dataset: TimeMmdDataset) -> None:
        """Tests that dataset metadata contains required information."""
        if len(dataset) == 0:
            pytest.skip("No dataset samples available")

        sample = dataset[0]
        metadata = sample["metadata"]

        # Verify required fields
        assert "series_id" in metadata
        assert "domain" in metadata
        assert "column" in metadata
        assert "start_index" in metadata

        # Verify field types and values
        assert metadata["domain"] == "Climate"
        assert isinstance(metadata["series_id"], str)
        assert isinstance(metadata["column"], str)
        assert isinstance(metadata["start_index"], int)


class TestEndToEndIntegration:
    """Integration tests for the complete pipeline."""

    def test_complete_pipeline(self) -> None:
        """Tests the complete pipeline from data loading to forecasting."""
        # Create dataset
        dataset = TimeMmdDataset(
            data_dir="/Users/himura/Desktop/Time-MMD", domain="Climate", split="train", context_len=512, horizon_len=128
        )

        if len(dataset) == 0:
            pytest.skip("No dataset samples available")

        # Create wrapper
        hparams = TimesFmHparams(
            backend="cpu",
            context_len=512,
            horizon_len=128,
            num_layers=50,
        )
        checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")
        wrapper = MultimodalTimesFM(hparams, checkpoint)

        # Process sample
        sample = dataset[0]
        inputs = [sample["time_series"].squeeze()]
        mean_forecast, quantile_forecast = wrapper.forecast(inputs)

        # Verify end-to-end success
        assert isinstance(mean_forecast, np.ndarray)
        assert mean_forecast.shape == (1, 128)
        assert not np.isnan(mean_forecast).any()

    def test_train_test_split_integration(self) -> None:
        """Tests that train/test splits work correctly with the wrapper."""
        # Create both splits using Environment dataset (has more data)
        train_dataset = TimeMmdDataset(
            data_dir="/Users/himura/Desktop/Time-MMD",
            domain="Environment",
            split="train",
            split_ratio=0.7,
            context_len=512,
            horizon_len=128,
        )

        test_dataset = TimeMmdDataset(
            data_dir="/Users/himura/Desktop/Time-MMD",
            domain="Environment",
            split="test",
            split_ratio=0.7,
            context_len=512,
            horizon_len=128,
        )

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            pytest.skip("Need both train and test samples")

        # Train should have more samples
        assert len(train_dataset) >= len(test_dataset)

        # Both should produce compatible samples
        train_sample = train_dataset[0]
        test_sample = test_dataset[0]

        assert train_sample["time_series"].shape == test_sample["time_series"].shape
        assert train_sample["target"].shape == test_sample["target"].shape
