"""Tests for MultimodalTimesFM wrapper class and model functionality."""

import numpy as np
import pytest
import torch
from timesfm import TimesFmCheckpoint, TimesFmHparams

from src.data.time_mmd_dataset import TimeMmdDataset
from src.models.multimodal_timesfm import MultimodalTimesFM
from src.models.text_encoder import MultimodalFusion, TextEncoder


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


class TestTextEncoder:
    """Test cases for TextEncoder component."""

    @pytest.fixture
    def text_encoder(self) -> TextEncoder:
        """Creates TextEncoder instance for testing."""
        return TextEncoder(model_name="all-MiniLM-L6-v2", embedding_dim=384)

    def test_text_encoder_initialization(self) -> None:
        """Tests TextEncoder initialization."""
        encoder = TextEncoder(embedding_dim=256)
        assert encoder.embedding_dim == 256
        assert encoder.sentence_transformer is not None

    def test_text_encoding(self, text_encoder: TextEncoder) -> None:
        """Tests text encoding functionality."""
        texts = ["This is a test sentence.", "Another test sentence with more words."]

        embeddings = text_encoder(texts)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (2, 384)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_single_text_encoding(self, text_encoder: TextEncoder) -> None:
        """Tests encoding of single text."""
        text = ["Climate data shows temperature trends."]

        embeddings = text_encoder(text)

        assert embeddings.shape == (1, 384)
        assert not torch.isnan(embeddings).any()

    def test_empty_text_handling(self, text_encoder: TextEncoder) -> None:
        """Tests handling of empty text list."""
        # sentence-transformers handles empty lists gracefully, returning a 1D tensor
        result = text_encoder([])
        assert result.shape == torch.Size([0]) or result.shape == (0, 384)
        assert result.numel() == 0  # Should be empty tensor

    def test_embedding_dimension_consistency(self) -> None:
        """Tests that different embedding dimensions work correctly."""
        # Use CPU to avoid MPS device issues in tests
        encoder_128 = TextEncoder(embedding_dim=128)
        encoder_512 = TextEncoder(embedding_dim=512)

        # Move to CPU explicitly
        encoder_128.to("cpu")
        encoder_512.to("cpu")

        texts = ["Test sentence"]

        with torch.no_grad():  # Disable gradients for inference
            emb_128 = encoder_128(texts)
            emb_512 = encoder_512(texts)

        assert emb_128.shape == (1, 128)
        assert emb_512.shape == (1, 512)


class TestMultimodalFusion:
    """Test cases for MultimodalFusion component."""

    @pytest.fixture
    def fusion_concat(self) -> MultimodalFusion:
        """Creates concatenation fusion module."""
        return MultimodalFusion(ts_feature_dim=256, text_feature_dim=384, output_dim=512, fusion_type="concat")

    @pytest.fixture
    def fusion_attention(self) -> MultimodalFusion:
        """Creates attention fusion module."""
        return MultimodalFusion(ts_feature_dim=256, text_feature_dim=384, output_dim=256, fusion_type="attention")

    @pytest.fixture
    def fusion_gated(self) -> MultimodalFusion:
        """Creates gated fusion module."""
        return MultimodalFusion(ts_feature_dim=256, text_feature_dim=384, output_dim=256, fusion_type="gated")

    def test_concat_fusion(self, fusion_concat: MultimodalFusion) -> None:
        """Tests concatenation fusion."""
        batch_size, seq_len = 2, 10
        ts_features = torch.randn(batch_size, seq_len, 256)
        text_features = torch.randn(batch_size, 384)

        fused = fusion_concat(ts_features, text_features)

        assert fused.shape == (batch_size, seq_len, 512)
        assert not torch.isnan(fused).any()

    def test_attention_fusion(self, fusion_attention: MultimodalFusion) -> None:
        """Tests attention-based fusion."""
        batch_size, seq_len = 2, 10
        ts_features = torch.randn(batch_size, seq_len, 256)
        text_features = torch.randn(batch_size, 384)

        fused = fusion_attention(ts_features, text_features)

        assert fused.shape == (batch_size, seq_len, 256)
        assert not torch.isnan(fused).any()

    def test_gated_fusion(self, fusion_gated: MultimodalFusion) -> None:
        """Tests gated fusion."""
        batch_size, seq_len = 2, 10
        ts_features = torch.randn(batch_size, seq_len, 256)
        text_features = torch.randn(batch_size, 384)

        fused = fusion_gated(ts_features, text_features)

        assert fused.shape == (batch_size, seq_len, 256)
        assert not torch.isnan(fused).any()

    def test_invalid_fusion_type(self) -> None:
        """Tests error handling for invalid fusion type."""
        with pytest.raises(ValueError):
            MultimodalFusion(ts_feature_dim=256, text_feature_dim=384, output_dim=512, fusion_type="invalid")


class TestMultimodalTimesFMEnhanced:
    """Test cases for enhanced MultimodalTimesFM with text support."""

    @pytest.fixture
    def hparams(self) -> TimesFmHparams:
        """Creates TimesFM hyperparameters for testing."""
        return TimesFmHparams(
            backend="cpu",
            context_len=512,
            horizon_len=128,
            num_layers=50,
            model_dims=1280,
        )

    @pytest.fixture
    def checkpoint(self) -> TimesFmCheckpoint:
        """Creates TimesFM checkpoint for testing."""
        return TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")

    @pytest.fixture
    def multimodal_wrapper(self, hparams: TimesFmHparams, checkpoint: TimesFmCheckpoint) -> MultimodalTimesFM:
        """Creates MultimodalTimesFM wrapper with multimodal enabled."""
        return MultimodalTimesFM(hparams, checkpoint, enable_multimodal=True)

    @pytest.fixture
    def unimodal_wrapper(self, hparams: TimesFmHparams, checkpoint: TimesFmCheckpoint) -> MultimodalTimesFM:
        """Creates MultimodalTimesFM wrapper with multimodal disabled."""
        return MultimodalTimesFM(hparams, checkpoint, enable_multimodal=False)

    def test_multimodal_initialization(self, multimodal_wrapper: MultimodalTimesFM) -> None:
        """Tests multimodal wrapper initialization."""
        assert multimodal_wrapper.is_multimodal_enabled()
        assert multimodal_wrapper.get_text_encoder() is not None
        assert multimodal_wrapper.get_fusion_module() is not None

    def test_unimodal_initialization(self, unimodal_wrapper: MultimodalTimesFM) -> None:
        """Tests unimodal wrapper initialization."""
        assert not unimodal_wrapper.is_multimodal_enabled()
        assert unimodal_wrapper.get_text_encoder() is None
        assert unimodal_wrapper.get_fusion_module() is None

    def test_text_encoding_functionality(self, multimodal_wrapper: MultimodalTimesFM) -> None:
        """Tests text encoding through wrapper."""
        texts = ["Climate data shows temperature trends.", "Weather patterns indicate seasonal changes."]

        embeddings = multimodal_wrapper.encode_text(texts)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (2, 384)
        assert not torch.isnan(embeddings).any()

    def test_text_encoding_disabled(self, unimodal_wrapper: MultimodalTimesFM) -> None:
        """Tests that text encoding raises error when disabled."""
        texts = ["Test text"]

        with pytest.raises(RuntimeError):
            unimodal_wrapper.encode_text(texts)

    def test_forecast_with_text_inputs(self, multimodal_wrapper: MultimodalTimesFM) -> None:
        """Tests forecasting with text inputs (currently falls back to TimesFM)."""
        # Create dummy time series data
        ts_data = [np.random.randn(512).astype(np.float32)]
        text_data = ["Climate data showing temperature trends over time."]

        # This should work but currently falls back to regular TimesFM
        mean_forecast, quantile_forecast = multimodal_wrapper.forecast(ts_data, text_inputs=text_data)

        assert isinstance(mean_forecast, np.ndarray)
        assert isinstance(quantile_forecast, np.ndarray)
        assert mean_forecast.shape == (1, 128)

    def test_forecast_without_text_inputs(self, multimodal_wrapper: MultimodalTimesFM) -> None:
        """Tests forecasting without text inputs."""
        ts_data = [np.random.randn(512).astype(np.float32)]

        mean_forecast, quantile_forecast = multimodal_wrapper.forecast(ts_data)

        assert isinstance(mean_forecast, np.ndarray)
        assert isinstance(quantile_forecast, np.ndarray)
        assert mean_forecast.shape == (1, 128)
