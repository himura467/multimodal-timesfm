"""Tests for TextEncoder class."""

import numpy as np
import pytest
import torch

from src.models.text_encoder import TextEncoder


class TestTextEncoder:
    """Test cases for TextEncoder class using real sentence transformers."""

    def test_init_default_parameters(self) -> None:
        """Tests initialization with default parameters."""
        encoder = TextEncoder()

        assert encoder.embedding_dim == 384

    def test_init_dimension_mismatch_error(self) -> None:
        """Tests initialization fails with mismatched embedding dimensions."""
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            TextEncoder(embedding_dim=512)

    def test_init_matching_dimensions(self) -> None:
        """Tests initialization when actual and desired dimensions match."""
        encoder = TextEncoder(model_name="sentence-transformers/all-MiniLM-L12-v2", embedding_dim=384)

        assert encoder.embedding_dim == 384

    def test_forward_with_string_input(self) -> None:
        """Tests forward pass with single string input."""
        encoder = TextEncoder()
        text = "This is a single string input."
        result = encoder(text)

        assert result.shape == (384,)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_forward_single_text(self) -> None:
        """Tests forward pass with single text."""
        encoder = TextEncoder()
        texts: list[str] = ["This is a test sentence."]
        result = encoder(texts)

        assert result.shape == (1, 384)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_forward_multiple_texts(self) -> None:
        """Tests forward pass with multiple texts."""
        encoder = TextEncoder()
        texts: list[str] = ["First sentence.", "Second sentence.", "Third sentence."]
        result = encoder(texts)

        assert result.shape == (3, 384)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_forward_empty_list(self) -> None:
        """Tests forward pass with empty text list."""
        encoder = TextEncoder()
        texts: list[str] = []
        result = encoder(texts)

        assert result.shape == (0,)
        assert isinstance(result, torch.Tensor)

    def test_forward_with_numpy_array(self) -> None:
        """Tests forward pass with numpy array input."""
        encoder = TextEncoder()
        texts_array = np.array(["First text", "Second text", "Third text"])
        result = encoder(texts_array)

        assert result.shape == (3, 384)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_device_consistency(self) -> None:
        """Tests that output tensor is on the correct device."""
        encoder = TextEncoder()
        texts: list[str] = ["Device test."]
        result = encoder(texts)

        # Result should be on the same device type as the encoder
        assert result.device.type == encoder.device.type

    def test_explicit_device_selection(self) -> None:
        """Tests that explicit device selection works."""
        encoder = TextEncoder(device="cpu")
        texts: list[str] = ["CPU device test."]
        result = encoder(texts)

        assert result.device.type == "cpu"
        assert encoder.device.type == "cpu"
