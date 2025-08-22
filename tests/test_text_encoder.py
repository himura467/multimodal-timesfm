"""Tests for text encoding components."""

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
        encoder = TextEncoder(embedding_dim=384)

        assert encoder.embedding_dim == 384

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

    def test_no_gradient_computation(self) -> None:
        """Tests that encoder works without gradient computation (inference only)."""
        encoder = TextEncoder()
        texts: list[str] = ["Test no gradient computation."]

        with torch.no_grad():
            result = encoder(texts)

        assert result.shape == (1, 384)
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

    def test_reproducibility(self) -> None:
        """Tests that TextEncoder produces consistent results."""
        encoder1 = TextEncoder()
        encoder2 = TextEncoder()

        texts: list[str] = ["Reproducibility test sentence."]

        result1 = encoder1(texts)
        result2 = encoder2(texts)

        # Results should be very similar (sentence transformer is deterministic)
        assert torch.allclose(result1, result2, atol=1e-4)

    def test_with_different_text_lengths(self) -> None:
        """Tests TextEncoder with texts of different lengths."""
        encoder = TextEncoder()
        texts: list[str] = [
            "Short text.",
            "This is a much longer text that contains more information and words.",
            "Medium length text with some details.",
        ]

        result = encoder(texts)

        assert result.shape == (3, 384)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

        # All embeddings should be different
        assert not torch.allclose(result[0], result[1])
        assert not torch.allclose(result[1], result[2])
        assert not torch.allclose(result[0], result[2])
