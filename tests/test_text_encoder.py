"""Tests for text encoder classes."""

import numpy as np
import pytest
import torch

from multimodal_timesfm.text_encoder import EnglishTextEncoder, JapaneseTextEncoder


class TestEnglishTextEncoder:
    """Test cases for EnglishTextEncoder class using real sentence transformers."""

    def test_init_default_parameters(self) -> None:
        """Tests initialization with default parameters."""
        encoder = EnglishTextEncoder()

        assert encoder.embedding_dim == 384

    def test_init_dimension_mismatch_error(self) -> None:
        """Tests initialization fails with mismatched embedding dimensions."""
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            EnglishTextEncoder(embedding_dim=512)

    def test_init_matching_dimensions(self) -> None:
        """Tests initialization when actual and desired dimensions match."""
        encoder = EnglishTextEncoder(model_name="sentence-transformers/all-MiniLM-L12-v2", embedding_dim=384)

        assert encoder.embedding_dim == 384

    def test_forward_with_string_input(self) -> None:
        """Tests forward pass with single string input."""
        encoder = EnglishTextEncoder()
        text = "This is a single string input."
        result = encoder(text)

        assert result.shape == (384,)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_forward_single_text(self) -> None:
        """Tests forward pass with single text."""
        encoder = EnglishTextEncoder()
        texts: list[str] = ["This is a test sentence."]
        result = encoder(texts)

        assert result.shape == (1, 384)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_forward_multiple_texts(self) -> None:
        """Tests forward pass with multiple texts."""
        encoder = EnglishTextEncoder()
        texts: list[str] = ["First sentence.", "Second sentence.", "Third sentence."]
        result = encoder(texts)

        assert result.shape == (3, 384)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_forward_empty_list(self) -> None:
        """Tests forward pass with empty text list."""
        encoder = EnglishTextEncoder()
        texts: list[str] = []
        result = encoder(texts)

        assert result.shape == (0,)
        assert isinstance(result, torch.Tensor)

    def test_forward_with_numpy_array(self) -> None:
        """Tests forward pass with numpy array input."""
        encoder = EnglishTextEncoder()
        texts_array = np.array(["First text", "Second text", "Third text"])
        result = encoder(texts_array)

        assert result.shape == (3, 384)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_device_consistency(self) -> None:
        """Tests that output tensor is on the correct device."""
        encoder = EnglishTextEncoder()
        texts: list[str] = ["Device test."]
        result = encoder(texts)

        # Result should be on the same device type as the encoder
        assert result.device.type == encoder.device.type

    def test_explicit_device_selection(self) -> None:
        """Tests that explicit device selection works."""
        encoder = EnglishTextEncoder(device="cpu")
        texts: list[str] = ["CPU device test."]
        result = encoder(texts)

        assert result.device.type == "cpu"
        assert encoder.device.type == "cpu"

    def test_freeze_unfreeze_parameters(self) -> None:
        """Tests freezing and unfreezing parameters."""
        encoder = EnglishTextEncoder()

        # Initially parameters should be unfrozen
        assert not encoder.is_frozen()

        # Freeze parameters
        encoder.freeze_parameters()
        assert encoder.is_frozen()

        # Unfreeze parameters
        encoder.unfreeze_parameters()
        assert not encoder.is_frozen()


class TestJapaneseTextEncoder:
    """Test cases for JapaneseTextEncoder class using Ruri models."""

    def test_init_default_parameters(self) -> None:
        """Tests initialization with default parameters."""
        encoder = JapaneseTextEncoder()

        assert encoder.embedding_dim == 768

    def test_init_dimension_mismatch_error(self) -> None:
        """Tests initialization fails with mismatched embedding dimensions."""
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            JapaneseTextEncoder(embedding_dim=384)

    def test_forward_with_string_input(self) -> None:
        """Tests forward pass with single string input."""
        encoder = JapaneseTextEncoder()
        text = "これはテストです。"
        result = encoder(text)

        assert result.shape == (768,)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_forward_multiple_texts(self) -> None:
        """Tests forward pass with multiple texts."""
        encoder = JapaneseTextEncoder()
        texts: list[str] = ["最初の文です。", "二番目の文です。", "三番目の文です。"]
        result = encoder(texts)

        assert result.shape == (3, 768)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_forward_with_numpy_array(self) -> None:
        """Tests forward pass with numpy array input."""
        encoder = JapaneseTextEncoder()
        texts_array = np.array(["最初のテキスト", "二番目のテキスト", "三番目のテキスト"])
        result = encoder(texts_array)

        assert result.shape == (3, 768)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_device_consistency(self) -> None:
        """Tests that output tensor is on the correct device."""
        encoder = JapaneseTextEncoder()
        texts: list[str] = ["デバイスのテスト"]
        result = encoder(texts)

        # Result should be on the same device type as the encoder
        assert result.device.type == encoder.device.type

    def test_freeze_unfreeze_parameters(self) -> None:
        """Tests freezing and unfreezing parameters."""
        encoder = JapaneseTextEncoder()

        # Initially parameters should be unfrozen
        assert not encoder.is_frozen()

        # Freeze parameters
        encoder.freeze_parameters()
        assert encoder.is_frozen()

        # Unfreeze parameters
        encoder.unfreeze_parameters()
        assert not encoder.is_frozen()
