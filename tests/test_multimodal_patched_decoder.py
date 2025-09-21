"""Tests for MultimodalPatchedDecoder."""

from typing import cast

import pytest
import torch

from src.models.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from src.models.text_encoder import EnglishTextEncoder, JapaneseTextEncoder


class TestMultimodalTimesFMConfig:
    """Test MultimodalTimesFMConfig configuration class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MultimodalTimesFMConfig()

        # Test multimodal-specific defaults
        assert config.text_encoder_type == "english"

        # Test inherited TimesFM defaults
        assert config.num_layers == 20

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MultimodalTimesFMConfig(
            text_encoder_type="japanese",
            num_layers=50,
        )

        assert config.text_encoder_type == "japanese"
        assert config.num_layers == 50


class TestMultimodalPatchedDecoder:
    """Test MultimodalPatchedDecoder class."""

    @pytest.fixture(scope="session")
    def config(self) -> MultimodalTimesFMConfig:
        """Create test configuration."""
        return MultimodalTimesFMConfig(
            num_layers=2,  # Reduced for faster testing
            hidden_size=128,
            intermediate_size=128,
            patch_len=8,
            horizon_len=16,
        )

    @pytest.fixture(scope="session")
    def decoder(self, config: MultimodalTimesFMConfig) -> MultimodalPatchedDecoder:
        """Create decoder instance."""
        return MultimodalPatchedDecoder(config)

    @pytest.fixture(scope="session")
    def sample_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]]:
        """Create sample input data."""
        batch_size = 2
        seq_len = 64
        patch_len = 8
        num_patches = seq_len // patch_len

        input_ts = torch.randn(batch_size, seq_len)
        input_padding = torch.zeros(batch_size, seq_len, dtype=torch.float)
        freq = torch.zeros(batch_size, 1, dtype=torch.long)

        # Create patch-level text descriptions: [batch][patch][texts]
        text_descriptions = [
            [  # Batch 1
                [f"Batch 1 Patch {i} text"] for i in range(num_patches)
            ],
            [  # Batch 2
                [f"Batch 2 Patch {i} text"] for i in range(num_patches)
            ],
        ]

        return input_ts, input_padding, freq, text_descriptions

    def test_initialization(self, config: MultimodalTimesFMConfig) -> None:
        """Test decoder initialization."""
        decoder = MultimodalPatchedDecoder(config)

        assert decoder.config == config
        assert decoder.text_encoder is not None
        assert decoder.multimodal_fusion is not None
        assert hasattr(decoder, "input_ff_layer")
        assert hasattr(decoder, "freq_emb")
        assert hasattr(decoder, "horizon_ff_layer")
        assert hasattr(decoder, "stacked_transformer")
        assert hasattr(decoder, "position_emb")

    def test_preprocess_multimodal_input(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test multimodal input preprocessing."""
        input_ts, input_padding, _, text_descriptions = sample_data

        # Move input tensors to the same device as decoder
        device = decoder.device
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)

        with torch.no_grad():
            model_input, patched_padding, stats, patched_inputs = decoder._preprocess_multimodal_input(
                input_ts=input_ts,
                input_padding=input_padding,
                text_descriptions=text_descriptions,
            )

        batch_size = input_ts.shape[0]
        num_patches = input_ts.shape[1] // decoder.config.patch_len

        assert model_input.shape == (batch_size, num_patches, decoder.config.hidden_size)
        assert not torch.isnan(model_input).any()
        assert patched_padding.shape == (batch_size, num_patches)
        assert stats is not None
        assert len(stats) == 2  # mean, std
        assert patched_inputs is not None
        assert patched_inputs.shape == (batch_size, num_patches, decoder.config.patch_len)
        assert not torch.isnan(patched_inputs).any()

    def test_encode_patch_text_features(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test patch-level text encoding functionality."""
        _, _, _, text_descriptions = sample_data
        target_shape = torch.Size([2, 8, 128])  # batch_size, num_patches, hidden_size

        # Get decoder's device
        device = torch.device(decoder.device)

        with torch.no_grad():
            text_features = decoder._encode_patch_text_features(text_descriptions, target_shape, device)

        expected_shape = (2, 8, decoder.text_encoder.embedding_dim)
        assert text_features.shape == expected_shape
        assert not torch.isnan(text_features).any()

    def test_encode_patch_text_features_batch_size_mismatch(self, decoder: MultimodalPatchedDecoder) -> None:
        """Test patch-level text encoding with batch size mismatch."""
        text_descriptions = [[["Text 1"]]]  # Batch size 1 with 1 patch
        target_shape = torch.Size([2, 8, 128])  # Batch size 2 with 8 patches

        # Get decoder's device
        device = torch.device(decoder.device)

        with pytest.raises(ValueError, match="Batch size mismatch"):
            decoder._encode_patch_text_features(text_descriptions, target_shape, device)

    def test_batch_encoding_consistency(self, decoder: MultimodalPatchedDecoder) -> None:
        """Test that batch encoding produces same results as one-by-one encoding."""
        # Create test text descriptions
        text_descriptions = [
            [  # Batch 1
                ["First batch first patch text", "additional text"],
                ["First batch second patch text"],
                ["First batch third patch with longer description text"],
                ["First batch fourth patch text"],
            ],
            [  # Batch 2
                ["Second batch first patch text"],
                ["Second batch second patch text", "more text", "even more"],
                ["Second batch third patch text"],
                ["Second batch fourth patch text"],
            ],
        ]

        target_shape = torch.Size([2, 4, 128])  # batch_size=2, num_patches=4, hidden_size=128
        device = torch.device(decoder.device)

        # Method 1: Batch encoding (current implementation)
        with torch.no_grad():
            batch_embeddings = decoder._encode_patch_text_features(text_descriptions, target_shape, device)

        # Method 2: One-by-one encoding for comparison
        individual_embeddings = []
        for batch_patches in text_descriptions:
            batch_individual_embeddings = []
            for patch_texts in batch_patches:
                if patch_texts:
                    text = " ".join(patch_texts)
                else:
                    text = ""
                # Encode single text
                single_embedding = decoder.text_encoder([text])  # Single text in list
                batch_individual_embeddings.append(single_embedding[0])  # Extract single embedding
            individual_embeddings.append(torch.stack(batch_individual_embeddings))

        individual_result = torch.stack(individual_embeddings)

        # Compare results - they should be identical
        assert batch_embeddings.shape == individual_result.shape
        assert torch.allclose(batch_embeddings, individual_result, atol=1e-6), (
            "Batch encoding should produce identical results to one-by-one encoding"
        )

    def test_forward_pass(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test forward pass."""
        input_ts, input_padding, freq, text_descriptions = sample_data

        # Move input tensors to the same device as decoder
        device = decoder.device
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)
        freq = freq.to(device)

        with torch.no_grad():
            output = decoder(input_ts, input_padding, freq, text_descriptions)

        # Check output shape: [batch_size, num_patches, horizon_len, num_outputs]
        batch_size = input_ts.shape[0]
        num_patches = input_ts.shape[1] // decoder.config.patch_len
        num_outputs = len(decoder.config.quantiles) + 1

        expected_shape = (batch_size, num_patches, decoder.config.horizon_len, num_outputs)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()

    def test_decode(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test auto-regressive decoding."""
        input_ts, _, freq, text_descriptions = sample_data
        batch_size, context_len = input_ts.shape
        horizon_len = 32

        # Move input tensors to the same device as decoder
        device = decoder.device
        input_ts = input_ts.to(device)
        freq = freq.to(device)

        # Create paddings for context + horizon
        paddings = torch.zeros(batch_size, context_len + horizon_len, device=device)

        with torch.no_grad():
            mean_output, full_output = decoder.decode(
                input_ts=input_ts,
                paddings=paddings,
                freq=freq,
                horizon_len=horizon_len,
                text_descriptions=text_descriptions,
            )

        # Check output shapes
        assert mean_output.shape == (batch_size, horizon_len)
        num_outputs = len(decoder.config.quantiles) + 1
        assert full_output.shape == (batch_size, horizon_len, num_outputs)
        assert not torch.isnan(mean_output).any()
        assert not torch.isnan(full_output).any()

    def test_decode_padding_length_mismatch(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test decode with incorrect padding length."""
        input_ts, _, freq, text_descriptions = sample_data
        batch_size, context_len = input_ts.shape
        horizon_len = 32

        # Move input tensors to the same device as decoder
        device = decoder.device
        input_ts = input_ts.to(device)
        freq = freq.to(device)

        # Incorrect padding length
        paddings = torch.zeros(batch_size, context_len + horizon_len - 5, device=device)  # Too short

        with pytest.raises(ValueError, match="Length of paddings must match"):
            decoder.decode(
                input_ts=input_ts,
                paddings=paddings,
                freq=freq,
                horizon_len=horizon_len,
                text_descriptions=text_descriptions,
            )

    def test_freeze_unfreeze_all_parameters(self, decoder: MultimodalPatchedDecoder) -> None:
        """Test freezing and unfreezing all parameters."""
        # Initially unfrozen - check a few key parameters
        text_encoder = cast(EnglishTextEncoder | JapaneseTextEncoder, decoder.text_encoder)
        text_encoder_param = next(text_encoder.sentence_transformer.parameters())
        assert text_encoder_param.requires_grad
        assert decoder.multimodal_fusion.text_projection.weight.requires_grad
        assert decoder.multimodal_fusion.text_projection.bias.requires_grad

        # Check that some TimesFM parameters are also unfrozen
        timesfm_param = next(decoder.input_ff_layer.parameters())
        assert timesfm_param.requires_grad

        # Freeze all parameters
        decoder.freeze_parameters()

        # Check that text components are frozen
        assert not text_encoder_param.requires_grad
        assert not decoder.multimodal_fusion.text_projection.weight.requires_grad
        assert not decoder.multimodal_fusion.text_projection.bias.requires_grad

        # Check that TimesFM parameters are also frozen
        assert not timesfm_param.requires_grad

        # Verify is_frozen status
        assert decoder.is_frozen()

        # Unfreeze all parameters
        decoder.unfreeze_parameters()

        # Check that text components are unfrozen
        assert text_encoder_param.requires_grad
        assert decoder.multimodal_fusion.text_projection.weight.requires_grad
        assert decoder.multimodal_fusion.text_projection.bias.requires_grad

        # Check that TimesFM parameters are also unfrozen
        assert timesfm_param.requires_grad

        # Verify is_frozen status
        assert not decoder.is_frozen()

    def test_freeze_unfreeze_text_components(self, decoder: MultimodalPatchedDecoder) -> None:
        """Test freezing and unfreezing text components."""
        # Initially unfrozen - check actual parameters
        text_encoder = cast(EnglishTextEncoder | JapaneseTextEncoder, decoder.text_encoder)
        text_encoder_param = next(text_encoder.sentence_transformer.parameters())
        assert text_encoder_param.requires_grad
        assert decoder.multimodal_fusion.text_projection.weight.requires_grad
        assert decoder.multimodal_fusion.text_projection.bias.requires_grad

        # Check that components are not frozen initially
        frozen_status = decoder.is_text_frozen()
        assert not frozen_status["encoder"]
        assert not frozen_status["fusion"]

        # Freeze text components
        decoder.freeze_text_components()

        assert not text_encoder_param.requires_grad
        assert not decoder.multimodal_fusion.text_projection.weight.requires_grad
        assert not decoder.multimodal_fusion.text_projection.bias.requires_grad

        # Check that components are frozen
        frozen_status = decoder.is_text_frozen()
        assert frozen_status["encoder"]
        assert frozen_status["fusion"]

        # Unfreeze text components
        decoder.unfreeze_text_components()

        assert text_encoder_param.requires_grad
        assert decoder.multimodal_fusion.text_projection.weight.requires_grad
        assert decoder.multimodal_fusion.text_projection.bias.requires_grad

        # Check that components are unfrozen
        frozen_status = decoder.is_text_frozen()
        assert not frozen_status["encoder"]
        assert not frozen_status["fusion"]

    def test_gradient_computation(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test that gradients can be computed."""
        input_ts, input_padding, freq, text_descriptions = sample_data

        # Move input tensors to the same device as decoder
        device = decoder.device
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)
        freq = freq.to(device)

        # Enable gradients
        input_ts.requires_grad_(True)

        output = decoder(input_ts, input_padding, freq, text_descriptions)
        loss = output.sum()  # Simple loss for gradient test

        loss.backward()

        # Check that gradients are computed
        assert input_ts.grad is not None
        assert not torch.isnan(input_ts.grad).any()
