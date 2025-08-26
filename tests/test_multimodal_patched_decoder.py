"""Tests for MultimodalPatchedDecoder."""

import pytest
import torch

from src.models.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig


class TestMultimodalTimesFMConfig:
    """Test MultimodalTimesFMConfig configuration class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MultimodalTimesFMConfig()

        # Test multimodal-specific defaults
        assert config.text_encoder_model == "all-MiniLM-L6-v2"
        assert config.text_embedding_dim == 384

        # Test inherited TimesFM defaults
        assert config.num_layers == 20
        assert config.hidden_size == 1280
        assert config.patch_len == 32

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MultimodalTimesFMConfig(
            text_encoder_model="sentence-transformers/all-MiniLM-L12-v2",
            text_embedding_dim=512,
            hidden_size=512,
        )

        assert config.text_encoder_model == "sentence-transformers/all-MiniLM-L12-v2"
        assert config.text_embedding_dim == 512
        assert config.hidden_size == 512


class TestMultimodalPatchedDecoder:
    """Test MultimodalPatchedDecoder class."""

    @pytest.fixture
    def config(self) -> MultimodalTimesFMConfig:
        """Create test configuration."""
        return MultimodalTimesFMConfig(
            num_layers=2,  # Reduced for faster testing
            hidden_size=128,
            intermediate_size=128,
            patch_len=8,
            horizon_len=16,
            text_embedding_dim=384,
        )

    @pytest.fixture
    def decoder(self, config: MultimodalTimesFMConfig) -> MultimodalPatchedDecoder:
        """Create decoder instance."""
        return MultimodalPatchedDecoder(config)

    @pytest.fixture
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

    def test_initialization_with_text(self, config: MultimodalTimesFMConfig) -> None:
        """Test decoder initialization with text fusion enabled."""
        decoder = MultimodalPatchedDecoder(config)

        assert decoder.config == config
        assert decoder.text_encoder is not None
        assert decoder.multimodal_fusion is not None
        assert hasattr(decoder, "input_ff_layer")
        assert hasattr(decoder, "stacked_transformer")

    def test_forward_pass_with_text(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test forward pass with text inputs."""
        input_ts, input_padding, freq, text_descriptions = sample_data

        # Move input tensors to the same device as decoder
        device = next(decoder.parameters()).device
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

    def test_forward_pass_without_text(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test forward pass without text inputs."""
        input_ts, input_padding, freq, _ = sample_data

        # Move input tensors to the same device as decoder
        device = next(decoder.parameters()).device
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)
        freq = freq.to(device)

        with torch.no_grad():
            output = decoder(input_ts, input_padding, freq)

        batch_size = input_ts.shape[0]
        num_patches = input_ts.shape[1] // decoder.config.patch_len
        num_outputs = len(decoder.config.quantiles) + 1

        expected_shape = (batch_size, num_patches, decoder.config.horizon_len, num_outputs)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()

    def test_encode_patch_text_features(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test patch-level text encoding functionality."""
        _, _, _, text_descriptions = sample_data
        target_shape = torch.Size([2, 8, 128])  # batch_size, num_patches, hidden_size

        # Get decoder's device
        device = next(decoder.parameters()).device

        with torch.no_grad():
            text_features = decoder._encode_patch_text_features(text_descriptions, target_shape, device)

        expected_shape = (2, 8, decoder.config.text_embedding_dim)
        assert text_features.shape == expected_shape
        assert not torch.isnan(text_features).any()

    def test_encode_patch_text_features_batch_size_mismatch(self, decoder: MultimodalPatchedDecoder) -> None:
        """Test patch-level text encoding with batch size mismatch."""
        text_descriptions = [[["Text 1"]]]  # Batch size 1 with 1 patch
        target_shape = torch.Size([2, 8, 128])  # Batch size 2 with 8 patches

        # Get decoder's device
        device = next(decoder.parameters()).device

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
            ],
            [  # Batch 2
                ["Second batch first patch text"],
                ["Second batch second patch text", "more text", "even more"],
                ["Second batch third patch text"],
            ],
        ]

        target_shape = torch.Size([2, 3, 128])  # batch_size=2, num_patches=3, hidden_size=128
        device = next(decoder.parameters()).device

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

    def test_decode_with_text(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test auto-regressive decoding with text."""
        input_ts, _, freq, text_descriptions = sample_data
        batch_size, context_len = input_ts.shape
        horizon_len = 32

        # Move input tensors to the same device as decoder
        device = next(decoder.parameters()).device
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

    def test_decode_without_text(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test auto-regressive decoding without text."""
        input_ts, _, freq, _ = sample_data
        batch_size, context_len = input_ts.shape
        horizon_len = 32

        # Move input tensors to the same device as decoder
        device = next(decoder.parameters()).device
        input_ts = input_ts.to(device)
        freq = freq.to(device)

        paddings = torch.zeros(batch_size, context_len + horizon_len, device=device)

        with torch.no_grad():
            mean_output, full_output = decoder.decode(
                input_ts=input_ts,
                paddings=paddings,
                freq=freq,
                horizon_len=horizon_len,
            )

        assert mean_output.shape == (batch_size, horizon_len)
        num_outputs = len(decoder.config.quantiles) + 1
        assert full_output.shape == (batch_size, horizon_len, num_outputs)

    def test_freeze_unfreeze_text_components(self, decoder: MultimodalPatchedDecoder) -> None:
        """Test freezing and unfreezing text components."""
        # Initially unfrozen
        assert not decoder.is_text_frozen()

        # Freeze
        decoder.freeze_text_components()
        assert decoder.is_text_frozen()

        # Unfreeze
        decoder.unfreeze_text_components()
        assert not decoder.is_text_frozen()

    def test_text_fusion_parameters(self, decoder: MultimodalPatchedDecoder) -> None:
        """Test getting and setting text fusion parameters."""
        # Get original parameters
        original_params = decoder.get_text_fusion_parameters()
        assert original_params is not None
        assert "weight" in original_params
        assert "bias" in original_params

        # Modify parameters
        new_weight = torch.randn_like(original_params["weight"])
        new_bias = torch.randn_like(original_params["bias"])
        new_params = {"weight": new_weight, "bias": new_bias}

        # Set new parameters
        decoder.set_text_fusion_parameters(new_params)

        # Verify parameters were set
        current_params = decoder.get_text_fusion_parameters()
        assert torch.allclose(current_params["weight"], new_weight)
        assert torch.allclose(current_params["bias"], new_bias)

    def test_preprocess_multimodal_input(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test multimodal input preprocessing."""
        input_ts, input_padding, _, text_descriptions = sample_data

        # Move input tensors to the same device as decoder
        device = next(decoder.parameters()).device
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
        assert patched_padding.shape == (batch_size, num_patches)
        assert stats is not None
        assert len(stats) == 2  # mean, std
        assert not torch.isnan(model_input).any()

    def test_preprocess_multimodal_input_no_text(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test multimodal preprocessing without text."""
        input_ts, input_padding, _, _ = sample_data

        # Move input tensors to the same device as decoder
        device = next(decoder.parameters()).device
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)

        with torch.no_grad():
            model_input, patched_padding, stats, patched_inputs = decoder._preprocess_multimodal_input(
                input_ts=input_ts,
                input_padding=input_padding,
                text_descriptions=None,
            )

        batch_size = input_ts.shape[0]
        num_patches = input_ts.shape[1] // decoder.config.patch_len

        assert model_input.shape == (batch_size, num_patches, decoder.config.hidden_size)
        assert not torch.isnan(model_input).any()

    def test_decode_padding_length_mismatch(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test decode with incorrect padding length."""
        input_ts, _, freq, _ = sample_data
        batch_size, context_len = input_ts.shape
        horizon_len = 32

        # Move input tensors to the same device as decoder
        device = next(decoder.parameters()).device
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
            )

    def test_gradient_computation(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test that gradients can be computed."""
        input_ts, input_padding, freq, text_descriptions = sample_data

        # Move input tensors to the same device as decoder
        device = next(decoder.parameters()).device
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

    def test_device_consistency(
        self,
        decoder: MultimodalPatchedDecoder,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[str]]]],
    ) -> None:
        """Test device consistency for inputs and outputs."""
        input_ts, input_padding, freq, text_descriptions = sample_data

        # Move to CPU explicitly (default should be CPU)
        device = torch.device("cpu")
        decoder = decoder.to(device)
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)
        freq = freq.to(device)

        with torch.no_grad():
            output = decoder(input_ts, input_padding, freq, text_descriptions)

        assert output.device == device

    def test_different_batch_sizes(self, decoder: MultimodalPatchedDecoder) -> None:
        """Test decoder with different batch sizes."""
        batch_sizes = [1, 3, 8]
        seq_len = 64

        # Get decoder's device
        device = next(decoder.parameters()).device

        for batch_size in batch_sizes:
            input_ts = torch.randn(batch_size, seq_len, device=device)
            input_padding = torch.zeros(batch_size, seq_len, dtype=torch.float, device=device)
            freq = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

            # Create patch-level text descriptions for different batch sizes
            num_patches = seq_len // decoder.config.patch_len
            text_descriptions = [[[f"Batch {b} Patch {p} text"] for p in range(num_patches)] for b in range(batch_size)]

            with torch.no_grad():
                output = decoder(input_ts, input_padding, freq, text_descriptions)

            num_patches = seq_len // decoder.config.patch_len
            num_outputs = len(decoder.config.quantiles) + 1
            expected_shape = (batch_size, num_patches, decoder.config.horizon_len, num_outputs)

            assert output.shape == expected_shape
            assert not torch.isnan(output).any()
