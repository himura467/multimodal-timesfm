"""Tests for text encoding components."""

import pytest
import torch

from src.models.multimodal_fusion import MultimodalFusion
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


class TestMultimodalFusion:
    """Test cases for MultimodalFusion class."""

    def test_init_valid_dimensions(self) -> None:
        """Tests initialization with valid dimensions."""
        fusion = MultimodalFusion(ts_feature_dim=1280, text_feature_dim=384)

        assert fusion.ts_feature_dim == 1280
        assert fusion.text_feature_dim == 384
        assert isinstance(fusion.text_projection, torch.nn.Linear)
        assert isinstance(fusion.activation, torch.nn.ReLU)
        assert fusion.text_projection.in_features == 384
        assert fusion.text_projection.out_features == 1280

    def test_init_invalid_dimensions(self) -> None:
        """Tests initialization fails with invalid dimensions."""
        with pytest.raises(ValueError, match="ts_feature_dim must be a positive integer"):
            MultimodalFusion(ts_feature_dim=0, text_feature_dim=384)

        with pytest.raises(ValueError, match="ts_feature_dim must be a positive integer"):
            MultimodalFusion(ts_feature_dim=-1, text_feature_dim=384)

        with pytest.raises(ValueError, match="text_feature_dim must be a positive integer"):
            MultimodalFusion(ts_feature_dim=1280, text_feature_dim=0)

        with pytest.raises(ValueError, match="text_feature_dim must be a positive integer"):
            MultimodalFusion(ts_feature_dim=1280, text_feature_dim=-1)

    def test_weight_initialization(self) -> None:
        """Tests that weights are properly initialized."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Check that weights are not zero (should be Xavier initialized)
        assert not torch.allclose(fusion.text_projection.weight, torch.zeros_like(fusion.text_projection.weight))

        # Check that bias is zero
        assert torch.allclose(fusion.text_projection.bias, torch.zeros_like(fusion.text_projection.bias))

    def test_forward_input_validation_wrong_types(self) -> None:
        """Tests forward pass fails with wrong input types."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        with pytest.raises(ValueError, match="ts_features must be a torch.Tensor"):
            fusion("not_a_tensor", torch.randn(2, 10, 64))

        with pytest.raises(ValueError, match="text_features must be a torch.Tensor"):
            fusion(torch.randn(2, 10, 128), "not_a_tensor")

    def test_forward_input_validation_wrong_dimensions(self) -> None:
        """Tests forward pass fails with wrong tensor dimensions."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # 2D tensors instead of 3D
        with pytest.raises(ValueError, match="ts_features must be 3D"):
            fusion(torch.randn(2, 128), torch.randn(2, 10, 64))

        with pytest.raises(ValueError, match="text_features must be 3D"):
            fusion(torch.randn(2, 10, 128), torch.randn(2, 64))

        # 4D tensors instead of 3D
        with pytest.raises(ValueError, match="ts_features must be 3D"):
            fusion(torch.randn(2, 5, 10, 128), torch.randn(2, 10, 64))

    def test_forward_batch_size_mismatch(self) -> None:
        """Tests forward pass fails with mismatched batch sizes."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        with pytest.raises(ValueError, match="Batch size mismatch"):
            fusion(torch.randn(2, 10, 128), torch.randn(3, 10, 64))

    def test_forward_sequence_length_mismatch(self) -> None:
        """Tests forward pass fails with mismatched sequence lengths."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        with pytest.raises(ValueError, match="Sequence length mismatch"):
            fusion(torch.randn(2, 10, 128), torch.randn(2, 8, 64))

    def test_forward_feature_dimension_mismatch(self) -> None:
        """Tests forward pass fails with mismatched feature dimensions."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        with pytest.raises(ValueError, match="Time series feature dimension mismatch"):
            fusion(torch.randn(2, 10, 100), torch.randn(2, 10, 64))

        with pytest.raises(ValueError, match="Text feature dimension mismatch"):
            fusion(torch.randn(2, 10, 128), torch.randn(2, 10, 50))

    def test_forward_device_mismatch(self) -> None:
        """Tests forward pass fails with device mismatch."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128, device="cpu")
        text_features = torch.randn(2, 10, 64)

        # Move one tensor to a different device if available
        if torch.cuda.is_available():
            text_features = text_features.to("cuda")
            with pytest.raises(RuntimeError, match="Device mismatch"):
                fusion(ts_features, text_features)
        elif torch.backends.mps.is_available():
            text_features = text_features.to("mps")
            with pytest.raises(RuntimeError, match="Device mismatch"):
                fusion(ts_features, text_features)

    def test_forward_pass_basic_functionality(self) -> None:
        """Tests basic forward pass functionality."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128)
        text_features = torch.randn(2, 10, 64)

        result = fusion(ts_features, text_features)

        # Check output shape
        assert result.shape == (2, 10, 128)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_forward_pass_mathematical_correctness(self) -> None:
        """Tests that forward pass implements correct mathematical operations."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128)
        text_features = torch.randn(2, 10, 64)

        # Manual computation for verification
        projected_text = fusion.text_projection(text_features)
        activated_text = fusion.activation(projected_text)
        expected_result = ts_features + activated_text

        # Forward pass result
        actual_result = fusion(ts_features, text_features)

        # Should be identical
        assert torch.allclose(actual_result, expected_result, atol=1e-6)

    def test_forward_pass_different_batch_sizes(self) -> None:
        """Tests forward pass with different batch sizes."""
        fusion = MultimodalFusion(ts_feature_dim=64, text_feature_dim=32)

        for batch_size in [1, 5, 16]:
            ts_features = torch.randn(batch_size, 8, 64)
            text_features = torch.randn(batch_size, 8, 32)

            result = fusion(ts_features, text_features)
            assert result.shape == (batch_size, 8, 64)

    def test_forward_pass_different_sequence_lengths(self) -> None:
        """Tests forward pass with different sequence lengths."""
        fusion = MultimodalFusion(ts_feature_dim=64, text_feature_dim=32)

        for seq_len in [1, 10, 50]:
            ts_features = torch.randn(3, seq_len, 64)
            text_features = torch.randn(3, seq_len, 32)

            result = fusion(ts_features, text_features)
            assert result.shape == (3, seq_len, 64)

    def test_forward_pass_gradient_flow(self) -> None:
        """Tests that gradients flow through the fusion layer."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128, requires_grad=True)
        text_features = torch.randn(2, 10, 64, requires_grad=True)

        result = fusion(ts_features, text_features)
        loss = result.sum()
        loss.backward()

        # Check that gradients are computed
        assert ts_features.grad is not None
        assert text_features.grad is not None
        assert not torch.allclose(ts_features.grad, torch.zeros_like(ts_features.grad))
        assert not torch.allclose(text_features.grad, torch.zeros_like(text_features.grad))

    def test_get_projection_parameters(self) -> None:
        """Tests getting projection layer parameters."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        params = fusion.get_projection_parameters()

        assert "weight" in params
        assert "bias" in params
        assert params["weight"].shape == (128, 64)
        assert params["bias"].shape == (128,)
        assert isinstance(params["weight"], torch.Tensor)
        assert isinstance(params["bias"], torch.Tensor)

        # Should be copies, not references
        assert params["weight"] is not fusion.text_projection.weight
        assert params["bias"] is not fusion.text_projection.bias

    def test_set_projection_parameters_valid(self) -> None:
        """Tests setting projection layer parameters with valid inputs."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Create new parameters
        new_weight = torch.randn(128, 64)
        new_bias = torch.randn(128)
        new_params = {"weight": new_weight, "bias": new_bias}

        # Set parameters
        fusion.set_projection_parameters(new_params)

        # Check that parameters were set correctly
        assert torch.allclose(fusion.text_projection.weight, new_weight)
        assert torch.allclose(fusion.text_projection.bias, new_bias)

    def test_set_projection_parameters_missing_keys(self) -> None:
        """Tests setting projection layer parameters fails with missing keys."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        with pytest.raises(KeyError, match="Missing 'weight' parameter"):
            fusion.set_projection_parameters({"bias": torch.randn(128)})

        with pytest.raises(KeyError, match="Missing 'bias' parameter"):
            fusion.set_projection_parameters({"weight": torch.randn(128, 64)})

    def test_set_projection_parameters_wrong_shapes(self) -> None:
        """Tests setting projection layer parameters fails with wrong shapes."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Wrong weight shape
        with pytest.raises(ValueError, match="Weight shape mismatch"):
            fusion.set_projection_parameters(
                {
                    "weight": torch.randn(64, 128),  # Transposed
                    "bias": torch.randn(128),
                }
            )

        # Wrong bias shape
        with pytest.raises(ValueError, match="Bias shape mismatch"):
            fusion.set_projection_parameters(
                {
                    "weight": torch.randn(128, 64),
                    "bias": torch.randn(64),  # Wrong size
                }
            )

    def test_freeze_unfreeze_projection(self) -> None:
        """Tests freezing and unfreezing projection layer parameters."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Initially parameters should be trainable
        assert not fusion.is_projection_frozen()
        for param in fusion.text_projection.parameters():
            assert param.requires_grad

        # Test freezing
        fusion.freeze_projection()
        assert fusion.is_projection_frozen()
        for param in fusion.text_projection.parameters():
            assert not param.requires_grad

        # Test unfreezing
        fusion.unfreeze_projection()
        assert not fusion.is_projection_frozen()
        for param in fusion.text_projection.parameters():
            assert param.requires_grad

    def test_frozen_parameters_no_gradients(self) -> None:
        """Tests that frozen parameters don't accumulate gradients."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128, requires_grad=True)
        text_features = torch.randn(2, 10, 64, requires_grad=True)

        # Freeze projection layer
        fusion.freeze_projection()

        # Forward and backward pass
        result = fusion(ts_features, text_features)
        loss = result.sum()
        loss.backward()

        # Projection layer parameters should not have gradients
        for param in fusion.text_projection.parameters():
            assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))

        # Input features should still have gradients
        assert ts_features.grad is not None
        assert text_features.grad is not None

    def test_parameter_state_preservation_after_freeze_unfreeze(self) -> None:
        """Tests that parameter values are preserved during freeze/unfreeze operations."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Store original parameters
        original_weight = fusion.text_projection.weight.clone()
        original_bias = fusion.text_projection.bias.clone()

        # Freeze and unfreeze
        fusion.freeze_projection()
        fusion.unfreeze_projection()

        # Parameters should be unchanged
        assert torch.allclose(fusion.text_projection.weight, original_weight)
        assert torch.allclose(fusion.text_projection.bias, original_bias)

    def test_compute_fusion_loss_default_mse(self) -> None:
        """Tests fusion loss computation with default MSE loss."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128)
        text_features = torch.randn(2, 10, 64)
        target = torch.randn(2, 10, 128)

        loss = fusion.compute_fusion_loss(ts_features, text_features, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # MSE loss is non-negative

    def test_compute_fusion_loss_custom_loss_function(self) -> None:
        """Tests fusion loss computation with custom loss function."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128)
        text_features = torch.randn(2, 10, 64)
        target = torch.randn(2, 10, 128)

        # Use L1 loss instead of MSE
        l1_loss = torch.nn.L1Loss()
        loss = fusion.compute_fusion_loss(ts_features, text_features, target, l1_loss)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # L1 loss is non-negative

    def test_compute_fusion_loss_mathematical_correctness(self) -> None:
        """Tests that fusion loss computation is mathematically correct."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128)
        text_features = torch.randn(2, 10, 64)
        target = torch.randn(2, 10, 128)

        # Manual computation
        fused_features = fusion(ts_features, text_features)
        mse_loss = torch.nn.MSELoss()
        expected_loss = mse_loss(fused_features, target)

        # Loss from method
        actual_loss = fusion.compute_fusion_loss(ts_features, text_features, target)

        # Should be identical
        assert torch.allclose(actual_loss, expected_loss, atol=1e-6)

    def test_compute_fusion_loss_gradient_flow(self) -> None:
        """Tests that gradients flow through fusion loss computation."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128, requires_grad=True)
        text_features = torch.randn(2, 10, 64, requires_grad=True)
        target = torch.randn(2, 10, 128)

        loss = fusion.compute_fusion_loss(ts_features, text_features, target)
        loss.backward()  # type: ignore[no-untyped-call]

        # Check that gradients are computed
        assert ts_features.grad is not None
        assert text_features.grad is not None
        assert not torch.allclose(ts_features.grad, torch.zeros_like(ts_features.grad))
        assert not torch.allclose(text_features.grad, torch.zeros_like(text_features.grad))

    def test_edge_case_zero_features(self) -> None:
        """Tests fusion with zero features."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.zeros(2, 10, 128)
        text_features = torch.zeros(2, 10, 64)

        result = fusion(ts_features, text_features)

        assert result.shape == (2, 10, 128)
        assert isinstance(result, torch.Tensor)
        # Result should be zero (relu(0) + 0 = 0)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_edge_case_single_sample(self) -> None:
        """Tests fusion with single sample batch."""
        fusion = MultimodalFusion(ts_feature_dim=64, text_feature_dim=32)

        ts_features = torch.randn(1, 1, 64)
        text_features = torch.randn(1, 1, 32)

        result = fusion(ts_features, text_features)

        assert result.shape == (1, 1, 64)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_edge_case_large_sequence_length(self) -> None:
        """Tests fusion with large sequence length."""
        fusion = MultimodalFusion(ts_feature_dim=32, text_feature_dim=16)

        ts_features = torch.randn(1, 1000, 32)
        text_features = torch.randn(1, 1000, 16)

        result = fusion(ts_features, text_features)

        assert result.shape == (1, 1000, 32)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_device_consistency_cpu(self) -> None:
        """Tests device consistency on CPU."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 10, 128, device="cpu")
        text_features = torch.randn(2, 10, 64, device="cpu")

        result = fusion(ts_features, text_features)

        assert result.device.type == "cpu"

    def test_device_consistency_same_device_different_from_model(self) -> None:
        """Tests that output follows input device even if model is on different device."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Inputs on CPU
        ts_features = torch.randn(2, 10, 128, device="cpu")
        text_features = torch.randn(2, 10, 64, device="cpu")

        result = fusion(ts_features, text_features)

        # Output should be on same device as inputs
        assert result.device.type == "cpu"

    def test_numerical_stability_extreme_values(self) -> None:
        """Tests numerical stability with extreme input values."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Very large values
        ts_features = torch.full((2, 10, 128), 1e6)
        text_features = torch.full((2, 10, 64), 1e6)

        result = fusion(ts_features, text_features)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # Very small values
        ts_features = torch.full((2, 10, 128), 1e-6)
        text_features = torch.full((2, 10, 64), 1e-6)

        result = fusion(ts_features, text_features)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_memory_efficiency_large_batch(self) -> None:
        """Tests memory efficiency with large batch size."""
        fusion = MultimodalFusion(ts_feature_dim=64, text_feature_dim=32)

        # Large batch size
        batch_size = 128
        ts_features = torch.randn(batch_size, 10, 64)
        text_features = torch.randn(batch_size, 10, 32)

        result = fusion(ts_features, text_features)

        assert result.shape == (batch_size, 10, 64)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()
