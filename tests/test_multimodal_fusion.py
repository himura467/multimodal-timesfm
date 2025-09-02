"""Tests for MultimodalFusion class."""

import pytest
import torch
import torch.nn as nn

from src.models.multimodal_fusion import MultimodalFusion


class TestMultimodalFusion:
    """Test cases for MultimodalFusion class."""

    def test_init_valid_parameters(self) -> None:
        """Tests initialization with valid parameters."""
        fusion = MultimodalFusion(ts_feature_dim=1280, text_feature_dim=384)

        assert fusion.ts_feature_dim == 1280
        assert fusion.text_feature_dim == 384
        assert isinstance(fusion.text_projection, nn.Linear)
        assert isinstance(fusion.activation, nn.ReLU)
        assert fusion.text_projection.in_features == 384
        assert fusion.text_projection.out_features == 1280

    def test_init_invalid_ts_dimension(self) -> None:
        """Tests initialization with invalid time series dimension."""
        with pytest.raises(ValueError, match="ts_feature_dim must be a positive integer, got 0"):
            MultimodalFusion(ts_feature_dim=0, text_feature_dim=384)

        with pytest.raises(ValueError, match="ts_feature_dim must be a positive integer, got -1"):
            MultimodalFusion(ts_feature_dim=-1, text_feature_dim=384)

    def test_init_invalid_text_dimension(self) -> None:
        """Tests initialization with invalid text dimension."""
        with pytest.raises(ValueError, match="text_feature_dim must be a positive integer, got 0"):
            MultimodalFusion(ts_feature_dim=1280, text_feature_dim=0)

        with pytest.raises(ValueError, match="text_feature_dim must be a positive integer, got -5"):
            MultimodalFusion(ts_feature_dim=1280, text_feature_dim=-5)

    def test_weight_initialization(self) -> None:
        """Tests that weights are properly initialized."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Check that weights are initialized (not zeros)
        assert not torch.allclose(fusion.text_projection.weight, torch.zeros_like(fusion.text_projection.weight))

        # Check that bias is initialized to zeros
        assert torch.allclose(fusion.text_projection.bias, torch.zeros_like(fusion.text_projection.bias))

    def test_forward_valid_inputs(self) -> None:
        """Tests forward pass with valid inputs."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 32, 128)  # (batch, seq_len, ts_dim)
        text_features = torch.randn(2, 32, 64)  # (batch, seq_len, text_dim)

        result = fusion(ts_features, text_features)

        assert result.shape == (2, 32, 128)
        assert result.dtype == ts_features.dtype
        assert result.device == ts_features.device

    def test_forward_invalid_ts_tensor_type(self) -> None:
        """Tests forward pass with invalid time series tensor type."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = [[1, 2, 3]]  # Invalid type
        text_features = torch.randn(2, 32, 64)

        with pytest.raises(ValueError, match="ts_features must be a torch.Tensor"):
            fusion(ts_features, text_features)

    def test_forward_invalid_text_tensor_type(self) -> None:
        """Tests forward pass with invalid text tensor type."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 32, 128)
        text_features = None  # Invalid type

        with pytest.raises(ValueError, match="text_features must be a torch.Tensor"):
            fusion(ts_features, text_features)

    def test_forward_invalid_ts_dimensions(self) -> None:
        """Tests forward pass with invalid time series tensor dimensions."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # 2D tensor instead of 3D
        ts_features = torch.randn(32, 128)
        text_features = torch.randn(2, 32, 64)

        with pytest.raises(ValueError, match="ts_features must be 3D.*got 2D"):
            fusion(ts_features, text_features)

    def test_forward_invalid_text_dimensions(self) -> None:
        """Tests forward pass with invalid text tensor dimensions."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 32, 128)
        # 4D tensor instead of 3D
        text_features = torch.randn(2, 32, 64, 1)

        with pytest.raises(ValueError, match="text_features must be 3D.*got 4D"):
            fusion(ts_features, text_features)

    def test_forward_batch_size_mismatch(self) -> None:
        """Tests forward pass with mismatched batch sizes."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 32, 128)
        text_features = torch.randn(3, 32, 64)  # Different batch size

        with pytest.raises(ValueError, match="Batch size mismatch: ts_features has 2, text_features has 3"):
            fusion(ts_features, text_features)

    def test_forward_sequence_length_mismatch(self) -> None:
        """Tests forward pass with mismatched sequence lengths."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 32, 128)
        text_features = torch.randn(2, 16, 64)  # Different sequence length

        with pytest.raises(ValueError, match="Sequence length mismatch: ts_features has 32, text_features has 16"):
            fusion(ts_features, text_features)

    def test_forward_ts_feature_dimension_mismatch(self) -> None:
        """Tests forward pass with mismatched time series feature dimensions."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 32, 256)  # Wrong feature dimension
        text_features = torch.randn(2, 32, 64)

        with pytest.raises(ValueError, match="Time series feature dimension mismatch: expected 128, got 256"):
            fusion(ts_features, text_features)

    def test_forward_text_feature_dimension_mismatch(self) -> None:
        """Tests forward pass with mismatched text feature dimensions."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 32, 128)
        text_features = torch.randn(2, 32, 128)  # Wrong feature dimension

        with pytest.raises(ValueError, match="Text feature dimension mismatch: expected 64, got 128"):
            fusion(ts_features, text_features)

    def test_forward_device_mismatch(self) -> None:
        """Tests forward pass with mismatched devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")

        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        ts_features = torch.randn(2, 32, 128)  # CPU tensor
        text_features = torch.randn(2, 32, 64).cuda()  # GPU tensor

        with pytest.raises(RuntimeError, match="Device mismatch"):
            fusion(ts_features, text_features)

    def test_fusion_correctness(self) -> None:
        """Tests that fusion produces mathematically correct results."""
        fusion = MultimodalFusion(ts_feature_dim=4, text_feature_dim=2)

        # Set known weights for predictable output
        with torch.no_grad():
            fusion.text_projection.weight.fill_(1.0)  # All weights = 1
            fusion.text_projection.bias.fill_(0.0)  # All biases = 0

        ts_features = torch.ones(1, 1, 4)  # [1, 1, 1, 1]
        text_features = torch.ones(1, 1, 2)  # [1, 1]

        result = fusion(ts_features, text_features)

        # Expected: projection gives [2, 2, 2, 2], ReLU keeps it, add with ts gives [3, 3, 3, 3]
        expected = torch.tensor([[[3.0, 3.0, 3.0, 3.0]]])
        assert torch.allclose(result, expected)

    def test_activation_function(self) -> None:
        """Tests that ReLU activation is applied correctly."""
        fusion = MultimodalFusion(ts_feature_dim=2, text_feature_dim=2)

        # Set weights to produce negative values
        with torch.no_grad():
            fusion.text_projection.weight.fill_(-1.0)
            fusion.text_projection.bias.fill_(0.0)

        ts_features = torch.zeros(1, 1, 2)
        text_features = torch.ones(1, 1, 2)  # Will become [-2, -2] after projection

        result = fusion(ts_features, text_features)

        # ReLU should clamp negative values to 0
        # ts_features (0, 0) + ReLU([-2, -2]) = (0, 0) + (0, 0) = (0, 0)
        expected = torch.zeros(1, 1, 2)
        assert torch.allclose(result, expected)

    def test_get_projection_parameters(self) -> None:
        """Tests getting projection parameters."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        params = fusion.get_projection_parameters()

        assert "weight" in params
        assert "bias" in params
        assert params["weight"].shape == (128, 64)
        assert params["bias"].shape == (128,)

        # Should be copies, not references
        params["weight"].fill_(999.0)
        assert not torch.allclose(fusion.text_projection.weight, params["weight"])

    def test_set_projection_parameters_valid(self) -> None:
        """Tests setting projection parameters with valid inputs."""
        fusion = MultimodalFusion(ts_feature_dim=4, text_feature_dim=2)

        new_weight = torch.ones(4, 2) * 2.0
        new_bias = torch.ones(4) * 0.5

        fusion.set_projection_parameters({"weight": new_weight, "bias": new_bias})

        assert torch.allclose(fusion.text_projection.weight, new_weight)
        assert torch.allclose(fusion.text_projection.bias, new_bias)

    def test_set_projection_parameters_missing_weight(self) -> None:
        """Tests setting projection parameters with missing weight."""
        fusion = MultimodalFusion(ts_feature_dim=4, text_feature_dim=2)

        with pytest.raises(KeyError, match="Missing 'weight' parameter"):
            fusion.set_projection_parameters({"bias": torch.ones(4)})

    def test_set_projection_parameters_missing_bias(self) -> None:
        """Tests setting projection parameters with missing bias."""
        fusion = MultimodalFusion(ts_feature_dim=4, text_feature_dim=2)

        with pytest.raises(KeyError, match="Missing 'bias' parameter"):
            fusion.set_projection_parameters({"weight": torch.ones(4, 2)})

    def test_set_projection_parameters_wrong_weight_shape(self) -> None:
        """Tests setting projection parameters with wrong weight shape."""
        fusion = MultimodalFusion(ts_feature_dim=4, text_feature_dim=2)

        wrong_weight = torch.ones(2, 4)  # Transposed shape
        bias = torch.ones(4)

        with pytest.raises(ValueError, match=r"Weight shape mismatch: expected \(4, 2\), got torch\.Size\(\[2, 4\]\)"):
            fusion.set_projection_parameters({"weight": wrong_weight, "bias": bias})

    def test_set_projection_parameters_wrong_bias_shape(self) -> None:
        """Tests setting projection parameters with wrong bias shape."""
        fusion = MultimodalFusion(ts_feature_dim=4, text_feature_dim=2)

        weight = torch.ones(4, 2)
        wrong_bias = torch.ones(2)  # Wrong size

        with pytest.raises(ValueError, match=r"Bias shape mismatch: expected \(4,\), got torch\.Size\(\[2\]\)"):
            fusion.set_projection_parameters({"weight": weight, "bias": wrong_bias})

    def test_freeze_projection(self) -> None:
        """Tests freezing projection parameters."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Initially unfrozen
        assert not fusion.is_projection_frozen()
        assert fusion.text_projection.weight.requires_grad
        assert fusion.text_projection.bias.requires_grad

        # Freeze
        fusion.freeze_projection()

        assert fusion.is_projection_frozen()
        assert not fusion.text_projection.weight.requires_grad
        assert not fusion.text_projection.bias.requires_grad

    def test_unfreeze_projection(self) -> None:
        """Tests unfreezing projection parameters."""
        fusion = MultimodalFusion(ts_feature_dim=128, text_feature_dim=64)

        # Freeze first
        fusion.freeze_projection()
        assert fusion.is_projection_frozen()

        # Unfreeze
        fusion.unfreeze_projection()

        assert not fusion.is_projection_frozen()
        assert fusion.text_projection.weight.requires_grad
        assert fusion.text_projection.bias.requires_grad

    def test_gradient_flow(self) -> None:
        """Tests that gradients flow correctly through the fusion module."""
        fusion = MultimodalFusion(ts_feature_dim=4, text_feature_dim=2)

        ts_features = torch.randn(2, 3, 4, requires_grad=True)
        text_features = torch.randn(2, 3, 2, requires_grad=True)

        result = fusion(ts_features, text_features)
        loss = result.sum()
        loss.backward()

        # Check that gradients exist
        assert ts_features.grad is not None
        assert text_features.grad is not None
        assert fusion.text_projection.weight.grad is not None
        assert fusion.text_projection.bias.grad is not None

    def test_no_gradient_when_frozen(self) -> None:
        """Tests that no gradients flow when projection is frozen."""
        fusion = MultimodalFusion(ts_feature_dim=4, text_feature_dim=2)
        fusion.freeze_projection()

        ts_features = torch.randn(2, 3, 4, requires_grad=True)
        text_features = torch.randn(2, 3, 2, requires_grad=True)

        result = fusion(ts_features, text_features)
        loss = result.sum()
        loss.backward()

        # Check that fusion parameters don't have gradients
        assert fusion.text_projection.weight.grad is None
        assert fusion.text_projection.bias.grad is None

        # But input gradients should still exist
        assert ts_features.grad is not None
        assert text_features.grad is not None

    def test_different_dtypes(self) -> None:
        """Tests fusion with different data types."""
        fusion = MultimodalFusion(ts_feature_dim=4, text_feature_dim=2)

        # Convert model to float64 to match input dtype
        fusion = fusion.double()

        # Test with float64
        ts_features = torch.randn(1, 2, 4, dtype=torch.float64)
        text_features = torch.randn(1, 2, 2, dtype=torch.float64)

        result = fusion(ts_features, text_features)

        assert result.dtype == torch.float64
        assert result.shape == (1, 2, 4)

    def test_reproducibility(self) -> None:
        """Tests that results are reproducible with same inputs."""
        torch.manual_seed(42)
        fusion1 = MultimodalFusion(ts_feature_dim=8, text_feature_dim=4)

        torch.manual_seed(42)
        fusion2 = MultimodalFusion(ts_feature_dim=8, text_feature_dim=4)

        ts_features = torch.randn(2, 3, 8)
        text_features = torch.randn(2, 3, 4)

        result1 = fusion1(ts_features, text_features)
        result2 = fusion2(ts_features, text_features)

        assert torch.allclose(result1, result2)
