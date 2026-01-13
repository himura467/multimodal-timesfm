"""Multimodal fusion mechanisms for combining time series and text features."""

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """Addition-based fusion mechanism for combining time series and text features.

    This module implements a temporally-aware fusion strategy where text features
    with temporal dimension are projected to match time series feature dimensions,
    then added element-wise. The projection can be a single-layer or multi-layer
    network, designed to be trainable within TimesFM's loss function.

    Architecture (num_layers=1):
        text_features(batch, seq_len, text_dim) -> Linear(text_dim -> ts_dim) -> ReLU -> add with ts_features

    Architecture (num_layers=2):
        text_features -> Linear(text_dim -> hidden) -> ReLU -> Linear(hidden -> ts_dim) -> ReLU -> add with ts_features

    Architecture (num_layers=3):
        text_features -> Linear(text_dim -> hidden) -> ReLU -> Linear(hidden -> hidden) -> ReLU -> Linear(hidden -> ts_dim) -> ReLU -> add with ts_features

    Args:
        ts_feature_dim: Dimension of time series features.
        text_feature_dim: Dimension of text features.
        num_layers: Number of linear layers in the projection network (1-3). Defaults to 1.
        use_bias: Whether to use bias in the projection layers. Defaults to True.

    Example:
        >>> # Single-layer projection
        >>> fusion = MultimodalFusion(ts_feature_dim=1280, text_feature_dim=384, num_layers=1)
        >>> ts_features = torch.randn(2, 32, 1280)  # (batch, seq_len, ts_dim)
        >>> text_features = torch.randn(2, 32, 384)     # (batch, seq_len, text_dim)
        >>> fused = fusion(ts_features, text_features)
        >>> print(fused.shape)  # torch.Size([2, 32, 1280])

        >>> # Multi-layer projection
        >>> fusion = MultimodalFusion(ts_feature_dim=1280, text_feature_dim=384, num_layers=2)
        >>> fused = fusion(ts_features, text_features)
        >>> print(fused.shape)  # torch.Size([2, 32, 1280])
    """

    def __init__(self, ts_feature_dim: int, text_feature_dim: int, num_layers: int = 1, use_bias: bool = True) -> None:
        """Initialize the addition-based fusion module.

        Args:
            ts_feature_dim: Dimension of time series features.
            text_feature_dim: Dimension of text features.
            num_layers: Number of linear layers in the projection network (1-3). Defaults to 1.
            use_bias: Whether to use bias in the projection layers. Defaults to True.

        Raises:
            ValueError: If feature dimensions are not positive integers or num_layers is not in [1, 3].
        """
        super().__init__()

        # Validate input dimensions
        if ts_feature_dim <= 0:
            raise ValueError(f"ts_feature_dim must be a positive integer, got {ts_feature_dim}")
        if text_feature_dim <= 0:
            raise ValueError(f"text_feature_dim must be a positive integer, got {text_feature_dim}")
        if num_layers < 1 or num_layers > 3:
            raise ValueError(f"num_layers must be between 1 and 3, got {num_layers}")

        self.ts_feature_dim = ts_feature_dim
        self.text_feature_dim = text_feature_dim
        self.num_layers = num_layers
        self.use_bias = use_bias

        # Build multi-layer projection network
        layers: list[nn.Module] = []

        if num_layers == 1:
            # Single-layer projection: text_dim -> ts_dim
            layers.append(nn.Linear(in_features=text_feature_dim, out_features=ts_feature_dim, bias=use_bias))
            layers.append(nn.ReLU())
        else:
            # Multi-layer projection with hidden dimension
            # Use the average of input and output dimensions as hidden dimension
            hidden_dim = (text_feature_dim + ts_feature_dim) // 2

            # First layer: text_dim -> hidden_dim
            layers.append(nn.Linear(in_features=text_feature_dim, out_features=hidden_dim, bias=use_bias))
            layers.append(nn.ReLU())

            # Middle layers (if num_layers == 3): hidden_dim -> hidden_dim
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=use_bias))
                layers.append(nn.ReLU())

            # Final layer: hidden_dim -> ts_dim
            layers.append(nn.Linear(in_features=hidden_dim, out_features=ts_feature_dim, bias=use_bias))
            layers.append(nn.ReLU())

        # Create sequential projection network
        self.text_projection = nn.Sequential(*layers)

        # Initialize projection weights with Xavier uniform initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize projection layer weights using Xavier uniform initialization."""
        for module in self.text_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if self.use_bias and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(self, ts_features: torch.Tensor, text_features: torch.Tensor) -> None:
        """Validate input tensor shapes, types, and compatibility.

        Args:
            ts_features: Time series features tensor.
            text_features: Text features tensor.

        Raises:
            ValueError: If input tensors have incorrect shapes, types, or incompatible dimensions.
            RuntimeError: If input tensors are not on the same device.
        """
        # Validate tensor types
        if not isinstance(ts_features, torch.Tensor):
            raise ValueError(f"ts_features must be a torch.Tensor, got {type(ts_features)}")
        if not isinstance(text_features, torch.Tensor):
            raise ValueError(f"text_features must be a torch.Tensor, got {type(text_features)}")

        # Validate tensor dimensionality
        if ts_features.dim() != 3:
            raise ValueError(
                f"ts_features must be 3D (batch_size, seq_len, feature_dim), "
                f"got {ts_features.dim()}D with shape {ts_features.shape}"
            )
        if text_features.dim() != 3:
            raise ValueError(
                f"text_features must be 3D (batch_size, seq_len, feature_dim), "
                f"got {text_features.dim()}D with shape {text_features.shape}"
            )

        batch_size, seq_len, ts_dim = ts_features.shape
        text_batch_size, text_seq_len, text_dim = text_features.shape

        # Validate batch sizes and sequence lengths match
        if batch_size != text_batch_size:
            raise ValueError(f"Batch size mismatch: ts_features has {batch_size}, text_features has {text_batch_size}")
        if seq_len != text_seq_len:
            raise ValueError(f"Sequence length mismatch: ts_features has {seq_len}, text_features has {text_seq_len}")

        # Validate feature dimensions match expected
        if ts_dim != self.ts_feature_dim:
            raise ValueError(f"Time series feature dimension mismatch: expected {self.ts_feature_dim}, got {ts_dim}")
        if text_dim != self.text_feature_dim:
            raise ValueError(f"Text feature dimension mismatch: expected {self.text_feature_dim}, got {text_dim}")

        # Validate tensors are on the same device
        if ts_features.device != text_features.device:
            raise RuntimeError(
                f"Device mismatch: ts_features on {ts_features.device}, text_features on {text_features.device}"
            )

    def forward(self, ts_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Fuse time series and text features using addition.

        Args:
            ts_features: Time series features of shape (batch_size, seq_len, ts_feature_dim).
            text_features: Text features of shape (batch_size, seq_len, text_feature_dim).

        Returns:
            Fused features of shape (batch_size, seq_len, ts_feature_dim).

        Raises:
            ValueError: If input tensor dimensions don't match expected shapes.
            RuntimeError: If input tensors are not on the same device.
        """
        # Validate input requirements
        self._validate_inputs(ts_features, text_features)

        # Project text features to time series dimension using multi-layer network
        # (batch_size, seq_len, text_dim) -> (batch_size, seq_len, ts_dim)
        projected_text = self.text_projection(text_features)

        # Add time series and text features element-wise
        fused_features = ts_features + projected_text

        return torch.as_tensor(fused_features)

    def get_projection_parameters(self) -> dict[str, torch.Tensor]:
        """Get projection layer parameters for TimesFM integration.

        Returns:
            Dictionary containing parameters of all linear layers in the projection network.
            Keys are in the format 'layer_i_weight' and 'layer_i_bias' where i is the layer index.
        """
        params: dict[str, torch.Tensor] = {}
        layer_idx = 0
        for module in self.text_projection.modules():
            if isinstance(module, nn.Linear):
                params[f"layer_{layer_idx}_weight"] = module.weight.clone()
                if self.use_bias and module.bias is not None:
                    params[f"layer_{layer_idx}_bias"] = module.bias.clone()
                layer_idx += 1
        return params

    def set_projection_parameters(self, parameters: dict[str, torch.Tensor]) -> None:
        """Set projection layer parameters for TimesFM integration.

        Args:
            parameters: Dictionary containing parameters for all linear layers.
                       Keys should be in the format 'layer_i_weight' and 'layer_i_bias'.

        Raises:
            KeyError: If required parameter keys are missing.
            ValueError: If parameters don't match expected shapes.
        """
        layer_idx = 0
        for module in self.text_projection.modules():
            if isinstance(module, nn.Linear):
                weight_key = f"layer_{layer_idx}_weight"
                bias_key = f"layer_{layer_idx}_bias"

                if weight_key not in parameters:
                    raise KeyError(f"Missing '{weight_key}' parameter")

                weight = parameters[weight_key]
                if weight.shape != module.weight.shape:
                    raise ValueError(
                        f"Weight shape mismatch for {weight_key}: expected {module.weight.shape}, got {weight.shape}"
                    )

                with torch.no_grad():
                    module.weight.copy_(weight)

                    if self.use_bias and module.bias is not None:
                        if bias_key not in parameters:
                            raise KeyError(f"Missing '{bias_key}' parameter")

                        bias = parameters[bias_key]
                        if bias.shape != module.bias.shape:
                            raise ValueError(
                                f"Bias shape mismatch for {bias_key}: expected {module.bias.shape}, got {bias.shape}"
                            )

                        module.bias.copy_(bias)

                layer_idx += 1

    def freeze_projection(self) -> None:
        """Freeze projection layer parameters for selective training."""
        for param in self.text_projection.parameters():
            param.requires_grad = False

    def unfreeze_projection(self) -> None:
        """Unfreeze projection layer parameters for training."""
        for param in self.text_projection.parameters():
            param.requires_grad = True

    def is_projection_frozen(self) -> bool:
        """Check if projection layer parameters are frozen.

        Returns:
            True if all projection parameters are frozen, False otherwise.
        """
        return all(not param.requires_grad for param in self.text_projection.parameters())
