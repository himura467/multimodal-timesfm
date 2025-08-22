"""Text encoding components for multimodal TimesFM."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    """Text encoder using sentence transformers for generating text embeddings.

    This component converts text descriptions into dense vector representations
    that can be fused with time series features in the multimodal TimesFM model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384, device: str | None = None) -> None:
        """Initialize the text encoder.

        Args:
            model_name: Name of the sentence transformer model to use.
            embedding_dim: Dimension of the output embeddings.
            device: Device to use for computations. If None, uses CUDA if available, then MPS, then CPU.
        """
        super().__init__()
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.sentence_transformer = SentenceTransformer(model_name, device=device)
        self.embedding_dim = embedding_dim

        # Get the actual embedding dimension from the model
        actual_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        if actual_dim is None:
            raise ValueError("Could not determine embedding dimension from sentence transformer")

        # Add a projection layer if needed to match desired embedding_dim
        if actual_dim != embedding_dim:
            self.projection: nn.Module = nn.Linear(actual_dim, embedding_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode text inputs into embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            Tensor of shape (batch_size, embedding_dim) containing text embeddings.
        """
        # Generate embeddings using sentence transformer
        embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=True)

        # Apply projection if needed
        projected_embeddings = self.projection(embeddings)

        return torch.as_tensor(projected_embeddings)


class MultimodalFusion(nn.Module):
    """Addition-based fusion mechanism for combining time series and text features.

    This module implements a temporally-aware fusion strategy where text features
    with temporal dimension are projected to match time series feature dimensions,
    then added element-wise. The projection layer is designed to be trainable within
    TimesFM's loss function.

    Architecture:
        text_features(batch, seq_len, text_dim) -> Linear(text_dim -> ts_dim) -> ReLU -> add with ts_features

    Args:
        ts_feature_dim: Dimension of time series features.
        text_feature_dim: Dimension of text features.

    Example:
        >>> fusion = MultimodalFusion(ts_feature_dim=1280, text_feature_dim=384)
        >>> ts_features = torch.randn(2, 32, 1280)  # (batch, seq_len, ts_dim)
        >>> text_features = torch.randn(2, 32, 384)     # (batch, seq_len, text_dim)
        >>> fused = fusion(ts_features, text_features)
        >>> print(fused.shape)  # torch.Size([2, 32, 1280])
    """

    def __init__(self, ts_feature_dim: int, text_feature_dim: int) -> None:
        """Initialize the addition-based fusion module.

        Args:
            ts_feature_dim: Dimension of time series features.
            text_feature_dim: Dimension of text features.

        Raises:
            ValueError: If feature dimensions are not positive integers.
        """
        super().__init__()

        # Validate input dimensions
        if ts_feature_dim <= 0:
            raise ValueError(f"ts_feature_dim must be a positive integer, got {ts_feature_dim}")
        if text_feature_dim <= 0:
            raise ValueError(f"text_feature_dim must be a positive integer, got {text_feature_dim}")

        self.ts_feature_dim = ts_feature_dim
        self.text_feature_dim = text_feature_dim

        # Projection layer: text_dim -> ts_dim
        self.text_projection = nn.Linear(text_feature_dim, ts_feature_dim)

        # ReLU activation
        self.activation = nn.ReLU()

        # Initialize projection weights with Xavier uniform initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize projection layer weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)

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
        # Validate input shapes
        self._validate_inputs(ts_features, text_features)

        batch_size, seq_len, ts_dim = ts_features.shape
        text_batch_size, text_seq_len, text_dim = text_features.shape

        # Ensure batch sizes and sequence lengths match
        if batch_size != text_batch_size:
            raise ValueError(f"Batch size mismatch: ts_features has {batch_size}, text_features has {text_batch_size}")
        if seq_len != text_seq_len:
            raise ValueError(f"Sequence length mismatch: ts_features has {seq_len}, text_features has {text_seq_len}")

        # Ensure feature dimensions match expected
        if ts_dim != self.ts_feature_dim:
            raise ValueError(f"Time series feature dimension mismatch: expected {self.ts_feature_dim}, got {ts_dim}")
        if text_dim != self.text_feature_dim:
            raise ValueError(f"Text feature dimension mismatch: expected {self.text_feature_dim}, got {text_dim}")

        # Ensure tensors are on the same device
        if ts_features.device != text_features.device:
            raise RuntimeError(
                f"Device mismatch: ts_features on {ts_features.device}, text_features on {text_features.device}"
            )

        # Project text features to time series dimension: (batch_size, seq_len, text_dim) -> (batch_size, seq_len, ts_dim)
        projected_text = self.text_projection(text_features)

        # Apply ReLU activation
        projected_text = self.activation(projected_text)

        # Add time series and text features element-wise
        fused_features = ts_features + projected_text

        return torch.as_tensor(fused_features)

    def _validate_inputs(self, ts_features: torch.Tensor, text_features: torch.Tensor) -> None:
        """Validate input tensor shapes and types.

        Args:
            ts_features: Time series features tensor.
            text_features: Text features tensor.

        Raises:
            ValueError: If input tensors have incorrect shapes or types.
        """
        if not isinstance(ts_features, torch.Tensor):
            raise ValueError(f"ts_features must be a torch.Tensor, got {type(ts_features)}")
        if not isinstance(text_features, torch.Tensor):
            raise ValueError(f"text_features must be a torch.Tensor, got {type(text_features)}")

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

    def get_projection_parameters(self) -> dict[str, torch.Tensor]:
        """Get projection layer parameters for TimesFM integration.

        Returns:
            Dictionary containing 'weight' and 'bias' parameters of the projection layer.
        """
        return {"weight": self.text_projection.weight.clone(), "bias": self.text_projection.bias.clone()}

    def set_projection_parameters(self, parameters: dict[str, torch.Tensor]) -> None:
        """Set projection layer parameters for TimesFM integration.

        Args:
            parameters: Dictionary containing 'weight' and 'bias' tensors.

        Raises:
            ValueError: If parameters don't match expected shapes.
            KeyError: If required parameter keys are missing.
        """
        if "weight" not in parameters:
            raise KeyError("Missing 'weight' parameter")
        if "bias" not in parameters:
            raise KeyError("Missing 'bias' parameter")

        weight = parameters["weight"]
        bias = parameters["bias"]

        # Validate parameter shapes
        expected_weight_shape = (self.ts_feature_dim, self.text_feature_dim)
        expected_bias_shape = (self.ts_feature_dim,)

        if weight.shape != expected_weight_shape:
            raise ValueError(f"Weight shape mismatch: expected {expected_weight_shape}, got {weight.shape}")
        if bias.shape != expected_bias_shape:
            raise ValueError(f"Bias shape mismatch: expected {expected_bias_shape}, got {bias.shape}")

        # Set parameters
        with torch.no_grad():
            self.text_projection.weight.copy_(weight)
            self.text_projection.bias.copy_(bias)

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

    def compute_fusion_loss(
        self,
        ts_features: torch.Tensor,
        text_features: torch.Tensor,
        target: torch.Tensor,
        loss_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """Compute fusion loss for TimesFM integration.

        Args:
            ts_features: Time series features.
            text_features: Text features.
            target: Target tensor for loss computation.
            loss_fn: Loss function to use. If None, uses MSE loss.

        Returns:
            Computed loss tensor.
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        fused_features = self(ts_features, text_features)
        loss_result = loss_fn(fused_features, target)
        return torch.as_tensor(loss_result)
