"""Text encoding components for multimodal TimesFM."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    """Text encoder using sentence transformers for generating text embeddings.

    This component converts text descriptions into dense vector representations
    that can be fused with time series features in the multimodal TimesFM model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384) -> None:
        """Initialize the text encoder.

        Args:
            model_name: Name of the sentence transformer model to use.
            embedding_dim: Dimension of the output embeddings.
        """
        super().__init__()
        self.sentence_transformer = SentenceTransformer(model_name)
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
        try:
            device_str = str(next(self.parameters()).device)
        except StopIteration:
            device_str = "cpu"
        embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=True, device=device_str)

        # Apply projection if needed
        projected_embeddings = self.projection(embeddings)

        return torch.as_tensor(projected_embeddings)

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.embedding_dim


class MultimodalFusion(nn.Module):
    """Fusion mechanism for combining time series and text features.

    This module implements various strategies for fusing time series features
    with text embeddings, including concatenation and attention-based fusion.
    """

    def __init__(
        self, ts_feature_dim: int, text_feature_dim: int, output_dim: int, fusion_type: str = "concat"
    ) -> None:
        """Initialize the fusion module.

        Args:
            ts_feature_dim: Dimension of time series features.
            text_feature_dim: Dimension of text features.
            output_dim: Dimension of the fused output.
            fusion_type: Type of fusion ('concat', 'attention', 'gated').
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.ts_feature_dim = ts_feature_dim
        self.text_feature_dim = text_feature_dim
        self.output_dim = output_dim

        if fusion_type == "concat":
            self.fusion_layer = nn.Linear(ts_feature_dim + text_feature_dim, output_dim)

        elif fusion_type == "attention":
            self.attention = nn.MultiheadAttention(embed_dim=ts_feature_dim, num_heads=8, batch_first=True)
            self.text_projection = nn.Linear(text_feature_dim, ts_feature_dim)
            self.output_projection = nn.Linear(ts_feature_dim, output_dim)

        elif fusion_type == "gated":
            self.gate = nn.Sequential(nn.Linear(ts_feature_dim * 2, ts_feature_dim), nn.Sigmoid())
            self.fusion_layer = nn.Linear(ts_feature_dim, output_dim)
            self.text_projection = nn.Linear(text_feature_dim, ts_feature_dim)

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, ts_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Fuse time series and text features.

        Args:
            ts_features: Time series features of shape (batch_size, seq_len, ts_feature_dim).
            text_features: Text features of shape (batch_size, text_feature_dim).

        Returns:
            Fused features of shape (batch_size, seq_len, output_dim).
        """
        batch_size, seq_len, _ = ts_features.shape

        if self.fusion_type == "concat":
            # Expand text features to match time series sequence length
            text_expanded = text_features.unsqueeze(1).expand(-1, seq_len, -1)
            # Concatenate along feature dimension
            concatenated = torch.cat([ts_features, text_expanded], dim=-1)
            result = self.fusion_layer(concatenated)
            return torch.as_tensor(result)

        elif self.fusion_type == "attention":
            # Project text features to match time series dimension
            text_projected = self.text_projection(text_features).unsqueeze(1)  # (batch, 1, ts_dim)

            # Use text as query, time series as key and value
            attended_features, _ = self.attention(query=text_projected, key=ts_features, value=ts_features)

            # Expand attended features to sequence length
            attended_expanded = attended_features.expand(-1, seq_len, -1)
            result = self.output_projection(attended_expanded)
            return torch.as_tensor(result)

        elif self.fusion_type == "gated":
            # Project text to time series dimension
            text_projected = self.text_projection(text_features).unsqueeze(1).expand(-1, seq_len, -1)

            # Compute gate weights
            gate_input = torch.cat([ts_features, text_projected], dim=-1)
            gate_weights = self.gate(gate_input)

            # Apply gating
            gated_features = ts_features * gate_weights + text_projected * (1 - gate_weights)
            result = self.fusion_layer(gated_features)
            return torch.as_tensor(result)

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
