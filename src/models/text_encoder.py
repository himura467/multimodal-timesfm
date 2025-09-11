"""Text encoder component for multimodal TimesFM."""

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.utils.device import resolve_device


class TextEncoder(nn.Module):
    """Text encoder using sentence transformers for generating text embeddings.

    This component converts text descriptions into dense vector representations
    that can be fused with time series features in the multimodal TimesFM model.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384, device: torch.device | str | None = None
    ) -> None:
        """Initialize the text encoder.

        Args:
            model_name: Name of the sentence transformer model to use.
            embedding_dim: Dimension of the output embeddings.
            device: Device to use for computations. Can be str, torch.device, or None for auto-detection.
        """
        super().__init__()
        self.device = resolve_device(device)
        self.sentence_transformer = SentenceTransformer(model_name, device=self.device.type)
        self.embedding_dim = embedding_dim

        # Get the actual embedding dimension from the model
        actual_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        if actual_dim is None:
            raise ValueError("Could not determine embedding dimension from sentence transformer")

        # Require exact dimension match - raise error if different
        if actual_dim != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: model produces {actual_dim}-dimensional embeddings, "
                f"but {embedding_dim} was requested. Please use embedding_dim={actual_dim}."
            )

    def forward(self, texts: str | list[str] | np.ndarray) -> torch.Tensor:
        """Encode text inputs into embeddings.

        Args:
            texts: Text input(s) to encode. Can be:
                - Single string
                - List of strings
                - NumPy array of strings

        Returns:
            Tensor containing text embeddings:
            - For single string: shape (embedding_dim,)
            - For multiple strings: shape (num_inputs, embedding_dim)
        """
        # Generate embeddings using sentence transformer
        embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=True)

        return embeddings.clone()

    def freeze_parameters(self) -> None:
        """Freeze all parameters of the sentence transformer for selective training."""
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters of the sentence transformer for training."""
        for param in self.sentence_transformer.parameters():
            param.requires_grad = True

    def is_frozen(self) -> bool:
        """Check if sentence transformer parameters are frozen.

        Returns:
            True if all parameters are frozen, False otherwise.
        """
        return all(not param.requires_grad for param in self.sentence_transformer.parameters())
