"""Text encoder component for multimodal TimesFM."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from multimodal_timesfm.utils.device import resolve_device


class TextEncoderBase(nn.Module, ABC):
    """Abstract base class for text encoders."""

    def __init__(self, embedding_dim: int, device: torch.device | str | None = None) -> None:
        """Initialize the base text encoder.

        Args:
            embedding_dim: Dimension of the output embeddings.
            device: Device to use for computations. Can be str, torch.device, or None for auto-detection.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = resolve_device(device)

    @abstractmethod
    def forward(self, texts: str | list[str] | np.ndarray) -> torch.Tensor:
        """Encode text inputs into embeddings.

        Args:
            texts: Text input(s) to encode.

        Returns:
            Tensor containing text embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def freeze_parameters(self) -> None:
        """Freeze all parameters for selective training."""
        raise NotImplementedError

    @abstractmethod
    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters for training."""
        raise NotImplementedError


class EnglishTextEncoder(TextEncoderBase):
    """Text encoder for English text using SentenceTransformer models."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384, device: torch.device | str | None = None
    ) -> None:
        """Initialize the English text encoder.

        Args:
            model_name: Name of the SentenceTransformer model to use.
            embedding_dim: Dimension of the output embeddings.
            device: Device to use for computations.
        """
        super().__init__(embedding_dim, device)
        self.sentence_transformer = SentenceTransformer(model_name, device=self.device.type)

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


class JapaneseTextEncoder(TextEncoderBase):
    """Text encoder for Japanese text using Ruri models."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-310m",
        embedding_dim: int = 768,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the Japanese text encoder.

        Args:
            model_name: Name of the Ruri model to use.
            embedding_dim: Dimension of the output embeddings.
            device: Device to use for computations.
        """
        super().__init__(embedding_dim, device)
        self.sentence_transformer = SentenceTransformer(model_name, device=self.device.type)

        # Get the actual embedding dimension from the model
        actual_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        if actual_dim is None:
            raise ValueError("Could not determine embedding dimension from Ruri model")

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
        embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=True)
        return embeddings.clone()

    def freeze_parameters(self) -> None:
        """Freeze all parameters of the Japanese text encoder for selective training."""
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters of the Japanese text encoder for training."""
        for param in self.sentence_transformer.parameters():
            param.requires_grad = True
