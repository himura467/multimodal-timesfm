"""Multimodal version of TimesFM's patched decoder that supports text inputs."""

from dataclasses import dataclass
from typing import Literal

import torch
from timesfm.pytorch_patched_decoder import (
    PatchedTimeSeriesDecoder,
    TimesFMConfig,
)

from multimodal_timesfm.multimodal_fusion import MultimodalFusion
from multimodal_timesfm.text_encoder import EnglishTextEncoder, JapaneseTextEncoder, TextEncoderBase
from multimodal_timesfm.utils.device import resolve_device


@dataclass
class MultimodalTimesFMConfig(TimesFMConfig):  # type: ignore[misc]
    """Config for initializing MultimodalPatchedDecoder that extends TimesFMConfig.

    Attributes:
        text_encoder_type: Type of text encoder to use ('english' or 'japanese').
        num_fusion_layers: Number of linear layers in the fusion projection network (1-3). Defaults to 1.
        use_bias: Whether to use bias in the fusion projection layers. Defaults to True.
    """

    text_encoder_type: Literal["english", "japanese"] = "english"
    num_fusion_layers: int = 1
    use_bias: bool = True


class MultimodalPatchedDecoder(PatchedTimeSeriesDecoder):  # type: ignore[misc]
    """Multimodal version of PatchedTimeSeriesDecoder that supports text inputs.

    This decoder extends the original TimesFM patched decoder to handle both time series
    and text data. It uses a text encoder to convert text descriptions into embeddings
    and fuses them with time series features using an addition-based fusion mechanism.

    The decoder maintains all the original functionality of PatchedTimeSeriesDecoder
    while adding multimodal capabilities through:
    1. Text encoding using sentence transformers
    2. Text feature projection and fusion with time series features
    3. Enhanced preprocessing to handle text inputs alongside time series data

    Architecture:
        - Original TimesFM decoder architecture is preserved
        - Text encoder converts text descriptions to embeddings
        - Fusion mechanism combines text and time series features at the input level
        - All transformer layers and output processing remain unchanged
    """

    def __init__(self, config: MultimodalTimesFMConfig, device: torch.device | str | None = None):
        """Initialize MultimodalPatchedDecoder.

        Args:
            config: Multimodal configuration containing both TimesFM and text encoding parameters.
            device: Device to use for the model. If None, will auto-resolve the best available device.
        """
        # Initialize parent class with base TimesFM config
        super().__init__(config)

        self.config = config
        self.device = resolve_device(device)

        # Initialize text encoder based on type
        self.text_encoder: TextEncoderBase
        if config.text_encoder_type == "english":
            self.text_encoder = EnglishTextEncoder(device=self.device)
        elif config.text_encoder_type == "japanese":
            self.text_encoder = JapaneseTextEncoder(device=self.device)
        else:
            raise ValueError(
                f"Unsupported text encoder type: {config.text_encoder_type}. Must be 'english' or 'japanese'."
            )

        self.multimodal_fusion = MultimodalFusion(
            ts_feature_dim=config.hidden_size,
            text_feature_dim=self.text_encoder.embedding_dim,
            num_layers=config.num_fusion_layers,
            use_bias=config.use_bias,
        )

        # Move the entire decoder to the selected device
        self.to(self.device)

    def _preprocess_multimodal_input(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.Tensor,
        text_descriptions: list[list[list[str]]] | None = None,
        text_embeddings: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor,
    ]:
        """Preprocess multimodal input for stacked transformer.

        This method extends the original _preprocess_input to handle text inputs
        by encoding them and fusing with time series features.

        Args:
            input_ts: Input time series tensor of shape (batch_size, sequence_length).
            input_padding: Padding tensor of shape (batch_size, sequence_length).
            text_descriptions: List of text descriptions organized as
                              [batch][patch] where each patch can have multiple text strings.
                              Shape: (batch_size, num_patches, variable_texts_per_patch).
                              If None, text_embeddings must be provided.
            text_embeddings: Pre-computed text embeddings of shape (batch_size, num_patches, embedding_dim).
                            If provided, text_descriptions is ignored.

        Returns:
            Tuple containing:
            - model_input: Preprocessed input tensor for transformer
            - patched_padding: Padding tensor for patches
            - stats: Normalization statistics (mean, std)
            - patched_inputs: Original patched inputs

        Raises:
            ValueError: If neither text_descriptions nor text_embeddings is provided.
        """
        # Use the original preprocessing for time series data
        model_input, patched_padding, stats, patched_inputs = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )

        # Get or compute text embeddings
        if text_embeddings is not None:
            # Use pre-computed embeddings
            embeddings = text_embeddings
        elif text_descriptions is not None:
            # Encode text descriptions for each patch
            embeddings = self._encode_patch_text_features(text_descriptions, model_input.shape, model_input.device)
        else:
            raise ValueError("Either text_descriptions or text_embeddings must be provided")

        # Fuse text features with time series features
        model_input = self.multimodal_fusion(model_input, embeddings)

        return model_input, patched_padding, stats, patched_inputs

    def _encode_patch_text_features(
        self, text_descriptions: list[list[list[str]]], target_shape: torch.Size, device: torch.device
    ) -> torch.Tensor:
        """Encode patch-level text descriptions to match time series patch structure.

        This method processes text descriptions at the patch level, where each patch
        can have multiple associated text strings. All texts for a patch are joined
        and encoded as a single embedding to match the granularity of time series patches.

        Args:
            text_descriptions: Nested list structure [batch][patch][texts] where:
                - batch: Index of the sample in the current batch
                - patch: Index of the time series patch
                - texts: List of text strings associated with that patch
            target_shape: Target tensor shape (batch_size, num_patches, feature_dim)
                         from preprocessed time series data.
            device: PyTorch device to place the resulting text embeddings on.

        Returns:
            Text embedding tensor of shape (batch_size, num_patches, text_embedding_dim)
            where text_embedding_dim matches the text encoder's output dimension.

        Raises:
            ValueError: If batch sizes don't match between text_descriptions and target_shape,
                       or if number of patches per batch doesn't match expected num_patches.
        """
        batch_size, num_patches, _ = target_shape

        # Validate batch size
        if len(text_descriptions) != batch_size:
            raise ValueError(
                f"Batch size mismatch: got {len(text_descriptions)} batch text descriptions for batch size {batch_size}"
            )

        # Validate number of patches for each batch item
        for batch_idx, batch_patches in enumerate(text_descriptions):
            if len(batch_patches) != num_patches:
                raise ValueError(
                    f"Patch number mismatch for batch {batch_idx}: got {len(batch_patches)} patch descriptions "
                    f"for {num_patches} patches"
                )

        # Flatten all text descriptions for batch encoding
        all_texts: list[str] = []
        for batch_patches in text_descriptions:
            for patch_texts in batch_patches:
                # Join multiple texts for each patch with space
                if patch_texts:
                    text = " ".join(patch_texts)
                else:
                    text = ""  # Empty text for patches without descriptions
                all_texts.append(text)

        # Encode all texts at once for efficiency
        if all_texts:
            all_embeddings = self.text_encoder(all_texts)  # Shape: (batch_size * num_patches, text_embedding_dim)
        else:
            # Handle empty case
            all_embeddings = torch.zeros((batch_size * num_patches, self.text_encoder.embedding_dim), device=device)

        # Reshape to (batch_size, num_patches, text_embedding_dim)
        text_embeddings: torch.Tensor = all_embeddings.reshape(batch_size, num_patches, self.text_encoder.embedding_dim)

        return text_embeddings

    def forward(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
        text_descriptions: list[list[list[str]]] | None = None,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for multimodal decoder.

        Args:
            input_ts: Input time series tensor.
            input_padding: Input padding tensor.
            freq: Frequency encoding tensor.
            text_descriptions: Patch-level text descriptions organized as [batch][patch][texts].
                             If None, text_embeddings must be provided.
            text_embeddings: Pre-computed text embeddings of shape (batch_size, num_patches, embedding_dim).
                            If provided, text_descriptions is ignored.

        Returns:
            Output tensor with forecasting predictions.
        """
        num_outputs = len(self.config.quantiles) + 1

        # Preprocess inputs with multimodal support
        model_input, patched_padding, stats, _ = self._preprocess_multimodal_input(
            input_ts=input_ts,
            input_padding=input_padding,
            text_descriptions=text_descriptions,
            text_embeddings=text_embeddings,
        )

        # Add frequency embedding (same as original)
        f_emb = self.freq_emb(freq)  # B x 1 x D
        model_input = model_input + f_emb

        # Pass through stacked transformer (same as original)
        model_output = self.stacked_transformer(model_input, patched_padding)

        # Postprocess output (same as original)
        output_ts: torch.Tensor = self._postprocess_output(model_output, num_outputs, stats)

        return output_ts

    def decode(
        self,
        input_ts: torch.Tensor,
        paddings: torch.Tensor,
        freq: torch.Tensor,
        horizon_len: int,
        text_descriptions: list[list[list[str]]] | None = None,
        text_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct multi-step forecasting with multimodal support following TimesFM interface.

        This method extends the original TimesFM decode method to incorporate text descriptions
        during forecasting. It performs one-shot prediction for the entire horizon rather than
        auto-regressive generation, maintaining the same interface and behavior as the base
        TimesFM decoder while adding multimodal capabilities.

        Args:
            input_ts: Input time series tensor of shape (batch_size, context_length).
            paddings: Binary padding tensor of shape (batch_size, context_length + horizon_len)
                     where 1 indicates valid data and 0 indicates padding.
            freq: Frequency encoding tensor of shape (batch_size, 1) with values
                 0=high frequency, 1=medium frequency, 2=low frequency.
            horizon_len: Number of time steps to forecast into the future. If this exceeds
                        the model's configured horizon_len, only the first min(horizon_len,
                        config.horizon_len) predictions will be returned.
            text_descriptions: Nested list [batch][patch][texts] containing text descriptions
                              for each patch of each batch sample. If None, text_embeddings must be provided.
            text_embeddings: Pre-computed text embeddings of shape (batch_size, num_patches, embedding_dim).
                            If provided, text_descriptions is ignored.

        Returns:
            Tuple containing:
            - Point forecasts: Tensor of shape (batch_size, actual_horizon_len) with mean predictions
            - Full forecasts: Tensor of shape (batch_size, actual_horizon_len, 1 + num_quantiles)
              where actual_horizon_len = min(horizon_len, config.horizon_len), the first channel
              is mean and remaining channels are quantile predictions

        Raises:
            ValueError: If paddings length doesn't match input_ts length + horizon_len.
        """
        context_len = input_ts.shape[1]

        if paddings.shape[1] != input_ts.shape[1] + horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {paddings.shape[1]} != {input_ts.shape[1]} + {horizon_len}"
            )

        context_input = input_ts[:, -context_len:]
        context_padding = paddings[:, 0 : input_ts.shape[1]][:, -context_len:]

        fprop_outputs = self(context_input, context_padding, freq, text_descriptions, text_embeddings)

        # The model outputs predictions for self.config.horizon_len steps
        # We need to handle the case where requested horizon_len differs from model's horizon_len
        model_horizon_len = fprop_outputs.shape[2]
        actual_horizon_len = min(horizon_len, model_horizon_len)

        new_full_ts = fprop_outputs[:, -1, :actual_horizon_len, :]

        return (new_full_ts[:, :, 0], new_full_ts)

    def freeze_parameters(self) -> None:
        """Freeze all parameters in the MultimodalPatchedDecoder model.

        This includes all TimesFM decoder parameters, text encoder parameters,
        and fusion layer parameters.
        """
        # Freeze all model parameters
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters in the MultimodalPatchedDecoder model.

        This includes all TimesFM decoder parameters, text encoder parameters,
        and fusion layer parameters.
        """
        # Unfreeze all model parameters
        for param in self.parameters():
            param.requires_grad = True

    def is_frozen(self) -> bool:
        """Check if all parameters in the MultimodalPatchedDecoder model are frozen.

        Returns:
            True if all parameters are frozen, False otherwise.
        """
        return all(not param.requires_grad for param in self.parameters())

    def freeze_text_components(self, freeze_encoder: bool = True, freeze_fusion: bool = True) -> None:
        """Freeze text encoder and/or fusion components for selective training.

        Args:
            freeze_encoder: Whether to freeze the text encoder parameters.
            freeze_fusion: Whether to freeze the fusion projection parameters.
        """
        if freeze_encoder:
            self.text_encoder.freeze_parameters()

        if freeze_fusion:
            self.multimodal_fusion.freeze_projection()

    def unfreeze_text_components(self, unfreeze_encoder: bool = True, unfreeze_fusion: bool = True) -> None:
        """Unfreeze text encoder and/or fusion components for training.

        Args:
            unfreeze_encoder: Whether to unfreeze the text encoder parameters.
            unfreeze_fusion: Whether to unfreeze the fusion projection parameters.
        """
        if unfreeze_encoder:
            self.text_encoder.unfreeze_parameters()

        if unfreeze_fusion:
            self.multimodal_fusion.unfreeze_projection()

    def is_text_frozen(self) -> dict[str, bool]:
        """Check if text components are frozen.

        Returns:
            Dictionary with freeze status of each component:
            - 'encoder': True if text encoder is frozen, False otherwise
            - 'fusion': True if fusion projection is frozen, False otherwise
        """
        return {"encoder": self.text_encoder.is_frozen(), "fusion": self.multimodal_fusion.is_projection_frozen()}
