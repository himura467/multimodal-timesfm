"""Multimodal version of TimesFM's patched decoder that supports text inputs."""

from dataclasses import dataclass

import torch
from timesfm.pytorch_patched_decoder import (
    PatchedTimeSeriesDecoder,
    TimesFMConfig,
)

from src.models.multimodal_fusion import MultimodalFusion
from src.models.text_encoder import TextEncoder


@dataclass
class MultimodalTimesFMConfig(TimesFMConfig):  # type: ignore[misc]
    """Config for initializing MultimodalPatchedDecoder that extends TimesFMConfig.

    Attributes:
        text_encoder_model: Name of the sentence transformer model for text encoding.
        text_embedding_dim: Dimension of text embeddings.
    """

    text_encoder_model: str = "all-MiniLM-L6-v2"
    text_embedding_dim: int = 384


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

    def __init__(self, config: MultimodalTimesFMConfig):
        """Initialize MultimodalPatchedDecoder.

        Args:
            config: Multimodal configuration containing both TimesFM and text encoding parameters.
        """
        # Initialize parent class with base TimesFM config
        super().__init__(config)

        self.config = config

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Initialize text encoder and fusion components
        self.text_encoder = TextEncoder(
            model_name=config.text_encoder_model, embedding_dim=config.text_embedding_dim, device=device
        )
        self.multimodal_fusion = MultimodalFusion(
            ts_feature_dim=config.hidden_size,
            text_feature_dim=config.text_embedding_dim,
        )

        # Move the entire decoder to the selected device
        self.to(device)

    def _preprocess_multimodal_input(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.Tensor,
        text_descriptions: list[list[list[str]]],
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

        Returns:
            Tuple containing:
            - model_input: Preprocessed input tensor for transformer
            - patched_padding: Padding tensor for patches
            - stats: Normalization statistics (mean, std)
            - patched_inputs: Original patched inputs
        """
        # Use the original preprocessing for time series data
        model_input, patched_padding, stats, patched_inputs = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )

        # Encode text descriptions for each patch
        text_embeddings = self._encode_patch_text_features(text_descriptions, model_input.shape, model_input.device)

        # Fuse text features with time series features
        model_input = self.multimodal_fusion(model_input, text_embeddings)

        return model_input, patched_padding, stats, patched_inputs

    def _encode_patch_text_features(
        self, text_descriptions: list[list[list[str]]], target_shape: torch.Size, device: torch.device
    ) -> torch.Tensor:
        """Encode patch-level text descriptions and match time series features shape.

        Args:
            text_descriptions: List of text descriptions organized as [batch][patch][texts].
                              Each batch item contains patches, each patch contains multiple text strings.
            target_shape: Target shape (batch_size, num_patches, feature_dim) to match.
            device: Device to place the text embeddings on.

        Returns:
            Text feature tensor of shape (batch_size, num_patches, text_embedding_dim).

        Raises:
            ValueError: If batch sizes or patch numbers don't match.
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
            all_embeddings = torch.zeros((batch_size * num_patches, self.config.text_embedding_dim), device=device)

        # Reshape to (batch_size, num_patches, text_embedding_dim)
        text_embeddings: torch.Tensor = all_embeddings.reshape(batch_size, num_patches, self.config.text_embedding_dim)

        return text_embeddings

    def forward(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
        text_descriptions: list[list[list[str]]],
    ) -> torch.Tensor:
        """Forward pass for multimodal decoder.

        Args:
            input_ts: Input time series tensor.
            input_padding: Input padding tensor.
            freq: Frequency encoding tensor.
            text_descriptions: Patch-level text descriptions organized as [batch][patch][texts].

        Returns:
            Output tensor with forecasting predictions.
        """
        num_outputs = len(self.config.quantiles) + 1

        # Preprocess inputs with multimodal support
        model_input, patched_padding, stats, _ = self._preprocess_multimodal_input(
            input_ts=input_ts,
            input_padding=input_padding,
            text_descriptions=text_descriptions,
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
        text_descriptions: list[list[list[str]]],
        output_patch_len: int | None = None,
        max_len: int | None = None,
        return_forecast_on_context: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Auto-regressive decoding with multimodal support.

        This method extends the original decode method to support text descriptions
        during auto-regressive forecasting.

        Args:
            input_ts: Input time-series tensor of shape B x C.
            paddings: Padding tensor of shape B x (C + H) where H is prediction length.
            freq: Frequency tensor of shape B x 1.
            horizon_len: Prediction length.
            text_descriptions: Patch-level text descriptions organized as [batch][patch][texts].
            output_patch_len: Output length per decoding step.
            max_len: Maximum training context length.
            return_forecast_on_context: Whether to return forecast on context.

        Returns:
            Tuple of:
            - Point (mean) output predictions as tensor with shape B x H'.
            - Full predictions (mean and quantiles) as tensor with shape B x H' x (1 + # quantiles).
        """
        final_out = input_ts
        context_len = final_out.shape[1]
        full_outputs = []

        if max_len is None:
            max_len = context_len
        if paddings.shape[1] != final_out.shape[1] + horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}"
            )
        if output_patch_len is None:
            output_patch_len = self.config.horizon_len

        num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len

        for step_index in range(num_decode_patches):
            current_padding = paddings[:, 0 : final_out.shape[1]]
            input_ts_step = final_out[:, -max_len:]
            input_padding = current_padding[:, -max_len:]

            # Use multimodal forward pass
            fprop_outputs = self(input_ts_step, input_padding, freq, text_descriptions)

            if return_forecast_on_context and step_index == 0:
                # Collect model forecast on context except unavailable first batch forecast
                new_full_ts = fprop_outputs[:, 0:-1, 0 : self.config.patch_len, :]
                new_full_ts = new_full_ts.reshape(new_full_ts.size(0), -1, new_full_ts.size(3))
                full_outputs.append(new_full_ts)

            # Extract predictions for next step
            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
            full_outputs.append(new_full_ts)
            final_out = torch.cat([final_out, new_ts], dim=-1)

        if return_forecast_on_context:
            full_outputs_tensor = torch.cat(full_outputs, dim=1)[
                :, : (context_len - self.config.patch_len + horizon_len), :
            ]
        else:
            full_outputs_tensor = torch.cat(full_outputs, dim=1)[:, 0:horizon_len, :]

        return (full_outputs_tensor[:, :, 0], full_outputs_tensor)

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
