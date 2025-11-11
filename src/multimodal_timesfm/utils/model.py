"""Model creation and loading utilities for multimodal TimesFM."""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder, TimesFMConfig

from multimodal_timesfm.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from multimodal_timesfm.utils.logging import get_logger


def create_multimodal_model(
    config: MultimodalTimesFMConfig,
    device: torch.device,
    load_pretrained: bool = True,
    pretrained_repo: str = "google/timesfm-2.0-500m-pytorch",
) -> MultimodalPatchedDecoder:
    """Create multimodal TimesFM model from configuration.

    Args:
        config: MultimodalTimesFMConfig instance.
        device: Device to place the model on.
        load_pretrained: Whether to load pretrained TimesFM weights.
        pretrained_repo: Hugging Face repository ID for pretrained weights.

    Returns:
        Multimodal TimesFM model instance.
    """
    logger = get_logger()

    # Create multimodal model
    model = MultimodalPatchedDecoder(config, device)

    # Load pretrained TimesFM weights if requested
    if load_pretrained:
        logger.info(f"Loading pretrained TimesFM weights from {pretrained_repo}")
        try:
            model_dir = Path(snapshot_download(pretrained_repo))
            checkpoint_path = model_dir / "torch_model.ckpt"
            pretrained_weights = torch.load(checkpoint_path, weights_only=True)

            # Load weights into the TimesFM components (excluding text components)
            model_state_dict = model.state_dict()
            pretrained_keys = set(pretrained_weights.keys())
            model_keys = set(model_state_dict.keys())

            # Find keys that match between pretrained and multimodal model
            matching_keys = pretrained_keys.intersection(model_keys)
            non_matching_keys = model_keys - pretrained_keys

            logger.info(f"Loading {len(matching_keys)} pretrained parameters")
            logger.info(f"Initializing {len(non_matching_keys)} new parameters (text components)")

            # Load matching weights
            for key in matching_keys:
                model_state_dict[key].copy_(pretrained_weights[key])

            logger.info("Successfully loaded pretrained TimesFM weights")

        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.warning("Continuing with randomly initialized weights")

    return model


def create_baseline_timesfm_model(
    config: TimesFMConfig,
    load_pretrained: bool = True,
    pretrained_repo: str = "google/timesfm-2.0-500m-pytorch",
) -> PatchedTimeSeriesDecoder:
    """Create original TimesFM model (without text) from configuration.

    Args:
        config: TimesFMConfig instance.
        load_pretrained: Whether to load pretrained TimesFM weights.
        pretrained_repo: Hugging Face repository ID for pretrained weights.

    Returns:
        Original TimesFM model instance.
    """
    logger = get_logger()

    # Create model
    model = PatchedTimeSeriesDecoder(config)

    # Load pretrained TimesFM weights if requested
    if load_pretrained:
        logger.info(f"Loading pretrained TimesFM weights from {pretrained_repo}")
        try:
            model_dir = Path(snapshot_download(pretrained_repo))
            checkpoint_path = model_dir / "torch_model.ckpt"
            pretrained_weights = torch.load(checkpoint_path, weights_only=True)

            # Load weights
            model_state_dict = model.state_dict()
            pretrained_keys = set(pretrained_weights.keys())
            model_keys = set(model_state_dict.keys())

            matching_keys = pretrained_keys.intersection(model_keys)

            logger.info(f"Loading {len(matching_keys)} pretrained parameters")

            # Load matching weights
            for key in matching_keys:
                model_state_dict[key].copy_(pretrained_weights[key])

            logger.info("Successfully loaded pretrained TimesFM weights")

        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.warning("Continuing with randomly initialized weights")

    return model


def load_multimodal_checkpoint(
    checkpoint_path: Path,
    config: MultimodalTimesFMConfig,
    device: torch.device,
) -> MultimodalPatchedDecoder:
    """Load trained multimodal TimesFM model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file.
        config: MultimodalTimesFMConfig instance.
        device: Device to place the model on.

    Returns:
        Loaded multimodal TimesFM model instance.
    """
    logger = get_logger()

    # Create model
    model = MultimodalPatchedDecoder(config, device)

    # Load checkpoint
    logger.info(f"Loading multimodal model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Successfully loaded multimodal model checkpoint")

    return model


def load_baseline_checkpoint(
    checkpoint_path: Path,
    config: TimesFMConfig,
    device: torch.device,
) -> PatchedTimeSeriesDecoder:
    """Load trained baseline TimesFM model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file.
        config: TimesFMConfig instance.
        device: Device to place the model on.

    Returns:
        Loaded baseline TimesFM model instance.
    """
    logger = get_logger()

    # Create model
    model = PatchedTimeSeriesDecoder(config)
    model.to(device)

    # Load checkpoint
    logger.info(f"Loading baseline model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Successfully loaded baseline model checkpoint")

    return model
