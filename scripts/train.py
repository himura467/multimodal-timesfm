#!/usr/bin/env python3
"""Training script for multimodal TimesFM on Time-MMD dataset."""

import argparse
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch.utils.data import ConcatDataset

from src.configs import ModelConfig, TrainingConfig
from src.data.time_mmd_dataset import TimeMmdDataset
from src.models.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from src.train.trainer import MultimodalTrainer
from src.utils.device import resolve_device
from src.utils.logging import get_logger, setup_logger
from src.utils.seed import set_seed


def create_datasets(
    data_path: Path,
    domains: list[str],
    split_ratio: float,
    patch_len: int,
    context_len: int,
    horizon_len: int,
) -> tuple[ConcatDataset[dict[str, Any]], ConcatDataset[dict[str, Any]]]:
    """Create concatenated datasets from multiple domains."""
    train_datasets = []
    val_datasets = []

    logger = get_logger()
    logger.info(f"Loading datasets from domains: {domains}")

    for domain in domains:
        logger.info(f"Loading domain: {domain}")

        # Create train dataset for this domain
        train_dataset = TimeMmdDataset(
            data_dir=data_path,
            domain=domain,
            split_ratio=split_ratio,
            split="train",
            patch_len=patch_len,
            context_len=context_len,
            horizon_len=horizon_len,
        )
        train_datasets.append(train_dataset)

        # Create validation dataset for this domain
        val_dataset = TimeMmdDataset(
            data_dir=data_path,
            domain=domain,
            split_ratio=split_ratio,
            split="test",
            patch_len=patch_len,
            context_len=context_len,
            horizon_len=horizon_len,
        )
        val_datasets.append(val_dataset)

        logger.info(f"Domain {domain}: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Concatenate all domain datasets
    combined_train: ConcatDataset[dict[str, Any]] = ConcatDataset(train_datasets)
    combined_val: ConcatDataset[dict[str, Any]] = ConcatDataset(val_datasets)

    logger.info(f"Total: {len(combined_train)} train, {len(combined_val)} val samples")

    return combined_train, combined_val


def create_model(model_config: ModelConfig, device: torch.device) -> MultimodalPatchedDecoder:
    """Create multimodal model from configuration and load pretrained TimesFM weights."""
    logger = get_logger()

    # Create multimodal config
    config = MultimodalTimesFMConfig(
        num_layers=model_config.timesfm.num_layers,
        num_heads=model_config.timesfm.num_heads,
        num_kv_heads=model_config.timesfm.num_kv_heads,
        hidden_size=model_config.timesfm.model_dims,
        intermediate_size=model_config.timesfm.model_dims,
        head_dim=model_config.timesfm.model_dims // model_config.timesfm.num_heads,
        rms_norm_eps=model_config.timesfm.rms_norm_eps,
        patch_len=model_config.timesfm.input_patch_len,
        horizon_len=model_config.timesfm.output_patch_len,
        quantiles=model_config.timesfm.quantiles,
        pad_val=model_config.timesfm.pad_val,
        tolerance=model_config.timesfm.tolerance,
        dtype=model_config.timesfm.dtype,
        use_positional_embedding=model_config.timesfm.use_positional_embedding,
        text_encoder_type=model_config.text_encoder.text_encoder_type,
    )

    # Create multimodal model
    model = MultimodalPatchedDecoder(config, device)

    # Load pretrained TimesFM weights
    repo_id = "google/timesfm-2.0-500m-pytorch"
    logger.info(f"Loading pretrained TimesFM weights from {repo_id}")

    try:
        model_dir = Path(snapshot_download(repo_id))
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


def train_model(
    model: MultimodalPatchedDecoder,
    train_dataset: ConcatDataset[dict[str, Any]],
    val_dataset: ConcatDataset[dict[str, Any]],
    training_config: TrainingConfig,
    device: torch.device,
) -> Path:
    """Train a model and return the path to the best checkpoint."""

    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_config.runner.batch_size,
        gradient_accumulation_steps=training_config.runner.gradient_accumulation_steps,
        max_grad_norm=training_config.runner.max_grad_norm,
        device=device,
        learning_rate=training_config.runner.learning_rate,
        weight_decay=training_config.runner.weight_decay,
        log_dir=Path(training_config.log.save_dir),
        checkpoint_dir=Path(training_config.checkpoint.save_dir),
        wandb_run_name=training_config.runner.wandb_run_name,
    )

    # Setup logger
    logger = get_logger()
    logger.info(f"Using device: {device}")

    # Only train fusion
    trainer.freeze_pretrained_parameters()

    epochs = training_config.runner.num_epochs
    logger.info(f"Training fusion components only for {epochs} epochs (TimesFM and text encoder frozen)")

    trainer.train(
        num_epochs=epochs,
        save_every=training_config.checkpoint.save_frequency,
    )

    # Return path to best checkpoint
    return trainer.checkpoint_dir / "best_model.pt"


def main() -> int:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train multimodal TimesFM on Time-MMD dataset")

    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to model configuration file",
    )

    parser.add_argument(
        "--training-config",
        type=str,
        help="Path to training configuration file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (if not provided, no seed will be set)",
    )

    args = parser.parse_args()

    # Load configurations
    if args.model_config:
        model_config = ModelConfig.from_yaml(Path(args.model_config))
    else:
        model_config = ModelConfig()
    if args.training_config:
        training_config = TrainingConfig.from_yaml(Path(args.training_config))
    else:
        training_config = TrainingConfig()

    # Set random seed for reproducibility if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Setup logging
    setup_logger(log_file=Path(training_config.log.save_dir) / f"{training_config.log.experiment_name}.log")

    logger = get_logger()
    logger.info("Starting multimodal TimesFM training")
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Training config: {args.training_config}")

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        data_path=Path(training_config.data.data_path),
        domains=training_config.data.domains,
        split_ratio=training_config.data.split_ratio,
        patch_len=training_config.data.patch_len,
        context_len=training_config.data.context_len,
        horizon_len=training_config.data.horizon_len,
    )

    # Train multimodal model
    logger.info("=" * 50)
    logger.info("Training multimodal TimesFM model")
    logger.info("=" * 50)

    # Setup device for model creation
    device = resolve_device(training_config.hardware.device)
    logger.info(f"Using device: {device}")

    multimodal_model = create_model(model_config, device)

    try:
        checkpoint_path = train_model(
            model=multimodal_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=training_config,
            device=device,
        )
        logger.info(f"Multimodal model training completed. Checkpoint: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Multimodal model training failed: {e}")
        return 1

    logger.info("Training completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())
