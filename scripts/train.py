#!/usr/bin/env python3
"""Training script for multimodal TimesFM on Time-MMD dataset."""

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import ConcatDataset

from src.data.time_mmd_dataset import TimeMmdDataset
from src.models.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from src.train.trainer import MultimodalTrainer
from src.utils.logging import get_logger, setup_logger
from src.utils.yaml import load_yaml


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def create_model(model_config: dict[str, Any]) -> MultimodalPatchedDecoder:
    """Create multimodal model from configuration and load pretrained TimesFM weights."""
    logger = get_logger()

    # Create multimodal config
    config = MultimodalTimesFMConfig(
        num_layers=model_config["timesfm"]["num_layers"],
        num_heads=model_config["timesfm"]["num_heads"],
        num_kv_heads=model_config["timesfm"]["num_kv_heads"],
        hidden_size=model_config["timesfm"]["model_dims"],
        intermediate_size=model_config["timesfm"]["model_dims"],
        head_dim=model_config["timesfm"]["model_dims"] // model_config["timesfm"]["num_heads"],
        rms_norm_eps=float(model_config["timesfm"]["rms_norm_eps"]),
        patch_len=model_config["timesfm"]["input_patch_len"],
        horizon_len=model_config["timesfm"]["output_patch_len"],
        quantiles=model_config["timesfm"]["quantiles"],
        pad_val=float(model_config["timesfm"]["pad_val"]),
        tolerance=float(model_config["timesfm"]["tolerance"]),
        dtype=model_config["timesfm"]["dtype"],
        use_positional_embedding=model_config["timesfm"]["use_positional_embedding"],
        text_encoder_model=model_config["text_encoder"]["model_name"],
        text_embedding_dim=model_config["text_encoder"]["embedding_dim"],
    )

    # Create multimodal model
    model = MultimodalPatchedDecoder(config)

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
    training_config: dict[str, Any],
) -> Path:
    """Train a model and return the path to the best checkpoint."""

    # Setup training configuration
    hardware_config = training_config["hardware"]
    train_config = training_config["train"]
    log_config = training_config["log"]
    checkpoint_config = training_config["checkpoint"]

    # Setup device
    device = hardware_config["device"]
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        max_grad_norm=float(train_config["max_grad_norm"]),
        device=device,
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        log_dir=Path(log_config["save_dir"]),
        checkpoint_dir=Path(checkpoint_config["save_dir"]),
        wandb_run_name=train_config["wandb_run_name"],
    )

    # Setup logger
    logger = get_logger()
    logger.info(f"Using device: {device}")

    # Only train fusion
    trainer.freeze_pretrained_parameters()

    epochs = train_config["num_epochs"]
    logger.info(f"Training fusion components only for {epochs} epochs (TimesFM and text encoder frozen)")

    trainer.train(
        num_epochs=epochs,
        save_every=checkpoint_config["save_frequency"],
    )

    # Return path to best checkpoint
    return trainer.checkpoint_dir / "best_model.pt"


def main() -> int:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train multimodal TimesFM on Time-MMD dataset")

    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yml",
        help="Path to model configuration file",
    )

    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training.yml",
        help="Path to training configuration file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (if not provided, no seed will be set)",
    )

    args = parser.parse_args()

    # Load configurations
    model_config = load_yaml(Path(args.model_config))
    training_config = load_yaml(Path(args.training_config))

    # Set random seed for reproducibility if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Setup logging
    log_config = training_config["log"]
    setup_logger(log_file=Path(log_config["save_dir"]) / f"{log_config['experiment_name']}.log")

    logger = get_logger()
    logger.info("Starting multimodal TimesFM training")
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Training config: {args.training_config}")

    # Create datasets
    data_config = training_config["data"]
    train_dataset, val_dataset = create_datasets(
        data_path=Path(data_config["data_path"]),
        domains=data_config["domains"],
        split_ratio=float(data_config["split_ratio"]),
        patch_len=data_config["patch_len"],
        context_len=data_config["context_len"],
        horizon_len=data_config["horizon_len"],
    )

    # Train multimodal model
    logger.info("=" * 50)
    logger.info("Training multimodal TimesFM model")
    logger.info("=" * 50)

    multimodal_model = create_model(model_config)

    try:
        checkpoint_path = train_model(
            model=multimodal_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=training_config,
        )
        logger.info(f"Multimodal model training completed. Checkpoint: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Multimodal model training failed: {e}")
        return 1

    logger.info("Training completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())
