#!/usr/bin/env python3
"""Cross-validation training script for multimodal TimesFM on Time-MMD dataset."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset

from examples.time_mmd.configs import ModelConfig, TrainingConfig
from examples.time_mmd.data.cross_validation import create_fold_datasets, get_all_domains, get_cross_validation_splits
from multimodal_timesfm.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from multimodal_timesfm.trainer import MultimodalTrainer
from multimodal_timesfm.utils.device import resolve_device
from multimodal_timesfm.utils.logging import get_logger, setup_logger
from multimodal_timesfm.utils.model import create_multimodal_model
from multimodal_timesfm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multimodal TimesFM with cross-validation on Time-MMD dataset")

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

    return parser.parse_args()


def create_model(model_config: ModelConfig, device: torch.device) -> MultimodalPatchedDecoder:
    """Create multimodal model from configuration and load pretrained TimesFM weights."""
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

    return create_multimodal_model(config=config, device=device, load_pretrained=True)


def train_fold(
    training_config: TrainingConfig,
    fold_idx: int,
    model: MultimodalPatchedDecoder,
    train_dataset: ConcatDataset[dict[str, Any]],
    val_dataset: ConcatDataset[dict[str, Any]],
    device: torch.device,
) -> tuple[Path, dict[str, float]]:
    """Train a model for one fold and return the path to the best checkpoint and validation metrics.

    Args:
        training_config: Training configuration.
        fold_idx: Index of the current fold.
        model: Model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        device: Device to use for training.

    Returns:
        Tuple of (checkpoint_path, validation_metrics).
    """
    logger = get_logger()

    # Create fold-specific directories
    fold_log_dir = Path(training_config.log.save_dir) / f"fold_{fold_idx}"
    fold_checkpoint_dir = Path(training_config.checkpoint.save_dir) / f"fold_{fold_idx}"

    fold_log_dir.mkdir(parents=True, exist_ok=True)
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    wandb_run_name = f"{training_config.runner.wandb_run_name}_fold_{fold_idx}"
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
        log_dir=fold_log_dir,
        checkpoint_dir=fold_checkpoint_dir,
        wandb_run_name=wandb_run_name,
    )

    logger.info(f"Training fold {fold_idx}")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")

    # Freeze pretrained parameters (only train fusion)
    trainer.freeze_pretrained_parameters()

    epochs = training_config.runner.num_epochs
    logger.info(f"Training fusion components only for {epochs} epochs (TimesFM and text encoder frozen)")

    trainer.train(
        num_epochs=epochs,
        save_every=training_config.checkpoint.save_frequency,
    )

    # Get best validation metrics
    best_checkpoint = fold_checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(best_checkpoint, weights_only=True)
    val_metrics = {
        "val_loss": checkpoint["best_val_loss"],
    }

    return best_checkpoint, val_metrics


def main() -> int:
    """Main cross-validation training function."""
    args = parse_args()

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
    setup_logger(log_file=Path(training_config.log.save_dir) / f"{training_config.log.experiment_name}_cv.log")

    logger = get_logger()
    logger.info("Starting cross-validation training on Time-MMD dataset")
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Training config: {args.training_config}")
    logger.info(f"Number of folds: {training_config.cross_validation.n_folds}")
    logger.info(
        f"Train/Val/Test ratio: {training_config.cross_validation.train_ratio}/"
        f"{training_config.cross_validation.val_ratio}/{training_config.cross_validation.test_ratio}"
    )

    # Get all available domains
    data_path = Path(training_config.data.data_path)
    all_domains = get_all_domains(data_path)
    logger.info(f"Found {len(all_domains)} domains in dataset")

    # Generate cross-validation splits
    cv_splits = get_cross_validation_splits(
        all_domains=all_domains,
        n_folds=training_config.cross_validation.n_folds,
        train_ratio=training_config.cross_validation.train_ratio,
        val_ratio=training_config.cross_validation.val_ratio,
        test_ratio=training_config.cross_validation.test_ratio,
        seed=args.seed,
    )

    logger.info(f"Generated {len(cv_splits)} cross-validation splits")

    # Setup device
    device = resolve_device(training_config.hardware.device)
    logger.info(f"Using device: {device}")

    # Store results for all folds
    all_results: list[dict[str, Any]] = []

    # Train each fold
    for fold_idx, (train_domains, val_domains, test_domains) in enumerate(cv_splits):
        logger.info("=" * 50)
        logger.info(f"Training fold {fold_idx + 1}/{len(cv_splits)}")
        logger.info("=" * 50)
        logger.info(f"Train domains: {len(train_domains)}")
        logger.info(f"Val domains: {len(val_domains)}")
        logger.info(f"Test domains: {len(test_domains)}")

        # Create datasets for this fold
        train_dataset, val_dataset, _ = create_fold_datasets(
            data_path=data_path,
            train_domains=train_domains,
            val_domains=val_domains,
            test_domains=test_domains,
            patch_len=training_config.data.patch_len,
            context_len=training_config.data.context_len,
            horizon_len=training_config.data.horizon_len,
        )

        # Create a fresh model for this fold
        model = create_model(model_config, device)

        try:
            checkpoint_path, val_metrics = train_fold(
                fold_idx=fold_idx,
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                training_config=training_config,
                device=device,
            )
            logger.info(f"Fold {fold_idx} training completed. Checkpoint: {checkpoint_path}")
            logger.info(f"Fold {fold_idx} validation metrics: {val_metrics}")

            # Store results
            fold_results = {
                "fold": fold_idx,
                "checkpoint": str(checkpoint_path),
                "train_domains": train_domains,
                "val_domains": val_domains,
                "test_domains": test_domains,
                "val_metrics": val_metrics,
            }
            all_results.append(fold_results)

        except Exception as e:
            logger.error(f"Fold {fold_idx} training failed: {e}")
            return 1

    # Save cross-validation results
    cv_results_path = Path(training_config.log.save_dir) / "cv_results.json"
    with open(cv_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Cross-validation results saved to {cv_results_path}")

    # Compute average metrics across folds
    avg_val_loss = sum(r["val_metrics"]["val_loss"] for r in all_results) / len(all_results)
    logger.info("=" * 50)
    logger.info("Cross-validation summary:")
    logger.info(f"Average validation loss: {avg_val_loss:.6f}")
    logger.info("=" * 50)

    logger.info("Cross-validation training completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())
