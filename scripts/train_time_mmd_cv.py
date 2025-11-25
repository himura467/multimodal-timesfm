#!/usr/bin/env python3
"""Cross-validation training script for multimodal TimesFM on Time-MMD dataset."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder, TimesFMConfig
from torch.utils.data import ConcatDataset

from examples.time_mmd.configs import ModelConfig
from examples.time_mmd.data.cross_validation import create_fold_datasets, get_all_domains, get_cross_validation_splits
from multimodal_timesfm.baseline_trainer import BaselineTrainer
from multimodal_timesfm.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from multimodal_timesfm.trainer import MultimodalTrainer
from multimodal_timesfm.training_args import TrainingArguments
from multimodal_timesfm.utils.device import resolve_device
from multimodal_timesfm.utils.logging import get_logger, setup_logger
from multimodal_timesfm.utils.model import create_baseline_timesfm_model
from multimodal_timesfm.utils.model import create_multimodal_model as create_multimodal_model_core
from multimodal_timesfm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multimodal TimesFM with cross-validation on Time-MMD dataset")

    parser.add_argument(
        "--training-args",
        type=str,
        help="Path to training arguments file",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to model configuration file",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/Time-MMD",
        help="Path to Time-MMD dataset",
    )

    parser.add_argument(
        "--train-baseline",
        action="store_true",
        help="Also train baseline TimesFM model (fine-tuned, no text) on same splits",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (if not provided, no seed will be set)",
    )

    return parser.parse_args()


def create_multimodal_model(model_config: ModelConfig, device: torch.device) -> MultimodalPatchedDecoder:
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

    return create_multimodal_model_core(config=config, device=device, load_pretrained=True)


def create_baseline_model(model_config: ModelConfig, load_pretrained: bool = True) -> PatchedTimeSeriesDecoder:
    """Create baseline TimesFM model from configuration."""
    config = TimesFMConfig(
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
    )

    return create_baseline_timesfm_model(config=config, load_pretrained=load_pretrained)


def train_multimodal_fold(
    args: TrainingArguments,
    fold_idx: int,
    model: MultimodalPatchedDecoder,
    train_dataset: ConcatDataset[dict[str, Any]],
    val_dataset: ConcatDataset[dict[str, Any]],
) -> tuple[Path, dict[str, float]]:
    """Train a multimodal model for one fold and return the path to the best checkpoint and validation metrics.

    Args:
        args: Training arguments.
        fold_idx: Index of the current fold.
        model: Multimodal model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.

    Returns:
        Tuple of (checkpoint_path, validation_metrics).
    """
    logger = get_logger()

    # Create fold-specific output directory
    fold_output_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    fold_args = replace(
        args,
        output_dir=str(fold_output_dir),
        run_name=f"{args.run_name}_fold_{fold_idx}" if args.run_name else f"multimodal_fold_{fold_idx}",
    )

    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        args=fold_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    logger.info(f"Training fold {fold_idx}")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")

    # Freeze pretrained parameters (only train fusion)
    trainer.freeze_pretrained_parameters()

    logger.info(f"Training fusion components only for {args.num_train_epochs} epochs (TimesFM and text encoder frozen)")

    trainer.train()

    # Get best validation metrics
    best_checkpoint = fold_args.checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(best_checkpoint, weights_only=True)
    val_metrics = {
        "val_loss": checkpoint["best_val_loss"],
    }

    return best_checkpoint, val_metrics


def train_baseline_fold(
    args: TrainingArguments,
    fold_idx: int,
    model: PatchedTimeSeriesDecoder,
    train_dataset: ConcatDataset[dict[str, Any]],
    val_dataset: ConcatDataset[dict[str, Any]],
) -> tuple[Path, dict[str, float]]:
    """Train a baseline model for one fold and return the path to the best checkpoint and validation metrics.

    Args:
        args: Training arguments.
        fold_idx: Index of the current fold.
        model: Baseline TimesFM model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.

    Returns:
        Tuple of (checkpoint_path, validation_metrics).
    """
    logger = get_logger()

    # Create fold-specific output directory for baseline
    fold_output_dir = Path(args.output_dir) / "baseline_finetuned" / f"fold_{fold_idx}"
    fold_args = replace(
        args,
        output_dir=str(fold_output_dir),
        run_name=f"{args.run_name}_baseline_finetuned_fold_{fold_idx}"
        if args.run_name
        else f"baseline_finetuned_fold_{fold_idx}",
    )

    # Create baseline trainer
    trainer = BaselineTrainer(
        model=model,
        args=fold_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        freeze_timesfm=False,
    )

    logger.info(f"Training baseline (fine-tuned) fold {fold_idx}")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")

    logger.info(f"Fine-tuning TimesFM for {args.num_train_epochs} epochs")

    trainer.train()

    # Get best validation metrics
    best_checkpoint = fold_args.checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(best_checkpoint, weights_only=True)
    val_metrics = {
        "val_loss": checkpoint["best_val_loss"],
    }

    return best_checkpoint, val_metrics


def main() -> int:
    """Main cross-validation training function."""
    parsed_args = parse_args()

    # Load configurations
    if parsed_args.training_args:
        training_args = TrainingArguments.from_yaml(Path(parsed_args.training_args))
    else:
        training_args = TrainingArguments()

    if parsed_args.model_config:
        model_config = ModelConfig.from_yaml(Path(parsed_args.model_config))
    else:
        model_config = ModelConfig()

    # Override seed from command line if provided
    if parsed_args.seed is not None:
        training_args = replace(training_args, seed=parsed_args.seed)

    # Set random seed for reproducibility if provided
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Setup logging
    setup_logger(log_file=training_args.logging_dir / "cv_training.log")

    logger = get_logger()
    logger.info("Starting cross-validation training on Time-MMD dataset")
    logger.info(f"Training arguments: {parsed_args.training_args}")
    logger.info(f"Model config: {parsed_args.model_config}")
    logger.info(f"Number of folds: {training_args.n_folds}")
    logger.info(
        f"Train/Val/Test ratio: {training_args.train_ratio}/{training_args.val_ratio}/{training_args.test_ratio}"
    )

    # Get all available domains
    data_path = Path(parsed_args.data_path)
    all_domains = get_all_domains(data_path)
    logger.info(f"Found {len(all_domains)} domains in dataset")

    # Generate cross-validation splits
    cv_splits = get_cross_validation_splits(
        all_domains=all_domains,
        n_folds=training_args.n_folds,
        train_ratio=training_args.train_ratio,
        val_ratio=training_args.val_ratio,
        test_ratio=training_args.test_ratio,
        seed=training_args.seed,
    )

    logger.info(f"Generated {len(cv_splits)} cross-validation splits")

    # Setup device
    device = resolve_device(training_args.device)
    logger.info(f"Using device: {device}")

    # Store results for all folds
    multimodal_results: list[dict[str, Any]] = []
    baseline_finetuned_results: list[dict[str, Any]] = []

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
            patch_len=training_args.patch_len,
            context_len=training_args.context_len,
            horizon_len=training_args.horizon_len,
        )

        # Train multimodal model
        try:
            logger.info("=" * 50)
            logger.info("TRAINING MULTIMODAL MODEL")
            logger.info("=" * 50)
            model = create_multimodal_model(model_config, device)
            checkpoint_path, val_metrics = train_multimodal_fold(
                args=training_args,
                fold_idx=fold_idx,
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            logger.info(f"Multimodal fold {fold_idx} completed. Checkpoint: {checkpoint_path}")
            logger.info(f"Multimodal fold {fold_idx} validation metrics: {val_metrics}")

            fold_results = {
                "fold": fold_idx,
                "checkpoint": str(checkpoint_path),
                "train_domains": train_domains,
                "val_domains": val_domains,
                "test_domains": test_domains,
                "val_metrics": val_metrics,
            }
            multimodal_results.append(fold_results)

        except Exception as e:
            logger.error(f"Multimodal fold {fold_idx} training failed: {e}")
            return 1

        # Train fine-tuned baseline if requested
        if parsed_args.train_baseline:
            try:
                logger.info("=" * 50)
                logger.info("TRAINING BASELINE (FINE-TUNED)")
                logger.info("=" * 50)
                baseline_model = create_baseline_model(model_config, load_pretrained=True)
                checkpoint_path, val_metrics = train_baseline_fold(
                    args=training_args,
                    fold_idx=fold_idx,
                    model=baseline_model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                )
                logger.info(f"Baseline (fine-tuned) fold {fold_idx} completed. Checkpoint: {checkpoint_path}")
                logger.info(f"Baseline (fine-tuned) fold {fold_idx} validation metrics: {val_metrics}")

                fold_results = {
                    "fold": fold_idx,
                    "checkpoint": str(checkpoint_path),
                    "train_domains": train_domains,
                    "val_domains": val_domains,
                    "test_domains": test_domains,
                    "val_metrics": val_metrics,
                }
                baseline_finetuned_results.append(fold_results)

            except Exception as e:
                logger.error(f"Baseline (fine-tuned) fold {fold_idx} training failed: {e}")
                return 1

    # Save cross-validation results for each model type
    logger.info("=" * 50)
    logger.info("Saving cross-validation results")
    logger.info("=" * 50)

    if multimodal_results:
        cv_results_path = training_args.logging_dir / "cv_results.json"
        with open(cv_results_path, "w") as f:
            json.dump(multimodal_results, f, indent=2)
        logger.info(f"Multimodal CV results saved to {cv_results_path}")

        avg_val_loss = sum(r["val_metrics"]["val_loss"] for r in multimodal_results) / len(multimodal_results)
        logger.info(f"Multimodal average validation loss: {avg_val_loss:.6f}")

    if baseline_finetuned_results:
        baseline_cv_path = training_args.logging_dir / "baseline_finetuned_cv_results.json"
        with open(baseline_cv_path, "w") as f:
            json.dump(baseline_finetuned_results, f, indent=2)
        logger.info(f"Baseline (fine-tuned) CV results saved to {baseline_cv_path}")

        avg_val_loss = sum(r["val_metrics"]["val_loss"] for r in baseline_finetuned_results) / len(
            baseline_finetuned_results
        )
        logger.info(f"Baseline (fine-tuned) average validation loss: {avg_val_loss:.6f}")

    logger.info("=" * 50)
    logger.info("Cross-validation training completed successfully!")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
