#!/usr/bin/env python3
"""WandB Sweep script for hyperparameter tuning of multimodal TimesFM on Time-MMD dataset.

This script uses Climate as validation dataset, Energy as test dataset, and all other
datasets as training dataset for hyperparameter optimization.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.data.cross_validation import create_fold_datasets, get_all_domains
from multimodal_timesfm.evaluation import evaluate_multimodal_model
from multimodal_timesfm.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from multimodal_timesfm.trainer import MultimodalTrainer
from multimodal_timesfm.training_args import TrainingArguments
from multimodal_timesfm.utils.collate import multimodal_collate_fn
from multimodal_timesfm.utils.device import get_pin_memory, resolve_device
from multimodal_timesfm.utils.logging import get_logger, setup_logger
from multimodal_timesfm.utils.model import create_multimodal_model as create_multimodal_model_core
from multimodal_timesfm.utils.seed import set_seed
from multimodal_timesfm.utils.yaml import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for multimodal TimesFM using WandB Sweeps on Time-MMD dataset"
    )

    parser.add_argument(
        "--sweep-config",
        type=str,
        help="Path to WandB sweep configuration file (YAML)",
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
        "--sweep-id",
        type=str,
        help="Existing WandB sweep ID to continue (if not provided, creates a new sweep)",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs to execute (if not provided, runs indefinitely until sweep completes)",
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


def train_and_evaluate(
    base_training_args: TrainingArguments,
    model_config: ModelConfig,
    data_path: Path,
    train_domains: list[str],
    val_domains: list[str],
    test_domains: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Train model with hyperparameters from WandB config and evaluate on test dataset.

    Args:
        base_training_args: Base training arguments (will be overridden by wandb.config).
        model_config: Model configuration.
        data_path: Path to Time-MMD dataset.
        train_domains: List of training domain names.
        val_domains: List of validation domain names (Climate).
        test_domains: List of test domain names (Energy).
        device: Device to train on.

    Returns:
        Dictionary of test metrics (MSE and MAE).
    """
    logger = get_logger()

    # Get hyperparameters from wandb.config (set by sweep agent)
    config = wandb.config

    # Create a unique run name for this sweep trial
    run_name = f"sweep_{config.learning_rate:.0e}_bs{config.batch_size}_wd{config.weight_decay:.0e}"

    # Override training args with sweep hyperparameters
    training_args = replace(
        base_training_args,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        run_name=run_name,
        output_dir=str(Path(base_training_args.output_dir) / run_name),
    )

    logger.info("Training with hyperparameters:")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Weight decay: {training_args.weight_decay}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Num epochs: {training_args.num_train_epochs}")

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_fold_datasets(
        data_path=data_path,
        train_domains=train_domains,
        val_domains=val_domains,
        test_domains=test_domains,
        patch_len=training_args.patch_len,
        context_len=training_args.context_len,
        horizon_len=training_args.horizon_len,
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create model
    model = create_multimodal_model(model_config, device)

    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        init_wandb=False,
    )

    # Freeze pretrained parameters (only train fusion)
    trainer.freeze_pretrained_parameters()

    logger.info(f"Training fusion components only for {training_args.num_train_epochs} epochs")

    # Train
    trainer.train()

    # Load best model checkpoint for evaluation
    best_checkpoint = training_args.checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(best_checkpoint, weights_only=True)
    val_loss = checkpoint["best_val_loss"]
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Training completed. Best validation loss: {val_loss:.6f}")
    logger.info("Evaluating best model on test dataset (Energy)...")

    # Create test dataloader for evaluation
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multimodal_collate_fn,
        pin_memory=get_pin_memory(device),
    )

    # Evaluate on test dataset using the same logic as evaluate_time_mmd_cv.py
    test_metrics = evaluate_multimodal_model(model, test_dataloader, device)

    logger.info(f"Test metrics: MSE={test_metrics['mse']:.6f}, MAE={test_metrics['mae']:.6f}")

    # Log metrics to WandB
    wandb.log(
        {
            "val_loss": val_loss,
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
        }
    )

    return {
        "val_loss": val_loss,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
    }


def main() -> int:
    """Main sweep training function."""
    parsed_args = parse_args()

    # Load model configuration
    if parsed_args.model_config:
        model_config = ModelConfig.from_yaml(Path(parsed_args.model_config))
    else:
        model_config = ModelConfig()

    # Base training arguments (will be overridden by sweep config)
    # Use values from model_config for context_len and horizon_len
    base_training_args = TrainingArguments(
        output_dir="outputs/sweep",
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,
        seed=parsed_args.seed,
        context_len=model_config.timesfm.context_len,
        horizon_len=model_config.timesfm.horizon_len,
    )

    # Set random seed for reproducibility if provided
    if parsed_args.seed is not None:
        set_seed(parsed_args.seed)

    # Setup logging
    setup_logger(log_file=base_training_args.logging_dir / "sweep_training.log")

    logger = get_logger()
    logger.info("Starting WandB Sweep for hyperparameter tuning on Time-MMD dataset")
    logger.info(f"Model config: {parsed_args.model_config}")
    logger.info("Data split: Climate (val), Energy (test), others (train)")

    # Get all available domains
    data_path = Path(parsed_args.data_path)
    all_domains = get_all_domains(data_path)
    logger.info(f"Found {len(all_domains)} domains in dataset: {all_domains}")

    # Define fixed split: Climate=val, Energy=test, others=train
    val_domains = ["Climate"]
    test_domains = ["Energy"]
    train_domains = [d for d in all_domains if d not in val_domains and d not in test_domains]

    logger.info(f"Train domains ({len(train_domains)}): {train_domains}")
    logger.info(f"Validation domains ({len(val_domains)}): {val_domains}")
    logger.info(f"Test domains ({len(test_domains)}): {test_domains}")

    # Setup device
    device = resolve_device(base_training_args.device)
    logger.info(f"Using device: {device}")

    # Define function for each sweep run
    def run() -> None:
        """Execute a single hyperparameter configuration trial."""
        with wandb.init(project="multimodal-timesfm-time-mmd-sweep"):
            logger.info(f"Starting sweep run: {wandb.run.id}")
            logger.info(f"Hyperparameters: {dict(wandb.config)}")

            try:
                metrics = train_and_evaluate(
                    base_training_args=base_training_args,
                    model_config=model_config,
                    data_path=data_path,
                    train_domains=train_domains,
                    val_domains=val_domains,
                    test_domains=test_domains,
                    device=device,
                )

                logger.info(f"Sweep run {wandb.run.id} completed. Test metrics: {metrics}")

            except Exception as e:
                logger.error(f"Sweep run {wandb.run.id} failed: {e}")
                raise

    # Initialize or continue sweep
    if parsed_args.sweep_id:
        # Continue existing sweep
        sweep_id = parsed_args.sweep_id
        logger.info(f"Continuing existing sweep: {sweep_id}")
    else:
        # Create new sweep from config file
        if not parsed_args.sweep_config:
            logger.error("Must provide either --sweep-config or --sweep-id")
            return 1

        # Load sweep configuration
        sweep_config = load_yaml(Path(parsed_args.sweep_config))

        logger.info(f"Creating new sweep with config from: {parsed_args.sweep_config}")
        logger.info(f"Sweep configuration: {json.dumps(sweep_config, indent=2)}")

        # Create sweep
        sweep_id = wandb.sweep(sweep=sweep_config, project="multimodal-timesfm-time-mmd-sweep")
        logger.info(f"Created new sweep: {sweep_id}")

    # Run sweep agent
    logger.info(f"Starting sweep agent (count={parsed_args.count})")
    wandb.agent(sweep_id, function=run, count=parsed_args.count, project="multimodal-timesfm-time-mmd-sweep")

    logger.info("Sweep completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())
