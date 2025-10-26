#!/usr/bin/env python3
"""Visualization script for multimodal TimesFM predictions on Time-MMD dataset.

This script generates predictions using trained models and visualizes them with matplotlib,
comparing multimodal model predictions against baseline model predictions and ground truth.

Usage:
    # Visualize all folds from cross-validation results
    PYTHONPATH=. uv run python scripts/visualize_predictions.py \\
        --cv-results logs/cv_results.json \\
        --num-samples 5 \\
        --output-dir visualizations

    # Visualize a specific fold only
    PYTHONPATH=. uv run python scripts/visualize_predictions.py \\
        --cv-results logs/cv_results.json \\
        --fold 0 \\
        --num-samples 3 \\
        --output-dir visualizations/fold_0

    # With custom configurations
    PYTHONPATH=. uv run python scripts/visualize_predictions.py \\
        --cv-results logs/cv_results.json \\
        --model-config examples/time_mmd/configs/model.yaml \\
        --training-config examples/time_mmd/configs/training.yaml \\
        --num-samples 10 \\
        --batch-size 16 \\
        --seed 42

Output:
    The script generates two types of visualizations for each fold:
    - fold_N_predictions.png: Time series plots showing context, ground truth, and predictions
    - fold_N_metrics_comparison.png: Bar charts comparing MSE and MAE metrics
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from timesfm.pytorch_patched_decoder import TimesFMConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.time_mmd.configs import ModelConfig, TrainingConfig
from examples.time_mmd.data.cross_validation import create_fold_datasets
from multimodal_timesfm.multimodal_patched_decoder import MultimodalTimesFMConfig
from multimodal_timesfm.utils.collate import multimodal_collate_fn
from multimodal_timesfm.utils.device import get_pin_memory, move_to_device, resolve_device
from multimodal_timesfm.utils.logging import get_logger, setup_logger
from multimodal_timesfm.utils.model import create_baseline_timesfm_model, load_multimodal_checkpoint
from multimodal_timesfm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize predictions on Time-MMD dataset")

    parser.add_argument(
        "--cv-results",
        type=str,
        required=True,
        help="Path to cross-validation results JSON file",
    )

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
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for prediction generation",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize per fold",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization plots",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (if not provided, no seed will be set)",
    )

    parser.add_argument(
        "--fold",
        type=int,
        help="Specific fold to visualize (if not provided, all folds will be visualized)",
    )

    return parser.parse_args()


def generate_predictions(
    multimodal_model: Any,
    baseline_model: Any,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
    num_samples: int,
) -> dict[str, Any]:
    """Generate predictions from both multimodal and baseline models.

    Args:
        multimodal_model: Multimodal TimesFM model instance.
        baseline_model: Baseline TimesFM model instance.
        dataloader: DataLoader providing batches of samples.
        device: Device to use for prediction.
        num_samples: Maximum number of samples to predict.

    Returns:
        Dictionary containing predictions, ground truth, and metadata.
    """
    multimodal_model.eval()
    baseline_model.eval()
    multimodal_model.to(device)
    baseline_model.to(device)

    results: dict[str, Any] = {
        "contexts": [],
        "futures": [],
        "multimodal_predictions": [],
        "baseline_predictions": [],
        "texts": [],
        "domains": [],
    }

    samples_collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            if samples_collected >= num_samples:
                break

            batch_tensors = move_to_device(
                {"context": batch["context"], "future": batch["future"], "freq": batch["freq"]}, device
            )
            context = batch_tensors["context"]
            future = batch_tensors["future"]
            freq = batch_tensors["freq"]
            patched_texts = batch["patched_texts"]

            # Create input_padding tensor
            input_padding = torch.zeros_like(context)

            # Generate predictions from multimodal model
            multimodal_preds = multimodal_model(context, input_padding.float(), freq, patched_texts)
            multimodal_preds_mean = multimodal_preds[..., 0][:, -1, :]  # [B, horizon_len]

            # Generate predictions from baseline model
            baseline_preds = baseline_model(context, input_padding.float(), freq)
            baseline_preds_mean = baseline_preds[..., 0][:, -1, :]  # [B, horizon_len]

            # Store results
            batch_size = context.size(0)
            for i in range(batch_size):
                results["contexts"].append(context[i].cpu().numpy())
                results["futures"].append(future[i].cpu().numpy())
                results["multimodal_predictions"].append(multimodal_preds_mean[i].cpu().numpy())
                results["baseline_predictions"].append(baseline_preds_mean[i].cpu().numpy())
                results["texts"].append(patched_texts[i] if patched_texts else [""] * len(context[i]))
                results["domains"].append(batch["metadata"][i]["domain"])

                samples_collected += 1

    return results


def plot_predictions(
    results: dict[str, Any],
    fold_idx: int,
    output_dir: Path,
) -> Path:
    """Create visualization plots comparing multimodal and baseline predictions.

    Args:
        results: Dictionary containing predictions and ground truth.
        fold_idx: Fold index for labeling.
        output_dir: Directory to save plots.
    """
    num_samples = len(results["contexts"])

    # Create figure with subplots
    _, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for idx in range(num_samples):
        ax = axes[idx]

        context = results["contexts"][idx]
        future = results["futures"][idx]
        multimodal_pred = results["multimodal_predictions"][idx]
        baseline_pred = results["baseline_predictions"][idx]
        domain = results["domains"][idx]

        context_len = len(context)
        horizon_len = len(future)

        # Create x-axis positions
        context_x = np.arange(context_len)
        future_x = np.arange(context_len, context_len + horizon_len)

        # Plot context (historical data)
        ax.plot(context_x, context, "k-", label="Context (Historical)", linewidth=2)

        # Plot ground truth future
        ax.plot(future_x, future, "g-", label="Ground Truth", linewidth=2, marker="o", markersize=4)

        # Plot multimodal predictions
        ax.plot(future_x, multimodal_pred, "b--", label="Multimodal Model", linewidth=2, marker="s", markersize=4)

        # Plot baseline predictions
        ax.plot(future_x, baseline_pred, "r--", label="Baseline Model", linewidth=2, marker="^", markersize=4)

        # Add vertical line separating context and future
        ax.axvline(x=context_len - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        # Calculate metrics for this sample
        multimodal_mse = np.mean((multimodal_pred - future) ** 2)
        baseline_mse = np.mean((baseline_pred - future) ** 2)
        multimodal_mae = np.mean(np.abs(multimodal_pred - future))
        baseline_mae = np.mean(np.abs(baseline_pred - future))

        # Set title with domain and metrics
        ax.set_title(
            f"Sample {idx + 1} - Domain: {domain}\n"
            f"Multimodal: MSE={multimodal_mse:.4f}, MAE={multimodal_mae:.4f} | "
            f"Baseline: MSE={baseline_mse:.4f}, MAE={baseline_mae:.4f}",
            fontsize=10,
        )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"fold_{fold_idx}_predictions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def plot_metrics_comparison(
    results: dict[str, Any],
    fold_idx: int,
    output_dir: Path,
) -> Path:
    """Create bar plot comparing metrics between multimodal and baseline models.

    Args:
        results: Dictionary containing predictions and ground truth.
        fold_idx: Fold index for labeling.
        output_dir: Directory to save plots.
    """
    num_samples = len(results["contexts"])

    multimodal_mses = []
    baseline_mses = []
    multimodal_maes = []
    baseline_maes = []

    for idx in range(num_samples):
        future = results["futures"][idx]
        multimodal_pred = results["multimodal_predictions"][idx]
        baseline_pred = results["baseline_predictions"][idx]

        multimodal_mses.append(np.mean((multimodal_pred - future) ** 2))
        baseline_mses.append(np.mean((baseline_pred - future) ** 2))
        multimodal_maes.append(np.mean(np.abs(multimodal_pred - future)))
        baseline_maes.append(np.mean(np.abs(baseline_pred - future)))

    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(num_samples)
    width = 0.35

    # MSE comparison
    ax1.bar(x - width / 2, multimodal_mses, width, label="Multimodal", color="blue", alpha=0.7)
    ax1.bar(x + width / 2, baseline_mses, width, label="Baseline", color="red", alpha=0.7)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("MSE")
    ax1.set_title(f"Fold {fold_idx}: MSE Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i + 1}" for i in range(num_samples)])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # MAE comparison
    ax2.bar(x - width / 2, multimodal_maes, width, label="Multimodal", color="blue", alpha=0.7)
    ax2.bar(x + width / 2, baseline_maes, width, label="Baseline", color="red", alpha=0.7)
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("MAE")
    ax2.set_title(f"Fold {fold_idx}: MAE Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{i + 1}" for i in range(num_samples)])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"fold_{fold_idx}_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def main() -> int:
    """Main visualization function."""
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
    setup_logger()

    logger = get_logger()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Visualizations will be saved to: {output_dir}")

    # Load cross-validation results
    cv_results_path = Path(args.cv_results)
    if not cv_results_path.exists():
        logger.error(f"Cross-validation results file not found: {cv_results_path}")
        return 1

    with open(cv_results_path) as f:
        cv_results = json.load(f)

    logger.info(f"Loaded cross-validation results from {cv_results_path}")

    # Setup device
    device = resolve_device()
    logger.info(f"Using device: {device}")

    # Create baseline model
    logger.info("Creating baseline TimesFM model...")
    baseline_config = TimesFMConfig(
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
    baseline_model = create_baseline_timesfm_model(baseline_config, load_pretrained=True)

    # Filter folds if specific fold is requested
    folds_to_process = cv_results if args.fold is None else [f for f in cv_results if f["fold"] == args.fold]

    if not folds_to_process:
        logger.error(f"Fold {args.fold} not found in cross-validation results")
        return 1

    # Process each fold
    all_visualization_paths = []
    data_path = Path(training_config.data.data_path)

    for fold_data in folds_to_process:
        fold_idx = fold_data["fold"]
        logger.info("=" * 50)
        logger.info(f"Visualizing fold {fold_idx}")
        logger.info("=" * 50)

        # Load test dataset for this fold
        _, _, test_dataset = create_fold_datasets(
            data_path=data_path,
            train_domains=fold_data["train_domains"],
            val_domains=fold_data["val_domains"],
            test_domains=fold_data["test_domains"],
            patch_len=training_config.data.patch_len,
            context_len=training_config.data.context_len,
            horizon_len=training_config.data.horizon_len,
        )

        logger.info(f"Test samples: {len(test_dataset)}")

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=multimodal_collate_fn,
            pin_memory=get_pin_memory(device),
        )

        # Load multimodal model for this fold
        checkpoint_path = Path(fold_data["checkpoint"])
        multimodal_config = MultimodalTimesFMConfig(
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
        multimodal_model = load_multimodal_checkpoint(checkpoint_path, multimodal_config, device)

        # Generate predictions
        logger.info(f"Generating predictions for {args.num_samples} samples...")
        results = generate_predictions(
            multimodal_model,
            baseline_model,
            test_dataloader,
            device,
            args.num_samples,
        )

        # Create visualizations
        logger.info("Creating prediction visualizations...")
        pred_plot_path = plot_predictions(results, fold_idx, output_dir)
        logger.info(f"Saved prediction plot: {pred_plot_path}")

        logger.info("Creating metrics comparison plot...")
        metrics_plot_path = plot_metrics_comparison(results, fold_idx, output_dir)
        logger.info(f"Saved metrics comparison plot: {metrics_plot_path}")

        all_visualization_paths.extend([pred_plot_path, metrics_plot_path])

    logger.info("=" * 50)
    logger.info("Visualization completed successfully!")
    logger.info(f"Total plots created: {len(all_visualization_paths)}")
    logger.info(f"Plots saved in: {output_dir}")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
