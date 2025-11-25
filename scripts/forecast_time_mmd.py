#!/usr/bin/env python3
"""Forecasting script for multimodal TimesFM on Time-MMD dataset."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from timesfm import TimesFm, TimesFmCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.data.cross_validation import create_fold_datasets
from multimodal_timesfm.multimodal_patched_decoder import MultimodalTimesFMConfig
from multimodal_timesfm.multimodal_timesfm import MultimodalTimesFM, TimesFmHparams
from multimodal_timesfm.training_args import TrainingArguments
from multimodal_timesfm.utils.collate import multimodal_collate_fn
from multimodal_timesfm.utils.device import get_pin_memory, resolve_device
from multimodal_timesfm.utils.logging import get_logger, setup_logger
from multimodal_timesfm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run forecasting with MultimodalTimesFM on Time-MMD dataset")

    parser.add_argument(
        "--context-len",
        type=int,
        required=True,
        help="Context length for forecasting",
    )

    parser.add_argument(
        "--horizon-len",
        type=int,
        required=True,
        help="Horizon length for forecasting",
    )

    parser.add_argument(
        "--cv-results",
        type=str,
        required=True,
        help="Path to cross-validation results JSON file",
    )

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
        "--fold",
        type=int,
        help="Specific fold to use for forecasting (if not provided, all folds will be processed)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize per fold",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for forecasting",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (if not provided, no seed will be set)",
    )

    return parser.parse_args()


def generate_forecasts(
    multimodal_model: MultimodalTimesFM,
    baseline_model: TimesFm,
    dataloader: DataLoader[dict[str, Any]],
    context_len: int,
) -> dict[str, Any]:
    """Generate forecasts from both models using forecast method.

    Args:
        multimodal_model: Multimodal TimesFM wrapper.
        baseline_model: Baseline TimesFM wrapper.
        dataloader: DataLoader providing samples.
        context_len: Context length for forecasting.

    Returns:
        Dictionary containing forecasts and ground truth.
    """
    results: dict[str, Any] = {
        "contexts": [],
        "futures": [],
        "multimodal_forecasts": [],
        "baseline_forecasts": [],
    }

    for batch in tqdm(dataloader, desc="Generating forecasts"):
        # Extract data from batch
        contexts = batch["context"].cpu().numpy()  # [B, context_len]
        futures = batch["future"].cpu().numpy()  # [B, horizon_len]
        freqs = batch["freq"].cpu().numpy().squeeze(-1).tolist()  # List of frequency values
        patched_texts = batch["patched_texts"]  # [B][patch_idx][text_list]

        batch_size = contexts.shape[0]

        # Convert contexts to list of arrays for forecast API
        context_list = [contexts[i] for i in range(batch_size)]

        # Generate multimodal forecasts using forecast method
        multimodal_forecasts, _ = multimodal_model.forecast(
            inputs=context_list,
            text_descriptions=patched_texts,
            freq=freqs,
            forecast_context_len=context_len,
        )

        # Generate baseline forecasts using original TimesFM forecast method (no text)
        baseline_forecasts, _ = baseline_model.forecast(
            inputs=context_list,
            freq=freqs,
            forecast_context_len=context_len,
        )

        # Collect results
        for i in range(batch_size):
            results["contexts"].append(contexts[i])
            results["futures"].append(futures[i])
            results["multimodal_forecasts"].append(multimodal_forecasts[i])
            results["baseline_forecasts"].append(baseline_forecasts[i])

    return results


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics."""
    mse = float(np.mean((predictions - ground_truth) ** 2))
    mae = float(np.mean(np.abs(predictions - ground_truth)))
    return {"mse": mse, "mae": mae}


def plot_forecasts(
    results: dict[str, Any],
    output_dir: Path,
    fold: int,
    num_samples: int,
) -> None:
    """Create visualization plots for forecasts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = min(num_samples, len(results["contexts"]))

    # Compute metrics for each sample
    mm_metrics = []
    bl_metrics = []
    for i in range(len(results["futures"])):
        mm_metrics.append(compute_metrics(results["multimodal_forecasts"][i], results["futures"][i]))
        bl_metrics.append(compute_metrics(results["baseline_forecasts"][i], results["futures"][i]))

    # Time series plots
    _, axes = plt.subplots(num_samples, 1, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for idx in range(num_samples):
        ax = axes[idx]
        context = results["contexts"][idx]
        future = results["futures"][idx]
        mm_forecast = results["multimodal_forecasts"][idx]
        bl_forecast = results["baseline_forecasts"][idx]

        context_len = len(context)
        horizon_len = len(future)
        context_x = np.arange(context_len)
        future_x = np.arange(context_len, context_len + horizon_len)

        ax.plot(context_x, context, "k-", label="Context", linewidth=2, alpha=0.8)
        ax.plot(future_x, future, "g-", label="Ground Truth", linewidth=2.5, marker="o", markersize=5)
        ax.plot(future_x, mm_forecast, "b--", label="Multimodal", linewidth=2.5, marker="s", markersize=5, alpha=0.8)
        ax.plot(future_x, bl_forecast, "r--", label="Baseline", linewidth=2.5, marker="^", markersize=5, alpha=0.8)
        ax.axvline(x=context_len - 0.5, color="gray", linestyle=":", linewidth=2, alpha=0.7)

        ax.set_title(
            f"Sample {idx + 1}\n"
            f"Multimodal: MSE={mm_metrics[idx]['mse']:.4f}, MAE={mm_metrics[idx]['mae']:.4f} | "
            f"Baseline: MSE={bl_metrics[idx]['mse']:.4f}, MAE={bl_metrics[idx]['mae']:.4f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / f"fold_{fold}_forecasts.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Metrics comparison
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    mm_mse = [mm_metrics[i]["mse"] for i in range(num_samples)]
    mm_mae = [mm_metrics[i]["mae"] for i in range(num_samples)]
    bl_mse = [bl_metrics[i]["mse"] for i in range(num_samples)]
    bl_mae = [bl_metrics[i]["mae"] for i in range(num_samples)]

    x = np.arange(num_samples)
    width = 0.35

    ax1.bar(x - width / 2, mm_mse, width, label="Multimodal", color="blue", alpha=0.7)
    ax1.bar(x + width / 2, bl_mse, width, label="Baseline", color="red", alpha=0.7)
    ax1.set_xlabel("Sample Index", fontsize=11)
    ax1.set_ylabel("MSE", fontsize=11)
    ax1.set_title("MSE Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(range(1, num_samples + 1))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x - width / 2, mm_mae, width, label="Multimodal", color="blue", alpha=0.7)
    ax2.bar(x + width / 2, bl_mae, width, label="Baseline", color="red", alpha=0.7)
    ax2.set_xlabel("Sample Index", fontsize=11)
    ax2.set_ylabel("MAE", fontsize=11)
    ax2.set_title("MAE Comparison", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(range(1, num_samples + 1))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / f"fold_{fold}_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    """Main forecasting function."""
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

    # Set random seed for reproducibility if provided
    if parsed_args.seed is not None:
        set_seed(parsed_args.seed)

    # Setup logging
    setup_logger()

    logger = get_logger()

    # Create output directory for forecasts
    output_dir = Path(training_args.output_dir) / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cross-validation results
    cv_results_path = Path(parsed_args.cv_results)
    if not cv_results_path.exists():
        logger.error(f"Cross-validation results file not found: {cv_results_path}")
        return 1

    with open(cv_results_path) as f:
        cv_results = json.load(f)

    # Setup device
    device = resolve_device()
    logger.info(f"Using device: {device}")

    # Filter folds if specific fold is requested
    folds_to_process = (
        cv_results if parsed_args.fold is None else [f for f in cv_results if f["fold"] == parsed_args.fold]
    )

    if not folds_to_process:
        logger.error(f"Fold {parsed_args.fold} not found in cross-validation results")
        return 1

    logger.info(f"Context length: {parsed_args.context_len}, Horizon length: {parsed_args.horizon_len}")

    # Create TimesFM hyperparameters for wrappers
    # Note: per_core_batch_size must divide evenly into the actual batch size
    # Setting it to 1 allows any batch size
    hparams = TimesFmHparams(
        context_len=parsed_args.context_len,
        horizon_len=parsed_args.horizon_len,
        input_patch_len=model_config.timesfm.input_patch_len,
        output_patch_len=model_config.timesfm.output_patch_len,
        num_layers=model_config.timesfm.num_layers,
        num_heads=model_config.timesfm.num_heads,
        model_dims=model_config.timesfm.model_dims,
        per_core_batch_size=1,  # Set to 1 to allow any batch size
        backend="gpu",
        quantiles=model_config.timesfm.quantiles,
        use_positional_embedding=model_config.timesfm.use_positional_embedding,
    )

    # Process each fold
    for fold_data in folds_to_process:
        fold_idx = fold_data["fold"]
        checkpoint_path = Path(fold_data["checkpoint"])
        test_domains = fold_data["test_domains"]

        logger.info("=" * 50)
        logger.info(f"Processing fold {fold_idx}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Test domains: {test_domains}")
        logger.info("=" * 50)

        # Create datasets with custom context/horizon lengths
        # We only need test_dataset, but create_fold_datasets requires non-empty train_domains
        # Use fold data to get all domains
        train_domains = fold_data["train_domains"]
        val_domains = fold_data["val_domains"]

        _, _, test_dataset = create_fold_datasets(
            data_path=Path(parsed_args.data_path),
            train_domains=train_domains,
            val_domains=val_domains,
            test_domains=test_domains,
            patch_len=training_args.patch_len,
            context_len=parsed_args.context_len,
            horizon_len=parsed_args.horizon_len,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=parsed_args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=multimodal_collate_fn,
            pin_memory=get_pin_memory(device),
        )

        # Create multimodal model wrapper for this fold
        logger.info("Loading multimodal model...")
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
        multimodal_model = MultimodalTimesFM(hparams, multimodal_config, checkpoint_path, device)

        # Create baseline TimesFM model (without text inputs)
        logger.info("Creating baseline TimesFM model...")
        baseline_model = TimesFm(
            hparams=hparams,
            checkpoint=TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )

        # Run forecasting
        logger.info("Generating forecasts...")
        results = generate_forecasts(multimodal_model, baseline_model, test_loader, parsed_args.context_len)

        # Compute summary metrics
        mm_metrics = [
            compute_metrics(results["multimodal_forecasts"][i], results["futures"][i])
            for i in range(len(results["contexts"]))
        ]
        bl_metrics = [
            compute_metrics(results["baseline_forecasts"][i], results["futures"][i])
            for i in range(len(results["contexts"]))
        ]

        mm_avg_mse = np.mean([m["mse"] for m in mm_metrics])
        mm_avg_mae = np.mean([m["mae"] for m in mm_metrics])
        bl_avg_mse = np.mean([m["mse"] for m in bl_metrics])
        bl_avg_mae = np.mean([m["mae"] for m in bl_metrics])

        logger.info("=" * 50)
        logger.info(f"Fold {fold_idx} - Multimodal model metrics:")
        logger.info(f"  Average MSE: {mm_avg_mse:.6f}")
        logger.info(f"  Average MAE: {mm_avg_mae:.6f}")
        logger.info(f"Fold {fold_idx} - Baseline model metrics:")
        logger.info(f"  Average MSE: {bl_avg_mse:.6f}")
        logger.info(f"  Average MAE: {bl_avg_mae:.6f}")
        logger.info(f"Fold {fold_idx} - Improvement (Multimodal vs Baseline):")
        logger.info(f"  MSE: {((bl_avg_mse - mm_avg_mse) / bl_avg_mse) * 100:+.2f}%")
        logger.info(f"  MAE: {((bl_avg_mae - mm_avg_mae) / bl_avg_mae) * 100:+.2f}%")
        logger.info("=" * 50)

        # Generate plots
        logger.info("Generating visualization plots...")
        plot_forecasts(results, output_dir, fold_idx, parsed_args.num_samples)
        logger.info(f"Plots saved to: {output_dir}")

        # Save forecasts
        output_data = {
            "metadata": {
                "fold": fold_idx,
                "checkpoint": str(checkpoint_path),
                "domains": test_domains,
                "context_len": parsed_args.context_len,
                "horizon_len": parsed_args.horizon_len,
                "num_samples": len(results["contexts"]),
            },
            "summary_metrics": {
                "multimodal": {"mse": float(mm_avg_mse), "mae": float(mm_avg_mae)},
                "baseline": {"mse": float(bl_avg_mse), "mae": float(bl_avg_mae)},
            },
            "samples": [
                {
                    "context": results["contexts"][i].tolist(),
                    "future": results["futures"][i].tolist(),
                    "multimodal_forecast": results["multimodal_forecasts"][i].tolist(),
                    "baseline_forecast": results["baseline_forecasts"][i].tolist(),
                    "multimodal_metrics": mm_metrics[i],
                    "baseline_metrics": bl_metrics[i],
                }
                for i in range(len(results["contexts"]))
            ],
        }

        output_path = output_dir / f"fold_{fold_idx}_forecasts.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Forecasts saved to: {output_path}")

    logger.info("=" * 50)
    logger.info("Forecasting completed successfully!")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
