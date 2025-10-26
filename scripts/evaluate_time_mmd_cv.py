#!/usr/bin/env python3
"""Cross-validation evaluation script for multimodal TimesFM on Time-MMD dataset."""

import argparse
import json
from pathlib import Path
from typing import Any

from timesfm.pytorch_patched_decoder import TimesFMConfig
from torch.utils.data import DataLoader

from examples.time_mmd.configs import ModelConfig, TrainingConfig
from examples.time_mmd.data.cross_validation import create_fold_datasets
from multimodal_timesfm.evaluation import evaluate_baseline_model, evaluate_multimodal_model
from multimodal_timesfm.multimodal_patched_decoder import MultimodalTimesFMConfig
from multimodal_timesfm.utils.collate import multimodal_collate_fn
from multimodal_timesfm.utils.device import get_pin_memory, resolve_device
from multimodal_timesfm.utils.logging import get_logger, setup_logger
from multimodal_timesfm.utils.model import create_baseline_timesfm_model, load_multimodal_checkpoint
from multimodal_timesfm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate cross-validation results on Time-MMD dataset")

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
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (if not provided, no seed will be set)",
    )

    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline TimesFM model (without text)",
    )

    return parser.parse_args()


def main() -> int:
    """Main cross-validation evaluation function."""
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

    # Load cross-validation results
    cv_results_path = Path(args.cv_results)
    if not cv_results_path.exists():
        logger.error(f"Cross-validation results file not found: {cv_results_path}")
        return 1

    with open(cv_results_path) as f:
        cv_results = json.load(f)

    logger.info(f"Loaded cross-validation results from {cv_results_path}")
    logger.info(f"Number of folds: {len(cv_results)}")

    # Setup device
    device = resolve_device()
    logger.info(f"Using device: {device}")

    # Create baseline model if needed
    baseline_model = None
    if args.compare_baseline:
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

    # Evaluate each fold
    fold_results = []
    data_path = Path(training_config.data.data_path)

    for fold_data in cv_results:
        fold_idx = fold_data["fold"]
        logger.info("=" * 50)
        logger.info(f"Evaluating fold {fold_idx}")
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

        # Evaluate multimodal model
        logger.info("Evaluating multimodal model...")
        multimodal_metrics = evaluate_multimodal_model(multimodal_model, test_dataloader, device)
        logger.info(f"Multimodal metrics: MSE={multimodal_metrics['mse']:.6f}, MAE={multimodal_metrics['mae']:.6f}")

        # Evaluate baseline if requested
        baseline_metrics = None
        if baseline_model is not None:
            logger.info("Evaluating baseline model...")
            baseline_metrics = evaluate_baseline_model(baseline_model, test_dataloader, device)
            logger.info(f"Baseline metrics: MSE={baseline_metrics['mse']:.6f}, MAE={baseline_metrics['mae']:.6f}")

        # Store results
        result = {
            "fold": fold_idx,
            "test_domains": fold_data["test_domains"],
            "num_test_samples": len(test_dataset),
            "multimodal": multimodal_metrics,
        }
        if baseline_metrics is not None:
            result["baseline"] = baseline_metrics
        fold_results.append(result)

    # Compute average metrics across folds
    avg_multimodal_mse = sum(r["multimodal"]["mse"] for r in fold_results) / len(fold_results)
    avg_multimodal_mae = sum(r["multimodal"]["mae"] for r in fold_results) / len(fold_results)

    logger.info("=" * 50)
    logger.info("Cross-validation evaluation summary:")
    logger.info("=" * 50)
    logger.info("Multimodal model:")
    logger.info(f"  Average test MSE: {avg_multimodal_mse:.6f}")
    logger.info(f"  Average test MAE: {avg_multimodal_mae:.6f}")

    if args.compare_baseline:
        avg_baseline_mse = sum(r["baseline"]["mse"] for r in fold_results) / len(fold_results)
        avg_baseline_mae = sum(r["baseline"]["mae"] for r in fold_results) / len(fold_results)

        logger.info("Baseline model:")
        logger.info(f"  Average test MSE: {avg_baseline_mse:.6f}")
        logger.info(f"  Average test MAE: {avg_baseline_mae:.6f}")

        logger.info("Improvement:")
        mse_improvement = ((avg_baseline_mse - avg_multimodal_mse) / avg_baseline_mse) * 100
        mae_improvement = ((avg_baseline_mae - avg_multimodal_mae) / avg_baseline_mae) * 100
        logger.info(f"  MSE improvement: {mse_improvement:+.2f}%")
        logger.info(f"  MAE improvement: {mae_improvement:+.2f}%")

    # Save evaluation results
    eval_results: dict[str, Any] = {
        "fold_results": fold_results,
        "average_metrics": {
            "multimodal_mse": avg_multimodal_mse,
            "multimodal_mae": avg_multimodal_mae,
        },
    }

    if args.compare_baseline:
        eval_results["average_metrics"]["baseline_mse"] = avg_baseline_mse
        eval_results["average_metrics"]["baseline_mae"] = avg_baseline_mae
        eval_results["average_metrics"]["mse_improvement_pct"] = mse_improvement
        eval_results["average_metrics"]["mae_improvement_pct"] = mae_improvement

    eval_results_path = cv_results_path.parent / "cv_evaluation_results.json"
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"Evaluation results saved to {eval_results_path}")

    logger.info("Cross-validation evaluation completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())
