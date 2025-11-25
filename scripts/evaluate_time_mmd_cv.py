#!/usr/bin/env python3
"""Cross-validation evaluation script for multimodal TimesFM on Time-MMD dataset."""

import argparse
import json
from pathlib import Path
from typing import Any

from timesfm.pytorch_patched_decoder import TimesFMConfig
from torch.utils.data import DataLoader

from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.data.cross_validation import create_fold_datasets
from multimodal_timesfm.arima_baseline import evaluate_arima_model
from multimodal_timesfm.evaluation import evaluate_baseline_model, evaluate_multimodal_model
from multimodal_timesfm.multimodal_patched_decoder import MultimodalTimesFMConfig
from multimodal_timesfm.training_args import TrainingArguments
from multimodal_timesfm.utils.collate import multimodal_collate_fn
from multimodal_timesfm.utils.device import get_pin_memory, resolve_device
from multimodal_timesfm.utils.logging import get_logger, setup_logger
from multimodal_timesfm.utils.model import (
    create_baseline_timesfm_model,
    load_baseline_checkpoint,
    load_multimodal_checkpoint,
)
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
        "--baseline-cv-results",
        type=str,
        help="Path to baseline cross-validation results JSON file (for fine-tuned baseline comparison)",
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
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline TimesFM model (without text)",
    )

    parser.add_argument(
        "--compare-arima",
        action="store_true",
        help="Compare with ARIMA baseline model",
    )

    parser.add_argument(
        "--arima-order",
        type=int,
        nargs=3,
        default=[32, 1, 1],
        help="ARIMA order (p, d, q) for baseline comparison (default: 32 1 1)",
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

    return parser.parse_args()


def main() -> int:
    """Main cross-validation evaluation function."""
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

    # Load cross-validation results
    cv_results_path = Path(parsed_args.cv_results)
    if not cv_results_path.exists():
        logger.error(f"Cross-validation results file not found: {cv_results_path}")
        return 1

    with open(cv_results_path) as f:
        cv_results = json.load(f)

    logger.info(f"Loaded cross-validation results from {cv_results_path}")
    logger.info(f"Number of folds: {len(cv_results)}")

    # Load baseline CV results if provided (for fine-tuned baseline comparison)
    baseline_cv_results = None
    if parsed_args.baseline_cv_results:
        baseline_cv_results_path = Path(parsed_args.baseline_cv_results)
        if not baseline_cv_results_path.exists():
            logger.error(f"Baseline CV results file not found: {baseline_cv_results_path}")
            return 1
        with open(baseline_cv_results_path) as f:
            baseline_cv_results = json.load(f)
        logger.info(f"Loaded baseline CV results from {baseline_cv_results_path}")
        logger.info(f"Baseline folds: {len(baseline_cv_results)}")
        # Validate that baseline and multimodal have same number of folds
        if len(baseline_cv_results) != len(cv_results):
            logger.error(f"Baseline folds ({len(baseline_cv_results)}) != multimodal folds ({len(cv_results)})")
            return 1

    # Setup device
    device = resolve_device()
    logger.info(f"Using device: {device}")

    # Create baseline TimesFM config
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

    # Create pretrained baseline model if needed
    pretrained_baseline_model = None
    if parsed_args.compare_baseline:
        logger.info("Creating pretrained baseline TimesFM model (no fine-tuning)...")
        pretrained_baseline_model = create_baseline_timesfm_model(baseline_config, load_pretrained=True)

    # Evaluate each fold
    fold_results = []
    data_path = Path(parsed_args.data_path)

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
            patch_len=training_args.patch_len,
            context_len=training_args.context_len,
            horizon_len=training_args.horizon_len,
        )

        logger.info(f"Test samples: {len(test_dataset)}")

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=parsed_args.batch_size,
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

        # Evaluate baseline models if requested
        pretrained_baseline_metrics = None
        finetuned_baseline_metrics = None
        arima_metrics = None

        # Evaluate ARIMA baseline if requested
        if parsed_args.compare_arima:
            logger.info(f"Evaluating ARIMA baseline (order={tuple(parsed_args.arima_order)})...")
            arima_metrics = evaluate_arima_model(test_dataloader, device, order=tuple(parsed_args.arima_order))
            logger.info(f"ARIMA baseline metrics: MSE={arima_metrics['mse']:.6f}, MAE={arima_metrics['mae']:.6f}")

        # Evaluate pretrained baseline (no fine-tuning)
        if pretrained_baseline_model is not None:
            logger.info("Evaluating pretrained baseline model (no fine-tuning)...")
            pretrained_baseline_metrics = evaluate_baseline_model(pretrained_baseline_model, test_dataloader, device)
            logger.info(
                f"Pretrained baseline metrics: MSE={pretrained_baseline_metrics['mse']:.6f}, "
                f"MAE={pretrained_baseline_metrics['mae']:.6f}"
            )

        # Evaluate fine-tuned baseline (if checkpoints provided)
        if baseline_cv_results is not None:
            baseline_fold_data = baseline_cv_results[fold_idx]
            baseline_checkpoint_path = Path(baseline_fold_data["checkpoint"])
            logger.info(f"Loading fine-tuned baseline model from {baseline_checkpoint_path}")
            finetuned_baseline_model = load_baseline_checkpoint(baseline_checkpoint_path, baseline_config, device)

            logger.info("Evaluating fine-tuned baseline model...")
            finetuned_baseline_metrics = evaluate_baseline_model(finetuned_baseline_model, test_dataloader, device)
            logger.info(
                f"Fine-tuned baseline metrics: MSE={finetuned_baseline_metrics['mse']:.6f}, "
                f"MAE={finetuned_baseline_metrics['mae']:.6f}"
            )

        # Store results
        result = {
            "fold": fold_idx,
            "test_domains": fold_data["test_domains"],
            "num_test_samples": len(test_dataset),
            "multimodal": multimodal_metrics,
        }
        if arima_metrics is not None:
            result["arima"] = arima_metrics
        if pretrained_baseline_metrics is not None:
            result["pretrained_baseline"] = pretrained_baseline_metrics
        if finetuned_baseline_metrics is not None:
            result["finetuned_baseline"] = finetuned_baseline_metrics
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

    # Prepare evaluation results dictionary
    eval_results: dict[str, Any] = {
        "fold_results": fold_results,
        "average_metrics": {
            "multimodal_mse": avg_multimodal_mse,
            "multimodal_mae": avg_multimodal_mae,
        },
    }

    # Show ARIMA baseline comparison
    if parsed_args.compare_arima:
        avg_arima_mse = sum(r["arima"]["mse"] for r in fold_results) / len(fold_results)
        avg_arima_mae = sum(r["arima"]["mae"] for r in fold_results) / len(fold_results)

        logger.info(f"ARIMA baseline model (order={tuple(parsed_args.arima_order)}):")
        logger.info(f"  Average test MSE: {avg_arima_mse:.6f}")
        logger.info(f"  Average test MAE: {avg_arima_mae:.6f}")

        logger.info("Multimodal vs ARIMA improvement:")
        arima_mse_improvement = ((avg_arima_mse - avg_multimodal_mse) / avg_arima_mse) * 100
        arima_mae_improvement = ((avg_arima_mae - avg_multimodal_mae) / avg_arima_mae) * 100
        logger.info(f"  MSE improvement: {arima_mse_improvement:+.2f}%")
        logger.info(f"  MAE improvement: {arima_mae_improvement:+.2f}%")

        eval_results["average_metrics"]["arima_mse"] = avg_arima_mse
        eval_results["average_metrics"]["arima_mae"] = avg_arima_mae
        eval_results["average_metrics"]["arima_mse_improvement_pct"] = arima_mse_improvement
        eval_results["average_metrics"]["arima_mae_improvement_pct"] = arima_mae_improvement

    # Show pretrained baseline comparison
    if parsed_args.compare_baseline and pretrained_baseline_model is not None:
        avg_pretrained_baseline_mse = sum(r["pretrained_baseline"]["mse"] for r in fold_results) / len(fold_results)
        avg_pretrained_baseline_mae = sum(r["pretrained_baseline"]["mae"] for r in fold_results) / len(fold_results)

        logger.info("Pretrained baseline model (no fine-tuning):")
        logger.info(f"  Average test MSE: {avg_pretrained_baseline_mse:.6f}")
        logger.info(f"  Average test MAE: {avg_pretrained_baseline_mae:.6f}")

        logger.info("Multimodal vs Pretrained baseline improvement:")
        pretrained_mse_improvement = (
            (avg_pretrained_baseline_mse - avg_multimodal_mse) / avg_pretrained_baseline_mse
        ) * 100
        pretrained_mae_improvement = (
            (avg_pretrained_baseline_mae - avg_multimodal_mae) / avg_pretrained_baseline_mae
        ) * 100
        logger.info(f"  MSE improvement: {pretrained_mse_improvement:+.2f}%")
        logger.info(f"  MAE improvement: {pretrained_mae_improvement:+.2f}%")

        eval_results["average_metrics"]["pretrained_baseline_mse"] = avg_pretrained_baseline_mse
        eval_results["average_metrics"]["pretrained_baseline_mae"] = avg_pretrained_baseline_mae
        eval_results["average_metrics"]["pretrained_mse_improvement_pct"] = pretrained_mse_improvement
        eval_results["average_metrics"]["pretrained_mae_improvement_pct"] = pretrained_mae_improvement

    # Show fine-tuned baseline comparison
    if baseline_cv_results is not None:
        avg_finetuned_baseline_mse = sum(r["finetuned_baseline"]["mse"] for r in fold_results) / len(fold_results)
        avg_finetuned_baseline_mae = sum(r["finetuned_baseline"]["mae"] for r in fold_results) / len(fold_results)

        logger.info("Fine-tuned baseline model (TimesFM fine-tuned on data):")
        logger.info(f"  Average test MSE: {avg_finetuned_baseline_mse:.6f}")
        logger.info(f"  Average test MAE: {avg_finetuned_baseline_mae:.6f}")

        logger.info("Multimodal vs Fine-tuned baseline improvement:")
        finetuned_mse_improvement = (
            (avg_finetuned_baseline_mse - avg_multimodal_mse) / avg_finetuned_baseline_mse
        ) * 100
        finetuned_mae_improvement = (
            (avg_finetuned_baseline_mae - avg_multimodal_mae) / avg_finetuned_baseline_mae
        ) * 100
        logger.info(f"  MSE improvement: {finetuned_mse_improvement:+.2f}%")
        logger.info(f"  MAE improvement: {finetuned_mae_improvement:+.2f}%")

        eval_results["average_metrics"]["finetuned_baseline_mse"] = avg_finetuned_baseline_mse
        eval_results["average_metrics"]["finetuned_baseline_mae"] = avg_finetuned_baseline_mae
        eval_results["average_metrics"]["finetuned_mse_improvement_pct"] = finetuned_mse_improvement
        eval_results["average_metrics"]["finetuned_mae_improvement_pct"] = finetuned_mae_improvement

    eval_results_path = cv_results_path.parent / "cv_evaluation_results.json"
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"Evaluation results saved to {eval_results_path}")

    logger.info("Cross-validation evaluation completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())
