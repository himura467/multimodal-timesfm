#!/usr/bin/env python3
"""
Compare multimodal TimesFM with text vs original PatchedTimeSeriesDecoder without text.

This script loads a trained multimodal TimesFM model from a checkpoint and compares
its performance against the original TimesFM model on Time-MMD dataset.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder, TimesFMConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.configs import EvaluationConfig, ModelConfig
from src.data.time_mmd_dataset import TimeMmdDataset
from src.models.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from src.utils.collate import multimodal_collate_fn
from src.utils.device import get_pin_memory, move_to_device, resolve_device
from src.utils.logging import get_logger, setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare multimodal TimesFM with text vs original TimesFM without text"
    )

    parser.add_argument(
        "--multimodal-checkpoint",
        type=str,
        required=True,
        help="Path to trained multimodal TimesFM checkpoint",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yml",
        help="Path to model configuration file",
    )

    parser.add_argument(
        "--evaluation-config",
        type=str,
        default="configs/evaluation.yml",
        help="Path to evaluation configuration file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (if not provided, no seed will be set)",
    )

    return parser.parse_args()


def create_original_timesfm_model(model_config: ModelConfig) -> PatchedTimeSeriesDecoder:
    """Create and load original TimesFM model with pretrained weights."""
    logger = get_logger()

    # Create TimesFM config
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

    # Create model
    model = PatchedTimeSeriesDecoder(config)

    # Load pretrained TimesFM weights
    repo_id = "google/timesfm-2.0-500m-pytorch"
    logger.info(f"Loading pretrained TimesFM weights from {repo_id}")

    try:
        model_dir = Path(snapshot_download(repo_id))
        checkpoint_path = model_dir / "torch_model.ckpt"
        pretrained_weights = torch.load(checkpoint_path, weights_only=True)

        # Load weights
        model_state_dict = model.state_dict()
        pretrained_keys = set(pretrained_weights.keys())
        model_keys = set(model_state_dict.keys())

        matching_keys = pretrained_keys.intersection(model_keys)
        logger.info(f"Loading {len(matching_keys)} pretrained parameters")

        for key in matching_keys:
            model_state_dict[key].copy_(pretrained_weights[key])

        logger.info("Successfully loaded pretrained TimesFM weights")

    except Exception as e:
        logger.warning(f"Failed to load pretrained weights: {e}")
        logger.warning("Continuing with randomly initialized weights")

    return model


def load_multimodal_model(
    checkpoint_path: Path, model_config: ModelConfig, device: torch.device
) -> MultimodalPatchedDecoder:
    """Load trained multimodal TimesFM model from checkpoint."""
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

    # Create model
    model = MultimodalPatchedDecoder(config, device)

    # Load checkpoint
    logger.info(f"Loading multimodal model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Successfully loaded multimodal model checkpoint")

    return model


def create_dataset(
    data_path: Path,
    domain: str,
    model_config: ModelConfig,
) -> TimeMmdDataset:
    """Create Time-MMD dataset for evaluation."""
    logger = get_logger()

    logger.info(f"Creating dataset for domain: {domain}")

    # Extract values from model config
    patch_len = model_config.timesfm.input_patch_len
    context_len = model_config.timesfm.context_len
    horizon_len = model_config.timesfm.horizon_len

    logger.info(f"Using config values - patch_len: {patch_len}, context_len: {context_len}, horizon_len: {horizon_len}")

    dataset = TimeMmdDataset(
        data_dir=data_path,
        domain=domain,
        split_ratio=1.0,
        patch_len=patch_len,
        context_len=context_len,
        horizon_len=horizon_len,
    )

    logger.info(f"Created dataset with {len(dataset)} samples")
    return dataset


def evaluate_original_model(
    model: PatchedTimeSeriesDecoder,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate original TimesFM model (without text)."""
    logger = get_logger()
    model.eval()
    model.to(device)

    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0

    logger.info("Evaluating original TimesFM model...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Original TimesFM"):
            batch_tensors = move_to_device(
                {"context": batch["context"], "future": batch["future"], "freq": batch["freq"]}, device
            )
            context = batch_tensors["context"]
            future = batch_tensors["future"]
            freq = batch_tensors["freq"]

            # Create input_padding tensor (zeros for now)
            input_padding = torch.zeros_like(context)

            # Forward pass without text (similar to reference TimesFMFinetuner._process_batch)
            predictions = model(context, input_padding.float(), freq)
            predictions_mean = predictions[..., 0]  # [B, patches, horizon_len]
            last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

            # Compute metrics using same approach as reference
            mse = torch.mean((last_patch_pred - future) ** 2)
            mae = torch.mean(torch.abs(last_patch_pred - future))

            total_mse += mse.item() * context.size(0)
            total_mae += mae.item() * context.size(0)
            num_samples += context.size(0)

    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples

    return {
        "mse": avg_mse,
        "mae": avg_mae,
    }


def evaluate_multimodal_model(
    model: MultimodalPatchedDecoder,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate multimodal TimesFM model (with text)."""
    logger = get_logger()
    model.eval()
    model.to(device)

    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0

    logger.info("Evaluating multimodal TimesFM model...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Multimodal TimesFM"):
            batch_tensors = move_to_device(
                {"context": batch["context"], "future": batch["future"], "freq": batch["freq"]}, device
            )
            context = batch_tensors["context"]
            future = batch_tensors["future"]
            freq = batch_tensors["freq"]
            patched_texts = batch["patched_texts"]

            # Create input_padding tensor (zeros for now)
            input_padding = torch.zeros_like(context)

            # Forward pass with text
            predictions = model(context, input_padding.float(), freq, patched_texts)
            predictions_mean = predictions[..., 0]  # [B, patches, horizon_len]
            last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

            # Compute metrics
            mse = torch.mean((last_patch_pred - future) ** 2)
            mae = torch.mean(torch.abs(last_patch_pred - future))

            total_mse += mse.item() * context.size(0)
            total_mae += mae.item() * context.size(0)
            num_samples += context.size(0)

    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples

    return {
        "mse": avg_mse,
        "mae": avg_mae,
    }


def main() -> int:
    """Main comparison function."""
    args = parse_args()

    # Setup
    if args.seed is not None:
        set_seed(args.seed)
    setup_logger()
    logger = get_logger()

    # Load configurations
    model_config = ModelConfig.from_yaml(Path(args.model_config))
    eval_config = EvaluationConfig.from_yaml(Path(args.evaluation_config))

    logger.info("Starting model comparison...")
    logger.info(f"Multimodal checkpoint: {args.multimodal_checkpoint}")
    logger.info(f"Domain: {eval_config.data.domain}")

    # Setup device
    device = resolve_device()
    logger.info(f"Using device: {device}")

    # Create dataset using config values
    dataset = create_dataset(
        data_path=Path(eval_config.data.data_path),
        domain=eval_config.data.domain,
        model_config=model_config,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=eval_config.runner.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multimodal_collate_fn,
        pin_memory=get_pin_memory(device),
    )

    # Load models
    logger.info("Loading models...")
    original_model = create_original_timesfm_model(model_config)
    multimodal_model = load_multimodal_model(Path(args.multimodal_checkpoint), model_config, device)

    # Evaluate models
    logger.info("Running evaluations...")
    original_results = evaluate_original_model(original_model, dataloader, device)
    multimodal_results = evaluate_multimodal_model(multimodal_model, dataloader, device)

    # Compile results
    results: dict[str, Any] = {
        "config": {
            "domain": eval_config.data.domain,
            "patch_len": model_config.timesfm.input_patch_len,
            "context_len": model_config.timesfm.context_len,
            "horizon_len": model_config.timesfm.horizon_len,
            "batch_size": eval_config.runner.batch_size,
            "num_samples": len(dataset),
        },
        "original": original_results,
        "multimodal": multimodal_results,
    }

    # Display comprehensive results
    print("\n" + "=" * 50)
    print("TimesFM Model Comparison Results")
    print("=" * 50)

    print("\nConfiguration:")
    print(f"  Domain: {results['config']['domain']}")
    print(f"  Patch Length: {results['config']['patch_len']}")
    print(f"  Context Length: {results['config']['context_len']}")
    print(f"  Horizon Length: {results['config']['horizon_len']}")
    print(f"  Batch Size: {results['config']['batch_size']}")
    print(f"  Test Samples: {results['config']['num_samples']}")

    print("\nOriginal TimesFM (without text):")
    print(f"  MSE:  {original_results['mse']:.6f}")
    print(f"  MAE:  {original_results['mae']:.6f}")

    print("\nMultimodal TimesFM (with text):")
    print(f"  MSE:  {multimodal_results['mse']:.6f}")
    print(f"  MAE:  {multimodal_results['mae']:.6f}")

    print("\nImprovement Analysis:")
    mse_improvement = ((original_results["mse"] - multimodal_results["mse"]) / original_results["mse"]) * 100
    mae_improvement = ((original_results["mae"] - multimodal_results["mae"]) / original_results["mae"]) * 100

    print(f"  MSE Improvement:  {mse_improvement:+.2f}%")
    print(f"  MAE Improvement:  {mae_improvement:+.2f}%")

    if mse_improvement > 0:
        print("\n✅ Multimodal model shows improvement over original TimesFM")
    else:
        print("\n❌ Multimodal model does not improve over original TimesFM")

    logger.info("Model comparison completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
