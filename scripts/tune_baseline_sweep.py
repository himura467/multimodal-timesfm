#!/usr/bin/env python3
"""Hyperparameter tuning for baseline (fine-tuned) time series forecasting with W&B Sweeps."""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
import wandb
from torch.utils.data import DataLoader

from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.cross_validation import load_fold_datasets
from examples.time_mmd.data.time_mmd_dataset import TimeMmdDataset
from multimodal_timesfm.data.collate import baseline_collate_fn
from multimodal_timesfm.decoder import MultimodalDecoder, MultimodalDecoderConfig
from multimodal_timesfm.evaluator import MultimodalEvaluator
from multimodal_timesfm.trainer import MultimodalTrainer
from multimodal_timesfm.training_args import TrainingArguments
from multimodal_timesfm.tsfm.timesfm import TimesFM2p5Adapter
from multimodal_timesfm.types import BaselineCheckpoint
from multimodal_timesfm.utils.device import pin_memory, resolve_device
from multimodal_timesfm.utils.logging import setup_logger
from multimodal_timesfm.utils.seed import set_seed
from multimodal_timesfm.utils.yaml import load_yaml

_logger = setup_logger()


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a W&B Sweeps hyperparameter search for baseline time series forecasting.",
    )

    parser.add_argument("--sweep-id", type=str, help="Existing W&B sweep ID to join.")
    parser.add_argument("--sweep-config", type=str, help="Path to a W&B sweep YAML config file.")
    parser.add_argument("--count", type=int, help="Number of sweep runs for the agent to execute.")
    parser.add_argument("--model-config", type=str, help="Path to a model config YAML file.")
    parser.add_argument("--forecast-config", type=str, help="Path to a forecast config YAML file.")
    parser.add_argument("--data-path", type=str, default="data/Time-MMD", help="Root path of the dataset.")
    parser.add_argument("--cache-dir", type=str, default="data/cache", help="Directory with pre-computed cached datasets.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    return parser.parse_args()


def _create_baseline_model(model_config: ModelConfig, device: torch.device) -> MultimodalDecoder:
    """Build a MultimodalDecoder with a pretrained adapter for baseline fine-tuning.

    The fusion head is constructed from model_config but remains unused during
    baseline training; only the adapter parameters are fine-tuned.

    Args:
        model_config: Static model configuration (adapter repo, embedding dims).
        device: Device to load the model onto.

    Returns:
        MultimodalDecoder with a pretrained adapter ready for fine-tuning.
    """
    _logger.info(
        "Loading pretrained adapter from %s on %s",
        model_config.adapter.pretrained_repo,
        device,
    )
    adapter = TimesFM2p5Adapter.from_pretrained(device, repo_id=model_config.adapter.pretrained_repo)
    config = MultimodalDecoderConfig(
        ts_embedding_dims=model_config.fusion.ts_embedding_dims,
        text_embedding_dims=model_config.fusion.text_embedding_dims,
        num_fusion_layers=model_config.fusion.num_fusion_layers,
        fusion_hidden_dims=model_config.fusion.fusion_hidden_dims,
    )
    return MultimodalDecoder(adapter, config)


def _train_and_evaluate(
    run: wandb.Run,
    base_training_args: TrainingArguments,
    model_config: ModelConfig,
    forecast_config: ForecastConfig,
    train_domains: list[str],
    val_domains: list[str],
    test_domains: list[str],
    device: torch.device,
    cache_dir: Path,
) -> None:
    """Run one sweep trial: fine-tune the adapter and log metrics to W&B.

    Reads hyperparameters from the active W&B run config, fine-tunes the
    adapter, loads the best checkpoint, evaluates on the test set, and logs
    val/loss, test/mse, and test/mae.
    The checkpoint directory is removed after evaluation.

    Args:
        run: Active W&B run whose config provides this trial's hyperparameters.
        base_training_args: Base training arguments partially overridden by sweep config.
        model_config: Static model architecture configuration.
        forecast_config: Forecasting parameters (context / horizon lengths).
        train_domains: Domain names used for training.
        val_domains: Domain names used for validation.
        test_domains: Domain names used for test evaluation.
        device: Device to train and evaluate on.
        cache_dir: Directory containing pre-computed cached datasets.
    """
    config = run.config
    _logger.info("Starting sweep run %s with config: %s", run.id, dict(config))

    training_args = replace(
        base_training_args,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    _logger.info(
        "Loading datasets — train: %s, val: %s, test: %s",
        train_domains,
        val_domains,
        test_domains,
    )
    train_dataset, val_dataset, test_dataset = load_fold_datasets(
        train_domains=train_domains,
        val_domains=val_domains,
        test_domains=test_domains,
        text_encoder_type=model_config.fusion.text_encoder_type,
        patch_len=model_config.adapter.patch_len,
        context_len=forecast_config.context_len,
        horizon_len=forecast_config.horizon_len,
        cache_dir=cache_dir,
    )

    model = _create_baseline_model(model_config, device)

    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        mode="baseline",
        device=device,
        wandb_run=run,
    )

    trainer.train()

    best_checkpoint_path = training_args.checkpoint_dir / "best_model.pt"
    _logger.info("Loading best checkpoint from %s", best_checkpoint_path)
    checkpoint = cast(BaselineCheckpoint, torch.load(best_checkpoint_path, weights_only=True))
    val_loss = checkpoint["best_val_loss"]
    model.adapter.load_state_dict(checkpoint["adapter_state_dict"])

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=baseline_collate_fn,
        pin_memory=pin_memory(device),
    )

    _logger.info("Evaluating on test domains: %s", test_domains)
    evaluator = MultimodalEvaluator(model, device)
    test_metrics = evaluator.evaluate(test_dataloader)

    _logger.info(
        "Run %s — val_loss: %.6f, test_mse: %.6f, test_mae: %.6f",
        run.id,
        val_loss,
        test_metrics["mse"],
        test_metrics["mae"],
    )
    run.log(
        {"val/loss": val_loss, "test/mse": test_metrics["mse"], "test/mae": test_metrics["mae"]},
        step=trainer.global_step,
    )

    checkpoint_dir = training_args.checkpoint_dir
    if checkpoint_dir.exists():
        _logger.info("Removing checkpoint directory %s", checkpoint_dir)
        shutil.rmtree(checkpoint_dir)


def main() -> int:
    """Entry point: resolve the sweep ID and start the W&B agent.

    Returns:
        Exit code — 0 on success, 1 if neither --sweep-id nor
        --sweep-config is provided.
    """
    parsed_args = _parse_args()

    if parsed_args.model_config:
        model_config = ModelConfig.from_yaml(Path(parsed_args.model_config))
        _logger.info("Loaded model config from %s", parsed_args.model_config)
    else:
        model_config = ModelConfig()
        _logger.info("Using default ModelConfig")

    if parsed_args.forecast_config:
        forecast_config = ForecastConfig.from_yaml(Path(parsed_args.forecast_config))
        _logger.info("Loaded forecast config from %s", parsed_args.forecast_config)
    else:
        forecast_config = ForecastConfig()
        _logger.info("Using default ForecastConfig")

    base_training_args = TrainingArguments(
        output_dir="outputs/sweeps/baseline",
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="best",
        seed=parsed_args.seed,
    )

    if parsed_args.seed is not None:
        _logger.info("Setting random seed to %d", parsed_args.seed)
        set_seed(parsed_args.seed)

    data_path = Path(parsed_args.data_path)
    all_domains = TimeMmdDataset.get_domains(data_path)
    _logger.info("Discovered %d domains in %s: %s", len(all_domains), data_path, all_domains)

    val_domains = ["Climate"]
    test_domains = ["Energy"]
    train_domains = [d for d in all_domains if d not in val_domains and d not in test_domains]

    device = resolve_device()
    _logger.info("Using device: %s", device)

    def _sweep_fn() -> None:
        """Execute a single sweep trial inside a W&B run context."""
        with wandb.init(project="baseline-timesfm-time-mmd") as run:
            _train_and_evaluate(
                run=run,
                base_training_args=base_training_args,
                model_config=model_config,
                forecast_config=forecast_config,
                train_domains=train_domains,
                val_domains=val_domains,
                test_domains=test_domains,
                device=device,
                cache_dir=Path(parsed_args.cache_dir),
            )

    if parsed_args.sweep_id:
        sweep_id = parsed_args.sweep_id
        _logger.info("Joining existing sweep %s", sweep_id)
    else:
        if not parsed_args.sweep_config:
            _logger.error("Either --sweep-id or --sweep-config must be provided.")
            return 1
        sweep_config = load_yaml(Path(parsed_args.sweep_config))
        sweep_id = wandb.sweep(sweep=sweep_config, project="baseline-timesfm-time-mmd")
        _logger.info("Created new sweep %s", sweep_id)

    _logger.info("Starting W&B agent (count=%s)", parsed_args.count)
    wandb.agent(sweep_id, function=_sweep_fn, project="baseline-timesfm-time-mmd", count=parsed_args.count)
    _logger.info("Sweep agent finished")

    return 0


if __name__ == "__main__":
    exit(main())
