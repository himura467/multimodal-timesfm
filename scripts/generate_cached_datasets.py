#!/usr/bin/env python3
"""Generate cached datasets with pre-computed text embeddings for Time-MMD dataset.

This script pre-processes all Time-MMD datasets and stores them with pre-computed
text embeddings to avoid redundant data loading and text encoding during training.
"""

import argparse
from pathlib import Path

from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.data.cross_validation import get_all_domains
from examples.time_mmd.data.time_mmd_dataset import TimeMmdDataset
from multimodal_timesfm.text_encoder import EnglishTextEncoder, JapaneseTextEncoder, TextEncoderBase
from multimodal_timesfm.utils.cache import (
    DatasetCache,
    create_or_load_cached_dataset,
)
from multimodal_timesfm.utils.device import resolve_device
from multimodal_timesfm.utils.logging import get_logger, setup_logger
from multimodal_timesfm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate cached datasets with pre-computed text embeddings")

    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to model configuration file (optional)",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/Time-MMD",
        help="Path to Time-MMD dataset",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Directory to store cached datasets",
    )

    parser.add_argument(
        "--text-encoder-type",
        type=str,
        choices=["english", "japanese", "baseline"],
        help="Type of text encoder to use (or 'baseline' for no text encoding)",
    )

    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        help="Specific domains to cache (if not provided, caches all domains)",
    )

    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio",
    )

    parser.add_argument(
        "--patch-len",
        type=int,
        help="Patch length (from model config if not provided)",
    )

    parser.add_argument(
        "--context-len",
        type=int,
        help="Context length (from model config if not provided)",
    )

    parser.add_argument(
        "--horizon-len",
        type=int,
        help="Horizon length (from model config if not provided)",
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild cache even if it exists",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for text encoding (auto-detected if not provided)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main() -> int:
    """Main cache generation function."""
    args = parse_args()

    # Set random seed for reproducibility if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Setup logging
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=log_dir / "cache_generation.log")
    logger = get_logger()

    logger.info("Starting cache generation for Time-MMD datasets")

    # Load model configuration
    if args.model_config:
        model_config = ModelConfig.from_yaml(Path(args.model_config))
        logger.info(f"Loaded model config from: {args.model_config}")
    else:
        model_config = ModelConfig()
        logger.info("Using default model configuration")

    # Determine dataset parameters
    patch_len = args.patch_len if args.patch_len is not None else model_config.timesfm.input_patch_len
    context_len = args.context_len if args.context_len is not None else model_config.timesfm.context_len
    horizon_len = args.horizon_len if args.horizon_len is not None else model_config.timesfm.horizon_len

    logger.info(f"Dataset parameters: patch_len={patch_len}, context_len={context_len}, horizon_len={horizon_len}")

    # Setup device
    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")

    # Initialize text encoder
    text_encoder: TextEncoderBase | None = None
    if args.text_encoder_type == "english":
        logger.info("Initializing English text encoder")
        text_encoder = EnglishTextEncoder(device=device)
    elif args.text_encoder_type == "japanese":
        logger.info("Initializing Japanese text encoder")
        text_encoder = JapaneseTextEncoder(device=device)
    else:
        logger.info("Baseline mode: no text encoder (text will be ignored)")

    # Initialize cache manager
    cache_dir = Path(args.cache_dir)
    cache = DatasetCache(cache_dir)
    logger.info(f"Cache directory: {cache_dir}")

    # Get domains to process
    data_path = Path(args.data_path)
    if args.domains:
        domains = args.domains
        logger.info(f"Processing specified domains: {domains}")
    else:
        domains = get_all_domains(data_path)
        logger.info(f"Processing all domains ({len(domains)}): {domains}")

    # Process each domain
    for domain in domains:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing domain: {domain}")
        logger.info(f"{'=' * 50}")

        # Process both train and test splits
        for split in ["train", "test"]:
            logger.info(f"\nProcessing {split} split for {domain}")

            # Get cache path
            cache_path = cache.get_cache_path(
                dataset_name="time_mmd",
                domain=domain,
                split_ratio=args.split_ratio,
                split=split,
                patch_len=patch_len,
                context_len=context_len,
                horizon_len=horizon_len,
                text_encoder_type=None if args.text_encoder_type == "baseline" else args.text_encoder_type,
            )

            # Create raw dataset factory
            def create_raw_dataset() -> TimeMmdDataset:
                return TimeMmdDataset(
                    data_dir=data_path,
                    domain=domain,
                    split_ratio=args.split_ratio,
                    split=split,  # type: ignore[arg-type]
                    patch_len=patch_len,
                    context_len=context_len,
                    horizon_len=horizon_len,
                )

            # Create or load cached dataset
            try:
                cached_data = create_or_load_cached_dataset(
                    cache=cache,
                    cache_path=cache_path,
                    raw_dataset_factory=create_raw_dataset,
                    text_encoder=text_encoder,
                    device=device if text_encoder is not None else None,
                    force_rebuild=args.force_rebuild,
                )

                logger.info(f"Successfully processed {split} split: {len(cached_data)} samples")

            except Exception as e:
                logger.error(f"Failed to process {split} split for {domain}: {e}")
                continue

    logger.info(f"\n{'=' * 50}")
    logger.info("Cache generation completed successfully!")
    logger.info(f"Cached datasets stored in: {cache_dir}")
    logger.info(f"{'=' * 50}")

    return 0


if __name__ == "__main__":
    exit(main())
