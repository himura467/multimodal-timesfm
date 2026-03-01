#!/usr/bin/env python3
"""Pre-compute and cache text embeddings for all Time-MMD domains.

Text embeddings must be cached before training or evaluation.
This script iterates over every domain in the dataset, runs the text encoder once, and
persists the results as pickle files so that training scripts can load them without re-encoding.
"""

import argparse
from pathlib import Path
from typing import Literal

import torch

from examples.time_mmd.configs.forecast import ForecastConfig
from examples.time_mmd.configs.model import ModelConfig
from examples.time_mmd.data.time_mmd_dataset import TimeMmdDataset
from multimodal_timesfm.data.preprocess import PreprocessPipeline
from multimodal_timesfm.text_encoder.base import TextEncoderBase
from multimodal_timesfm.text_encoder.english import EnglishTextEncoder
from multimodal_timesfm.text_encoder.japanese import JapaneseTextEncoder
from multimodal_timesfm.utils.device import resolve_device
from multimodal_timesfm.utils.logging import setup_logger
from multimodal_timesfm.utils.seed import set_seed

_logger = setup_logger()


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache text embeddings for Time-MMD domains.",
    )

    parser.add_argument("--model-config", type=str, help="Path to a model config YAML file.")
    parser.add_argument("--forecast-config", type=str, help="Path to a forecast config YAML file.")
    parser.add_argument(
        "--text-encoder-type",
        type=str,
        choices=["english", "japanese"],
        required=True,
        help="Text encoder to use for embedding generation.",
    )
    parser.add_argument("--data-path", type=str, default="data/Time-MMD", help="Root path of the dataset.")
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        help="Subset of domains to cache. Defaults to all available domains.",
    )
    parser.add_argument("--cache-dir", type=str, default="data/cache", help="Directory to write cached datasets.")
    parser.add_argument("--force-rebuild", action="store_true", help="Overwrite existing cache files.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    return parser.parse_args()


def _build_text_encoder(
    text_encoder_type: Literal["english", "japanese"],
    device: torch.device,
) -> TextEncoderBase:
    """Instantiate a text encoder for the given type.

    Args:
        text_encoder_type: Which encoder to use — "english" or "japanese".
        device: Device to run the encoder on.

    Returns:
        Initialized TextEncoderBase instance.

    Raises:
        ValueError: If text_encoder_type is not recognized.
    """
    match text_encoder_type:
        case "english":
            _logger.info("Initializing EnglishTextEncoder")
            return EnglishTextEncoder(device=device)
        case "japanese":
            _logger.info("Initializing JapaneseTextEncoder")
            return JapaneseTextEncoder(device=device)
        case _:
            raise ValueError(f"Unknown text encoder type: {text_encoder_type!r}")


def main() -> int:
    """Entry point: cache text embeddings for all (or selected) Time-MMD domains.

    Returns:
        Exit code — 0 on success.
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

    if parsed_args.seed is not None:
        _logger.info("Setting random seed to %d", parsed_args.seed)
        set_seed(parsed_args.seed)

    text_encoder_type: Literal["english", "japanese"] = parsed_args.text_encoder_type

    device = resolve_device()
    _logger.info("Using device: %s", device)

    text_encoder = _build_text_encoder(text_encoder_type, device)

    data_path = Path(parsed_args.data_path)
    if parsed_args.domains:
        _logger.info("Caching specified domains: %s", parsed_args.domains)
    else:
        domains = TimeMmdDataset.get_domains(data_path)
        _logger.info("Caching all %d domains: %s", len(domains), domains)

    pipeline = PreprocessPipeline(Path(parsed_args.cache_dir))

    for domain in domains:
        _logger.info("Processing domain: %s", domain)
        cache_path = pipeline.get_path(
            dataset_name="time_mmd",
            entity=domain,
            text_encoder_type=text_encoder_type,
            patch_len=model_config.adapter.patch_len,
            context_len=forecast_config.context_len,
            horizon_len=forecast_config.horizon_len,
        )

        def _dataset_factory() -> TimeMmdDataset:
            return TimeMmdDataset(
                data_dir=data_path,
                domain=domain,
                patch_len=model_config.adapter.patch_len,
                context_len=forecast_config.context_len,
                horizon_len=forecast_config.horizon_len,
            )

        pipeline.prepare(
            path=cache_path,
            dataset_factory=_dataset_factory,
            text_encoder=text_encoder,
            device=device,
            force_rebuild=parsed_args.force_rebuild,
        )
        _logger.info("Done: %s -> %s", domain, cache_path)

    _logger.info("All domains cached successfully")
    return 0


if __name__ == "__main__":
    exit(main())
