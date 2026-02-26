"""Cross-validation utilities for Time-MMD dataset."""

from pathlib import Path
from typing import Literal

from torch.utils.data import ConcatDataset, Dataset

from multimodal_timesfm.data.dataset import PreprocessedDataset
from multimodal_timesfm.data.preprocess import PreprocessPipeline
from multimodal_timesfm.types import PreprocessedSample


def load_fold_datasets(
    train_domains: list[str],
    val_domains: list[str],
    test_domains: list[str],
    text_encoder_type: Literal["english", "japanese"],
    patch_len: int,
    context_len: int,
    horizon_len: int,
    cache_dir: Path,
) -> tuple[ConcatDataset[PreprocessedSample], ConcatDataset[PreprocessedSample], ConcatDataset[PreprocessedSample]]:
    """Load cached datasets for a single fold from pre-computed cache.

    Args:
        train_domains: List of domain names for training.
        val_domains: List of domain names for validation.
        test_domains: List of domain names for testing.
        text_encoder_type: Type of text encoder used for caching.
        patch_len: Length of input patches.
        context_len: Length of context.
        horizon_len: Length of horizon.
        cache_dir: Directory containing pre-computed cached datasets.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    cache = PreprocessPipeline(cache_dir)

    def load_cached_domains(domains: list[str]) -> list[Dataset[PreprocessedSample]]:
        datasets: list[Dataset[PreprocessedSample]] = []
        for domain in domains:
            cache_path = cache.get_path(
                dataset_name="time_mmd",
                entity=domain,
                text_encoder_type=text_encoder_type,
                patch_len=patch_len,
                context_len=context_len,
                horizon_len=horizon_len,
            )
            cached_data = cache.load(cache_path)
            datasets.append(PreprocessedDataset(cached_data, mode="multimodal"))
        return datasets

    train_datasets = load_cached_domains(train_domains)
    val_datasets = load_cached_domains(val_domains)
    test_datasets = load_cached_domains(test_domains)

    return (
        ConcatDataset(train_datasets),
        ConcatDataset(val_datasets),
        ConcatDataset(test_datasets),
    )
