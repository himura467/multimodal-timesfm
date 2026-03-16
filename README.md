# Multimodal TimesFM

> [!IMPORTANT]
> This repository is being renamed to **Mutex**. We encourage all users to migrate to the new repository.
>
> - New repository: [himura467/mutex](https://github.com/himura467/mutex)
> - New package: `pip install mutex`

A multimodal extension of Google's [TimesFM](https://github.com/google-research/timesfm) for time series forecasting with text inputs.

## Installation

```sh
pip install multimodal-timesfm[all]
```

## Quick Start

### 1. Setup

Clone the Time-MMD dataset:

```sh
./scripts/clone_time_mmd.sh
```

Split the dataset into train / val / test:

```sh
PYTHONPATH=. uv run python scripts/split_time_mmd_datasets.py \
    --train-ratio 0.6 \
    --val-ratio 0.2
```

### 2. Pre-compute Text Embeddings

```sh
PYTHONPATH=. uv run python scripts/cache_time_mmd_datasets.py --text-encoder-type english
```

### 3. Hyperparameter Tuning

Run a W&B Sweeps search for the multimodal model:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_sweep.py \
    --sweep-config examples/time_mmd/configs/sweeps/multimodal_1layer.yml
```

To compare against a fine-tuned baseline:

```sh
PYTHONPATH=. uv run python scripts/tune_baseline_sweep.py \
    --sweep-config examples/time_mmd/configs/sweeps/baseline.yml
```

## Acknowledgments

We thank the [Time-MMD](https://github.com/AdityaLab/Time-MMD) team for providing the multimodal time series dataset used in our examples and experiments.

## License

MIT
