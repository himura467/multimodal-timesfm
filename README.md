# Multimodal TimesFM

A multimodal extension of Google's [TimesFM](https://github.com/google-research/timesfm) for time series forecasting with text inputs.

## Installation

```sh
pip install multimodal-timesfm[all]
```

## Quick Start

### 1. Setup

Clone the Time-MMD dataset:

```sh
./setup_time_mmd.sh
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
