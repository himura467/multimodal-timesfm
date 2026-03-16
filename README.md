# Mutex

**MUTEX** (**M**UTEX **U**nifies **T**SFMs **E**nabling **X**-modalities) is a unified framework for multimodal time series foundation models. It brings together multiple TSFMs (including [TimesFM](https://github.com/google-research/timesfm) and [Chronos](https://github.com/amazon-science/chronos-forecasting)) and extends them to support diverse input modalities such as text.

## Installation

```sh
pip install mutex[all]
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
