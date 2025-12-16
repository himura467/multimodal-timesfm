# Multimodal TimesFM

## Overview

This project provides a multimodal extension of Google's [TimesFM](https://github.com/google-research/timesfm) to support time series forecasting with text inputs. It consists of a core Python package and example implementations using the [Time-MMD](https://github.com/AdityaLab/Time-MMD) dataset.

## Project Structure

```
multimodal-timesfm/
├── src/
│   └── multimodal_timesfm/                  # Core PyPI package
│       ├── __init__.py
│       ├── multimodal_timesfm.py            # Main wrapper class
│       ├── text_encoder.py                  # Text encoding components
│       ├── multimodal_fusion.py             # Fusion mechanism
│       ├── multimodal_patched_decoder.py    # Patched decoder for multimodal
│       ├── multimodal_dataset.py            # Base multimodal dataset class
│       ├── preprocessing.py                 # Data preprocessing utilities
│       ├── trainer.py                       # Multimodal training logic
│       ├── baseline_trainer.py              # Baseline (non-multimodal) training logic
│       ├── training_args.py                 # Training arguments configuration
│       ├── arima_baseline.py                # ARIMA baseline for comparison
│       ├── evaluation.py                    # Evaluation logic
│       ├── cross_validation.py              # Cross-validation utilities
│       └── utils/                           # Utility modules
│           ├── __init__.py
│           ├── collate.py                   # Data collation utilities (multimodal & baseline)
│           ├── device.py                    # Device utilities
│           ├── logging.py                   # Logging utilities
│           ├── seed.py                      # Random seed utilities
│           ├── model.py                     # Model utilities (creation & loading)
│           └── yaml.py                      # YAML configuration utilities
├── scripts/
│   ├── train_time_mmd_cv.py                 # Train models (multimodal & baseline) with cross-validation
│   ├── tune_time_mmd_sweep.py               # Hyperparameter tuning for multimodal TimesFM with WandB Sweep
│   ├── tune_baseline_sweep.py               # Hyperparameter tuning for baseline TimesFM with WandB Sweep
│   ├── evaluate_time_mmd_cv.py              # Evaluate models with cross-validation
│   ├── visualize_time_mmd_cv.py             # Visualize model predictions
│   └── forecast_time_mmd.py                 # Forecasting script with custom parameters
├── examples/
│   └── time_mmd/                            # Time-MMD dataset example components
│       ├── configs/                         # Time-MMD specific configurations
│       │   ├── __init__.py
│       │   ├── model.py                     # Model architecture settings
│       │   ├── domain_columns.py            # Domain-specific column configurations
│       │   ├── sweep_config.yml             # WandB sweep config for multimodal TimesFM
│       │   └── baseline_sweep_config.yml    # WandB sweep config for baseline TimesFM
│       └── data/
│           ├── __init__.py
│           ├── time_mmd_dataset.py          # Time-MMD dataset loader
│           └── cross_validation.py          # Cross-validation split logic
├── data/
│   └── Time-MMD/                            # Time-MMD dataset submodule
│       ├── numerical/                       # Time series data by domain
│       └── textual/                         # Text descriptions by domain
├── tests/
│   ├── __init__.py
│   ├── test_text_encoder.py
│   ├── test_time_mmd_dataset.py
│   ├── test_preprocessing.py
│   ├── test_trainer.py
│   ├── test_baseline_trainer.py
│   ├── test_multimodal_fusion.py
│   ├── test_multimodal_patched_decoder.py
│   └── test_cross_validation.py
├── .gitmodules                              # Git submodule configuration
├── .python-version                          # Python 3.11
├── pyproject.toml                           # Package configuration for PyPI
├── README.md
└── CLAUDE.md                                # This file
```

## Bash Commands

### Development

- `uv run mypy .`: Type checking
- `uv run ruff check`: Linting
- `uv run ruff format`: Code formatting
- `uv run pytest tests/ -v`: Run test suite

### Training

- `PYTHONPATH=. uv run python scripts/train_time_mmd_cv.py --train-baseline --seed 42`: Train both multimodal and fine-tuned baseline on same CV splits (recommended)
- `PYTHONPATH=. uv run python scripts/train_time_mmd_cv.py --seed 42`: Train only multimodal TimesFM

### Evaluation

- `PYTHONPATH=. uv run python scripts/evaluate_time_mmd_cv.py --cv-results logs/cv_results.json`: Evaluate multimodal model only
- `PYTHONPATH=. uv run python scripts/evaluate_time_mmd_cv.py --cv-results logs/cv_results.json --compare-baseline`: Compare with pretrained baseline (untrained)
- `PYTHONPATH=. uv run python scripts/evaluate_time_mmd_cv.py --cv-results logs/cv_results.json --baseline-cv-results logs/baseline_finetuned_cv_results.json`: Compare with fine-tuned baseline
- `PYTHONPATH=. uv run python scripts/evaluate_time_mmd_cv.py --cv-results logs/cv_results.json --compare-arima`: Compare with ARIMA baseline
- `PYTHONPATH=. uv run python scripts/evaluate_time_mmd_cv.py --cv-results logs/cv_results.json --compare-arima --arima-order 5 1 2`: Compare with ARIMA using custom order (p, d, q)
- `PYTHONPATH=. uv run python scripts/evaluate_time_mmd_cv.py --cv-results logs/cv_results.json --compare-baseline --baseline-cv-results logs/baseline_finetuned_cv_results.json --compare-arima`: Compare all models (multimodal, pretrained, fine-tuned, ARIMA)

### Visualization & Forecasting

- `PYTHONPATH=. uv run python scripts/visualize_time_mmd_cv.py`: Visualize model predictions
- `PYTHONPATH=. uv run python scripts/forecast_time_mmd.py --cv-results logs/cv_results.json --context-len 512 --horizon-len 128`: Generate forecasts with custom context/horizon lengths

## Data Access

The Time-MMD dataset is included as a git submodule in `data/Time-MMD/`. It contains:

- **Numerical data**: `data/Time-MMD/numerical/[domain]/[domain].csv` - Time series data for 10 domains
- **Textual data**: `data/Time-MMD/textual/[domain]/[domain]_report.csv` and `[domain]_search.csv` - Text descriptions

Domains include: Agriculture, Climate, Economy, Energy, Environment, Health_AFR, Health_US, Security, SocialGood, Traffic.

## Documentation Conventions

- Use docstrings for all functions, classes, and modules
- Include type hints for all function parameters and return values
- Follow Google-style docstring format
- Provide clear, concise descriptions of purpose and functionality
- Document any non-obvious implementation details or design choices
- Include examples where helpful
- Explain parameters, return values, and potential exceptions

# Home folder CLAUDE.md

- @~/.claude/CLAUDE.md
