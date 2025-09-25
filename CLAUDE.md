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
│       ├── trainer.py                       # Training logic
│       └── utils/                           # Utility modules
│           ├── __init__.py
│           ├── collate.py                   # Data collation utilities
│           ├── device.py                    # Device utilities
│           ├── logging.py                   # Logging utilities
│           ├── seed.py                      # Random seed utilities
│           └── yaml.py                      # YAML configuration utilities
├── scripts/
│   ├── train.py                             # Training script for Time-MMD
│   └── evaluate.py                          # Evaluation script for Time-MMD
├── examples/
│   └── time_mmd/                            # Time-MMD dataset example components
│       ├── configs/                         # Time-MMD specific configurations
│       │   ├── __init__.py
│       │   ├── training.py                  # Training configuration
│       │   ├── evaluation.py                # Evaluation configuration
│       │   └── model.py                     # Model architecture settings
│       └── data/
│           ├── __init__.py
│           └── time_mmd_dataset.py          # Time-MMD dataset loader
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
│   ├── test_multimodal_fusion.py
│   └── test_multimodal_patched_decoder.py
├── .gitmodules                              # Git submodule configuration
├── .python-version                          # Python 3.11
├── pyproject.toml                           # Package configuration for PyPI
├── README.md
└── CLAUDE.md                                # This file
```

## Bash Commands

- `uv run mypy .`: Type checking
- `uv run ruff check`: Linting
- `uv run ruff format`: Code formatting
- `uv run pytest tests/ -v`: Run test suite
- `PYTHONPATH=. uv run python scripts/train.py`: Train multimodal TimesFM on Time-MMD
- `PYTHONPATH=. uv run python scripts/evaluate.py`: Evaluate trained models

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
