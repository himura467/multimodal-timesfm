# Mutex

## Overview

This project provides a unified framework for multimodal time series foundation models (TSFMs), enabling diverse input modalities such as text. It consists of a core Python package (`mutex`) and example implementations using the [Time-MMD](https://github.com/AdityaLab/Time-MMD) dataset.

## Bash Commands

- `uv run ty check`: Type checking
- `uv run ruff check`: Linting
- `uv run ruff format`: Code formatting
- `uv run pytest tests/ -v`: Run test suite

## Data Access

The Time-MMD dataset is cloned into `data/Time-MMD/` via `scripts/clone_time_mmd.sh` and split into train/val/test via `scripts/split_time_mmd_datasets.py`. It contains:

- **Numerical data**: `data/Time-MMD/numerical/[domain]/[domain].csv` - Time series data for 10 domains
- **Textual data**: `data/Time-MMD/textual/[domain]/[domain]_report.csv` and `[domain]_search.csv` - Text descriptions

Domains include: Agriculture, Climate, Economy, Energy, Environment, Health_AFR, Health_US, Security, SocialGood, Traffic.

## Documentation Conventions

- Use docstrings for all functions, classes, and modules
- Include type hints for all function parameters and return values
- Follow Google-style docstring format
- Provide clear, concise descriptions of purpose and functionality
- Document any non-obvious implementation details or design choices
- Explain parameters, return values, and potential exceptions
