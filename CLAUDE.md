# Multimodal TimesFM

## Overview

This project provides a multimodal extension of Google's [TimesFM](https://github.com/google-research/timesfm) to support time series forecasting with text inputs. It consists of a core Python package and example implementations using the [Time-MMD](https://github.com/AdityaLab/Time-MMD) dataset.

## Bash Commands

- `uv run mypy src/ scripts/ examples/ tests/`: Type checking
- `uv run ruff check`: Linting
- `uv run ruff format`: Code formatting
- `uv run pytest tests/ -v`: Run test suite

## Data Access

The Time-MMD dataset is cloned into `data/Time-MMD/` via `setup_time_mmd.sh`. It contains:

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
