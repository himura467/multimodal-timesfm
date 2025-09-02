"""Shared pytest fixtures for all tests."""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_data_dir() -> Generator[Path, None, None]:
    """Creates a temporary directory with sample Time-MMD data structure.

    This fixture is shared across all test classes and creates a temporary directory
    with the Time-MMD dataset structure including numerical and textual data.

    Yields:
        Path: Path to the temporary data directory with sample Time-MMD structure.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)

        # Create directory structure
        numerical_dir = data_dir / "numerical" / "TestDomain"
        textual_dir = data_dir / "textual" / "TestDomain"
        numerical_dir.mkdir(parents=True)
        textual_dir.mkdir(parents=True)

        # Create sample numerical data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        numerical_data = pd.DataFrame(
            {
                "start_date": dates.astype(str),
                "end_date": dates.astype(str),
                "value1": np.sin(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100),
                "value2": np.cos(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100),
                "category": ["A"] * 50 + ["B"] * 50,  # Non-numeric column
            }
        )
        numerical_data.to_csv(numerical_dir / "TestDomain.csv", index=False)

        # Create sample textual data
        report_data = pd.DataFrame(
            {
                "start_date": ["2020-01-01", "2020-01-15", "2020-02-01"],
                "end_date": ["2020-01-14", "2020-01-31", "2020-02-15"],
                "fact": ["Report fact 1", "Report fact 2", "Report fact 3"],
                "preds": ["Report pred 1", np.nan, "Report pred 3"],
            }
        )
        report_data.to_csv(textual_dir / "TestDomain_report.csv", index=False)

        search_data = pd.DataFrame(
            {
                "start_date": ["2020-01-05", "2020-01-20", "2020-01-25"],
                "end_date": ["2020-01-10", "2020-01-25", "2020-01-30"],
                "fact": ["Search fact 1", "Search fact 2", np.nan],
                "preds": ["Search pred 1", "NA", "Search pred 3"],
            }
        )
        search_data.to_csv(textual_dir / "TestDomain_search.csv", index=False)

        yield data_dir
