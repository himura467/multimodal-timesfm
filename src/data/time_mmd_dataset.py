"""Time-MMD dataset loader for multimodal time series forecasting."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TimeMmdDataset(Dataset[dict[str, Any]]):
    """Dataset loader for Time-MMD dataset with time series and text data.

    This class loads multimodal time series data from the Time-MMD dataset structure,
    which contains numerical time series data in domain-specific CSV files and
    corresponding textual information (reports and search data) for each domain.

    Expected directory structure:
        data_dir/
        ├── numerical/
        │   ├── Agriculture/
        │   │   └── Agriculture.csv
        │   └── (Domain)/
        │       └── (Domain).csv
        └── textual/
            └── (Domain)/
                ├── (Domain)_report.csv
                └── (Domain)_search.csv
    """

    def __init__(
        self,
        data_dir: str | Path,
        domain: str,
        split_ratio: float = 0.7,
        split: str = "train",
        context_len: int = 512,
        horizon_len: int = 128,
    ) -> None:
        """Initializes Time-MMD dataset loader.

        Args:
            data_dir: Root directory containing Time-MMD dataset.
            domain: Domain name (e.g., 'Agriculture').
            split_ratio: Train/test split ratio (default 0.7 for 70% train).
            split: Dataset split ('train' or 'test').
            context_len: Length of context window for input sequences.
            horizon_len: Length of forecasting horizon (target sequence length).
        """
        self.data_dir = Path(data_dir)
        self.domain = domain
        self.split_ratio = split_ratio
        self.split = split
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.data: list[dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Loads Time-MMD dataset from files."""
        numerical_file = self.data_dir / "numerical" / self.domain / f"{self.domain}.csv"
        textual_dir = self.data_dir / "textual" / self.domain

        if not numerical_file.exists():
            raise FileNotFoundError(f"Numerical data file not found: {numerical_file}")

        # Load numerical time series data
        numerical_df = pd.read_csv(numerical_file)

        # Load textual data if available
        report_file = textual_dir / f"{self.domain}_report.csv"
        search_file = textual_dir / f"{self.domain}_search.csv"

        textual_data = {}
        if report_file.exists():
            textual_data["reports"] = pd.read_csv(report_file)
        if search_file.exists():
            textual_data["search"] = pd.read_csv(search_file)

        self._process_data(numerical_df, textual_data)

    def _process_data(self, numerical_df: pd.DataFrame, textual_data: dict[str, pd.DataFrame]) -> None:
        """Processes loaded dataframes into internal format.

        Args:
            numerical_df: Dataframe containing numerical time series data.
            textual_data: Dictionary containing textual dataframes (reports, search).
        """
        # Identify numeric columns (exclude date columns)
        date_cols = ["Date", "date", "start_date", "end_date"]
        numeric_cols = [
            col
            for col in numerical_df.columns
            if col not in date_cols and pd.api.types.is_numeric_dtype(numerical_df[col])
        ]

        # Process each numeric column as a separate time series
        for col_idx, column in enumerate(numeric_cols):
            # Extract time series from this column
            time_series_values = numerical_df[column].dropna().values

            # Skip if insufficient data for context + horizon
            min_length = self.context_len + self.horizon_len
            if len(time_series_values) < min_length:
                continue

            # Split data based on split_ratio
            split_idx = int(len(time_series_values) * self.split_ratio)

            if self.split == "train":
                ts_data = time_series_values[:split_idx]
            else:  # test
                ts_data = time_series_values[split_idx:]

            # Skip if insufficient data after split
            if len(ts_data) < min_length:
                continue

            # Create windowed samples from this time series
            # Step size is horizon_len to avoid too much overlap
            step_size = self.horizon_len

            for start_idx in range(0, len(ts_data) - min_length + 1, step_size):
                # Extract context window
                context_end = start_idx + self.context_len
                time_series = np.asarray(ts_data[start_idx:context_end]).reshape(-1, 1)

                # Extract target (next horizon_len values)
                target_end = context_end + self.horizon_len
                target = np.asarray(ts_data[context_end:target_end]).reshape(-1, 1)

                # Get associated text for this column/variable
                text_description = self._get_text_for_series(col_idx, textual_data, column)

                sample = {
                    "time_series": time_series.astype(np.float32),
                    "text": text_description,
                    "target": target.astype(np.float32),
                    "metadata": {
                        "series_id": f"{self.domain}_{column}_{start_idx}",
                        "domain": self.domain,
                        "column": column,
                        "start_index": start_idx,
                    },
                }
                self.data.append(sample)

    def _get_text_for_series(
        self, series_idx: int, textual_data: dict[str, pd.DataFrame], column_name: str = ""
    ) -> str:
        """Gets textual description for a time series.

        Args:
            series_idx: Index of the time series.
            textual_data: Dictionary containing textual dataframes.
            column_name: Name of the column/variable for this time series.

        Returns:
            Combined textual description for the series.
        """
        descriptions = []

        # Add report text if available
        if "reports" in textual_data:
            reports_df = textual_data["reports"]
            if len(reports_df) > 0:
                # Use the first available report entry (not tied to series_idx)
                row = reports_df.iloc[0]
                if "fact" in reports_df.columns and pd.notna(row["fact"]):
                    descriptions.append(f"Report: {str(row['fact'])}")
                if "preds" in reports_df.columns and pd.notna(row["preds"]):
                    descriptions.append(f"Prediction: {str(row['preds'])}")

        # Add search text if available
        if "search" in textual_data:
            search_df = textual_data["search"]
            if len(search_df) > 0:
                # Use the first available search entry (not tied to series_idx)
                row = search_df.iloc[0]
                if "fact" in search_df.columns and pd.notna(row["fact"]) and str(row["fact"]) != "NA":
                    descriptions.append(f"Search: {str(row['fact'])}")
                if "preds" in search_df.columns and pd.notna(row["preds"]) and str(row["preds"]) != "NA":
                    descriptions.append(f"Search prediction: {str(row['preds'])}")

        # Return combined description or default
        if descriptions:
            return " ".join(descriptions)
        else:
            return f"Time series from {self.domain} domain, variable: {column_name}"

    def __len__(self) -> int:
        """Returns dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Gets dataset item by index.

        Args:
            idx: Item index.

        Returns:
            Dictionary containing time series, text, target, and metadata.
        """
        return self.data[idx]

    def get_split_info(self) -> dict[str, Any]:
        """Gets information about the dataset split.

        Returns:
            Dictionary with split information including size, shapes, and metadata.
        """
        return {
            "domain": self.domain,
            "split": self.split,
            "split_ratio": self.split_ratio,
            "size": len(self.data),
            "data_dir": str(self.data_dir),
            "has_text": len(self.data) > 0 and "text" in self.data[0],
            "time_series_shape": self.data[0]["time_series"].shape if self.data else None,
            "target_shape": self.data[0]["target"].shape if self.data else None,
        }
