"""Time-MMD dataset loader for multimodal time series forecasting."""

from pathlib import Path
from typing import Any, Literal

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
        data_dir: Path,
        domain: str,
        split_ratio: float = 0.8,
        split: Literal["train", "test"] = "train",
        patch_len: int = 32,
        context_len: int = 128,
        horizon_len: int = 32,
    ) -> None:
        """Initializes Time-MMD dataset loader.

        Args:
            data_dir: Root directory containing Time-MMD dataset.
            domain: Domain name (e.g., 'Agriculture').
            split_ratio: Train/test split ratio (default 0.8 for 80% train).
            split: Dataset split ('train' or 'test').
            patch_len: Length of input patches for temporal alignment with time series data.
            context_len: Length of context window for input sequences.
                         context_len must be an integer multiple of patch_len.
            horizon_len: Length of forecasting horizon (target sequence length).
                         horizon_len must be an integer multiple of patch_len.
        """
        # Validate that context_len is an integer multiple of patch_len
        if context_len % patch_len != 0:
            raise ValueError(f"context_len ({context_len}) must be an integer multiple of patch_len ({patch_len})")

        # Validate that horizon_len is an integer multiple of patch_len
        if horizon_len % patch_len != 0:
            raise ValueError(f"horizon_len ({horizon_len}) must be an integer multiple of patch_len ({patch_len})")

        self.data_dir = data_dir
        self.domain = domain
        self.split_ratio = split_ratio
        self.split = split
        self.patch_len = patch_len
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

        # Sort numerical_df by end_date to ensure chronological order
        if "end_date" in numerical_df.columns:
            numerical_df = numerical_df.sort_values("end_date").reset_index(drop=True)

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

        # Prepare date series for efficient lookup
        if "start_date" in numerical_df.columns:
            full_start_dates = numerical_df["start_date"]
        elif "Date" in numerical_df.columns:
            full_start_dates = numerical_df["Date"]
        elif "date" in numerical_df.columns:
            full_start_dates = numerical_df["date"]
        else:
            raise ValueError("No start_date column found. Expected at least one of: 'start_date', 'Date', 'date'")

        if "end_date" in numerical_df.columns:
            full_end_dates = numerical_df["end_date"]
        else:
            raise ValueError("No end_date column found in numerical data")

        # Process each numeric column as a separate time series
        for column in numeric_cols:
            # Extract time series from this column
            time_series_values = numerical_df.loc[:, column].to_numpy()

            # Skip if insufficient data for context + horizon
            min_length = self.context_len + self.horizon_len
            if len(time_series_values) < min_length:
                continue

            # Split data based on split_ratio
            split_idx = int(len(time_series_values) * self.split_ratio)

            if self.split == "train":
                ts_data = time_series_values[:split_idx]
                start_dates = full_start_dates.iloc[:split_idx]
                end_dates = full_end_dates.iloc[:split_idx]
            else:  # test
                ts_data = time_series_values[split_idx:]
                start_dates = full_start_dates.iloc[split_idx:]
                end_dates = full_end_dates.iloc[split_idx:]

            # Skip if insufficient data after split
            if len(ts_data) < min_length:
                continue

            # Create windowed samples from this time series
            for start_idx in range(0, len(ts_data) - min_length + 1, self.horizon_len):
                # Extract context window
                context_end = start_idx + self.context_len
                time_series = ts_data[start_idx:context_end].reshape(-1, 1)

                # Extract target
                target_end = context_end + self.horizon_len
                target = ts_data[context_end:target_end].reshape(-1, 1)

                # Get associated text for this time period
                window_start_date = str(start_dates.iloc[start_idx])
                window_end_date = str(end_dates.iloc[target_end - 1])

                # Calculate frequency based on interval between end_date values
                freq = self._calculate_frequency_for_sample(end_dates, start_idx, target_end)

                # Calculate number of text patches based on context_len / patch_len
                text_patches_num = self.context_len // self.patch_len

                patched_texts = self._get_patched_texts_for_period(
                    window_start_date, window_end_date, textual_data, text_patches_num
                )

                sample = {
                    "time_series": time_series.astype(np.float32),
                    "patched_texts": patched_texts,
                    "target": target.astype(np.float32),
                    "freq": freq,
                    "metadata": {
                        "domain": self.domain,
                        "column": column,
                        "start_index": start_idx,
                    },
                }
                self.data.append(sample)

    def _get_patched_texts_for_period(
        self, start_date: str, end_date: str, textual_data: dict[str, pd.DataFrame], text_patches_num: int
    ) -> list[list[str]]:
        """Gets patched textual descriptions for a specific time period.

        Args:
            start_date: Start date of the time period (YYYY-MM-DD format).
            end_date: End date of the time period (YYYY-MM-DD format).
            textual_data: Dictionary containing textual dataframes.
            text_patches_num: Number of text patches to generate for this period.

        Returns:
            List of lists where each inner list contains text data for one patch period.
            Returns text_patches_num number of lists.
        """
        # Convert dates to pandas datetime for comparison
        period_start = pd.to_datetime(start_date)
        period_end = pd.to_datetime(end_date)

        # Divide the time period into equal parts
        period_duration = period_end - period_start
        patch_duration = period_duration / text_patches_num

        patches = []

        for i in range(text_patches_num):
            # Calculate patch time boundaries
            patch_start = period_start + i * patch_duration
            patch_end = period_start + (i + 1) * patch_duration

            patch_reports = []

            # Get reports that overlap with this patch period
            if "reports" in textual_data:
                reports_df = textual_data["reports"]
                if "start_date" in reports_df.columns and "end_date" in reports_df.columns:
                    reports_df = reports_df.copy()
                    reports_df["start_date"] = pd.to_datetime(reports_df["start_date"])
                    reports_df["end_date"] = pd.to_datetime(reports_df["end_date"])

                    matching_reports = reports_df[
                        (reports_df["start_date"] <= patch_end) & (reports_df["end_date"] >= patch_start)
                    ]

                    for _, row in matching_reports.iterrows():
                        if "fact" in reports_df.columns and pd.notna(row["fact"]):
                            patch_reports.append(f"Report: {str(row['fact'])}")
                        if "preds" in reports_df.columns and pd.notna(row["preds"]):
                            patch_reports.append(f"Prediction: {str(row['preds'])}")

            # Get search data that overlaps with this patch period
            if "search" in textual_data:
                search_df = textual_data["search"]
                if "start_date" in search_df.columns and "end_date" in search_df.columns:
                    search_df = search_df.copy()
                    search_df["start_date"] = pd.to_datetime(search_df["start_date"])
                    search_df["end_date"] = pd.to_datetime(search_df["end_date"])

                    matching_search = search_df[
                        (search_df["start_date"] <= patch_end) & (search_df["end_date"] >= patch_start)
                    ]

                    for _, row in matching_search.iterrows():
                        if "fact" in search_df.columns and pd.notna(row["fact"]) and str(row["fact"]) != "NA":
                            patch_reports.append(f"Search: {str(row['fact'])}")
                        if "preds" in search_df.columns and pd.notna(row["preds"]) and str(row["preds"]) != "NA":
                            patch_reports.append(f"Search prediction: {str(row['preds'])}")

            patches.append(patch_reports)

        return patches

    def _calculate_frequency_for_sample(self, end_dates: pd.Series, start_idx: int, target_end: int) -> int:
        """Calculate frequency value based on interval between end_date values.

        Args:
            end_dates: Series of end dates for the time series.
            start_idx: Starting index of the sample.
            target_end: Ending index of the sample.

        Returns:
            Frequency value:
            - 0 for daily or lower granularity
            - 1 for weekly or monthly granularity
            - 2 for quarterly or higher granularity
        """
        if target_end - start_idx < 2:
            return 0  # Default to daily if insufficient data

        # Convert to datetime and calculate intervals for the entire sample range
        sample_dates = pd.to_datetime(end_dates.iloc[start_idx:target_end])
        intervals = sample_dates.diff().dropna()

        if len(intervals) == 0:
            return 0  # Default to daily

        # Calculate average interval across all data points in the sample
        avg_interval = intervals.mean()
        avg_days = pd.Timedelta(avg_interval).total_seconds() / (24 * 3600)  # Convert to days

        # Classify based on average interval
        if avg_days < 3:
            return 0
        elif avg_days < 35:  # Weekly to monthly (up to ~5 weeks)
            return 1
        else:  # Quarterly or higher
            return 2

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
