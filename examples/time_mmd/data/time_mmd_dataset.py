"""Time-MMD dataset loader for multimodal time series forecasting."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from examples.time_mmd.configs.domain_columns import DEFAULT_TIME_MMD_CONFIGS, DomainColumnConfig
from multimodal_timesfm.multimodal_dataset import MultimodalDatasetBase


class TimeMmdDataset(MultimodalDatasetBase):
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
        column_config: DomainColumnConfig | None = None,
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
            horizon_len: Length of forecasting horizon.
                        horizon_len must be an integer multiple of patch_len.
            column_config: Optional column configuration for this domain.
                          If None, uses the default configuration from DEFAULT_TIME_MMD_CONFIGS.
        """
        self.domain = domain
        self.column_config = column_config or DEFAULT_TIME_MMD_CONFIGS.get_config_for_domain(domain)
        super().__init__(data_dir, split_ratio, split, patch_len, context_len, horizon_len)

    def _sanitize_time_series(
        self, time_series_values: np.ndarray, start_dates: pd.Series, end_dates: pd.Series
    ) -> tuple[np.ndarray, pd.Series, pd.Series] | None:
        """Sanitize time series by removing leading/trailing invalid values and interpolating all invalid values.

        This function performs the following operations:
        1. Strips leading and trailing NaN/inf/None values from the time series
        2. Interpolates any remaining invalid values (NaN/inf/None) in the middle of the series

        Args:
            time_series_values: Raw time series values from the dataset.
            start_dates: Series of start dates corresponding to each time series value.
            end_dates: Series of end dates corresponding to each time series value.

        Returns:
            Tuple of (sanitized_values, trimmed_start_dates, trimmed_end_dates) if successful,
            None if the series cannot be sanitized (e.g., no valid values exist).
        """
        # Convert to float for consistent handling
        sanitized_values = time_series_values.astype(float)

        # Strip leading and trailing invalid values (NaN/inf/None)
        valid_mask = pd.notna(sanitized_values) & np.isfinite(sanitized_values)

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None

        first_valid_idx = valid_indices[0]
        last_valid_idx = valid_indices[-1]

        # Trim to valid range (remove leading/trailing invalid values)
        sanitized_values = sanitized_values[first_valid_idx : last_valid_idx + 1]
        trimmed_start_dates = start_dates.iloc[first_valid_idx : last_valid_idx + 1].reset_index(drop=True)
        trimmed_end_dates = end_dates.iloc[first_valid_idx : last_valid_idx + 1].reset_index(drop=True)

        # Interpolate any remaining invalid values in the middle of the series
        # This handles NaN, inf, and -inf values uniformly
        if not np.all(np.isfinite(sanitized_values)):
            # Use pandas Series for easy interpolation
            ts_series = pd.Series(sanitized_values)
            # Replace inf/-inf with NaN so pandas can interpolate everything uniformly
            ts_series = ts_series.replace([np.inf, -np.inf], np.nan)
            # Interpolate: first try linear, then forward fill, then backward fill
            ts_series = ts_series.interpolate(method="linear", limit_direction="both")
            ts_series = ts_series.ffill().bfill()
            sanitized_values = ts_series.to_numpy()

        return sanitized_values, trimmed_start_dates, trimmed_end_dates

    def _normalize_sample(self, context: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Normalize context and future using z-score normalization based on context statistics.

        This ensures that samples from different domains with vastly different scales
        (e.g., Energy: 0.95-1.75 vs Security: millions-billions) are normalized to
        comparable ranges for stable training.

        Args:
            context: Context window values of shape (context_len, 1).
            future: Future values to predict of shape (horizon_len, 1).

        Returns:
            Tuple of (normalized_context, normalized_future, context_mean, context_std).
            The normalization parameters can be used to denormalize predictions.
        """
        # Calculate statistics from context window
        context_mean = np.mean(context)
        context_std = np.std(context)

        # Avoid division by zero
        epsilon = 1e-7
        if context_std < epsilon:
            context_std = 1.0

        # Normalize both context and future using context statistics
        context_normalized = (context - context_mean) / context_std
        future_normalized = (future - context_mean) / context_std

        return context_normalized, future_normalized, float(context_mean), float(context_std)

    def _load_data(self) -> None:
        """Loads Time-MMD dataset from files."""
        numerical_file = self.data_dir / "numerical" / self.domain / f"{self.domain}.csv"
        textual_dir = self.data_dir / "textual" / self.domain

        if not numerical_file.exists():
            raise FileNotFoundError(f"Numerical data file not found: {numerical_file}")

        # Load numerical time series data
        numerical_df = pd.read_csv(numerical_file)

        # Sort numerical_df by start_date to ensure chronological order
        start_date_col = self.column_config.start_date_col
        if start_date_col in numerical_df.columns:
            numerical_df = numerical_df.sort_values(start_date_col).reset_index(drop=True)

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
        # Use column configuration to determine which columns to use
        numeric_cols = self.column_config.get_time_series_columns(all_columns=numerical_df.columns.tolist())

        if not numeric_cols:
            raise ValueError(f"No time series columns found for domain '{self.domain}' with the given configuration")

        # Get date columns from configuration
        start_date_col = self.column_config.start_date_col
        end_date_col = self.column_config.end_date_col

        # Prepare date series for efficient lookup
        if start_date_col not in numerical_df.columns:
            raise ValueError(
                f"Start date column '{start_date_col}' not found in numerical data. "
                f"Available columns: {numerical_df.columns.tolist()}"
            )

        if end_date_col not in numerical_df.columns:
            raise ValueError(
                f"End date column '{end_date_col}' not found in numerical data. "
                f"Available columns: {numerical_df.columns.tolist()}"
            )

        full_start_dates = numerical_df[start_date_col]
        full_end_dates = numerical_df[end_date_col]

        # Process each numeric column as a separate time series
        for column in numeric_cols:
            # Extract time series from this column
            time_series_values = numerical_df.loc[:, column].to_numpy()

            # Sanitize time series: strip leading/trailing invalid values and interpolate middle ones
            sanitized = self._sanitize_time_series(time_series_values, full_start_dates, full_end_dates)
            if sanitized is None:
                continue
            sanitized_values, trimmed_start_dates, trimmed_end_dates = sanitized

            # Split data based on split_ratio
            split_idx = int(len(sanitized_values) * self.split_ratio)

            if self.split == "train":
                ts_data = sanitized_values[:split_idx]
                start_dates = trimmed_start_dates.iloc[:split_idx]
                end_dates = trimmed_end_dates.iloc[:split_idx]
            else:  # test
                ts_data = sanitized_values[split_idx:]
                start_dates = trimmed_start_dates.iloc[split_idx:]
                end_dates = trimmed_end_dates.iloc[split_idx:]

            # Skip if insufficient data after split
            if len(ts_data) < self.context_len + self.horizon_len:
                continue

            # Create windowed samples from this time series
            for start_idx in range(0, len(ts_data) - self.context_len - self.horizon_len + 1, self.horizon_len):
                # Extract context
                context_end = start_idx + self.context_len
                context = ts_data[start_idx:context_end].reshape(-1, 1)

                # Extract future
                future_end = context_end + self.horizon_len
                future = ts_data[context_end:future_end].reshape(-1, 1)

                # Normalize context and future using context statistics
                context_normalized, future_normalized, context_mean, context_std = self._normalize_sample(
                    context, future
                )

                # Get associated text for this context
                window_start_date = str(start_dates.iloc[start_idx])
                window_end_date = str(end_dates.iloc[context_end - 1])

                # Calculate frequency based on interval between end_date values
                freq = self._calculate_frequency_for_sample(end_dates, start_idx, context_end)

                # Calculate number of text patches based on context_len / patch_len
                text_patches_num = self.context_len // self.patch_len

                patched_texts = self._get_patched_texts_for_period(
                    window_start_date, window_end_date, textual_data, text_patches_num
                )

                sample = {
                    "context": context_normalized.astype(np.float32),
                    "future": future_normalized.astype(np.float32),
                    "freq": freq,
                    "patched_texts": patched_texts,
                    "metadata": {
                        "domain": self.domain,
                        "column": column,
                        "start_index": start_idx,
                        "mean": context_mean,
                        "std": context_std,
                    },
                }
                self.data.append(sample)

    def _calculate_frequency_for_sample(self, dates: pd.Series, start_idx: int, end_idx: int) -> int:
        """Calculate frequency value based on interval between date values.

        Args:
            dates: Series of dates for the time series.
            start_idx: Starting index of the sample.
            end_idx: Ending index of the sample.

        Returns:
            Frequency value:
            - 0 for daily or lower granularity
            - 1 for weekly or monthly granularity
            - 2 for quarterly or higher granularity
        """
        if end_idx - start_idx < 1:
            return 0  # Default to daily if insufficient data

        # Convert to datetime and calculate intervals for the entire sample range
        sample_dates = pd.to_datetime(dates.iloc[start_idx:end_idx])
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
                        if "fact" in reports_df.columns and pd.notna(row["fact"]) and str(row["fact"]) != "NA":
                            patch_reports.append(f"Report: {str(row['fact'])}")
                        if "preds" in reports_df.columns and pd.notna(row["preds"]) and str(row["preds"]) != "NA":
                            patch_reports.append(f"Report Prediction: {str(row['preds'])}")

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
