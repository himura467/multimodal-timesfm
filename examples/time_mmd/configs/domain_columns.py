"""Domain-specific column configuration for Time-MMD dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainColumnConfig:
    """Configuration for columns in a specific domain's dataset.

    Attributes:
        start_date_col: Column name to use for start date (e.g., 'start_date').
        end_date_col: Column name to use for end date (e.g., 'end_date').
        time_series_cols: List of column names to use as time series data.
    """

    start_date_col: str
    end_date_col: str
    time_series_cols: list[str]

    def get_time_series_columns(self, all_columns: list[str]) -> list[str]:
        """Get the list of columns to use as time series data.

        Args:
            all_columns: All columns in the dataframe.

        Returns:
            List of column names to use as time series data that exist in the dataframe.
        """
        return [col for col in self.time_series_cols if col in all_columns]


@dataclass
class DomainColumnsConfig:
    """Configuration for all domains in the Time-MMD dataset.

    Attributes:
        default: Fallback configuration applied to domains not listed in `domains`.
        domains: Per-domain overrides keyed by domain name (e.g., 'Agriculture').
    """

    default: DomainColumnConfig
    domains: dict[str, DomainColumnConfig] = field(default_factory=dict)

    def get_config_for_domain(self, domain: str) -> DomainColumnConfig:
        """Get column configuration for a specific domain.

        Args:
            domain: Domain name (e.g., 'Agriculture').

        Returns:
            DomainColumnConfig for the specified domain, or default if not configured.
        """
        return self.domains.get(domain, self.default)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> DomainColumnsConfig:
        """Create DomainColumnsConfig from a dictionary.

        Args:
            config_dict: Dictionary containing domain column configurations.

        Returns:
            DomainColumnsConfig instance.
        """
        default_config = DomainColumnConfig(**config_dict.get("default", {}))
        domains_dict = {}

        for domain, domain_config in config_dict.get("domains", {}).items():
            domains_dict[domain] = DomainColumnConfig(**domain_config)

        return cls(default=default_config, domains=domains_dict)


# Predefined column configurations for all Time-MMD domains.
# Most domains use 'start_date'/'end_date' date columns and 'OT' as the time series column.
# Health_AFR is the only exception: its start date column is named 'date' instead of 'start_date'.
DEFAULT_TIME_MMD_CONFIGS = DomainColumnsConfig(
    default=DomainColumnConfig(
        start_date_col="start_date",
        end_date_col="end_date",
        time_series_cols=["OT"],
    ),
    domains={
        "Health_AFR": DomainColumnConfig(
            start_date_col="date",
            end_date_col="end_date",
            time_series_cols=["OT"],
        ),
    },
)
