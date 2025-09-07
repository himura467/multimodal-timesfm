"""YAML utilities for reading and writing YAML files."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: Path) -> dict[str, Any]:
    """Load data from YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing the loaded data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML file to contain a dictionary, got {type(data).__name__}")
        return data
