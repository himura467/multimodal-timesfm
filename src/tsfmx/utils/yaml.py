"""YAML utilities."""

from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a raw dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the loaded data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        ValueError: If the file does not contain a top-level mapping.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML file to contain a mapping, got {type(data).__name__}")
    return data


def parse_yaml(path: Path, cls: type[T]) -> T:
    """Parse a YAML file and return an instance of the given type.

    Args:
        path: Path to the YAML file.
        cls: Type to construct from the top-level YAML mapping.

    Returns:
        Instance of cls constructed from the YAML data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
        ValueError: If the file does not contain a top-level mapping.
    """
    return cls(**load_yaml(path))
