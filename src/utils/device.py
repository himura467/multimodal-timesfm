"""Device management utilities for consistent device handling across the project."""

import torch


def get_default_device() -> torch.device:
    """Get the default device for the current system.

    Returns the best available device in priority order:
    1. CUDA (GPU) if available
    2. MPS (Apple Silicon GPU) if available
    3. CPU as fallback

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """Resolve device specification to a torch.device object.

    Args:
        device: Device specification. Can be:
            - None: Auto-detect best available device
            - str: Device string like "cuda", "mps", "cpu"
            - torch.device: Already resolved device object

    Returns:
        torch.device: Resolved device object.
    """
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return get_default_device()


def move_to_device(
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    exclude_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Move tensor dictionary to specified device.

    Args:
        tensors: Dictionary of tensors to move.
        device: Target device.
        exclude_keys: Keys to exclude from device movement.

    Returns:
        Dictionary with tensors moved to device.
    """
    if exclude_keys is None:
        exclude_keys = set()

    return {key: tensor.to(device) if key not in exclude_keys else tensor for key, tensor in tensors.items()}


def get_pin_memory(device: torch.device) -> bool:
    """Get appropriate pin_memory setting for device.

    Args:
        device: Target device.

    Returns:
        bool: Whether to use pinned memory.
    """
    return device.type == "cuda"
