"""Device management utilities."""

import torch


def _default_device() -> torch.device:
    """Select the best available device in priority order: cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """Resolve a device specification to a torch.device.

    When device is None, selects the best available device in priority order.
    """
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return _default_device()


def pin_memory(device: torch.device) -> bool:
    """Return whether pinned memory should be used for the given device."""
    return device.type == "cuda"
