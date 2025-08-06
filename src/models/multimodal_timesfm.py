"""MultimodalTimesFM wrapper class that extends TimesFM to support multimodal inputs."""

from typing import Any, Sequence

import numpy as np
from timesfm import TimesFmCheckpoint, TimesFmHparams
from timesfm.timesfm_torch import TimesFmTorch as TimesFm


class MultimodalTimesFM:
    """Wrapper class for TimesFM that delegates to underlying TimesFM model."""

    def __init__(self, hparams: TimesFmHparams, checkpoint: TimesFmCheckpoint) -> None:
        """Initialize MultimodalTimesFM wrapper.

        Args:
            hparams: Hyperparameters of the model
            checkpoint: Checkpoint to load. checkpoint.version decides which TimesFM version to use
        """
        # Initialize the underlying TimesFM model
        self.timesfm = TimesFm(hparams, checkpoint)

    def forecast(
        self, inputs: Sequence[Any], freq: Sequence[int] | None = None, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast method that delegates to underlying TimesFM."""
        return self.timesfm.forecast(inputs, freq=freq, **kwargs)  # type: ignore[no-any-return]
