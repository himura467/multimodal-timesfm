"""MultimodalTimesFM wrapper class that extends TimesFM to support multimodal inputs."""

from typing import Any, Sequence

import numpy as np
from timesfm import TimesFmCheckpoint, TimesFmHparams
from timesfm.timesfm_torch import TimesFmTorch as TimesFm


class MultimodalTimesFM:
    """Wrapper class for TimesFM that delegates to underlying TimesFM model.

    This class serves as a foundation for extending TimesFM to support multimodal inputs.
    Currently, it delegates all functionality to the underlying TimesFM model, but can be
    extended to handle text inputs and multimodal fusion in future iterations.
    """

    def __init__(self, hparams: TimesFmHparams, checkpoint: TimesFmCheckpoint) -> None:
        """Initializes MultimodalTimesFM wrapper.

        Args:
            hparams: Hyperparameters of the model.
            checkpoint: Checkpoint to load. checkpoint.version decides which TimesFM version to use.
        """
        # Initialize the underlying TimesFM model
        self.timesfm = TimesFm(hparams, checkpoint)

    def forecast(
        self, inputs: Sequence[Any], freq: Sequence[int] | None = None, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecasts time series using the underlying TimesFM model.

        Args:
            inputs: Input time series data sequences.
            freq: Optional frequency information for each input sequence.
            **kwargs: Additional keyword arguments passed to `TimesFM.forecast()`.

        Returns:
        A tuple for np.array:
        - The mean forecast of size (# inputs, # forecast horizon).
        - The full forecast (mean + quantiles) of size (# inputs, # forecast horizon, 1 + # quantiles).
        """
        return self.timesfm.forecast(inputs, freq=freq, **kwargs)  # type: ignore[no-any-return]
