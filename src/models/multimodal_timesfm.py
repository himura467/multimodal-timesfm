"""MultimodalTimesFM wrapper class that provides a high-level interface for multimodal forecasting."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch

from src.models.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from src.utils.device import resolve_device


@dataclass(kw_only=True)
class TimesFmHparams:
    """Hparams used to initialize a TimesFM model for inference.

    These are the sufficient subset of hparams to configure TimesFM inference
    agnostic to the checkpoint version, and are not necessarily the same as the
    hparams used to train the checkpoint.

    Attributes:
      context_len: Largest context length the model allows for each decode call.
        This technically can be any large, but practically should set to the
        context length the checkpoint was trained with.
      horizon_len: Forecast horizon.
      input_patch_len: Input patch len.
      output_patch_len: Output patch len. How many timepoints is taken from a
        single step of autoregressive decoding. Can be set as the training horizon
        of the checkpoint.
      num_layers: Number of transformer layers in the model.
      model_dims: Model dimension.
      per_core_batch_size: Batch size on each core for data parallelism.
      backend: One of "cpu", "gpu" or "tpu".
      quantiles: Which quantiles are output by the model.
    """

    context_len: int = 512
    horizon_len: int = 128
    input_patch_len: int = 32
    output_patch_len: int = 128
    num_layers: int = 20
    num_heads: int = 16
    model_dims: int = 1280
    per_core_batch_size: int = 32
    backend: Literal["cpu", "gpu", "tpu"] = "gpu"
    quantiles: Sequence[float] | None = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    use_positional_embedding: bool = True
    # Hparams beyond the model.
    point_forecast_mode: Literal["mean", "median"] = "median"


class MultimodalTimesFM:
    """High-level wrapper class for multimodal TimesFM forecasting.

    This class provides a clean interface for using the multimodal TimesFM architecture,
    which extends Google's TimesFM with text encoding capabilities. It wraps the
    MultimodalPatchedDecoder and provides methods for loading checkpoints and generating
    forecasts that incorporate both time series data and text descriptions.

    The wrapper maintains compatibility with the original TimesFM interface while adding
    multimodal functionality through:
    - Text encoding using sentence transformers
    - Multimodal fusion mechanisms
    - Enhanced preprocessing for text-augmented time series

    Example:
        ```python
        hparams = TimesFmHparams(context_len=512, horizon_len=128)
        config = MultimodalTimesFMConfig(text_encoder_type="english")
        model = MultimodalTimesFM(hparams, config, "checkpoint.pt")

        forecasts, quantiles = model.forecast(
            inputs=[time_series_data],
            text_descriptions=[[[["Market volatility high"]]]],
            freq=[0],
            forecast_context_len=128
        )
        ```
    """

    def __init__(
        self,
        hparams: TimesFmHparams,
        config: MultimodalTimesFMConfig,
        checkpoint_path: Path | str,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize MultimodalTimesFM wrapper with model configuration and checkpoint.

        Args:
            hparams: TimesFM hyperparameters controlling model architecture (context length,
                    horizon length, patch sizes, etc.) and inference behavior.
            config: Multimodal-specific configuration including text encoder type and
                   TimesFM decoder parameters.
            checkpoint_path: Path to saved model checkpoint containing trained weights.
                            Can be either a string path or Path object.
            device: PyTorch device for model execution. If None, automatically selects
                   the best available device (CUDA GPU if available, otherwise CPU).

        Raises:
            FileNotFoundError: If the specified checkpoint file doesn't exist.
        """
        self.hparams = hparams
        self.device = resolve_device(device)
        self.model = MultimodalPatchedDecoder(config, self.device)

        self.load_from_checkpoint(checkpoint_path)

    def _preprocess(
        self, inputs: Sequence[np.ndarray], freq: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Formats and pads raw time series inputs for batch processing.

        This function prepares input time series by:
        1. Padding/truncating each series to match the required context length
        2. Creating padding masks to indicate valid vs padded positions
        3. Ensuring batch size alignment for efficient parallel processing

        Args:
            inputs: List of 1D numpy arrays, each representing a time series context
                   for a single forecasting task. Series can have varying lengths.
            freq: List of frequency encodings corresponding to each input time series.
                 Values should be 0 (high freq), 1 (medium freq), or 2 (low freq).

        Returns:
            Tuple containing:
            - Padded input time series array of shape (batch_size, context_len)
            - Padding indicator array of shape (batch_size, context_len + horizon_len)
              where 1 indicates padding and 0 indicates valid data
            - Frequency array of shape (batch_size, 1) with frequency encodings
            - Number of padding examples added to align batch size with per_core_batch_size
        """

        input_ts, input_padding, inp_freq = [], [], []

        pmap_pad = ((len(inputs) - 1) // self.hparams.per_core_batch_size + 1) * self.hparams.per_core_batch_size - len(
            inputs
        )

        for i, ts in enumerate(inputs):
            input_len = ts.shape[0]
            padding = np.zeros(shape=(input_len + self.hparams.horizon_len,), dtype=float)
            if input_len < self.hparams.context_len:
                num_front_pad = self.hparams.context_len - input_len
                ts = np.concatenate([np.zeros(shape=(num_front_pad,), dtype=float), ts], axis=0)
                padding = np.concatenate([np.ones(shape=(num_front_pad,), dtype=float), padding], axis=0)
            elif input_len > self.hparams.context_len:
                ts = ts[-self.hparams.context_len :]
                padding = padding[-(self.hparams.context_len + self.hparams.horizon_len) :]

            input_ts.append(ts)
            input_padding.append(padding)
            inp_freq.append(freq[i])

        # Padding the remainder batch.
        for _ in range(pmap_pad):
            input_ts.append(input_ts[-1])
            input_padding.append(input_padding[-1])
            inp_freq.append(inp_freq[-1])

        return (
            np.stack(input_ts, axis=0),
            np.stack(input_padding, axis=0),
            np.array(inp_freq).astype(np.int32).reshape(-1, 1),
            pmap_pad,
        )

    def load_from_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def forecast(
        self,
        inputs: Sequence[Any],
        text_descriptions: list[list[list[str]]],
        freq: Sequence[int],
        forecast_context_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate multimodal forecasts following the TimesFM interface.

        This method extends the original TimesFM forecast functionality to incorporate
        text descriptions alongside time series data. The text information is used to
        enhance forecasting accuracy by providing contextual information about the
        time series patterns, events, or domain-specific knowledge.

        Args:
            inputs: List of time series contexts, where each element is a numpy array
                   or array-like object of shape (context_length,) for univariate series.
                   Series can have different lengths but will be padded/truncated to
                   forecast_context_len.
            text_descriptions: Nested list structure [input_idx][patch_idx][text_list]
                              where each patch of each input time series can have multiple
                              associated text descriptions. The structure must match the
                              number of inputs and patches generated from each input.
            freq: List of frequency encodings for each input time series, where
                 0=high frequency, 1=medium frequency, 2=low frequency.
            forecast_context_len: Maximum context length to use for forecasting.
                                 Time series longer than this will be truncated to
                                 the most recent forecast_context_len points.

        Returns:
            Tuple containing:
            - Point forecasts: numpy array of shape (num_inputs, horizon_len) containing
              mean/median predictions based on point_forecast_mode setting
            - Full forecasts: numpy array of shape (num_inputs, horizon_len, 1 + num_quantiles)
              where the first channel contains mean predictions and remaining channels
              contain quantile predictions as specified in the model configuration

        Raises:
            ValueError: If number of inputs is not compatible with per_core_batch_size
                       for parallel processing.
        """
        if forecast_context_len is None:
            forecast_context_len = self.hparams.context_len
        inputs = [np.array(ts)[-forecast_context_len:] for ts in inputs]

        input_ts, input_padding, inp_freq, pmap_pad = self._preprocess(inputs, freq)
        if pmap_pad > 0:
            raise ValueError("Number of inputs must be a multiple of per_core_batch_size.")

        with torch.no_grad():
            mean_outputs = []
            full_outputs = []
            for i in range(input_ts.shape[0] // self.hparams.per_core_batch_size):
                batch_start = i * self.hparams.per_core_batch_size
                batch_end = (i + 1) * self.hparams.per_core_batch_size

                t_input_ts = torch.Tensor(input_ts[batch_start:batch_end]).to(self.device)
                t_input_padding = torch.Tensor(input_padding[batch_start:batch_end]).to(self.device)
                t_inp_freq = torch.LongTensor(inp_freq[batch_start:batch_end, :]).to(self.device)
                batched_text_descriptions = text_descriptions[batch_start:batch_end]

                t_mean_output, t_full_output = self.model.decode(
                    input_ts=t_input_ts,
                    paddings=t_input_padding,
                    freq=t_inp_freq,
                    horizon_len=self.hparams.horizon_len,
                    text_descriptions=batched_text_descriptions,
                )

                if self.hparams.backend == "gpu":
                    t_mean_output = t_mean_output.cpu()
                    t_full_output = t_full_output.cpu()
                mean_output = t_mean_output.detach().numpy()
                full_output = t_full_output.detach().numpy()
                mean_outputs.append(mean_output)
                full_outputs.append(full_output)

        mean_outputs = np.concatenate(mean_outputs, axis=0)
        full_outputs = np.concatenate(full_outputs, axis=0)

        return mean_outputs, full_outputs
