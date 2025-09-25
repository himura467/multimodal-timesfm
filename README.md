# Multimodal TimesFM

A multimodal extension of Google's [TimesFM](https://github.com/google-research/timesfm) for time series forecasting with text inputs.

## Installation

```bash
pip install multimodal-timesfm
```

## Quick Start

```python
from multimodal_timesfm import MultimodalTimesFM, TimesFmHparams, MultimodalTimesFMConfig

# Configure model
hparams = TimesFmHparams(context_len=512, horizon_len=128)
config = MultimodalTimesFMConfig(text_encoder_type="english")
model = MultimodalTimesFM(hparams, config, "checkpoint.pt")

# Forecast with time series and text descriptions
forecasts, quantiles = model.forecast(
    inputs=[time_series_data],
    text_descriptions=[[[["Market volatility high"]]]],
    freq=[0],
    forecast_context_len=128
)
```

## Features

- **Multimodal forecasting**: Combines time series data with textual context
- **Built on TimesFM**: Leverages Google's state-of-the-art time series foundation model
- **Flexible text encoding**: Supports English and Japanese text inputs
- **Easy integration**: Simple API for adding text context to time series forecasting

## Examples

See the `examples/` directory for complete usage examples including training on the Time-MMD dataset.

## Acknowledgments

We thank the [Time-MMD](https://github.com/AdityaLab/Time-MMD) team for providing the multimodal time series dataset used in our examples and experiments.

## License

MIT
