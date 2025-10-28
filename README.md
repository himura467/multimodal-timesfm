# Multimodal TimesFM

A multimodal extension of Google's [TimesFM](https://github.com/google-research/timesfm) for time series forecasting with text inputs.

## Installation

```bash
pip install multimodal-timesfm
```

## Quick Start

### 1. Define Custom Dataset

Implement a custom dataset by extending `MultimodalDatasetBase`:

```python
from pathlib import Path
from typing import Literal
import numpy as np
from multimodal_timesfm.multimodal_dataset import MultimodalDatasetBase

class CustomDataset(MultimodalDatasetBase):
    """Custom dataset for your multimodal time series data."""

    def __init__(
        self,
        data_dir: Path,
        split_ratio: float = 0.8,
        split: Literal["train", "test"] = "train",
        patch_len: int = 32,
        context_len: int = 128,
        horizon_len: int = 32,
    ):
        super().__init__(data_dir, split_ratio, split, patch_len, context_len, horizon_len)

    def _load_data(self) -> None:
        """Load and process your custom data format.

        Populate self.data with dictionaries containing:
        - context: np.ndarray of shape (context_len,) - historical time series values
        - future: np.ndarray of shape (horizon_len,) - target future values
        - freq: int - frequency indicator (0=daily, 1=weekly/monthly, 2=quarterly+)
        - patched_texts: list of lists - text organized by temporal patches
        - metadata: dict - additional sample information
        """
        # Your custom data loading logic
        for sample in self._read_your_data_files():
            # Organize texts by patches (one list per patch)
            num_patches = self.context_len // self.patch_len
            patched_texts = [[] for _ in range(num_patches)]

            # Assign text descriptions to appropriate patches
            for text_item in sample["texts"]:
                patch_idx = self._get_patch_index(text_item["timestamp"])
                patched_texts[patch_idx].append(text_item["text"])

            self.data.append({
                "context": sample["historical_values"],  # shape: (context_len,)
                "future": sample["target_values"],       # shape: (horizon_len,)
                "freq": sample["frequency"],             # 0, 1, or 2
                "patched_texts": patched_texts,          # list of text lists
                "metadata": sample["info"]
            })
```

### 2. Train the Model

```python
from multimodal_timesfm.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from multimodal_timesfm.trainer import MultimodalTrainer

# Create datasets
train_dataset = CustomDataset(data_dir="path/to/data", split="train")
val_dataset = CustomDataset(data_dir="path/to/data", split="test")

# Initialize model with pretrained TimesFM weights
config = MultimodalTimesFMConfig(
    text_encoder_type="english",  # or "japanese"
    context_len=128,
    horizon_len=32,
    input_patch_len=32,
)
model = MultimodalPatchedDecoder(config)

# Load pretrained TimesFM checkpoint
model.load_pretrained_timesfm("path/to/timesfm_checkpoint.ckpt")

# Train
trainer = MultimodalTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    log_dir=Path("logs"),
    checkpoint_dir=Path("checkpoints"),
    wandb_project="my-project",
    wandb_run_name="experiment-1"
)

# Train for 10 epochs, saving checkpoints every 5 epochs
trainer.train(num_epochs=10, save_frequency=5)
```

### 3. Evaluate the Model

```python
from multimodal_timesfm.evaluation import evaluate_multimodal_model
from torch.utils.data import DataLoader
from multimodal_timesfm.utils.collate import multimodal_collate_fn

# Create test dataset
test_dataset = CustomDataset(data_dir="path/to/data", split="test")
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=multimodal_collate_fn
)

# Load trained model
model = MultimodalPatchedDecoder(config)
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluate
metrics = evaluate_multimodal_model(model, test_loader, device="cuda")
print(f"MSE: {metrics['mse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

### 4. Visualize Predictions

```python
import matplotlib.pyplot as plt
import numpy as np

# Get predictions for visualization
model.eval()
sample = test_dataset[0]
with torch.no_grad():
    prediction = model.forecast(
        context=torch.tensor(sample["context"]).unsqueeze(0),
        text_inputs=[sample["patched_texts"]],
        freq=torch.tensor([sample["freq"]])
    )

# Plot
plt.figure(figsize=(12, 4))
context_len = len(sample["context"])
horizon_len = len(sample["future"])

# Plot context
plt.plot(range(context_len), sample["context"], label="Context", color="blue")

# Plot ground truth
plt.plot(range(context_len, context_len + horizon_len),
         sample["future"], label="Ground Truth", color="green")

# Plot prediction
plt.plot(range(context_len, context_len + horizon_len),
         prediction[0].cpu().numpy(), label="Prediction", color="red", linestyle="--")

plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("Multimodal TimesFM Forecast")
plt.savefig("forecast_visualization.png")
```

### 5. Inference on New Data

```python
from multimodal_timesfm import MultimodalTimesFM, TimesFmHparams, MultimodalTimesFMConfig

# Load trained model for inference
hparams = TimesFmHparams(context_len=128, horizon_len=32)
config = MultimodalTimesFMConfig(text_encoder_type="english")
model = MultimodalTimesFM(hparams, config, "checkpoints/best_model.pt")

# Prepare new data
time_series_data = np.array([...])  # Your time series context
text_descriptions = [[
    ["High volatility expected"],        # Texts for patch 1
    ["Market uncertainty increasing"],   # Texts for patch 2
    ["Economic indicators show growth"]  # Texts for patch 3
]]

# Generate forecast
forecasts, quantiles = model.forecast(
    inputs=[time_series_data],
    text_descriptions=text_descriptions,
    freq=[0],  # 0=daily, 1=weekly/monthly, 2=quarterly+
    forecast_context_len=128
)

print(f"Forecast shape: {forecasts.shape}")
print(f"Point forecast: {forecasts[0]}")
print(f"Quantiles shape: {quantiles.shape}")
```

## Features

- **Multimodal forecasting**: Combines time series data with textual context
- **Built on TimesFM**: Leverages Google's state-of-the-art time series foundation model
- **Flexible text encoding**: Supports English and Japanese text inputs
- **Easy integration**: Simple API for adding text context to time series forecasting

## Time-MMD Dataset Example

The project includes complete scripts for training and evaluating on the [Time-MMD](https://github.com/AdityaLab/Time-MMD) dataset.

### Setup

Initialize the Time-MMD dataset submodule:

```bash
git submodule update --init
```

The dataset contains multimodal time series data across 10 domains: Agriculture, Climate, Economy, Energy, Environment, Health_AFR, Health_US, Security, SocialGood, and Traffic.

### Training with Cross-Validation

Train a multimodal TimesFM model using cross-validation:

```bash
PYTHONPATH=. uv run python scripts/train_time_mmd_cv.py \
    --seed 42
```

**Configuration:**

- Model config: TimesFM architecture parameters (layers, dimensions, context length, etc.)
- Training config: Batch size, learning rate, domains, cross-validation settings, etc.
- See [examples/time_mmd/configs/](examples/time_mmd/configs/) for configuration templates

The script will:

- Create train/validation splits for each cross-validation fold
- Train a separate model for each fold
- Save checkpoints and cross-validation results to JSON

### Evaluation

Evaluate trained models on the test set:

```bash
PYTHONPATH=. uv run python scripts/evaluate_time_mmd_cv.py \
    --cv-results logs/cv_results.json \
    --seed 42
```

This evaluates both the multimodal model and a baseline TimesFM model (without text inputs), reporting:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Per-fold and overall metrics

### Visualization

Visualize model predictions:

```bash
PYTHONPATH=. uv run python scripts/visualize_time_mmd_cv.py \
    --cv-results logs/cv_results.json
```

**Output:**

- Time series plots showing context, ground truth, and predictions
- Metric comparison bar charts (MSE and MAE)

### Forecasting with Custom Parameters

Generate forecasts with manually configurable context and horizon lengths:

```bash
# Forecast on all folds
PYTHONPATH=. uv run python scripts/forecast_time_mmd.py \
    --cv-results logs/cv_results.json \
    --context-len 512 \
    --horizon-len 128

# Forecast on a specific fold
PYTHONPATH=. uv run python scripts/forecast_time_mmd.py \
    --cv-results logs/cv_results.json \
    --fold 0 \
    --context-len 512 \
    --horizon-len 128

# Forecast with custom settings
PYTHONPATH=. uv run python scripts/forecast_time_mmd.py \
    --cv-results logs/cv_results.json \
    --context-len 256 \
    --horizon-len 64 \
    --num-samples 10 \
    --output-dir custom_plots
```

This script compares multimodal model forecasts against baseline TimesFM forecasts, providing:

- Time series plots comparing multimodal vs baseline predictions
- Bar charts comparing MSE and MAE metrics
- JSON output with all forecasts and metrics

## Acknowledgments

We thank the [Time-MMD](https://github.com/AdityaLab/Time-MMD) team for providing the multimodal time series dataset used in our examples and experiments.

## License

MIT
