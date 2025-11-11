"""ARIMA baseline forecasting for comparison with multimodal TimesFM."""

from typing import Any

import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal_timesfm.utils.device import move_to_device


def forecast_arima(
    context: np.ndarray,
    horizon_len: int,
    order: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    """Forecast using ARIMA model.

    Args:
        context: Historical time series data of shape (context_len,).
        horizon_len: Number of future steps to forecast.
        order: ARIMA order (p, d, q). Default (1, 1, 1).

    Returns:
        Forecasted values of shape (horizon_len,).
    """
    # Fit ARIMA model on context
    model = ARIMA(context, order=order)
    fitted_model = model.fit()

    # Forecast horizon_len steps ahead
    forecast = fitted_model.forecast(steps=horizon_len)

    return np.asarray(forecast)


def evaluate_arima_model(
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
    order: tuple[int, int, int] = (1, 1, 1),
) -> dict[str, float]:
    """Evaluate ARIMA model on test dataset.

    Args:
        dataloader: DataLoader providing batches of samples.
        device: Device to use for evaluation (for consistency with other models).
        order: ARIMA order (p, d, q). Default (1, 1, 1).

    Returns:
        Dictionary containing evaluation metrics (mse, mae).
    """
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating ARIMA"):
        batch_tensors = move_to_device({"context": batch["context"], "future": batch["future"]}, device)
        context = batch_tensors["context"]
        future = batch_tensors["future"]

        batch_size = context.size(0)

        # Process each sample in the batch
        for i in range(batch_size):
            # Extract context and future for this sample
            context_sample = context[i].squeeze().cpu().numpy()  # (context_len,)
            future_sample = future[i].squeeze().cpu().numpy()  # (horizon_len,)

            # Forecast using ARIMA
            horizon_len = len(future_sample)
            forecast = forecast_arima(context_sample, horizon_len, order)

            # Compute metrics
            mse = np.mean((forecast - future_sample) ** 2)
            mae = np.mean(np.abs(forecast - future_sample))

            total_mse += mse
            total_mae += mae
            num_samples += 1

    # Raise error if no samples were processed
    if num_samples == 0:
        raise ValueError("No samples were processed during ARIMA evaluation")

    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples

    return {
        "mse": float(avg_mse),
        "mae": float(avg_mae),
    }
