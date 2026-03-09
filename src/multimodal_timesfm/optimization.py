# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
"""PyTorch optimization utilities for multimodal time series forecasting."""

import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
) -> LRScheduler:
    """Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        LambdaLR with the appropriate schedule.
    """
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LRScheduler:
    """Create a schedule with a learning rate that decreases following the values of the cosine function between the
        initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
        initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of waves in the cosine schedule. Defaults to
            a half-cosine (decay from max to 0).
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        LambdaLR with the appropriate schedule.
    """
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
