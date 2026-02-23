"""Trainer for multimodal and baseline time series forecasting."""

from collections.abc import Iterator
from typing import cast

import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

from multimodal_timesfm.data.collate import baseline_collate_fn, multimodal_collate_fn
from multimodal_timesfm.multimodal_decoder import MultimodalDecoder
from multimodal_timesfm.training_args import TrainingArguments
from multimodal_timesfm.types import Batch, PreprocessedSample, TrainingMode
from multimodal_timesfm.utils.device import pin_memory
from multimodal_timesfm.utils.logging import get_logger

_logger = get_logger()


class MultimodalTrainer:
    """Trainer for multimodal and baseline time series forecasting.

    In multimodal mode, the adapter is frozen and only fusion parameters are trained.
    In baseline mode, the adapter is fine-tuned and fusion is unused.
    """

    def __init__(
        self,
        model: MultimodalDecoder,
        args: TrainingArguments,
        train_dataset: ConcatDataset[PreprocessedSample],
        val_dataset: ConcatDataset[PreprocessedSample],
        mode: TrainingMode,
        device: torch.device,
        wandb_run: wandb.Run | None,
    ) -> None:
        """Initialize MultimodalTrainer.

        Args:
            model: MultimodalDecoder model to train.
            args: Training arguments.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            mode: Training mode — 'multimodal' trains fusion only, 'baseline' fine-tunes adapter.
            device: Device to train on.
            wandb_run: W&B run instance for logging. If None, W&B logging is disabled.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.mode = mode
        self.device = device
        self._wandb_run = wandb_run

        self.model.to(self.device)

        if mode == "multimodal":
            self.model.adapter.freeze_parameters()
        else:
            self.model.adapter.unfreeze_parameters()

        collate_fn = multimodal_collate_fn if mode == "multimodal" else baseline_collate_fn
        self.train_loader: DataLoader[PreprocessedSample] = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=pin_memory(self.device),
        )
        self.val_loader: DataLoader[PreprocessedSample] = DataLoader(
            val_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=pin_memory(self.device),
        )

        self.optimizer = AdamW(
            self._get_trainable_params(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.loss_fn = nn.MSELoss()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def _get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return the parameters to optimize based on mode."""
        if self.mode == "multimodal":
            return self.model.fusion.parameters()
        return (p for p in self.model.adapter.parameters() if p.requires_grad)

    def _forward_batch(self, batch: Batch, horizon: int) -> torch.Tensor:
        """Run forward pass on a batch, handling text_embeddings if present.

        Returns:
            point_forecast tensor of shape (batch_size, horizon).
        """
        context = batch["context"].to(self.device)
        input_padding = torch.zeros_like(context)
        text_embeddings = batch["text_embeddings"].to(self.device) if "text_embeddings" in batch else None
        return cast(torch.Tensor, self.model(horizon, context, input_padding, text_embeddings))

    def train_epoch(self) -> float:
        """Train one epoch.

        Returns:
            Average training loss for the epoch.

        Raises:
            RuntimeError: If the training dataset is empty.
        """
        self.model.train()
        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise RuntimeError("Training dataset is empty.")

        total_loss = 0.0
        for i, batch in enumerate(self.train_loader):
            horizon = batch["horizon"].to(self.device)
            horizon_len = horizon.shape[-1]
            point_forecast = self._forward_batch(batch, horizon_len)

            loss = self.loss_fn(point_forecast, horizon)
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            scaled_loss = loss.item() * self.args.gradient_accumulation_steps

            if (i + 1) % self.args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self._get_trainable_params(), self.args.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if (
                    self.args.logging_strategy == "steps"
                    and self.global_step % self.args.logging_steps == 0
                    and self._wandb_run is not None
                ):
                    self._wandb_run.log(
                        {
                            "train/loss": scaled_loss,
                        },
                        step=self.global_step,
                    )

            total_loss += scaled_loss

            if i % self.args.logging_steps == 0:
                _logger.info(
                    "Epoch %d, Batch %d/%d, Loss: %.6f",
                    self.current_epoch,
                    i,
                    num_batches,
                    scaled_loss,
                )

        return total_loss / num_batches

    def validate_epoch(self) -> float:
        """Run one validation epoch.

        Returns:
            Average validation loss for the epoch.

        Raises:
            RuntimeError: If the validation dataset is empty.
        """
        self.model.eval()
        num_batches = len(self.val_loader)
        if num_batches == 0:
            raise RuntimeError("Validation dataset is empty.")

        total_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                horizon = batch["horizon"].to(self.device)
                horizon_len = horizon.shape[-1]
                point_forecast = self._forward_batch(batch, horizon_len)

                loss = self.loss_fn(point_forecast, horizon).item()
                total_loss += loss

                if i % self.args.logging_steps == 0:
                    _logger.info(
                        "Epoch %d, Batch %d/%d, Val Loss: %.6f",
                        self.current_epoch,
                        i,
                        num_batches,
                        loss,
                    )

        val_loss = total_loss / num_batches
        if self._wandb_run is not None:
            self._wandb_run.log({"val/loss": val_loss}, step=self.global_step)
        return val_loss
