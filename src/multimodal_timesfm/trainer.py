"""Trainer for multimodal and baseline time series forecasting."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import ConcatDataset, DataLoader

from multimodal_timesfm.data.collate import baseline_collate_fn, multimodal_collate_fn
from multimodal_timesfm.decoder import MultimodalDecoder
from multimodal_timesfm.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from multimodal_timesfm.training_args import TrainingArguments
from multimodal_timesfm.types import (
    BaselineCheckpoint,
    Batch,
    MultimodalCheckpoint,
    PreprocessedSample,
    TrainingMode,
)
from multimodal_timesfm.utils.device import pin_memory
from multimodal_timesfm.utils.logging import get_logger

if TYPE_CHECKING:
    import wandb

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
        optimizers: tuple[Optimizer | None, LRScheduler | None] = (None, None),
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
            optimizers: A tuple containing the optimizer and the scheduler to use.
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
        self.train_loader = cast(
            DataLoader[Batch],
            DataLoader(
                train_dataset,
                batch_size=args.per_device_train_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=pin_memory(self.device),
            ),
        )
        self.val_loader = cast(
            DataLoader[Batch],
            DataLoader(
                val_dataset,
                batch_size=args.per_device_eval_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=pin_memory(self.device),
            ),
        )

        self.loss_fn = nn.MSELoss()

        optimizer, lr_scheduler = optimizers
        num_training_steps = args.num_train_epochs * math.ceil(
            len(self.train_loader) / args.gradient_accumulation_steps
        )
        self.optimizer = self._resolve_optimizer(optimizer)
        self.lr_scheduler = self._resolve_scheduler(lr_scheduler, num_training_steps)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return the parameters to optimize based on mode."""
        if self.mode == "multimodal":
            return self.model.fusion.parameters()
        return (p for p in self.model.adapter.parameters() if p.requires_grad)

    def _create_optimizer(self) -> Optimizer:
        """Create the optimizer.

        Returns:
            The optimizer instance.
        """
        return AdamW(
            self._get_trainable_params(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _create_scheduler(
        self,
        num_training_steps: int,
    ) -> LRScheduler:
        """Create the scheduler.

        Args:
            num_training_steps: The number of training steps to do.

        Returns:
            The learning rate scheduler instance.
        """
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        match self.args.lr_scheduler_type:
            case "linear":
                return get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            case "cosine":
                return get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            case _:
                raise NotImplementedError(f"Unsupported lr_scheduler_type: {self.args.lr_scheduler_type!r}")

    def _resolve_optimizer(self, optimizer: Optimizer | None) -> Optimizer:
        """Return the provided optimizer or create one if None.

        Args:
            optimizer: An externally provided optimizer, or None to create a default one.

        Returns:
            The optimizer instance.
        """
        if optimizer is not None:
            return optimizer
        return self._create_optimizer()

    def _resolve_scheduler(self, lr_scheduler: LRScheduler | None, num_training_steps: int) -> LRScheduler:
        """Return the provided scheduler or create one if None.

        Args:
            lr_scheduler: An externally provided scheduler, or None to create a default one.
            num_training_steps: The total number of training steps.

        Returns:
            The learning rate scheduler instance.
        """
        if lr_scheduler is not None:
            return lr_scheduler
        return self._create_scheduler(num_training_steps)

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
            context = batch["context"].to(self.device)
            horizon = batch["horizon"].to(self.device)
            horizon_len = horizon.shape[-1]
            input_padding = torch.zeros_like(context, dtype=torch.bool)
            text_embeddings = batch["text_embeddings"].to(self.device) if "text_embeddings" in batch else None
            point_forecast = cast(torch.Tensor, self.model(horizon_len, context, input_padding, text_embeddings))

            loss = self.loss_fn(point_forecast, horizon)
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            scaled_loss = loss.item() * self.args.gradient_accumulation_steps

            if (i + 1) % self.args.gradient_accumulation_steps == 0 or (i + 1) == num_batches:
                if self.args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self._get_trainable_params(), self.args.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.global_step += 1

                if (
                    self.args.logging_strategy == "steps"
                    and self.global_step % self.args.logging_steps == 0
                    and self._wandb_run is not None
                ):
                    self._wandb_run.log(
                        {
                            "train/loss": scaled_loss,
                            "train/lr": self.optimizer.param_groups[0]["lr"],
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
                context = batch["context"].to(self.device)
                horizon = batch["horizon"].to(self.device)
                horizon_len = horizon.shape[-1]
                input_padding = torch.zeros_like(context, dtype=torch.bool)
                text_embeddings = batch["text_embeddings"].to(self.device) if "text_embeddings" in batch else None
                point_forecast = cast(torch.Tensor, self.model(horizon_len, context, input_padding, text_embeddings))

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

        return total_loss / num_batches

    def _build_checkpoint(self) -> MultimodalCheckpoint | BaselineCheckpoint:
        """Build a mode-specific checkpoint dict from current training state."""
        if self.mode == "multimodal":
            return MultimodalCheckpoint(
                epoch=self.current_epoch,
                global_step=self.global_step,
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.lr_scheduler.state_dict(),
                best_val_loss=self.best_val_loss,
                fusion_state_dict=self.model.fusion.state_dict(),
            )
        return BaselineCheckpoint(
            epoch=self.current_epoch,
            global_step=self.global_step,
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.lr_scheduler.state_dict(),
            best_val_loss=self.best_val_loss,
            adapter_state_dict=self.model.adapter.state_dict(),
        )

    def _load_checkpoint_state(self, checkpoint: MultimodalCheckpoint | BaselineCheckpoint) -> None:
        """Load mode-specific state dict from checkpoint."""
        if self.mode == "multimodal":
            self.model.fusion.load_state_dict(cast(MultimodalCheckpoint, checkpoint)["fusion_state_dict"])
        else:
            self.model.adapter.load_state_dict(cast(BaselineCheckpoint, checkpoint)["adapter_state_dict"])

    def _rotate_checkpoints(self) -> None:
        if self.args.save_total_limit is None:
            return

        checkpoints = sorted(
            self.args.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        if len(checkpoints) > self.args.save_total_limit:
            for checkpoint in checkpoints[: -self.args.save_total_limit]:
                checkpoint.unlink()
                _logger.info("Deleted old checkpoint: %s", checkpoint.name)

    def save_checkpoint(self, val_loss: float) -> None:
        """Save model checkpoint.

        For 'epoch' strategy, saves every epoch and separately tracks the best.
        For 'best' strategy, saves only when val_loss improves.

        Args:
            val_loss: Current validation loss.
        """
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        if self.args.save_strategy == "best" and not is_best:
            return

        checkpoint = self._build_checkpoint()

        if self.args.save_strategy == "epoch":
            checkpoint_path = self.args.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            _logger.info("Saved checkpoint at epoch %d", self.current_epoch)

            if self.args.save_total_limit is not None:
                self._rotate_checkpoints()

        if is_best:
            best_path = self.args.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            _logger.info("Saved best model checkpoint at epoch %d", self.current_epoch)

    def train(self) -> None:
        """Main training loop."""
        if self.args.eval_strategy != "epoch":
            raise NotImplementedError(
                f"eval_strategy={self.args.eval_strategy!r} is not supported; only 'epoch' is implemented."
            )

        _logger.info("Starting %s training for %d epochs", self.mode, self.args.num_train_epochs)
        _logger.info("Training on %s", self.device)
        _logger.info("Train dataset size: %d", len(self.train_dataset))
        _logger.info("Validation dataset size: %d", len(self.val_dataset))

        for epoch in range(self.args.num_train_epochs):
            self.current_epoch = epoch
            epoch_lr = self.optimizer.param_groups[0]["lr"]

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            _logger.info("Epoch %d: Train Loss = %.6f, Val Loss = %.6f", epoch, train_loss, val_loss)

            if self._wandb_run is not None:
                if self.args.logging_strategy == "epoch":
                    self._wandb_run.log(
                        {
                            "train/loss": train_loss,
                            "train/lr": epoch_lr,
                            "val/loss": val_loss,
                        },
                        step=self.global_step,
                    )
                else:
                    self._wandb_run.log({"val/loss": val_loss}, step=self.global_step)

            if self.args.save_strategy in ("epoch", "best"):
                self.save_checkpoint(val_loss)

        if self.args.load_best_model_at_end:
            best_path = self.args.checkpoint_dir / "best_model.pt"
            if best_path.exists():
                checkpoint = cast(MultimodalCheckpoint | BaselineCheckpoint, torch.load(best_path, weights_only=True))
                self._load_checkpoint_state(checkpoint)
                _logger.info("Loaded best model at end of training")

        _logger.info("Training completed")
