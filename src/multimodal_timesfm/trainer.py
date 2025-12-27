"""Multimodal trainer for TimesFM with text inputs."""

from typing import Any

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.types import FileLike
from torch.utils.data import ConcatDataset, DataLoader

from multimodal_timesfm.multimodal_dataset import MultimodalDatasetBase
from multimodal_timesfm.multimodal_patched_decoder import MultimodalPatchedDecoder
from multimodal_timesfm.training_args import TrainingArguments
from multimodal_timesfm.utils.collate import multimodal_collate_fn
from multimodal_timesfm.utils.device import get_pin_memory, move_to_device, resolve_device
from multimodal_timesfm.utils.logging import setup_logger


class MultimodalTrainer:
    """Trainer for multimodal TimesFM model.

    This trainer handles:
    1. Training loop with both text and time series inputs
    2. Loss computation for forecasting tasks
    3. Gradient accumulation for large batches
    4. Checkpointing and logging
    5. Validation loop with metrics
    """

    def __init__(
        self,
        model: MultimodalPatchedDecoder,
        args: TrainingArguments,
        train_dataset: MultimodalDatasetBase | ConcatDataset[dict[str, Any]],
        val_dataset: MultimodalDatasetBase | ConcatDataset[dict[str, Any]],
        init_wandb: bool = True,
    ) -> None:
        """Initialize MultimodalTrainer.

        Args:
            model: MultimodalPatchedDecoder model to train.
            args: Training arguments.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            init_wandb: Whether to initialize wandb (default: True). Set to False for sweep runs.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Set up device
        self.device = resolve_device(args.device)
        self.model.to(self.device)

        # Set up data loaders
        self.train_loader: DataLoader[dict[str, Any]] = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=multimodal_collate_fn,
            pin_memory=get_pin_memory(self.device),
        )
        self.val_loader: DataLoader[dict[str, Any]] = DataLoader(
            val_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=multimodal_collate_fn,
            pin_memory=get_pin_memory(self.device),
        )

        # Set up optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Set up loss function (MSE for forecasting)
        self.loss_fn = nn.MSELoss()

        # Set up logger
        self.logger = setup_logger(log_file=args.logging_dir / "training.log")

        # Initialize W&B (skip if already initialized, e.g., in sweep runs)
        if init_wandb:
            wandb.init(
                project="multimodal-timesfm",
                name=args.run_name,
                config=args.__dict__,
            )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.init_wandb = init_wandb

    def train_epoch(self) -> float:
        """Train one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move tensors to device
            batch_tensors = move_to_device(
                {"context": batch["context"], "future": batch["future"], "freq": batch["freq"]},
                self.device,
            )
            context = batch_tensors["context"]
            future = batch_tensors["future"]
            freq = batch_tensors["freq"]

            # Create input_padding tensor (zeros for now)
            input_padding = torch.zeros_like(context)

            # Forward pass - handle both raw text and pre-computed embeddings
            if "text_embeddings" in batch:
                # Using cached dataset with pre-computed embeddings
                text_embeddings = move_to_device({"text_embeddings": batch["text_embeddings"]}, self.device)[
                    "text_embeddings"
                ]
                predictions = self.model(
                    input_ts=context,
                    input_padding=input_padding.float(),
                    freq=freq,
                    text_embeddings=text_embeddings,
                )
            else:
                # Using raw dataset with text descriptions
                patched_texts = batch["patched_texts"]
                predictions = self.model(
                    input_ts=context,
                    input_padding=input_padding.float(),
                    freq=freq,
                    text_descriptions=patched_texts,
                )

            # Extract predictions following TimesFM implementation
            # Model output shape: (batch_size, num_patches, patch_len, num_quantiles)
            predictions_mean = predictions[..., 0]  # Get mean prediction: (batch_size, num_patches, patch_len)
            last_patch_pred = predictions_mean[:, -1, :]  # Extract last patch: (batch_size, patch_len)

            # Compute loss
            loss = self.loss_fn(last_patch_pred, future)

            # Scale loss for gradient accumulation
            loss = loss / self.args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Log training metrics
                if self.global_step % self.args.logging_steps == 0:
                    metrics = {
                        "train/loss": loss.item() * self.args.gradient_accumulation_steps,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    }

                    wandb.log(metrics)

            total_loss += loss.item() * self.args.gradient_accumulation_steps

            # Log progress
            if batch_idx % self.args.logging_steps == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item() * self.args.gradient_accumulation_steps:.6f}"
                )

        return total_loss / num_batches

    def validate(self) -> float:
        """Run validation loop.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        if num_batches == 0:
            self.logger.warning("Validation dataset is empty, skipping validation")
            return float("inf")

        with torch.no_grad():
            for batch in self.val_loader:
                # Move tensors to device
                batch_tensors = move_to_device(
                    {"context": batch["context"], "future": batch["future"], "freq": batch["freq"]},
                    self.device,
                )
                context = batch_tensors["context"]
                future = batch_tensors["future"]
                freq = batch_tensors["freq"]

                # Create input_padding tensor (zeros for now)
                input_padding = torch.zeros_like(context)

                # Forward pass - handle both raw text and pre-computed embeddings
                if "text_embeddings" in batch:
                    # Using cached dataset with pre-computed embeddings
                    text_embeddings = move_to_device({"text_embeddings": batch["text_embeddings"]}, self.device)[
                        "text_embeddings"
                    ]
                    predictions = self.model(
                        input_ts=context,
                        input_padding=input_padding,
                        freq=freq,
                        text_embeddings=text_embeddings,
                    )
                else:
                    # Using raw dataset with text descriptions
                    patched_texts = batch["patched_texts"]
                    predictions = self.model(
                        input_ts=context,
                        input_padding=input_padding,
                        freq=freq,
                        text_descriptions=patched_texts,
                    )

                # Extract predictions following TimesFM implementation
                # Model output shape: (batch_size, num_patches, patch_len, num_quantiles)
                predictions_mean = predictions[..., 0]  # Get mean prediction: (batch_size, num_patches, patch_len)
                last_patch_pred = predictions_mean[:, -1, :]  # Extract last patch: (batch_size, patch_len)

                # Compute loss
                loss = self.loss_fn(last_patch_pred, future)

                total_loss += loss.item()

        avg_val_loss = total_loss / num_batches
        return avg_val_loss

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best checkpoint so far.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "model_config": self.model.config.__dict__,
        }

        # Save regular checkpoint
        checkpoint_path = self.args.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint at epoch {self.current_epoch}")

        # Save best checkpoint
        if is_best:
            best_path = self.args.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model checkpoint at epoch {self.current_epoch}")

        # Clean up old checkpoints if save_total_limit is set
        if self.args.save_total_limit is not None:
            self._rotate_checkpoints()

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to maintain save_total_limit."""
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Get all checkpoint files (excluding best_model.pt)
        checkpoints = sorted(
            [p for p in self.args.checkpoint_dir.glob("checkpoint_epoch_*.pt")],
            key=lambda p: p.stat().st_mtime,
        )

        # Remove oldest checkpoints if we exceed the limit
        if len(checkpoints) > self.args.save_total_limit:
            for checkpoint in checkpoints[: -self.args.save_total_limit]:
                checkpoint.unlink()
                self.logger.info(f"Deleted old checkpoint: {checkpoint.name}")

    def load_checkpoint(self, checkpoint_path: FileLike) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, scheduler: torch.optim.lr_scheduler.LRScheduler | None = None) -> None:
        """Main training loop.

        Args:
            scheduler: Learning rate scheduler (optional).
        """
        self.logger.info(f"Starting training for {self.args.num_train_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Train dataset size: {len(self.train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(self.val_dataset)}")

        for epoch in range(self.args.num_train_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self.train_epoch()

            # Validate
            if self.args.eval_strategy == "epoch":
                val_loss = self.validate()
            else:
                val_loss = None

            # Log epoch metrics
            epoch_metrics = {"epoch/train_loss": train_loss, "epoch": epoch}
            if val_loss is not None:
                epoch_metrics["epoch/val_loss"] = val_loss

            wandb.log(epoch_metrics)

            if val_loss is not None:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")

            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            # Save checkpoint
            if self.args.save_strategy == "epoch":
                is_best = False
                if val_loss is not None:
                    if val_loss < self.best_val_loss:
                        is_best = True
                        self.best_val_loss = val_loss
                self.save_checkpoint(is_best=is_best)

        # Load best model at end if requested
        if self.args.load_best_model_at_end:
            best_path = self.args.checkpoint_dir / "best_model.pt"
            if best_path.exists():
                self.load_checkpoint(best_path)
                self.logger.info("Loaded best model at end of training")

        self.logger.info("Training completed!")

        # Close W&B run (skip if externally managed, e.g., in sweep runs)
        if self.init_wandb:
            wandb.finish()

    def freeze_pretrained_parameters(self) -> None:
        """Freeze pretrained TimesFM and text encoder parameters - only train fusion components."""
        # Use the model's built-in method to freeze all parameters first
        self.model.freeze_parameters()

        # Then unfreeze only the fusion component for training (keep text encoder frozen)
        self.model.unfreeze_text_components(unfreeze_encoder=False, unfreeze_fusion=True)

        # Log the status
        status = self.model.is_text_frozen()
        self.logger.info(f"Froze pretrained parameters - text components status: {status}")

    def unfreeze_all_parameters(self) -> None:
        """Unfreeze all parameters for full model training."""
        self.model.unfreeze_parameters()

        self.logger.info("Unfroze all parameters - training full model")
