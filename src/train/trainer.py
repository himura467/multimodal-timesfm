"""Multimodal trainer for TimesFM with text inputs."""

from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
from src.models.multimodal_patched_decoder import MultimodalPatchedDecoder
from src.utils.logging import setup_logger


class MultimodalDataset(Protocol):
    """Protocol for datasets compatible with multimodal training."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, Any]: ...


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
        train_dataset: MultimodalDataset,
        val_dataset: MultimodalDataset,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_dir: str | Path = "logs",
        checkpoint_dir: str | Path = "checkpoints",
        device: str | torch.device | None = None,
        wandb_project: str = "multimodal-timesfm",
        wandb_run_name: str | None = None,
    ) -> None:
        """Initialize MultimodalTrainer.

        Args:
            model: MultimodalPatchedDecoder model to train.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            max_grad_norm: Maximum gradient norm for clipping.
            log_dir: Directory for logs.
            checkpoint_dir: Directory for model checkpoints.
            device: Device to run training on (str or torch.device, auto-detected if None).
            wandb_project: W&B project name.
            wandb_run_name: W&B run name (auto-generated if None).
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Set up device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Set up data loaders
        self.train_loader: DataLoader[Any] = DataLoader(
            train_dataset,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.val_loader: DataLoader[Any] = DataLoader(
            val_dataset,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Set up optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Set up loss function (MSE for forecasting)
        self.loss_fn = nn.MSELoss(reduction="mean")

        # Set up logging
        self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
        self.checkpoint_dir = Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_grad_norm": max_grad_norm,
            },
        )

        # Set up logger
        self.logger = setup_logger(log_file=self.log_dir / "training.log")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate function for multimodal data.

        Args:
            batch: List of samples from the dataset.

        Returns:
            Batched data dictionary.
        """
        # Stack time series data
        time_series = torch.stack([torch.from_numpy(sample["time_series"]) for sample in batch])
        targets = torch.stack([torch.from_numpy(sample["target"]) for sample in batch])

        # Create padding tensors (assume no padding for now)
        batch_size, seq_len, _ = time_series.shape
        input_padding = torch.zeros(batch_size, seq_len, dtype=torch.float32)

        # Create frequency tensor (assume daily frequency for now)
        freq = torch.zeros(batch_size, 1, dtype=torch.long)

        # Collect text descriptions for each batch item
        text_descriptions = []
        for sample in batch:
            # Convert patch texts to the expected format: [batch][patch][texts]
            patched_texts = sample["patched_texts"]
            text_descriptions.append(patched_texts)

        return {
            "time_series": time_series.squeeze(-1),  # Remove last dimension for compatibility
            "input_padding": input_padding,
            "freq": freq,
            "text_descriptions": text_descriptions,
            "targets": targets.squeeze(-1),  # Remove last dimension for compatibility
        }

    def train_epoch(self) -> float:
        """Train one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move tensors to device
            time_series = batch["time_series"].to(self.device)
            input_padding = batch["input_padding"].to(self.device)
            freq = batch["freq"].to(self.device)
            targets = batch["targets"].to(self.device)
            text_descriptions = batch["text_descriptions"]  # Keep on CPU as list

            # Forward pass
            outputs = self.model(
                input_ts=time_series,
                input_padding=input_padding,
                freq=freq,
                text_descriptions=text_descriptions,
            )

            # Extract predictions and reshape properly
            # Model output shape: (batch_size, num_patches, patch_len, num_quantiles)
            # We want the mean prediction (index 0 in quantiles) and reshape to horizon_len
            batch_size, num_patches, patch_len, _ = outputs.shape
            predictions = outputs[:, :, :, 0]  # Get mean prediction: (batch_size, num_patches, patch_len)
            predictions = predictions.reshape(batch_size, num_patches * patch_len)  # (batch_size, total_len)

            # Slice to match horizon length (in case output is longer than expected)
            horizon_len = targets.shape[1]
            predictions = predictions[:, :horizon_len]  # (batch_size, horizon_len)

            # Compute loss
            loss = self.loss_fn(predictions, targets)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Log training metrics
                if self.global_step % 100 == 0:
                    metrics = {
                        "train/loss": loss.item() * self.gradient_accumulation_steps,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    }

                    wandb.log(metrics)

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Log progress
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item() * self.gradient_accumulation_steps:.6f}"
                )

        return total_loss / num_batches

    def validate(self) -> float:
        """Run validation loop.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move tensors to device
                time_series = batch["time_series"].to(self.device)
                input_padding = batch["input_padding"].to(self.device)
                freq = batch["freq"].to(self.device)
                targets = batch["targets"].to(self.device)
                text_descriptions = batch["text_descriptions"]  # Keep on CPU as list

                # Forward pass
                outputs = self.model(
                    input_ts=time_series,
                    input_padding=input_padding,
                    freq=freq,
                    text_descriptions=text_descriptions,
                )

                # Extract predictions and reshape properly
                # Model output shape: (batch_size, num_patches, patch_len, num_quantiles)
                batch_size, num_patches, patch_len, _ = outputs.shape
                predictions = outputs[:, :, :, 0]  # Get mean prediction: (batch_size, num_patches, patch_len)
                predictions = predictions.reshape(batch_size, num_patches * patch_len)  # (batch_size, total_len)

                # Slice to match horizon length (in case output is longer than expected)
                horizon_len = targets.shape[1]
                predictions = predictions[:, :horizon_len]  # (batch_size, horizon_len)

                # Compute loss
                loss = self.loss_fn(predictions, targets)

                total_loss += loss.item()
                num_batches += 1

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
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model checkpoint at epoch {self.current_epoch}")

        self.logger.info(f"Saved checkpoint at epoch {self.current_epoch}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(
        self,
        num_epochs: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        save_every: int = 5,
    ) -> None:
        """Main training loop.

        Args:
            num_epochs: Number of epochs to train.
            scheduler: Learning rate scheduler (optional).
            save_every: Save checkpoint every N epochs.
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Train dataset size: {len(self.train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(self.val_dataset)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Log epoch metrics
            epoch_metrics = {
                "epoch/train_loss": train_loss,
                "epoch/val_loss": val_loss,
                "epoch": epoch,
            }

            wandb.log(epoch_metrics)

            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

        self.logger.info("Training completed!")

        # Close W&B run
        wandb.finish()

    def freeze_pretrained_parameters(self) -> None:
        """Freeze TimesFM and text encoder parameters - only train fusion components."""
        for name, param in self.model.named_parameters():
            if not name.startswith("multimodal_fusion"):
                param.requires_grad = False

        self.logger.info("Froze TimesFM and text encoder parameters - only training fusion components")

    def unfreeze_all_parameters(self) -> None:
        """Unfreeze all parameters for full model training."""
        for param in self.model.parameters():
            param.requires_grad = True

        self.logger.info("Unfroze all parameters - training full model")
