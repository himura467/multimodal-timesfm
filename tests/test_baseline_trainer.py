"""Tests for baseline TimesFM trainer."""

import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder, TimesFMConfig

from examples.time_mmd.configs import ModelConfig
from multimodal_timesfm.baseline_trainer import BaselineTrainer
from multimodal_timesfm.multimodal_dataset import MultimodalDatasetBase
from multimodal_timesfm.training_args import TrainingArguments


class MockBaselineDataset(MultimodalDatasetBase):
    """Mock dataset for testing baseline training loop."""

    def __init__(self, size: int = 100, context_len: int = 128, horizon_len: int = 32):
        self.size = size

        # Initialize base class - will call _load_data()
        super().__init__(
            data_dir=Path(tempfile.mkdtemp()),
            split_ratio=0.8,
            split="train",
            patch_len=32,
            context_len=context_len,
            horizon_len=horizon_len,
        )

    def _load_data(self) -> None:
        """Load synthetic data."""

        # Generate synthetic data
        self.data = []
        for i in range(self.size):
            # Generate synthetic time series
            context = np.random.randn(self.context_len, 1).astype(np.float32)
            future = np.random.randn(self.horizon_len, 1).astype(np.float32)

            # Generate mock text patches
            num_patches = self.context_len // self.patch_len
            patched_texts = []
            for j in range(num_patches):
                patch_texts = [f"Sample {i} patch {j} text description"]
                patched_texts.append(patch_texts)

            sample = {
                "context": context,
                "future": future,
                "freq": 0,
                "patched_texts": patched_texts,
                "metadata": {
                    "domain": "Agriculture",
                    "column": f"test_col_{i}",
                    "start_index": 0,
                },
            }
            self.data.append(sample)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


class TestBaselineTrainer:
    """Test cases for BaselineTrainer class."""

    @pytest.fixture(scope="session")
    def model_config(self) -> TimesFMConfig:
        """Load model configuration from YAML file."""
        config = ModelConfig()

        return TimesFMConfig(
            num_layers=config.timesfm.num_layers,
            num_heads=config.timesfm.num_heads,
            num_kv_heads=config.timesfm.num_kv_heads,
            hidden_size=config.timesfm.model_dims,
            intermediate_size=config.timesfm.model_dims,
            head_dim=config.timesfm.model_dims // config.timesfm.num_heads,
            rms_norm_eps=config.timesfm.rms_norm_eps,
            patch_len=config.timesfm.input_patch_len,
            horizon_len=config.timesfm.output_patch_len,
            quantiles=config.timesfm.quantiles,
            pad_val=config.timesfm.pad_val,
            tolerance=config.timesfm.tolerance,
            dtype=config.timesfm.dtype,
            use_positional_embedding=config.timesfm.use_positional_embedding,
        )

    @pytest.fixture(scope="session")
    def mock_datasets(self) -> tuple[MockBaselineDataset, MockBaselineDataset]:
        """Create mock training and validation datasets."""
        # Set seeds for reproducible test data
        torch.manual_seed(42)
        np.random.seed(42)

        train_dataset = MockBaselineDataset(size=10, context_len=128, horizon_len=128)
        val_dataset = MockBaselineDataset(size=5, context_len=128, horizon_len=128)
        return train_dataset, val_dataset

    @pytest.fixture
    def model(self, model_config: TimesFMConfig) -> PatchedTimeSeriesDecoder:
        """Create baseline TimesFM model."""
        return PatchedTimeSeriesDecoder(model_config)

    @pytest.fixture(scope="session")
    def training_args(self) -> Generator[TrainingArguments, None, None]:
        """Create training arguments for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield TrainingArguments(
                output_dir=temp_dir,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=2,
                num_train_epochs=1,
                logging_steps=10,
                save_strategy="epoch",
                eval_strategy="epoch",
            )

    @pytest.fixture(scope="session", autouse=True)
    def mock_wandb(self) -> Generator[Mock, None, None]:
        """Mock wandb to avoid initialization and deprecation warnings during tests."""
        with patch("multimodal_timesfm.baseline_trainer.wandb") as mock_wandb:
            mock_wandb.init = Mock()
            mock_wandb.log = Mock()
            mock_wandb.finish = Mock()
            yield mock_wandb

    def test_trainer_initialization(
        self,
        model: PatchedTimeSeriesDecoder,
        mock_datasets: tuple[MockBaselineDataset, MockBaselineDataset],
        training_args: TrainingArguments,
    ) -> None:
        """Test baseline trainer initialization."""
        train_dataset, val_dataset = mock_datasets

        trainer = BaselineTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            freeze_timesfm=False,
        )

        # Test trainer properties
        assert trainer.model is not None
        assert len(trainer.train_loader) > 0
        assert len(trainer.val_loader) > 0
        assert trainer.device.type in ["cuda", "mps", "cpu"]
        assert trainer.freeze_timesfm is False

    def test_frozen_mode(
        self,
        model: PatchedTimeSeriesDecoder,
        mock_datasets: tuple[MockBaselineDataset, MockBaselineDataset],
        training_args: TrainingArguments,
    ) -> None:
        """Test baseline trainer with frozen TimesFM parameters."""
        train_dataset, val_dataset = mock_datasets

        trainer = BaselineTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            freeze_timesfm=True,
        )

        assert trainer.freeze_timesfm is True

        # Check that all parameters are frozen
        for param in trainer.model.parameters():
            assert param.requires_grad is False

        # Check that optimizer is None (no trainable parameters)
        assert trainer.optimizer is None

        # Verify that attempting to train raises an error
        with pytest.raises(RuntimeError, match="Cannot train with frozen model"):
            trainer.train_epoch()

    def test_training_mode(
        self,
        model: PatchedTimeSeriesDecoder,
        mock_datasets: tuple[MockBaselineDataset, MockBaselineDataset],
        training_args: TrainingArguments,
    ) -> None:
        """Test baseline trainer with trainable TimesFM parameters."""
        train_dataset, val_dataset = mock_datasets

        trainer = BaselineTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            freeze_timesfm=False,
        )

        assert trainer.freeze_timesfm is False

        # Check that parameters are trainable
        trainable_params = [p for p in trainer.model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0

    def test_forward_pass(
        self,
        model: PatchedTimeSeriesDecoder,
        mock_datasets: tuple[MockBaselineDataset, MockBaselineDataset],
        training_args: TrainingArguments,
    ) -> None:
        """Test single forward pass through the baseline model."""
        train_dataset, val_dataset = mock_datasets

        trainer = BaselineTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            freeze_timesfm=False,
        )

        # Get a batch from the trainer
        sample_batch = next(iter(trainer.train_loader))

        # Check batch structure
        assert "context" in sample_batch
        assert "future" in sample_batch
        assert "freq" in sample_batch

        # Move batch to device
        context = sample_batch["context"].to(trainer.device)
        freq = sample_batch["freq"].to(trainer.device)

        input_padding = torch.zeros_like(context)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ts=context,
                input_padding=input_padding,
                freq=freq,
            )

        # Check output shape
        batch_size = context.shape[0]
        expected_shape = (batch_size, 4, 128, 10)  # Based on model config
        assert outputs.shape == expected_shape

    def test_training_loop(
        self,
        model: PatchedTimeSeriesDecoder,
        mock_datasets: tuple[MockBaselineDataset, MockBaselineDataset],
        training_args: TrainingArguments,
    ) -> None:
        """Test training loop execution."""
        train_dataset, val_dataset = mock_datasets

        trainer = BaselineTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            freeze_timesfm=False,
        )

        # Count trainable parameters
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        # Should have all parameters trainable when not frozen
        assert trainable_before == total_params

        # Test short training run
        trainer.train()

        # Test checkpoint exists
        checkpoint_dir = training_args.checkpoint_dir
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) in [1, 2]  # Epoch checkpoint, possibly best model too
