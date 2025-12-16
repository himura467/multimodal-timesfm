"""Tests for multimodal trainer."""

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from examples.time_mmd.configs.model import ModelConfig
from multimodal_timesfm.multimodal_dataset import MultimodalDatasetBase
from multimodal_timesfm.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from multimodal_timesfm.trainer import MultimodalTrainer
from multimodal_timesfm.training_args import TrainingArguments


class MockMultimodalDataset(MultimodalDatasetBase):
    """Mock dataset for testing training loop."""

    def __init__(self, size: int = 100, context_len: int = 512, horizon_len: int = 128):
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


class TestMultimodalTrainer:
    """Test cases for MultimodalTrainer class."""

    @pytest.fixture(scope="session")
    def model_config(self) -> MultimodalTimesFMConfig:
        """Load model configuration from YAML file."""
        model_config = ModelConfig()

        return MultimodalTimesFMConfig(
            num_layers=model_config.timesfm.num_layers,
            num_heads=model_config.timesfm.num_heads,
            num_kv_heads=model_config.timesfm.num_kv_heads,
            hidden_size=model_config.timesfm.model_dims,
            intermediate_size=model_config.timesfm.model_dims,
            head_dim=model_config.timesfm.model_dims // model_config.timesfm.num_heads,
            rms_norm_eps=model_config.timesfm.rms_norm_eps,
            patch_len=model_config.timesfm.input_patch_len,
            horizon_len=model_config.timesfm.output_patch_len,
            quantiles=model_config.timesfm.quantiles,
            pad_val=model_config.timesfm.pad_val,
            tolerance=model_config.timesfm.tolerance,
            dtype=model_config.timesfm.dtype,
            use_positional_embedding=model_config.timesfm.use_positional_embedding,
            text_encoder_type=model_config.text_encoder.text_encoder_type,
        )

    @pytest.fixture(scope="session")
    def mock_datasets(self) -> tuple[MockMultimodalDataset, MockMultimodalDataset]:
        """Create mock training and validation datasets."""
        # Set seeds for reproducible test data
        torch.manual_seed(42)
        np.random.seed(42)

        train_dataset = MockMultimodalDataset(size=10, context_len=128, horizon_len=128)
        val_dataset = MockMultimodalDataset(size=5, context_len=128, horizon_len=128)
        return train_dataset, val_dataset

    @pytest.fixture(scope="session")
    def model(self, model_config: MultimodalTimesFMConfig) -> MultimodalPatchedDecoder:
        """Create multimodal model."""
        return MultimodalPatchedDecoder(model_config)

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
        with patch("multimodal_timesfm.trainer.wandb") as mock_wandb:
            mock_wandb.init = Mock()
            mock_wandb.log = Mock()
            mock_wandb.finish = Mock()
            yield mock_wandb

    def test_trainer_initialization(
        self,
        model: MultimodalPatchedDecoder,
        mock_datasets: tuple[MockMultimodalDataset, MockMultimodalDataset],
        training_args: TrainingArguments,
    ) -> None:
        """Test trainer initialization."""
        train_dataset, val_dataset = mock_datasets

        trainer = MultimodalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        # Test trainer properties
        assert len(trainer.train_loader) > 0
        assert len(trainer.val_loader) > 0
        assert trainer.device.type in ["cuda", "mps", "cpu"]

    def test_forward_pass(
        self,
        model: MultimodalPatchedDecoder,
        mock_datasets: tuple[MockMultimodalDataset, MockMultimodalDataset],
        training_args: TrainingArguments,
    ) -> None:
        """Test single forward pass through the model."""
        train_dataset, val_dataset = mock_datasets

        trainer = MultimodalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        # Get a batch from the trainer
        sample_batch = next(iter(trainer.train_loader))

        # Check batch structure
        assert "context" in sample_batch
        assert "future" in sample_batch
        assert "freq" in sample_batch
        assert "patched_texts" in sample_batch

        # Move batch to device
        context = sample_batch["context"].to(trainer.device)
        freq = sample_batch["freq"].to(trainer.device)
        patched_texts = sample_batch["patched_texts"]

        input_padding = torch.zeros_like(context)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ts=context,
                input_padding=input_padding,
                freq=freq,
                text_descriptions=patched_texts,
            )

        # Check output shape
        batch_size = context.shape[0]
        expected_shape = (batch_size, 4, 128, 10)  # Based on model config
        assert outputs.shape == expected_shape

    @pytest.mark.ci_unstable
    def test_training_loop(
        self,
        model: MultimodalPatchedDecoder,
        mock_datasets: tuple[MockMultimodalDataset, MockMultimodalDataset],
        training_args: TrainingArguments,
    ) -> None:
        """Test training loop execution."""
        train_dataset, val_dataset = mock_datasets

        trainer = MultimodalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        # Test parameter freezing
        trainer.freeze_pretrained_parameters()

        # Count trainable parameters
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        # Should have fewer trainable parameters when frozen
        assert trainable_before < total_params

        # Test short training run
        trainer.train()

        # Test checkpoint exists
        checkpoint_dir = training_args.checkpoint_dir
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) in [1, 2]  # Epoch checkpoint, possibly best model too

        # Test parameter unfreezing
        trainer.unfreeze_all_parameters()
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have all parameters trainable after unfreezing
        assert trainable_after == total_params

    @pytest.mark.ci_unstable
    def test_checkpoint_loading(
        self,
        model: MultimodalPatchedDecoder,
        mock_datasets: tuple[MockMultimodalDataset, MockMultimodalDataset],
        training_args: TrainingArguments,
    ) -> None:
        """Test checkpoint saving and loading."""
        train_dataset, val_dataset = mock_datasets

        # Use training args with 2 epochs for this test
        test_args = replace(training_args, num_train_epochs=2)

        trainer = MultimodalTrainer(
            model=model,
            args=test_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        # Train for two epochs
        trainer.train()

        # Find checkpoint files - specifically look for epoch checkpoints (not best_model.pt)
        checkpoint_dir = test_args.checkpoint_dir
        epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(epoch_checkpoints) == 2  # Epoch 0 and 1

        # Test loading the first epoch checkpoint
        checkpoint_path = epoch_checkpoints[0]
        trainer.load_checkpoint(checkpoint_path)

        # Verify checkpoint loaded - should be epoch 0
        assert trainer.current_epoch == 0

        # Test loading the second epoch checkpoint
        checkpoint_path = epoch_checkpoints[1]
        trainer.load_checkpoint(checkpoint_path)

        # Verify checkpoint loaded - should be epoch 1
        assert trainer.current_epoch == 1
