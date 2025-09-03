"""Tests for multimodal trainer."""

import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import Dataset

from src.models.multimodal_patched_decoder import MultimodalPatchedDecoder, MultimodalTimesFMConfig
from src.train.trainer import MultimodalTrainer


class MockTimeMmdDataset(Dataset[dict[str, Any]]):
    """Mock dataset for testing training loop."""

    def __init__(self, size: int = 100, context_len: int = 512, horizon_len: int = 128):
        self.size = size
        self.patch_len = 32
        self.context_len = context_len
        self.horizon_len = horizon_len

        # Generate synthetic data
        self.data = []
        for i in range(size):
            # Generate synthetic time series
            context = np.random.randn(context_len, 1).astype(np.float32)
            future = np.random.randn(horizon_len, 1).astype(np.float32)

            # Generate mock text patches
            num_patches = context_len // self.patch_len
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


@pytest.fixture
def model_config() -> MultimodalTimesFMConfig:
    """Load model configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "configs" / "model.yml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return MultimodalTimesFMConfig(
        num_layers=config_dict["timesfm"]["num_layers"],
        num_heads=config_dict["timesfm"]["num_heads"],
        num_kv_heads=config_dict["timesfm"]["num_kv_heads"],
        hidden_size=config_dict["timesfm"]["hidden_size"],
        intermediate_size=config_dict["timesfm"]["intermediate_size"],
        head_dim=config_dict["timesfm"]["head_dim"],
        rms_norm_eps=config_dict["timesfm"]["rms_norm_eps"],
        patch_len=config_dict["timesfm"]["patch_len"],
        horizon_len=config_dict["timesfm"]["horizon_len"],
        quantiles=config_dict["timesfm"]["quantiles"],
        pad_val=config_dict["timesfm"]["pad_val"],
        tolerance=config_dict["timesfm"]["tolerance"],
        dtype=config_dict["timesfm"]["dtype"],
        use_positional_embedding=config_dict["timesfm"]["use_positional_embedding"],
        text_encoder_model=config_dict["text_encoder"]["model_name"],
        text_embedding_dim=config_dict["text_encoder"]["embedding_dim"],
    )


@pytest.fixture
def mock_datasets() -> tuple[MockTimeMmdDataset, MockTimeMmdDataset]:
    """Create mock training and validation datasets."""
    # Set seeds for reproducible test data
    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset = MockTimeMmdDataset(size=10, context_len=512, horizon_len=128)  # Smaller for testing
    val_dataset = MockTimeMmdDataset(size=5, context_len=512, horizon_len=128)
    return train_dataset, val_dataset


@pytest.fixture
def model(model_config: MultimodalTimesFMConfig) -> MultimodalPatchedDecoder:
    """Create multimodal model."""
    return MultimodalPatchedDecoder(model_config)


@pytest.fixture
def temp_dirs() -> Generator[tuple[Path, Path], None, None]:
    """Create temporary directories for logging and checkpoints."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "logs"
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        yield log_dir, checkpoint_dir


@pytest.fixture(autouse=True)
def mock_wandb() -> Generator[Mock, None, None]:
    """Mock wandb to avoid initialization and deprecation warnings during tests."""
    with patch("src.train.trainer.wandb") as mock_wandb:
        mock_wandb.init = Mock()
        mock_wandb.log = Mock()
        mock_wandb.finish = Mock()
        yield mock_wandb


def test_mock_dataset() -> None:
    """Test mock dataset creation and access."""
    dataset = MockTimeMmdDataset(size=5, context_len=512, horizon_len=128)

    assert len(dataset) == 5

    sample = dataset[0]
    assert "time_series" in sample
    assert "target" in sample
    assert "patched_texts" in sample
    assert "metadata" in sample

    # Check shapes
    assert sample["time_series"].shape == (512, 1)
    assert sample["target"].shape == (128, 1)
    assert len(sample["patched_texts"]) == 16  # 512 / 32 patches


def test_model_creation(model: MultimodalPatchedDecoder) -> None:
    """Test model creation and basic properties."""
    # Test model exists and has parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1000000  # Should have many parameters

    # Test device placement
    device = next(model.parameters()).device
    assert device.type in ["cuda", "mps", "cpu"]


def test_trainer_creation(
    model: MultimodalPatchedDecoder,
    mock_datasets: tuple[MockTimeMmdDataset, MockTimeMmdDataset],
    temp_dirs: tuple[Path, Path],
) -> None:
    """Test trainer creation and initialization."""
    train_dataset, val_dataset = mock_datasets
    log_dir, checkpoint_dir = temp_dirs

    trainer = MultimodalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=2,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir),
    )

    # Test trainer properties
    assert trainer.device.type in ["cuda", "mps", "cpu"]
    assert len(trainer.train_loader) > 0
    assert trainer.val_loader is not None
    assert len(trainer.val_loader) > 0


def test_forward_pass(
    model: MultimodalPatchedDecoder,
    mock_datasets: tuple[MockTimeMmdDataset, MockTimeMmdDataset],
    temp_dirs: tuple[Path, Path],
) -> None:
    """Test single forward pass through the model."""
    train_dataset, val_dataset = mock_datasets
    log_dir, checkpoint_dir = temp_dirs

    trainer = MultimodalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=2,
        learning_rate=1e-4,
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir),
    )

    # Get a batch from the trainer
    sample_batch = next(iter(trainer.train_loader))

    # Check batch structure
    assert "time_series" in sample_batch
    assert "targets" in sample_batch
    assert "text_descriptions" in sample_batch
    assert "input_padding" in sample_batch
    assert "freq" in sample_batch

    # Move batch to device
    time_series = sample_batch["time_series"].to(trainer.device)
    input_padding = sample_batch["input_padding"].to(trainer.device)
    freq = sample_batch["freq"].to(trainer.device)
    text_descriptions = sample_batch["text_descriptions"]

    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ts=time_series,
            input_padding=input_padding,
            freq=freq,
            text_descriptions=text_descriptions,
        )

    # Check output shape
    batch_size = time_series.shape[0]
    expected_shape = (batch_size, 16, 128, 9)  # Based on model config
    assert outputs.shape == expected_shape


def test_training_loop(
    model: MultimodalPatchedDecoder,
    mock_datasets: tuple[MockTimeMmdDataset, MockTimeMmdDataset],
    temp_dirs: tuple[Path, Path],
) -> None:
    """Test training loop execution."""
    train_dataset, val_dataset = mock_datasets
    log_dir, checkpoint_dir = temp_dirs

    trainer = MultimodalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=2,
        learning_rate=1e-4,
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir),
    )

    # Test parameter freezing
    trainer.freeze_pretrained_parameters()

    # Count trainable parameters
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # Should have fewer trainable parameters when frozen
    assert trainable_before < total_params

    # Test short training run
    trainer.train(num_epochs=1, save_every=1)

    # Test checkpoint exists
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoint_files) > 0

    # Test parameter unfreezing
    trainer.unfreeze_all_parameters()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Should have all parameters trainable after unfreezing
    assert trainable_after == total_params


def test_checkpoint_loading(
    model: MultimodalPatchedDecoder,
    mock_datasets: tuple[MockTimeMmdDataset, MockTimeMmdDataset],
    temp_dirs: tuple[Path, Path],
) -> None:
    """Test checkpoint saving and loading."""
    train_dataset, val_dataset = mock_datasets
    log_dir, checkpoint_dir = temp_dirs

    trainer = MultimodalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=2,
        learning_rate=1e-4,
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir),
    )

    # Train for one epoch to create checkpoint
    trainer.train(num_epochs=1, save_every=1)

    # Find checkpoint file
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoint_files) > 0

    # Test loading checkpoint
    checkpoint_path = checkpoint_files[0]
    trainer.load_checkpoint(str(checkpoint_path))

    # Verify checkpoint loaded (epoch should match)
    assert trainer.current_epoch >= 0
