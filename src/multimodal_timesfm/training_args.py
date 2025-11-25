"""Training arguments for multimodal TimesFM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from multimodal_timesfm.utils.yaml import load_yaml


@dataclass(frozen=True)
class TrainingArguments:
    output_dir: str = field(default="outputs", metadata={"help": "Output directory for logs and checkpoints."})
    eval_strategy: Literal["no", "epoch"] = field(default="epoch", metadata={"help": "Evaluation strategy."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Training batch size per device."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Evaluation batch size per device."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm for clipping."})
    num_train_epochs: int = field(default=10, metadata={"help": "Total training epochs."})
    logging_steps: int = field(default=100, metadata={"help": "Log every N steps."})
    save_strategy: Literal["no", "epoch"] = field(default="epoch", metadata={"help": "Checkpoint save strategy."})
    save_total_limit: int | None = field(default=10, metadata={"help": "Limit total checkpoints."})
    run_name: str | None = field(default=None, metadata={"help": "Run name for W&B."})
    load_best_model_at_end: bool = field(default=False, metadata={"help": "Load best model at end."})
    device: str | None = field(default=None, metadata={"help": "Device (auto-detected if None)."})
    seed: int | None = field(default=None, metadata={"help": "Random seed for reproducibility."})
    patch_len: int = field(default=32, metadata={"help": "Patch length."})
    context_len: int = field(default=512, metadata={"help": "Context length."})
    horizon_len: int = field(default=128, metadata={"help": "Forecast horizon length."})
    n_folds: int = field(default=5, metadata={"help": "Number of CV folds."})
    train_ratio: float = field(default=0.6, metadata={"help": "Training ratio per fold."})
    val_ratio: float = field(default=0.2, metadata={"help": "Validation ratio per fold."})
    test_ratio: float = field(default=0.2, metadata={"help": "Test ratio per fold."})

    def __post_init__(self) -> None:
        """Validate and setup directories."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Validate CV ratios
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(ratio_sum - 1.0) < 1e-6:
            raise ValueError(f"CV ratios must sum to 1.0, got {ratio_sum}")

    @property
    def logging_dir(self) -> Path:
        """Directory for logs."""
        return Path(self.output_dir) / "logs"

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for model checkpoints."""
        return Path(self.output_dir) / "checkpoints"

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> TrainingArguments:
        """Load from YAML file."""
        config_dict = load_yaml(Path(yaml_path))
        return cls(**config_dict)
