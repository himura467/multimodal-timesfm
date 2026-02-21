"""Trainer for multimodal and baseline time series forecasting."""

from collections.abc import Iterator

from torch import nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

from multimodal_timesfm.data.collate import baseline_collate_fn, multimodal_collate_fn
from multimodal_timesfm.multimodal_decoder import MultimodalDecoder
from multimodal_timesfm.training_args import TrainingArguments
from multimodal_timesfm.types import PreprocessedSample, TrainingMode
from multimodal_timesfm.utils.device import pin_memory, resolve_device
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
        use_wandb: bool,
    ) -> None:
        """Initialize MultimodalTrainer.

        Args:
            model: MultimodalDecoder model to train.
            args: Training arguments.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            mode: Training mode — 'multimodal' trains fusion only, 'baseline' fine-tunes adapter.
            use_wandb: Whether to use W&B for logging.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.mode = mode
        self.use_wandb = use_wandb

        self.device = resolve_device(args.device)
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

    def _get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return the parameters to optimize based on mode."""
        if self.mode == "multimodal":
            return self.model.fusion.parameters()
        return (p for p in self.model.adapter.parameters() if p.requires_grad)
