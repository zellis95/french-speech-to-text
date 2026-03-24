"""Base trainer with shared training loop logic."""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _get_warmup_scheduler(optimizer: AdamW, warmup_steps: int) -> LambdaLR:
    """Constant LR after linear warmup (SLAM-ASR style)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class BaseTrainer(ABC):
    """Shared training loop: optimizer, scheduler, grad accum, W&B, checkpointing, early stopping.

    Subclasses implement train_step() and eval_step() for experiment-specific logic.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        # Training config
        t = cfg.training
        self.epochs = t.epochs
        self.grad_accum_steps = t.grad_accum_steps
        self.max_grad_norm = t.max_grad_norm
        self.log_every_n_steps = t.log_every_n_steps
        self.val_every_n_epochs = t.val_every_n_epochs
        self.wer_eval_every_n_epochs = t.wer_eval_every_n_epochs
        self.early_stopping_patience = t.early_stopping_patience

        # Optimizer — only trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=t.lr, weight_decay=t.weight_decay)
        self.scheduler = _get_warmup_scheduler(self.optimizer, t.warmup_steps)

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Checkpointing
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

        # W&B (lazy init in train())
        self._wandb_run = None

    def _init_wandb(self):
        """Initialize W&B run."""
        import wandb
        from omegaconf import OmegaConf

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        self._wandb_run = wandb.init(
            project="french-asr",
            name=self.cfg.run_name,
            config=dict(OmegaConf.to_container(self.cfg, resolve=True)),  # type: ignore[arg-type]
            tags=[self.cfg.experiment.type, self.cfg.experiment.name],
        )
        wandb.config.update(
            {
                "trainable_params": trainable_params,
                "total_params": total_params,
                "effective_batch_size": self.cfg.training.batch_size * self.grad_accum_steps,
            }
        )

    def _log_wandb(self, metrics: dict, step: int | None = None):
        """Log metrics to W&B if initialized."""
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step or self.global_step)

    def train(self):
        """Main training loop: epochs → train_epoch → evaluate."""
        self._init_wandb()
        train_start = time.time()

        log.info(
            f"Starting training: {self.epochs} epochs, "
            f"grad_accum={self.grad_accum_steps}, "
            f"trainable_params="
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        for epoch in range(1, self.epochs + 1):
            epoch_loss = self._train_epoch(epoch)
            log.info(f"Epoch {epoch}/{self.epochs} — train_loss={epoch_loss:.4f}")

            # Validation
            if epoch % self.val_every_n_epochs == 0:
                val_loss = self._validate(epoch)

                # Early stopping on val_loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        log.info(
                            f"Early stopping at epoch {epoch} "
                            f"(patience {self.early_stopping_patience} exhausted)"
                        )
                        break

            # WER/CER evaluation (slower, requires decoding)
            if epoch % self.wer_eval_every_n_epochs == 0 or epoch == self.epochs:
                self._evaluate_wer(epoch)

        total_time = time.time() - train_start
        log.info(f"Training complete in {total_time:.1f}s")
        self._log_wandb({"total_training_time_s": total_time})

        if self._wandb_run is not None:
            import wandb

            wandb.finish()

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        self.optimizer.zero_grad()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            step_start = time.time()

            # Move batch to device
            batch = self._to_device(batch)

            # Forward + loss (experiment-specific)
            loss = self.train_step(batch)
            loss = loss / self.grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    step_time = time.time() - step_start
                    lr = self.scheduler.get_last_lr()[0]
                    self._log_wandb(
                        {
                            "train_loss": loss.item() * self.grad_accum_steps,
                            "learning_rate": lr,
                            "grad_norm": (
                                grad_norm.item()
                                if isinstance(grad_norm, torch.Tensor)
                                else grad_norm
                            ),
                            "step_time_s": step_time,
                            "epoch": epoch,
                        }
                    )

        return epoch_loss / max(num_batches, 1)

    def _validate(self, epoch: int) -> float:
        """Run validation, return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        eval_start = time.time()

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                loss = self.train_step(batch)
                total_loss += loss.item()
                num_batches += 1

        val_loss = total_loss / max(num_batches, 1)
        eval_time = time.time() - eval_start

        log.info(f"  val_loss={val_loss:.4f} ({eval_time:.1f}s)")
        self._log_wandb(
            {
                "val_loss": val_loss,
                "eval_time_s": eval_time,
                "epoch": epoch,
            }
        )

        return val_loss

    def _evaluate_wer(self, epoch: int):
        """Run WER/CER evaluation with decoding (slower than validation)."""
        self.model.eval()
        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                preds, refs = self.eval_step(batch)
                all_predictions.extend(preds)
                all_references.extend(refs)

        from src.evaluation.metrics import compute_cer, compute_wer

        val_wer = compute_wer(all_references, all_predictions)
        val_cer = compute_cer(all_references, all_predictions)

        log.info(f"  val_wer={val_wer:.4f}, val_cer={val_cer:.4f}")
        self._log_wandb(
            {
                "val_wer": val_wer,
                "val_cer": val_cer,
                "epoch": epoch,
            }
        )

        # Log sample predictions as a W&B table
        if self._wandb_run is not None:
            import wandb

            table = wandb.Table(columns=["reference", "prediction"])
            for ref, pred in zip(all_references[:10], all_predictions[:10], strict=True):
                table.add_data(ref, pred)
            self._log_wandb({"sample_predictions": table})

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save trainable parameters only."""
        trainable_state = {
            k: v
            for k, v in self.model.state_dict().items()
            if any(v is p for p in self.model.parameters() if p.requires_grad)
        }

        path = self.checkpoint_dir / f"{self.cfg.experiment.name}_epoch{epoch}.pt"
        torch.save(trainable_state, path)
        log.info(f"  saved checkpoint: {path}")

        if is_best:
            best_path = self.checkpoint_dir / f"{self.cfg.experiment.name}_best.pt"
            torch.save(trainable_state, best_path)

    def _to_device(self, batch: dict) -> dict:
        """Move tensor values in batch to device, leave non-tensors as-is."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    @abstractmethod
    def train_step(self, batch: dict) -> torch.Tensor:
        """Experiment-specific forward pass + loss computation.

        Args:
            batch: collated batch (already on device)

        Returns:
            loss: scalar tensor
        """
        ...

    @abstractmethod
    def eval_step(self, batch: dict) -> tuple[list[str], list[str]]:
        """Experiment-specific decoding for WER/CER evaluation.

        Args:
            batch: collated batch (already on device)

        Returns:
            predictions: list of predicted transcript strings
            references: list of reference transcript strings
        """
        ...
