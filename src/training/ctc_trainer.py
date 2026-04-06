"""CTC trainer: encoder + linear head with CTC loss."""

import torch
from torch.utils.data import DataLoader

from src.evaluation.decode import ctc_greedy_decode
from src.models.ctc_model import CTCModel
from src.training.base import BaseTrainer


class CTCTrainer(BaseTrainer):
    """CTC-specific training: forward through encoder + projection, CTC loss, greedy decode."""

    def __init__(
        self,
        model: CTCModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        device: str = "cpu",
    ):
        super().__init__(model, train_loader, val_loader, cfg, device)
        self.model: CTCModel  # type narrowing

    def train_step(self, batch: dict) -> torch.Tensor:
        """CTC forward + loss.

        Batch keys (from ctc_collate_fn):
            input_values: (B, T) padded waveforms
            attention_mask: (B, T)
            labels: (N,) packed CTC label indices
            input_lengths: (B,) sample lengths
            label_lengths: (B,) label lengths
        """
        log_probs, frame_lengths = self.model(
            batch["input_values"],
            batch["attention_mask"],
        )

        loss = self.model.compute_loss(
            log_probs,
            frame_lengths,
            batch["labels"],
            batch["label_lengths"],
        )
        return loss

    def eval_step(self, batch: dict) -> tuple[list[str], list[str]]:
        """CTC greedy decode for WER/CER evaluation.

        Returns (predictions, references) as string lists.
        """
        log_probs, frame_lengths = self.model(
            batch["input_values"],
            batch["attention_mask"],
        )

        predictions = ctc_greedy_decode(log_probs, frame_lengths)

        # Recover reference transcripts from packed CTC labels
        from src.data.text_normalizer import decode_ctc_indices, normalize_text

        label_offset = 0
        references = []
        for length in batch["label_lengths"]:
            label_ids = batch["labels"][label_offset : label_offset + length].tolist()
            references.append(normalize_text(decode_ctc_indices(label_ids)))
            label_offset += length

        return predictions, references
