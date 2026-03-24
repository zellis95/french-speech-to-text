"""CTC model: frozen mHuBERT encoder + trainable linear projection head."""

import torch
import torch.nn as nn

from src.data.text_normalizer import CTC_VOCAB_SIZE
from src.models.encoder import EncoderWrapper


class CTCModel(nn.Module):
    """CTC baseline: encoder → linear projection → CTC loss.

    The encoder is frozen; only the projection head is trained (~35K params).
    """

    def __init__(self, encoder: EncoderWrapper):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Linear(encoder.hidden_dim, CTC_VOCAB_SIZE)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder + projection.

        Returns:
            log_probs: (B, T_frames, vocab_size) log softmax output
            lengths: (B,) number of valid frames per sample
        """
        hidden_states, lengths = self.encoder(input_values, attention_mask)
        logits = self.projection(hidden_states)  # (B, T, vocab_size)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs, lengths

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss, with CPU fallback for MPS.

        CTC loss requires (T, B, C) input format.
        """
        # CTC expects (T, B, C)
        log_probs_t = log_probs.transpose(0, 1)

        # CTC loss not supported on MPS — fall back to CPU
        if log_probs.device.type == "mps":
            loss = self.ctc_loss(
                log_probs_t.cpu(),
                labels.cpu(),
                lengths.cpu(),
                label_lengths.cpu(),
            )
        else:
            loss = self.ctc_loss(log_probs_t, labels, lengths, label_lengths)

        return loss
