"""Frozen mHuBERT encoder wrapper."""

import torch
import torch.nn as nn
from transformers import HubertModel


class EncoderWrapper(nn.Module):
    """Wraps mHuBERT-147 as a frozen feature extractor.

    Loads the model, freezes all parameters, and sets eval mode.
    Forward pass is wrapped in torch.no_grad() since nothing is trainable.

    Output: 50fps features (320x downsampling from 16kHz input).
    """

    def __init__(self, model_name: str = "utter-project/mHuBERT-147"):
        super().__init__()
        self.encoder = HubertModel.from_pretrained(model_name)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_dim = self.encoder.config.hidden_size  # 768

    @torch.no_grad()
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run frozen encoder.

        Args:
            input_values: (B, T_samples) raw waveform, 16kHz mono, zero-mean unit-var normalized
            attention_mask: (B, T_samples) optional, 1 for real samples, 0 for padding

        Returns:
            hidden_states: (B, T_frames, 768) encoder output
            lengths: (B,) number of valid frames per sample
        """
        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, T_frames, 768)

        # Compute output lengths from input lengths
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=-1)  # (B,)
            lengths = self._get_output_lengths(input_lengths)
        else:
            batch_size = input_values.shape[0]
            T_frames = hidden_states.shape[1]
            lengths = torch.full((batch_size,), T_frames, device=hidden_states.device)

        return hidden_states, lengths

    def _get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute frame-level output lengths from sample-level input lengths.

        Applies the same conv formula as the HuBERT feature extractor:
        output_length = floor((input_length - kernel_size) / stride) + 1
        for each of the 7 conv layers.
        """
        long_lengths: torch.LongTensor = input_lengths.long()  # type: ignore[assignment]
        return self.encoder._get_feat_extract_output_lengths(long_lengths).long()

    def train(self, mode: bool = True) -> "EncoderWrapper":
        """Override to keep encoder always in eval mode."""
        super().train(mode)
        self.encoder.eval()
        return self
