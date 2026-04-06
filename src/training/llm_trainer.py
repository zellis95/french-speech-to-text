"""LLM adapter trainer: inputs_embeds construction + frozen LLM forward."""

import torch
from torch.utils.data import DataLoader

from src.evaluation.decode import llm_generate
from src.models.llm_model import LLMModel
from src.training.base import BaseTrainer

IGNORE_INDEX = -100


class LLMTrainer(BaseTrainer):
    """LLM adapter training with inputs_embeds construction.

    Each train_step:
    1. Encode audio through frozen encoder + trainable adapter
    2. Assemble inputs_embeds: [prefix | audio | suffix | transcript | eos]
    3. Build labels with -100 masking for non-transcript positions
    4. Forward through frozen LLM → cross-entropy loss
    5. Backward → gradients flow through frozen LLM to adapter
    """

    def __init__(
        self,
        model: LLMModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        device: str = "cpu",
    ):
        super().__init__(model, train_loader, val_loader, cfg, device)
        self.model: LLMModel  # type narrowing

    def _build_inputs_embeds(
        self,
        adapted: torch.Tensor,
        audio_lengths: torch.Tensor,
        transcripts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble inputs_embeds from audio + chat template + transcripts.

        Template: <|im_start|>user\\n [audio] {prompt}<|im_end|>
        \\n<|im_start|>assistant\\n [transcript] <|im_end|>

        Args:
            adapted: (B, T_adapted, hidden_dim) adapter output
            audio_lengths: (B,) valid frame counts
            transcripts: list of raw transcript strings

        Returns:
            inputs_embeds: (B, max_seq_len, hidden_dim) padded
            attention_mask: (B, max_seq_len)
            labels: (B, max_seq_len) with IGNORE_INDEX for non-transcript positions
        """
        device = adapted.device
        prefix_emb, suffix_emb, eos_emb, _, _, eos_ids = self.model.get_template_embeds(device)

        all_embeds = []
        all_labels = []

        for i in range(adapted.shape[0]):
            audio_i = adapted[i, : audio_lengths[i]]  # (A_i, hidden_dim)
            transcript_emb, transcript_ids = self.model.embed_transcript(transcripts[i], device)

            # Concatenate: prefix + audio + suffix + transcript + eos
            seq_embeds = torch.cat([prefix_emb, audio_i, suffix_emb, transcript_emb, eos_emb])

            # Labels: -100 for prefix + audio + suffix, transcript IDs, eos ID
            n_masked = prefix_emb.shape[0] + audio_i.shape[0] + suffix_emb.shape[0]
            labels_i = torch.cat(
                [
                    torch.full((n_masked,), IGNORE_INDEX, dtype=torch.long, device=device),
                    transcript_ids,
                    eos_ids,
                ]
            )

            all_embeds.append(seq_embeds)
            all_labels.append(labels_i)

        # Pad to max sequence length in batch
        max_len = max(e.shape[0] for e in all_embeds)
        hidden_dim = adapted.shape[-1]
        B = len(all_embeds)

        inputs_embeds = torch.zeros(B, max_len, hidden_dim, device=device, dtype=adapted.dtype)
        attention_mask = torch.zeros(B, max_len, device=device, dtype=torch.long)
        labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long, device=device)

        for i, (emb, lab) in enumerate(zip(all_embeds, all_labels, strict=True)):
            seq_len = emb.shape[0]
            inputs_embeds[i, :seq_len] = emb
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = lab

        return inputs_embeds, attention_mask, labels

    def train_step(self, batch: dict) -> torch.Tensor:
        """LLM adapter forward + loss.

        Batch keys (from simple_audio_collate_fn):
            input_values: (B, T) padded waveforms
            attention_mask: (B, T) waveform mask
            transcripts: list[str] raw transcripts
        """
        # 1. Encode audio: frozen encoder + trainable adapter
        adapted, audio_lengths = self.model.encode_audio(
            batch["input_values"],
            batch["attention_mask"],
        )

        # 2. Build inputs_embeds with chat template
        inputs_embeds, attn_mask, labels = self._build_inputs_embeds(
            adapted,
            audio_lengths,
            batch["transcripts"],
        )

        # 3. Forward through frozen LLM
        loss = self.model(inputs_embeds, attn_mask, labels)
        return loss

    def eval_step(self, batch: dict) -> tuple[list[str], list[str]]:
        """LLM generate for WER/CER evaluation.

        Builds prompt (prefix + audio + suffix) without transcript,
        then calls model.generate() to produce predictions.
        """
        adapted, audio_lengths = self.model.encode_audio(
            batch["input_values"],
            batch["attention_mask"],
        )

        device = adapted.device
        prefix_emb, suffix_emb, _, _, _, _ = self.model.get_template_embeds(device)

        # Build prompt-only inputs_embeds (no transcript)
        all_embeds = []
        for i in range(adapted.shape[0]):
            audio_i = adapted[i, : audio_lengths[i]]
            seq = torch.cat([prefix_emb, audio_i, suffix_emb])
            all_embeds.append(seq)

        max_len = max(e.shape[0] for e in all_embeds)
        hidden_dim = adapted.shape[-1]
        B = len(all_embeds)

        inputs_embeds = torch.zeros(B, max_len, hidden_dim, device=device, dtype=adapted.dtype)
        attn_mask = torch.zeros(B, max_len, device=device, dtype=torch.long)

        for i, emb in enumerate(all_embeds):
            inputs_embeds[i, : emb.shape[0]] = emb
            attn_mask[i, : emb.shape[0]] = 1

        predictions = llm_generate(self.model, inputs_embeds, attn_mask)
        references = batch["transcripts"]

        return predictions, references
