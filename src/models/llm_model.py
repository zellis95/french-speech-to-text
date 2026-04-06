"""LLM-adapter model: frozen mHuBERT + trainable adapter + frozen Qwen2.5."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.adapters import BaseAdapter
from src.models.encoder import EncoderWrapper


class LLMModel(nn.Module):
    """Frozen encoder + trainable adapter + frozen instruct LLM.

    The encoder is wrapped in no_grad (fully frozen). The LLM has
    requires_grad=False but is NOT wrapped in no_grad — gradients must
    flow backward through it to reach the adapter.

    Provides helpers for building inputs_embeds with the chat template.
    The actual sequence assembly happens in LLMTrainer.train_step.
    """

    def __init__(
        self,
        encoder: EncoderWrapper,
        adapter: BaseAdapter,
        llm_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        llm_dtype: torch.dtype = torch.float16,
        prompt: str = "Transcrivez la parole en texte.",
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

        # Load LLM — frozen but gradients flow through for adapter training
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=llm_dtype)
        self.llm_dtype = llm_dtype
        for param in self.llm.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.embed_fn = self.llm.get_input_embeddings()
        self.prompt = prompt

        # Cache template token IDs (populated on first use, device-dependent)
        self._prefix_ids: torch.Tensor | None = None
        self._suffix_ids: torch.Tensor | None = None
        self._eos_ids: torch.Tensor | None = None

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text, return token IDs (no special tokens added)."""
        assert self.tokenizer is not None
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]

    def _get_template_ids(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cached token IDs for chat template parts.

        Template structure:
            <|im_start|>user\\n [audio] {prompt}<|im_end|>
            \\n<|im_start|>assistant\\n [transcript] <|im_end|>
        """
        if self._prefix_ids is None or self._prefix_ids.device != device:
            self._prefix_ids = self._tokenize("<|im_start|>user\n").to(device)
            self._suffix_ids = self._tokenize(
                f"{self.prompt}<|im_end|>\n<|im_start|>assistant\n"
            ).to(device)
            self._eos_ids = self._tokenize("<|im_end|>").to(device)
        assert self._suffix_ids is not None
        assert self._eos_ids is not None
        return self._prefix_ids, self._suffix_ids, self._eos_ids

    def get_template_embeds(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cached embeddings and token IDs for chat template parts.

        Returns:
            prefix_embeds: (P, hidden_dim)
            suffix_embeds: (S, hidden_dim)
            eos_embeds: (E, hidden_dim)
            prefix_ids, suffix_ids, eos_ids: corresponding token ID tensors
        """
        prefix_ids, suffix_ids, eos_ids = self._get_template_ids(device)
        with torch.no_grad():
            prefix_embeds = self.embed_fn(prefix_ids)
            suffix_embeds = self.embed_fn(suffix_ids)
            eos_embeds = self.embed_fn(eos_ids)
        return prefix_embeds, suffix_embeds, eos_embeds, prefix_ids, suffix_ids, eos_ids

    def encode_audio(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run frozen encoder + trainable adapter.

        Args:
            input_values: (B, T_samples) raw waveform
            attention_mask: (B, T_samples) optional

        Returns:
            adapted: (B, T_adapted, llm_hidden_dim) adapter output
            lengths: (B,) valid frame counts after adapter downsampling
        """
        hidden_states, lengths = self.encoder(input_values, attention_mask)
        adapted, out_lengths = self.adapter(hidden_states, lengths)
        return adapted, out_lengths

    def embed_transcript(
        self, transcript: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and embed a transcript string.

        Returns:
            embeds: (T, hidden_dim)
            token_ids: (T,)
        """
        token_ids = self._tokenize(transcript).to(device)
        with torch.no_grad():
            embeds = self.embed_fn(token_ids)
        return embeds, token_ids

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through frozen LLM. Returns loss.

        Args:
            inputs_embeds: (B, seq_len, hidden_dim) assembled sequence embeddings
            attention_mask: (B, seq_len) 1=real, 0=pad
            labels: (B, seq_len) token IDs with -100 for masked positions

        Returns:
            loss: scalar cross-entropy loss (labels shifted internally by HF)
        """
        # Cast to LLM dtype — adapter outputs fp32 but LLM is fp16.
        # MPS requires matching dtypes for matmul.
        inputs_embeds = inputs_embeds.to(self.llm_dtype)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def train(self, mode: bool = True) -> "LLMModel":
        """Override to keep encoder and LLM in eval mode. Only adapter trains."""
        super().train(mode)
        self.encoder.eval()
        self.llm.eval()
        return self
