"""Decoding utilities for CTC and LLM models."""

import torch

from src.data.text_normalizer import CTC_BLANK, IDX_TO_CHAR


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    """Greedy CTC decoding: argmax → collapse repeats → remove blanks.

    Args:
        log_probs: (B, T, vocab_size) log probabilities
        lengths: (B,) valid frame counts

    Returns:
        List of decoded strings, one per batch item.
    """
    predictions = log_probs.argmax(dim=-1)  # (B, T)
    decoded = []

    for i in range(predictions.shape[0]):
        seq = predictions[i, : lengths[i]].tolist()

        # Collapse repeats, then remove blanks
        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev:
                collapsed.append(idx)
            prev = idx

        chars = [IDX_TO_CHAR.get(idx, "") for idx in collapsed if idx != CTC_BLANK]
        decoded.append("".join(chars))

    return decoded


def llm_generate(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 256,
    num_beams: int = 4,
) -> list[str]:
    """Generate transcriptions from LLM using beam search.

    Args:
        model: LLMModel instance
        inputs_embeds: (B, seq_len, hidden_dim) pre-assembled prompt embeddings
        attention_mask: (B, seq_len)
        max_new_tokens: maximum tokens to generate
        num_beams: beam width (1 = greedy, 4 = beam search per plan)

    Returns:
        List of decoded transcript strings.
    """
    eos_token_id = model.tokenizer.encode("<|im_end|>", add_special_tokens=False)

    # Cast to LLM dtype (adapter outputs fp32, LLM is fp16 — MPS needs matching)
    inputs_embeds = inputs_embeds.to(model.llm_dtype)

    with torch.no_grad():
        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            num_beams=num_beams,
            do_sample=False,
            # Override Qwen's default sampling params to suppress warnings
            # (irrelevant when do_sample=False / beam search)
            temperature=None,
            top_p=None,
            top_k=None,
        )

    # generate() returns full sequence including prompt tokens (as IDs).
    # When using inputs_embeds, the "prompt" part appears as generated pad/eos
    # tokens. We decode everything and strip template artifacts.
    decoded = []
    for ids in output_ids:
        text = model.tokenizer.decode(ids, skip_special_tokens=True)
        decoded.append(text.strip())

    return decoded
