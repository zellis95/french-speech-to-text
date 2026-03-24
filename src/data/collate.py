"""Collate functions for batching audio + text samples."""

import torch

from src.data.text_normalizer import encode_for_ctc, normalize_text


def ctc_collate_fn(batch: list[dict]) -> dict:
    """Collate for CTC training. Pads waveforms, encodes transcripts to CTC indices.

    Input: list of {"input_values": Tensor[T], "transcript": str}
    Output: {
        "input_values": (B, T_max) padded waveforms,
        "attention_mask": (B, T_max) 1=real, 0=pad,
        "labels": (N,) packed CTC label indices (all samples concatenated),
        "input_lengths": (B,) sample lengths in samples,
        "label_lengths": (B,) label lengths per sample,
    }
    """
    waveforms = [s["input_values"] for s in batch]
    transcripts = [s["transcript"] for s in batch]

    # Pad waveforms to max length in batch
    input_lengths = torch.tensor([w.shape[0] for w in waveforms])
    max_len = int(input_lengths.max().item())
    padded = torch.zeros(len(batch), max_len)
    attention_mask = torch.zeros(len(batch), max_len)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[0]] = w
        attention_mask[i, : w.shape[0]] = 1.0

    # Encode transcripts to CTC label indices
    encoded = [encode_for_ctc(normalize_text(t)) for t in transcripts]
    label_lengths = torch.tensor([len(e) for e in encoded])
    labels = torch.cat([torch.tensor(e, dtype=torch.long) for e in encoded])

    return {
        "input_values": padded,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
    }


def simple_audio_collate_fn(batch: list[dict]) -> dict:
    """Collate for LLM training. Pads waveforms, keeps transcript strings.

    The LLM experiment builds inputs_embeds in the training loop (needs GPU +
    adapter forward), so collate just handles waveform padding.

    Input: list of {"input_values": Tensor[T], "transcript": str}
    Output: {
        "input_values": (B, T_max) padded waveforms,
        "attention_mask": (B, T_max) 1=real, 0=pad,
        "transcripts": list[str] of raw transcript strings,
    }
    """
    waveforms = [s["input_values"] for s in batch]
    transcripts = [s["transcript"] for s in batch]

    input_lengths = torch.tensor([w.shape[0] for w in waveforms])
    max_len = int(input_lengths.max().item())
    padded = torch.zeros(len(batch), max_len)
    attention_mask = torch.zeros(len(batch), max_len)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[0]] = w
        attention_mask[i, : w.shape[0]] = 1.0

    return {
        "input_values": padded,
        "attention_mask": attention_mask,
        "transcripts": transcripts,
    }
