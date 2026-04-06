"""Tests for collate functions: padding correctness and label masking."""

import torch

from src.data.collate import ctc_collate_fn, simple_audio_collate_fn


def _make_samples(lengths: list[int]) -> list[dict]:
    """Create fake samples with given waveform lengths."""
    return [
        {"input_values": torch.randn(length), "transcript": "bonjour le monde"}
        for length in lengths
    ]


class TestCTCCollateFn:
    def test_padding_shape(self):
        samples = _make_samples([100, 200, 150])
        batch = ctc_collate_fn(samples)
        assert batch["input_values"].shape == (3, 200)  # padded to max
        assert batch["attention_mask"].shape == (3, 200)

    def test_attention_mask_values(self):
        samples = _make_samples([100, 200])
        batch = ctc_collate_fn(samples)
        # First sample: 100 real, 100 padded
        assert batch["attention_mask"][0, 99].item() == 1.0
        assert batch["attention_mask"][0, 100].item() == 0.0
        # Second sample: all real
        assert batch["attention_mask"][1, 199].item() == 1.0

    def test_labels_are_packed(self):
        samples = _make_samples([100, 100])
        batch = ctc_collate_fn(samples)
        # Labels should be 1D packed tensor (all samples concatenated)
        assert batch["labels"].dim() == 1
        assert batch["labels"].shape[0] == batch["label_lengths"].sum().item()

    def test_input_lengths(self):
        samples = _make_samples([100, 200, 150])
        batch = ctc_collate_fn(samples)
        assert batch["input_lengths"].tolist() == [100, 200, 150]


class TestSimpleAudioCollateFn:
    def test_padding_shape(self):
        samples = _make_samples([100, 200])
        batch = simple_audio_collate_fn(samples)
        assert batch["input_values"].shape == (2, 200)
        assert batch["attention_mask"].shape == (2, 200)

    def test_transcripts_preserved(self):
        samples = _make_samples([100, 100])
        batch = simple_audio_collate_fn(samples)
        assert len(batch["transcripts"]) == 2
        assert batch["transcripts"][0] == "bonjour le monde"

    def test_no_labels_key(self):
        samples = _make_samples([100])
        batch = simple_audio_collate_fn(samples)
        assert "labels" not in batch
