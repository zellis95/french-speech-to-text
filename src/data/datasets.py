"""Dataset classes for Multilingual LibriSpeech (HuggingFace) and SPS corpus (local TSV + mp3)."""

import csv
import logging
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000


# MLS French parquet file layout on HuggingFace Hub
_MLS_BASE = "hf://datasets/facebook/multilingual_librispeech/french"
_MLS_SPLIT_FILES = {
    "dev": [f"{_MLS_BASE}/dev-00000-of-00001.parquet"],
    "test": [f"{_MLS_BASE}/test-00000-of-00001.parquet"],
    "train": [f"{_MLS_BASE}/train-{i:05d}-of-00034.parquet" for i in range(34)],
}


class MLSDataset(Dataset):
    """Wraps a HuggingFace Multilingual LibriSpeech French split.

    Downloads only the parquet files for the requested split, avoiding the
    default load_dataset behaviour of downloading all files for the config.

    Clips exceeding max_duration_s are filtered at init using the
    `audio_duration` metadata field (no audio decoding needed).
    Audio is resampled from native 48kHz to 16kHz on-the-fly.

    Uses targeted parquet downloads to avoid fetching all 34 train shards
    when only dev/test is needed.

    Available splits:
        - "dev":   2,416 samples, ~158MB, ~10h
        - "test":  2,426 samples, ~158MB, ~10h
        - "train": 258,213 samples, ~17GB, ~1,077h

    Returns {"input_values": Tensor[T], "transcript": str}.
    """

    def __init__(self, split: str = "dev", max_duration_s: float = 15.0):
        from datasets import load_dataset

        if split not in _MLS_SPLIT_FILES:
            raise ValueError(f"Unknown MLS split: {split!r}. Available: {list(_MLS_SPLIT_FILES)}")

        data_files = _MLS_SPLIT_FILES[split]
        log.info(f"MLSDataset: loading {len(data_files)} parquet file(s) for split='{split}'")

        # Load only the specific parquet files for this split.
        # The parquet loader puts everything into a single "train" split.
        ds = load_dataset("parquet", data_files=data_files, split="train")

        # Filter using audio_duration column — batch read, no per-row indexing
        durations = ds["audio_duration"]
        total = len(durations)
        self._valid_indices = [
            i for i, d in enumerate(durations)
            if d <= max_duration_s
        ]
        skipped = total - len(self._valid_indices)
        if skipped:
            log.info(f"MLSDataset: filtered {skipped}/{total} clips exceeding {max_duration_s}s")

        self._ds = ds
        self._resampler: torchaudio.transforms.Resample | None = None
        log.info(f"MLSDataset: {len(self)} samples ready")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        row = self._ds[self._valid_indices[idx]]
        audio = row["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]

        # Resample to 16kHz if needed (MLS native is 48kHz via parquet loader)
        if sr != TARGET_SAMPLE_RATE:
            if self._resampler is None or self._resampler.orig_freq != sr:
                self._resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
            waveform = self._resampler(waveform)

        return {
            "input_values": waveform,
            "transcript": row["transcript"],
        }


class SPSDataset(Dataset):
    """Local SPS corpus: TSV metadata + mp3 audio files.

    Filters to validated clips (non-empty transcription).
    Clips exceeding max_duration_s are filtered out at init using TSV duration_ms.
    Audio is mono 32kHz mp3, resampled to 16kHz on-the-fly.
    Returns {"input_values": Tensor[T], "transcript": str}.
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        max_duration_s: float | None = None,
    ):
        corpus_dir = Path(corpus_dir)
        self._audio_dir = corpus_dir / "audios"
        self._resampler_cache: dict[int, torchaudio.transforms.Resample] = {}

        # Load validated entries (non-empty transcription), optionally filter by duration
        tsv_path = corpus_dir / "ss-corpus-fr.tsv"
        self._entries: list[dict[str, str]] = []
        skipped = 0
        with open(tsv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if not row["transcription"].strip():
                    continue
                if max_duration_s is not None:
                    duration_s = float(row["duration_ms"]) / 1000
                    if duration_s > max_duration_s:
                        skipped += 1
                        continue
                self._entries.append(row)

        if skipped:
            log.info(f"SPSDataset: filtered {skipped} clips exceeding {max_duration_s}s")

    def __len__(self) -> int:
        return len(self._entries)

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(
                orig_sr, TARGET_SAMPLE_RATE
            )
        return self._resampler_cache[orig_sr]

    def __getitem__(self, idx: int) -> dict:
        entry = self._entries[idx]
        audio_path = self._audio_dir / entry["audio_file"]

        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)  # (1, T) → (T), all files are mono

        if sr != TARGET_SAMPLE_RATE:
            waveform = self._get_resampler(sr)(waveform)

        return {
            "input_values": waveform,
            "transcript": entry["transcription"],
        }
