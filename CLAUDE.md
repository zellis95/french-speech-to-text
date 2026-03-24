# CLAUDE.md

## Project Overview

French ASR experimentation pipeline comparing CTC baseline vs LLM-adapter (SLAM-ASR architecture). Custom PyTorch training, Hydra configs, W&B tracking.

Full implementation plan: `.claude/plans/implementation-plan.md`

## Key Models

| Component | Model | Hidden Dim |
|---|---|---|
| Encoder | `utter-project/mHuBERT-147` (frozen) | 768 |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` (frozen) | 896 |
| CTC head | `nn.Linear(768, 45)` | — |
| Adapter | ConcatMLP or ConvMLP (trainable) | → 896 |

mHuBERT expects: mono 16kHz float32, zero-mean unit-variance normalized. Output: 50fps (320x downsampling).

## Data

- **Training/eval (scripted)**: Multilingual LibriSpeech French (`facebook/multilingual_librispeech`)
  - Loaded via targeted parquet downloads (not `load_dataset` with config name — that downloads all 34 train shards)
  - Native 48kHz, resampled to 16kHz in dataset class
  - dev: 2,416 samples ~10h | test: 2,426 samples ~10h | train: 258K samples ~1,077h
- **Test (spontaneous)**: SPS corpus at `data/sps-corpus-3.0-2026-03-09-fr/`
  - 152 validated clips (non-empty transcription), mono 32kHz mp3, resampled to 16kHz
  - TSV metadata: `ss-corpus-fr.tsv`

## Running

```bash
# Local dev (uses dev split, ~158MB, cached)
python scripts/train.py mode=local experiment=ctc_baseline

# Remote (full train on Modal GPU)
modal run modal_app.py -- mode=remote experiment=llm_adapter adapter=conv_mlp

# Override anything
python scripts/train.py mode=local training.lr=5e-5
```

## Hydra Config Structure

```
conf/
├── config.yaml          # Defaults list + encoder/llm/device/seed
├── experiment/          # ctc_baseline.yaml, llm_adapter.yaml
├── adapter/             # concat_mlp.yaml, conv_mlp.yaml
├── data/                # mls.yaml (MLS splits/config), sps.yaml (SPS corpus path)
├── training/            # ctc.yaml, llm.yaml (hyperparams)
└── mode/                # local.yaml (dev split, 5 epochs), remote.yaml (full train, 30 epochs)
```

`mode=local` vs `mode=remote` controls data splits and training scale. Local uses MLS `dev` split to avoid downloading the 17GB train set. MLS and SPS configs both live in `conf/data/`, loaded via `data@data: mls` and `data@sps: sps` in defaults.

## Key File Paths

- `src/data/text_normalizer.py` — French normalizer + CTC charset (45 classes)
- `src/data/datasets.py` — MLSDataset (HF parquet), SPSDataset (local TSV+mp3)
- `src/utils/device.py` — CUDA → MPS → CPU auto-detection
- `tests/test_text_normalizer.py` — 41 tests for normalizer

## Environment

- Python 3.13.12 (pyenv), deps managed with uv
- `.env` file has `HF_TOKEN` (needed for HF downloads)
- Load with `from dotenv import load_dotenv; load_dotenv()` in entry points

## Code Quality

- **Linting/formatting**: `uv run ruff check src/ scripts/ tests/` and `uv run ruff format src/ scripts/ tests/`
- **Type checking**: `uv run ty check src/`
- **Tests**: `uv run pytest tests/`
- Run all three before committing. Ruff config in pyproject.toml (line-length=100, py313).

## Observations

Log interesting findings in `observations.md` — things spotted during development, experiments, unexpected model behaviours, etc.

## Gotchas

- **MLS downloads**: Do NOT use `load_dataset("facebook/multilingual_librispeech", "french", split="dev")` — this downloads ALL splits (~17GB). Use targeted parquet `data_files` approach in `MLSDataset`.
- **MLS small splits**: `1_hours` and `9_hours` are subsets of train stored in the same parquet shards — they also trigger full train download. Use `dev` for local training instead.
- **CTC loss on MPS**: Not supported natively. Use `PYTORCH_ENABLE_MPS_FALLBACK=1` or manual CPU fallback.
