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
- `src/data/collate.py` — ctc_collate_fn, simple_audio_collate_fn
- `src/models/encoder.py` — EncoderWrapper (frozen mHuBERT)
- `src/models/adapters.py` — ConcatMLP, ConvMLP, registry + build_adapter()
- `src/models/ctc_model.py` — CTCModel (encoder + linear head + CTC loss)
- `src/models/llm_model.py` — LLMModel (encoder + adapter + frozen Qwen, chat template)
- `src/training/run.py` — Shared training entry points (train_ctc, train_llm, build_dataloaders)
- `src/training/base.py` — BaseTrainer (shared loop, grad accum, W&B, checkpointing)
- `src/training/ctc_trainer.py` — CTCTrainer
- `src/training/llm_trainer.py` — LLMTrainer (inputs_embeds construction)
- `src/evaluation/metrics.py` — WER/CER via jiwer
- `src/evaluation/decode.py` — CTC greedy decode, LLM beam search (beam=4)
- `src/utils/device.py` — CUDA → MPS → CPU auto-detection
- `modal_app.py` — Modal serverless GPU entry point (T4, Hydra compose, checkpoint volume)
- `scripts/quick_decode.py` — Ad-hoc checkpoint decode test (load model + decode samples)
- `tests/` — 65 tests (normalizer, adapters, collate, metrics)

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

## Running

```bash
# Local dev (uses dev split, ~158MB, cached)
python scripts/train.py mode=local experiment=ctc_baseline

# Modal GPU (passes Hydra overrides via --overrides, comma-separated)
modal run modal_app.py --overrides "experiment=llm_adapter,training.epochs=5,data.max_duration_s=11.0"

# Override anything locally
python scripts/train.py mode=local training.lr=5e-5

# Disable W&B for quick local tests
python scripts/train.py mode=local wandb=false training.epochs=1
```

## Modal

- `modal_app.py` — single entry point: image (debian_slim + uv_sync + ffmpeg), T4 GPU, checkpoint volume
- Uses `hydra.compose()` to build config (same overrides as CLI), calls `src/training/run.py` directly
- Base overrides: `mode=local` (dev split). Override anything via `--overrides` flag.
- Secrets: `Secret.from_dotenv()` injects `.env` vars (HF_TOKEN, WANDB_API_KEY) into container
- Checkpoint volume: `asr-checkpoints` — persists across runs, accessible via `uv run modal volume ls/get`
- Image is cached (deps via uv_sync). Source code mounted at runtime — code changes don't trigger rebuild.
- Resource config: T4 GPU ($0.59/hr), 16GB RAM request, 32GB hard limit

### Modal volume commands

```bash
uv run modal volume ls asr-checkpoints              # list checkpoints
uv run modal volume get asr-checkpoints <file> ./checkpoints/ --force  # download
uv run modal volume rm asr-checkpoints <file>        # delete
```

## Gotchas

- **MLS downloads**: Do NOT use `load_dataset("facebook/multilingual_librispeech", "french", split="dev")` — this downloads ALL splits (~17GB). Use targeted parquet `data_files` approach in `MLSDataset`.
- **MLS small splits**: `1_hours` and `9_hours` are subsets of train stored in the same parquet shards — they also trigger full train download. Use `dev` for local training instead.
- **CTC loss on MPS**: Not supported natively. Use `PYTORCH_ENABLE_MPS_FALLBACK=1` or manual CPU fallback.
- **MPS dtype mismatch**: MPS requires matching dtypes for matmul. Adapter outputs fp32, LLM is fp16 — must cast `inputs_embeds` to fp16 before LLM forward and generate. Handled in `LLMModel.forward()` and `llm_generate()`.
