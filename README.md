# French Speech-to-Text

Experimentation pipeline comparing CTC baseline vs LLM-adapter ASR on French, following the [SLAM-ASR architecture](https://arxiv.org/abs/2402.08846).

**Core idea**: frozen speech encoder (mHuBERT) + frozen instruct LLM (Qwen2.5-0.5B) + trainable linear adapter achieves competitive ASR by leveraging the LLM's linguistic prior.

## Architecture

Two independent experiments, both using the same frozen encoder:

```
CTC Experiment:
  Audio (16kHz) ──► mHuBERT-147 (frozen) ──► Linear(768, 45) ──► CTC Loss

LLM Experiment:
  Audio (16kHz) ──► mHuBERT-147 (frozen) ──► Adapter (trainable) ──► Qwen2.5-0.5B (frozen) ──► CE Loss
                                              ConcatMLP / ConvMLP     (inputs_embeds)
```

## Setup

```bash
# Python 3.13 via pyenv
pyenv local 3.13

# Install dependencies
uv sync

# Set up HuggingFace token in .env
echo "HF_TOKEN=hf_your_token_here" > .env
```

## Data

| Dataset | Type | Split | Samples | Hours |
|---|---|---|---|---|
| [MLS French](https://huggingface.co/datasets/facebook/multilingual_librispeech) | Read audiobooks (scripted) | dev | 2,416 | ~10h |
| MLS French | Read audiobooks (scripted) | test | 2,426 | ~10h |
| MLS French | Read audiobooks (scripted) | train | 258,213 | ~1,077h |
| [SPS Corpus](https://datacollective.mozillafoundation.org/datasets/cmmytgij900f2nz07xm0wyzrd) | Spontaneous speech | — | 152 | ~0.6h |

Two test sets: scripted (in-distribution MLS) and spontaneous (out-of-distribution SPS) to measure generalisation.

MLS is CC-BY-4.0. SPS (Common Voice Spontaneous Speech 3.0) is CC0-1.0 but prohibits redistribution and speaker identification attempts — data is gitignored, not included in this repo.

## Usage

```bash
# Train CTC baseline locally (uses MLS dev split, ~158MB download)
python scripts/train.py mode=local experiment=ctc_baseline

# Train LLM adapter locally
python scripts/train.py mode=local experiment=llm_adapter adapter=concat_mlp

# Train on Modal GPU (comma-separated Hydra overrides)
modal run modal_app.py --overrides "experiment=llm_adapter,training.epochs=5"

# Override any config
python scripts/train.py mode=local training.lr=5e-5 training.epochs=10

# Quick decode test on a checkpoint
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/quick_decode.py
```

## Experiments

### Phase A: CTC Baselines
1. **CTC-direct**: mHuBERT → Linear(768, 45) → CTC loss (~35K params)
2. **CTC-adapter**: mHuBERT → ConcatMLP → Linear(896, 45) → CTC loss

### Phase B: LLM Adapter Comparison
3. **LLM-concat-mlp**: ConcatMLP k=5, hidden=[2048] (SLAM-ASR baseline)
4. **LLM-conv-mlp**: ConvMLP stride=5, kernel=5 (learned downsampling)
5. **LLM-deeper-mlp**: ConcatMLP k=5, hidden=[2048, 2048] (3-layer)

All evaluated on both test sets (MLS scripted + SPS spontaneous) with WER and CER.

### Future Directions

Based on findings from [ByteDance's comprehensive ASR study](https://arxiv.org/abs/2406.17272):

- **Matching loss**: auxiliary loss aligning adapter output embeddings with LLM text embeddings (cross-attention + MSE/cosine), giving the adapter a direct training signal
- **LoRA fine-tuning**: phased training — first adapter-only (Phase 1), then add LoRA on LLM (Phase 2, nearly free since gradients already flow), optionally LoRA on encoder (Phase 3, more memory)
- **NSET**: fine-tuning on non-speech audio with empty transcripts to reduce hallucination/repetition
- **Inference constraints**: n-gram non-repetition + length penalty during beam search

## Project Structure

```
conf/                   Hydra configs (experiment, adapter, data, training, mode)
src/
├── data/               Datasets (MLS, SPS), text normalizer, collate functions
├── models/             Encoder, adapters, CTC model, LLM model
├── training/           Base trainer, CTC/LLM trainers, shared run entry points
├── evaluation/         WER/CER metrics, decoding
└── utils/              Device detection
scripts/                Train, evaluate, inference, quick decode
modal_app.py            Modal serverless GPU entry point
tests/                  Unit tests (pytest)
```

## Infrastructure

- **Local** (`mode=local`): MPS backend, MLS `dev` split for training, `test` for validation. Intended for smoke testing and fast iteration — not for producing final results. Swap to `mode=remote` or download the full train split for real training runs.
- **Modal** (`modal run modal_app.py`): Serverless GPU (T4 $0.59/hr, L4 $0.80/hr). Checkpoint volume persists across runs. Secrets injected from `.env`. Image cached after first build — source code changes don't trigger rebuild. Currently POC setup (data/models re-downloaded each run). See `CLAUDE.md` for volume commands.

## Status

Training pipeline complete (CTC + LLM adapter, 65 tests passing). Modal POC verified (CTC + LLM training on T4, W&B logging, checkpoint persistence). Next: full Modal setup (data/model volumes, preprocessing), then run experiments.
