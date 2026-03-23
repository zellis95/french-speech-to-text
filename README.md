# French Speech-to-Text

Experimentation pipeline comparing CTC baseline vs LLM-adapter ASR on French, following the [SLAM-ASR architecture](https://arxiv.org/abs/2402.08846).

**Core idea**: frozen speech encoder (mHuBERT) + frozen instruct LLM (Qwen2.5-0.5B) + trainable linear adapter achieves competitive ASR by leveraging the LLM's linguistic prior.

## Architecture

```
                        CTC Experiment              LLM Experiment
                        ──────────────              ──────────────
Audio (16kHz) ──► mHuBERT-147 (frozen) ──► Linear(768,45) ──► CTC Loss
                        │
                        └──► Adapter (trainable) ──► Qwen2.5-0.5B (frozen) ──► CE Loss
                             ConcatMLP / ConvMLP       (inputs_embeds)
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
| [SPS Corpus](data/sps-corpus-3.0-2026-03-09-fr/) | Spontaneous speech | — | 152 | ~0.6h |

Two test sets: scripted (in-distribution MLS) and spontaneous (out-of-distribution SPS) to measure generalisation.

## Usage

```bash
# Train CTC baseline locally (uses MLS dev split, ~158MB download)
python scripts/train.py mode=local experiment=ctc_baseline

# Train LLM adapter locally
python scripts/train.py mode=local experiment=llm_adapter adapter=concat_mlp

# Train on Modal GPU (full dataset)
modal run modal_app.py -- mode=remote experiment=llm_adapter adapter=conv_mlp

# Override any config
python scripts/train.py mode=local training.lr=5e-5 training.epochs=10
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

## Project Structure

```
conf/                   Hydra configs (experiment, adapter, data, training, mode)
src/
├── data/               Datasets (MLS, SPS), text normalizer, collate functions
├── models/             Encoder, adapters, CTC model, LLM model
├── training/           Base trainer, CTC trainer, LLM trainer
├── evaluation/         WER/CER metrics, decoding
└── utils/              Device detection
scripts/                Train, evaluate, inference entry points
tests/                  Unit tests (pytest)
```

## Infrastructure

- **Local** (`mode=local`): MPS backend, MLS `dev` split for training, `test` for validation. Intended for smoke testing and fast iteration — not for producing final results. Swap to `mode=remote` or download the full train split for real training runs.
- **Remote** (`mode=remote`): Modal serverless GPU (L4/T4), full MLS `train` split pre-cached on Modal Volume, `dev` for validation, W&B logging.
