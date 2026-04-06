# Observations

Interesting findings from development and experiments.

## Smoke test: LLM generates prompt-like text with random adapter (2026-03-24)

With a completely random (untrained) adapter, the LLM generated:

> `'videotranscrivez la parole en texto'`

The audio embeddings are garbage (random adapter weights), but the LLM's linguistic prior is strong enough to pick up on the instruction text in the chat template suffix ("Transcrivez la parole en texte") and produce something related — almost like it's trying to translate the prompt. This suggests the frozen LLM is processing the template structure correctly even before any adapter training, and that the instruction-following behaviour is robust to noisy prefix embeddings.

On a separate run, the same setup generated Chinese:

> `'狄安娜的愛意，她的名字是艾琳娜。'`

Which roughly translates to "Diana's love, her name is Elena." Qwen2.5 is from Alibaba and was trained heavily on Chinese — with random adapter weights producing noise, the LLM falls back on its strongest prior (Chinese narrative text). Shows the model is very much alive and generating coherently, just in the wrong language and from garbage audio input. Once the adapter trains, it should steer the LLM toward French transcription.

Both outputs are good signs: the LLM processes the chat template correctly, generates fluent text, and just needs a trained adapter to condition it on actual audio content.

## CTC overfit test: loss goes down, output is "errrrrrr" (2026-03-24)

Ran 100 epochs of CTC training on 3 SPS samples (batch_size=1, lr=3e-4) to verify the training loop works. Loss decreased smoothly from 10.54 → 4.65:

```
Epoch   0: loss=10.5421
Epoch  50: loss=7.0946
Epoch  99: loss=4.6472
```

Decoded output after 100 epochs:

```
pred: errrrrrrrrrrerrrrrrrrr
pred: errrrrrrrr
pred: errrreerrrrrrr
```

Meanwhile the reference transcripts are beautiful French sentences like "Quand il pleut j'aime bien prendre un poncho plutôt qu'un parapluie..." — the contrast is very funny. CTC is notoriously slow to start producing readable text — it first needs to learn the blank/space structure before actual characters emerge. The smooth loss curve confirms the pipeline is working correctly; it just needs more training to get past the "errrrr" phase.

## CTC checkpoint decode: full garbage after 2 epochs on MLS dev (2026-03-25)

Ran 2 epochs of CTC training on the full MLS dev split (1,220 samples, batch_size=2, lr=3e-4). Loss went from ~11.7 → ~10.5 (train) and val_loss=8.68. WER=1.0072 (>100%), CER=0.75.

Loaded the epoch 2 checkpoint and decoded 5 test samples:

```
[0] ref: la nuit suivante appela sa soeur quand il en fut temps si vous ne dormez pas
    hyp: aqêêxgêêgêgêgêêûêûêûêûêêêêuuêuêxüêùêæyckycêéÿyéèïûwcîæôsïîêæülxûüôqpëïÿeûÿ

[1] ref: à l'aspect d'un monstre d'une grandeur si démesurée le pêcheur voulut prendre
    hyp: qêûêûêûêûêûêêêkkxsôîæùüébéüêùÿgæyéêpêvûïlüdÿæüflkxjÿékpêùôÿïûïcnlébükêsïÿ
```

Complete alphabet soup — no recognisable French at all. Interesting that the model has latched onto accented characters (ê, û, ï, ÿ, æ) quite heavily. Possibly because the encoder features for French speech correlate with the accented character space more than plain a-z at this early stage. With CTC, the learning progression is typically: first learn to output the right density of characters and blanks, then learn which characters, then learn word boundaries. We're still in the very early "which characters" phase.

## LLM adapter decode: barely trained adapter output (2026-03-25)

Ran 1 epoch of LLM adapter training (ConcatMLP, 9.7M trainable params) on 103 MLS samples (max_duration=10.5s). Train loss 4.42, val loss 4.44. Only 6 weight updates total (52 batches / 8 grad_accum).

Decoded 5 test samples:

```
[0] ref: ce discours affligea fort le pêcheur je suis bien malheureux...
    hyp: ![](https://www.example.com/image1)

[1] ref: schahriar y consentit et scheherazade reprenant son discours...
    hyp: (empty)

[3] ref: sire répondit la sultane le troisième vieillard raconta son histoire...
    hyp: anticipated anticipation

[4] ref: il tâcha encore d'apaiser le génie hélas reprit-il daignez avoir...
    hyp: anticipation anticipation anticipation anticipation anticipation...
```

The adapter has shifted from the pre-training behaviour (Chinese text / prompt parroting with random weights) to English patterns from Qwen's training data — markdown syntax, English words stuck in loops. Generation behaviour falls into three modes:
- **Immediate EOS**: model outputs `<|im_end|>` right away → empty string after special token stripping. The LLM just closes the assistant turn without saying anything.
- **Repetition loops**: gets stuck repeating a word ("anticipation") until hitting max_new_tokens. Common failure mode with barely-trained autoregressive models.
- **Pretraining artefacts**: markdown image links, random English — the LLM's prior dominates over the weak audio signal from 6 gradient updates.

With proper training (more data, more epochs), the adapter should learn to map audio features to the text embedding space, steering the LLM toward French transcription.

## LLM adapter with more data: French emerges! (2026-03-25)

1 epoch on ~300 MLS samples (max_duration=12s), ConcatMLP adapter. Decoded 5 test samples:

```
[0] hyp: ![](https://i.imgur.com/7ZJzQWu.png)          ← still imgur markdown
[2] hyp: compartment, container, receptacle, receptivity... ← English thesaurus mode
[3] hyp: ixi i xixi i xixi i xixi i xixi...              ← rhythmic repetition
[4] hyp: -C'est une phrase en français.                   ← ACTUAL FRENCH!
```

Sample 4 produces French: "C'est une phrase en français" = "It's a sentence in French." However, the prompt itself is in French ("Transcrivez la parole en texte"), so the LLM has a strong hint to respond in French regardless of what the adapter feeds it. This is more likely the LLM following instruction language than the adapter having learned anything meaningful about the audio content. The real signal would be output that matches the *content* of the audio, not just the language.

Still, the progression from Chinese/English to French output is interesting — fewer pretraining artefacts, more prompt-following:
1. **Random adapter**: Chinese text / prompt parroting (LLM's strongest prior)
2. **6 weight updates** (103 samples): English patterns, markdown, repetition loops
3. **~38 weight updates** (300 samples, 1 epoch): Mix of English artefacts + French appearing

Memory note: this run hit ~20GB during epoch 2, forcing a kill. Data preprocessing (48kHz→16kHz one-off) is essential before longer training runs on MPS.

## Modal POC: CTC + LLM training on T4 GPU (2026-04-06)

First training runs on Modal serverless GPU (T4, 16GB VRAM). Verified the full pipeline works remotely.

**CTC run** (2 epochs, 46 samples at max_duration=10.2s): Loss 10.75 → 10.75, barely moved. Tiny dataset, too few epochs. 14s total training time.

**LLM run** (3 epochs, 214 samples at max_duration=11.0s): Val loss 4.13 → 3.94 → 3.79, clear learning signal. 9.7M trainable adapter params. 144s total training time. GPU usage peaked at ~4.5GB VRAM, 68% utilisation, ~3GB RAM.

Downloaded the LLM checkpoint and decoded 5 test samples locally:

```
[0] hyp: ![](http://www.1234567890.com/1234567890/...)    ← markdown URLs (pretraining artefact)
[1] hyp: 01234567891011121314151617181920...                ← sequential digit counting (new pattern)
[2] hyp: 60000000000000000000000000000000...                ← degenerate repetition
```

Same stage as earlier observations: pretraining artefacts dominate, no meaningful audio→text mapping yet. The sequential digit counting is a new failure mode not seen before — the model's found a low-loss pattern of outputting numbers in sequence. Needs significantly more data and epochs to push past this.

**Infrastructure notes**: Image build cached after first run (~158s). Source code mounted at runtime (no rebuild on code changes). Checkpoint volume works — 37MB adapter weights persist correctly. W&B logging confirmed. Total cost for all POC runs: ~$0.15.

**Bug found**: `_save_checkpoint()` was using `v is p` (object identity) to filter trainable params from `state_dict()`. Since `state_dict()` returns detached copies, this was always False — checkpoints were empty dicts (1.3KB). Fixed to use `named_parameters()` key matching.
