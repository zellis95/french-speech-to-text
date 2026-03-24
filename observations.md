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
