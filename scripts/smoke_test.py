"""E2E smoke test: load all components, forward pass, verify gradients + generate."""

import logging
import sys

import torch
from dotenv import load_dotenv

from src.data.collate import ctc_collate_fn, simple_audio_collate_fn
from src.data.datasets import SPSDataset
from src.models.encoder import EncoderWrapper
from src.utils.device import get_device

log = logging.getLogger(__name__)


def smoke_test_ctc(device: str):
    """Verify CTC pipeline: encoder → linear → CTC loss."""
    from src.models.ctc_model import CTCModel

    log.info("=== CTC Smoke Test ===")

    encoder = EncoderWrapper()
    model = CTCModel(encoder).to(device)
    model.train()

    # Load one real sample
    ds = SPSDataset("data/sps-corpus-3.0-2026-03-09-fr", max_duration_s=15.0)
    batch = ctc_collate_fn([ds[0]])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Forward
    log_probs, lengths = model(batch["input_values"], batch["attention_mask"])
    log.info(f"  log_probs shape: {log_probs.shape}")
    log.info(f"  frame lengths: {lengths}")

    # Loss
    loss = model.compute_loss(log_probs, lengths, batch["labels"], batch["label_lengths"])
    log.info(f"  CTC loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "CTC loss is not finite!"

    # Backward
    loss.backward()
    grad_ok = model.projection.weight.grad is not None
    log.info(f"  projection grad exists: {grad_ok}")
    assert grad_ok, "No gradient on projection head!"

    # Decode
    from src.evaluation.decode import ctc_greedy_decode

    with torch.no_grad():
        decoded = ctc_greedy_decode(log_probs, lengths)
    log.info(f"  decoded (random weights): {decoded[0][:50]!r}")

    log.info("CTC smoke test PASSED")


def smoke_test_llm(device: str):
    """Verify LLM pipeline: encoder → adapter → LLM forward + generate."""
    from src.models.adapters import ConcatMLP
    from src.models.llm_model import LLMModel

    log.info("=== LLM Smoke Test ===")

    encoder = EncoderWrapper()
    adapter = ConcatMLP(encoder_dim=768, output_dim=896)
    model = LLMModel(encoder=encoder, adapter=adapter).to(device)
    model.train()

    # Load one real sample
    ds = SPSDataset("data/sps-corpus-3.0-2026-03-09-fr", max_duration_s=15.0)
    batch = simple_audio_collate_fn([ds[0]])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Encode audio
    adapted, audio_lengths = model.encode_audio(batch["input_values"], batch["attention_mask"])
    log.info(f"  adapted shape: {adapted.shape}")
    log.info(f"  audio lengths: {audio_lengths}")

    # Build inputs_embeds (mimicking trainer)
    prefix_emb, suffix_emb, eos_emb, _, _, eos_ids = model.get_template_embeds(device)
    audio_0 = adapted[0, : audio_lengths[0]]
    transcript_emb, transcript_ids = model.embed_transcript(batch["transcripts"][0], device)

    inputs_embeds = torch.cat([prefix_emb, audio_0, suffix_emb, transcript_emb, eos_emb]).unsqueeze(
        0
    )
    n_masked = prefix_emb.shape[0] + audio_0.shape[0] + suffix_emb.shape[0]
    labels = torch.cat(
        [
            torch.full((n_masked,), -100, dtype=torch.long, device=device),
            transcript_ids,
            eos_ids,
        ]
    ).unsqueeze(0)
    attn_mask = torch.ones(1, inputs_embeds.shape[1], device=device, dtype=torch.long)

    log.info(f"  inputs_embeds shape: {inputs_embeds.shape}")
    log.info(f"  labels shape: {labels.shape}")

    # Forward — loss
    loss = model(inputs_embeds, attn_mask, labels)
    log.info(f"  LLM loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "LLM loss is not finite!"

    # Backward — check adapter gradients
    loss.backward()
    adapter_grads = [p.grad is not None for p in adapter.parameters() if p.requires_grad]
    log.info(f"  adapter params with grad: {sum(adapter_grads)}/{len(adapter_grads)}")
    assert all(adapter_grads), "Some adapter params have no gradient!"

    # Generate (prompt only, no transcript)
    model.eval()
    prompt_embeds = torch.cat([prefix_emb, audio_0, suffix_emb]).unsqueeze(0)
    prompt_mask = torch.ones(1, prompt_embeds.shape[1], device=device, dtype=torch.long)

    from src.evaluation.decode import llm_generate

    with torch.no_grad():
        generated = llm_generate(
            model,
            prompt_embeds,
            prompt_mask,
            max_new_tokens=20,
            num_beams=1,
        )
    log.info(f"  generated (random adapter): {generated[0][:80]!r}")

    # Memory usage
    if device == "mps":
        mem = torch.mps.current_allocated_memory() / 1e9
        log.info(f"  MPS memory: {mem:.2f} GB")
    elif device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        log.info(f"  CUDA peak memory: {mem:.2f} GB")

    log.info("LLM smoke test PASSED")


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = get_device()
    log.info(f"Device: {device}")

    try:
        smoke_test_ctc(device)
    except Exception as e:
        log.error(f"CTC smoke test FAILED: {e}")
        sys.exit(1)

    try:
        smoke_test_llm(device)
    except Exception as e:
        log.error(f"LLM smoke test FAILED: {e}")
        sys.exit(1)

    log.info("\nAll smoke tests PASSED!")


if __name__ == "__main__":
    main()
