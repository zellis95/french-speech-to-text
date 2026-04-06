"""Quick decode: load LLM checkpoint and run inference on a few MLS test samples."""

import logging

import torch
from dotenv import load_dotenv

from src.data.collate import simple_audio_collate_fn
from src.data.datasets import MLSDataset
from src.evaluation.decode import llm_generate
from src.models.adapters import ConcatMLP
from src.models.encoder import EncoderWrapper
from src.models.llm_model import LLMModel
from src.utils.device import get_device

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    load_dotenv()
    device = get_device()
    log.info(f"Device: {device}")

    # Build model with same architecture as training
    encoder = EncoderWrapper("utter-project/mHuBERT-147")
    adapter = ConcatMLP(encoder_dim=768, output_dim=896, concat_k=5)
    model = LLMModel(encoder=encoder, adapter=adapter)

    # Load trained adapter weights
    ckpt = torch.load("checkpoints/llm_concat_mlp_best.pt", weights_only=True, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    log.info(f"Loaded checkpoint with {len(ckpt)} parameter tensors")

    # Load a few test samples
    ds = MLSDataset(split="test", max_duration_s=11.0)
    n_samples = min(5, len(ds))
    samples = [ds[i] for i in range(n_samples)]
    batch = simple_audio_collate_fn(samples)

    # Encode audio
    with torch.no_grad():
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        adapted, audio_lengths = model.encode_audio(input_values, attention_mask)

    # Decode each sample individually
    prefix_emb, suffix_emb, _, _, _, _ = model.get_template_embeds(torch.device(device))

    for i in range(n_samples):
        audio_emb = adapted[i, : audio_lengths[i]]
        prompt_embeds = torch.cat([prefix_emb, audio_emb, suffix_emb]).unsqueeze(0)
        attn_mask = torch.ones(1, prompt_embeds.shape[1], device=device, dtype=torch.long)

        with torch.no_grad():
            text = llm_generate(model, prompt_embeds, attn_mask, max_new_tokens=128, num_beams=1)[0]

        ref = batch["transcripts"][i]
        log.info(f"[{i}] ref: {ref[:80]}")
        log.info(f"[{i}] hyp: {text[:80]}")
        log.info("")


if __name__ == "__main__":
    main()
