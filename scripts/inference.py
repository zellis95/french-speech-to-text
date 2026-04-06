"""Run inference on a single audio file."""

import argparse
import logging

import torch
import torchaudio
from dotenv import load_dotenv

from src.data.datasets import TARGET_SAMPLE_RATE
from src.models.encoder import EncoderWrapper
from src.utils.device import get_device

log = logging.getLogger(__name__)


def infer_ctc(audio_path: str, checkpoint_path: str, device: str):
    """Run CTC inference on a single audio file."""
    from src.evaluation.decode import ctc_greedy_decode
    from src.models.ctc_model import CTCModel

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.squeeze(0)  # mono
    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(waveform)

    # Load model
    encoder = EncoderWrapper()
    model = CTCModel(encoder)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
    model.to(device).eval()

    # Inference
    with torch.no_grad():
        input_values = waveform.unsqueeze(0).to(device)  # (1, T)
        log_probs, lengths = model(input_values)
        text = ctc_greedy_decode(log_probs, lengths)[0]

    print(f"Transcription: {text}")
    return text


def infer_llm(audio_path: str, checkpoint_path: str, device: str, adapter_cfg=None):
    """Run LLM adapter inference on a single audio file."""
    from src.evaluation.decode import llm_generate
    from src.models.adapters import build_adapter
    from src.models.llm_model import LLMModel

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.squeeze(0)
    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(waveform)

    # Load model — need adapter config to reconstruct
    encoder = EncoderWrapper()
    adapter = build_adapter(adapter_cfg)
    model = LLMModel(encoder=encoder, adapter=adapter)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
    model.to(device).eval()

    # Build prompt
    input_values = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        adapted, audio_lengths = model.encode_audio(input_values)

    prefix_emb, suffix_emb, _, _, _, _ = model.get_template_embeds(device)
    audio_emb = adapted[0, : audio_lengths[0]]
    prompt_embeds = torch.cat([prefix_emb, audio_emb, suffix_emb]).unsqueeze(0)
    attn_mask = torch.ones(1, prompt_embeds.shape[1], device=device, dtype=torch.long)

    text = llm_generate(model, prompt_embeds, attn_mask, max_new_tokens=256)[0]
    print(f"Transcription: {text}")
    return text


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to audio file (.wav, .mp3, .flac)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--type", choices=["ctc", "llm"], required=True)
    args = parser.parse_args()

    device = get_device()

    if args.type == "ctc":
        infer_ctc(args.audio, args.checkpoint, device)
    else:
        raise NotImplementedError("LLM inference requires adapter config — use Hydra entry point")


if __name__ == "__main__":
    main()
