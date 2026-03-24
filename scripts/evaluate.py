"""Evaluate a trained model on test sets (MLS scripted + SPS spontaneous)."""

import argparse
import logging

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data.collate import ctc_collate_fn
from src.data.datasets import MLSDataset, SPSDataset
from src.evaluation.decode import ctc_greedy_decode
from src.evaluation.metrics import compute_cer, compute_wer
from src.models.encoder import EncoderWrapper
from src.utils.device import get_device

log = logging.getLogger(__name__)


def evaluate_ctc(checkpoint_path: str, device: str, batch_size: int = 4):
    """Evaluate CTC model on both test sets."""
    from src.models.ctc_model import CTCModel

    encoder = EncoderWrapper()
    model = CTCModel(encoder)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
    model.to(device).eval()

    # MLS test set
    mls_test = MLSDataset(split="test", max_duration_s=15.0)
    mls_loader = DataLoader(mls_test, batch_size=batch_size, collate_fn=ctc_collate_fn)
    _evaluate_ctc_loader(model, mls_loader, device, "MLS test")

    # SPS test set
    sps_test = SPSDataset("data/sps-corpus-3.0-2026-03-09-fr")
    sps_loader = DataLoader(sps_test, batch_size=batch_size, collate_fn=ctc_collate_fn)
    _evaluate_ctc_loader(model, sps_loader, device, "SPS test")


def _evaluate_ctc_loader(model, loader, device, name):
    """Run CTC evaluation on a single dataloader."""
    all_preds, all_refs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            log_probs, lengths = model(batch["input_values"], batch["attention_mask"])
            preds = ctc_greedy_decode(log_probs, lengths)
            all_preds.extend(preds)

            # Recover references from packed labels
            from src.data.text_normalizer import decode_ctc_indices

            offset = 0
            for length in batch["label_lengths"]:
                ids = batch["labels"][offset : offset + length].tolist()
                all_refs.append(decode_ctc_indices(ids))
                offset += length

    wer_val = compute_wer(all_refs, all_preds)
    cer_val = compute_cer(all_refs, all_preds)
    print(f"{name}: WER={wer_val:.4f}, CER={cer_val:.4f} ({len(all_refs)} samples)")


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--type", choices=["ctc", "llm"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    log.info(f"Device: {device}")

    if args.type == "ctc":
        evaluate_ctc(args.checkpoint, device, args.batch_size)
    else:
        raise NotImplementedError("LLM evaluation — use scripts/inference.py for now")


if __name__ == "__main__":
    main()
