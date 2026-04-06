"""WER and CER computation via jiwer."""

import logging

from jiwer import cer, wer

from src.data.text_normalizer import normalize_text

log = logging.getLogger(__name__)


def compute_wer(references: list[str], hypotheses: list[str], normalize: bool = True) -> float:
    """Compute Word Error Rate.

    Args:
        references: ground truth transcripts
        hypotheses: predicted transcripts
        normalize: whether to apply French text normalization before comparison

    Returns:
        WER as a float (0.0 = perfect, 1.0 = 100% error)
    """
    if normalize:
        references = [normalize_text(r) for r in references]
        hypotheses = [normalize_text(h) for h in hypotheses]

    # Filter out empty references (jiwer crashes on empty reference strings)
    total = len(references)
    pairs = [(r, h) for r, h in zip(references, hypotheses, strict=True) if r.strip()]
    skipped = total - len(pairs)
    if skipped:
        log.warning(f"compute_wer: skipped {skipped}/{total} empty references")
    if not pairs:
        log.warning("compute_wer: no non-empty references — returning 0.0")
        return 0.0

    refs, hyps = zip(*pairs, strict=False)
    return wer(list(refs), list(hyps))


def compute_cer(references: list[str], hypotheses: list[str], normalize: bool = True) -> float:
    """Compute Character Error Rate.

    Args:
        references: ground truth transcripts
        hypotheses: predicted transcripts
        normalize: whether to apply French text normalization before comparison

    Returns:
        CER as a float (0.0 = perfect)
    """
    if normalize:
        references = [normalize_text(r) for r in references]
        hypotheses = [normalize_text(h) for h in hypotheses]

    # Filter out empty references (jiwer crashes on empty reference strings)
    total = len(references)
    pairs = [(r, h) for r, h in zip(references, hypotheses, strict=True) if r.strip()]
    skipped = total - len(pairs)
    if skipped:
        log.warning(f"compute_cer: skipped {skipped}/{total} empty references")
    if not pairs:
        log.warning("compute_cer: no non-empty references — returning 0.0")
        return 0.0

    refs, hyps = zip(*pairs, strict=False)
    return cer(list(refs), list(hyps))
