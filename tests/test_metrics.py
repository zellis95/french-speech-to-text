"""Tests for WER/CER metrics."""

from src.evaluation.metrics import compute_cer, compute_wer


class TestWER:
    def test_perfect_match(self):
        refs = ["bonjour le monde"]
        hyps = ["bonjour le monde"]
        assert compute_wer(refs, hyps, normalize=False) == 0.0

    def test_complete_mismatch(self):
        refs = ["bonjour le monde"]
        hyps = ["au revoir la terre"]
        wer = compute_wer(refs, hyps, normalize=False)
        assert wer > 0.0

    def test_normalization_makes_match(self):
        refs = ["Bonjour, le monde!"]
        hyps = ["bonjour le monde"]
        # With normalization, punctuation/case stripped → should match
        assert compute_wer(refs, hyps, normalize=True) == 0.0

    def test_empty_reference_skipped(self):
        refs = ["", "bonjour"]
        hyps = ["au revoir", "bonjour"]
        # Empty reference filtered out, only "bonjour"=="bonjour" counted
        assert compute_wer(refs, hyps, normalize=False) == 0.0

    def test_multiple_samples(self):
        refs = ["un deux trois", "quatre cinq"]
        hyps = ["un deux trois", "quatre six"]
        wer = compute_wer(refs, hyps, normalize=False)
        # First pair: 0/3 errors. Second: 1/2 errors.
        # Total: 1 error / 5 words = 0.2
        assert abs(wer - 0.2) < 0.01


class TestCER:
    def test_perfect_match(self):
        refs = ["bonjour"]
        hyps = ["bonjour"]
        assert compute_cer(refs, hyps, normalize=False) == 0.0

    def test_one_char_error(self):
        refs = ["bonjour"]
        hyps = ["bonjoir"]
        cer = compute_cer(refs, hyps, normalize=False)
        assert cer > 0.0
        assert cer < 0.5  # 1 substitution in 7 chars

    def test_empty_hypothesis(self):
        refs = ["bonjour"]
        hyps = [""]
        cer = compute_cer(refs, hyps, normalize=False)
        assert cer == 1.0  # all chars deleted
