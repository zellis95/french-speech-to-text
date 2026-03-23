"""French text normalization and CTC charset for ASR evaluation and training."""

import re
import unicodedata

from num2words import num2words

# CTC charset: 45 classes
# Index 0 = blank (CTC), 1 = space, 2-27 = a-z, 28-43 = French accented, 44 = apostrophe
CTC_CHARS = list(
    " abcdefghijklmnopqrstuvwxyzàâæçéèêëîïôœùûüÿ'"
)
CTC_BLANK = 0
CTC_VOCAB_SIZE = len(CTC_CHARS) + 1  # +1 for blank at index 0

# Char-to-index mapping (blank=0, chars start at 1)
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CTC_CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CTC_CHARS)}
IDX_TO_CHAR[CTC_BLANK] = ""  # blank decodes to empty string

# Characters to keep after normalization (the CTC charset)
_KEEP_CHARS = set(CTC_CHARS)

# Regex to find French ordinals: 1er, 1ère, 2e, 3ème, 2nd, 2nde, etc.
# Must be checked BEFORE plain numbers so "1er" doesn't become "un er"
_ordinal_re = re.compile(r"(\d+)\s*(ère|er|re|ème|nde|nd|e)\b", re.IGNORECASE)

# Regex to find plain numbers (integers and decimals)
_number_re = re.compile(r"\d+(?:[.,]\d+)?")

# Regex to strip bracketed/parenthetical/angled annotations (e.g. [rires], (musique), <inaudible>)
_bracketed_re = re.compile(r"\[.*?\]|\(.*?\)|<.*?>")



def _normalize_apostrophes(text: str) -> str:
    """Normalize all apostrophe variants to standard ASCII apostrophe."""
    return text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02BC", "'")


def _ordinals_to_words(text: str) -> str:
    """Convert French ordinal patterns to words. E.g. '1er' → 'premier', '25e' → 'vingt-cinquième'.

    Handles feminine "1ère" → "première" (only ordinal with gendered forms in French).
    """

    def _replace(match: re.Match) -> str:
        try:
            n = int(match.group(1))
            suffix = match.group(2).lower()
            # 1ère/1re → première (feminine of premier)
            if n == 1 and suffix in ("ère", "re"):
                return "première"
            return num2words(n, lang="fr", to="ordinal")
        except (ValueError, OverflowError):
            return match.group()

    return _ordinal_re.sub(_replace, text)


def _numbers_to_words(text: str) -> str:
    """Convert digit sequences to French words. E.g. '42' → 'quarante-deux'."""

    def _replace(match: re.Match) -> str:
        s = match.group()
        # Handle decimal with comma (French style: 3,5) or dot
        s = s.replace(",", ".")
        try:
            n = float(s) if "." in s else int(s)
            return num2words(n, lang="fr")
        except (ValueError, OverflowError):
            return s

    return _number_re.sub(_replace, text)


def normalize_text(text: str) -> str:
    """Full French text normalization pipeline for ASR.

    Steps:
        1. Strip bracketed annotations ([rires], (musique), <inaudible>)
        2. Normalize apostrophes (curly → straight)
        3. Convert ordinals to French words (1er → premier, 25e → vingt-cinquième)
        4. Convert remaining digits to French words
        5. Lowercase
        6. Unicode NFC normalization
        7. Replace any character not in CTC charset with space
        8. Collapse whitespace
    """
    text = _bracketed_re.sub(" ", text)
    text = _normalize_apostrophes(text)
    text = _ordinals_to_words(text)
    text = _numbers_to_words(text)
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    # Keep only CTC charset characters; everything else (punctuation, hyphens,
    # brackets, etc.) becomes a space to preserve word boundaries
    text = "".join(c if c in _KEEP_CHARS else " " for c in text)
    text = " ".join(text.split())
    return text


def encode_for_ctc(text: str) -> list[int]:
    """Encode normalized text to CTC label indices. Unknown chars are skipped."""
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_ctc_indices(indices: list[int]) -> str:
    """Decode CTC indices back to text (no blank/dedup — raw mapping only)."""
    return "".join(IDX_TO_CHAR.get(i, "") for i in indices)
