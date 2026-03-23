"""Tests for French text normalization and CTC charset."""

import pytest

from src.data.text_normalizer import (
    CTC_BLANK,
    CTC_CHARS,
    CTC_VOCAB_SIZE,
    CHAR_TO_IDX,
    IDX_TO_CHAR,
    decode_ctc_indices,
    encode_for_ctc,
    normalize_text,
)


# --- CTC charset ---


class TestCTCCharset:
    def test_vocab_size(self):
        assert CTC_VOCAB_SIZE == 45  # 44 chars + 1 blank

    def test_blank_is_zero(self):
        assert CTC_BLANK == 0

    def test_space_is_first_char(self):
        assert CTC_CHARS[0] == " "
        assert CHAR_TO_IDX[" "] == 1  # blank=0, space=1

    def test_all_french_accented_chars_present(self):
        accented = "àâæçéèêëîïôœùûüÿ"
        for c in accented:
            assert c in CHAR_TO_IDX, f"Missing accented char: {c!r}"

    def test_apostrophe_present(self):
        assert "'" in CHAR_TO_IDX

    def test_no_duplicate_indices(self):
        indices = list(CHAR_TO_IDX.values())
        assert len(indices) == len(set(indices))

    def test_idx_to_char_roundtrip(self):
        for c, i in CHAR_TO_IDX.items():
            assert IDX_TO_CHAR[i] == c


# --- Encoding / decoding ---


class TestEncoding:
    def test_encode_simple(self):
        indices = encode_for_ctc("abc")
        assert indices == [CHAR_TO_IDX["a"], CHAR_TO_IDX["b"], CHAR_TO_IDX["c"]]

    def test_encode_with_space(self):
        indices = encode_for_ctc("a b")
        assert indices == [CHAR_TO_IDX["a"], CHAR_TO_IDX[" "], CHAR_TO_IDX["b"]]

    def test_encode_skips_unknown_chars(self):
        # Uppercase, digits, punctuation should be absent after normalization,
        # but if passed raw, they're silently skipped
        indices = encode_for_ctc("a!b@c")
        assert indices == [CHAR_TO_IDX["a"], CHAR_TO_IDX["b"], CHAR_TO_IDX["c"]]

    def test_decode_roundtrip(self):
        text = "bonjour ça va"
        indices = encode_for_ctc(text)
        assert decode_ctc_indices(indices) == text

    def test_decode_blank_is_empty(self):
        assert decode_ctc_indices([CTC_BLANK]) == ""

    def test_encode_accented(self):
        text = "éèêëàâîïôùûüçæœÿ"
        indices = encode_for_ctc(text)
        assert decode_ctc_indices(indices) == text


# --- Text normalization ---


class TestNormalization:
    def test_lowercase(self):
        assert normalize_text("BONJOUR") == "bonjour"

    def test_punctuation_removed(self):
        assert normalize_text("Bonjour, monde!") == "bonjour monde"

    def test_preserves_french_accents(self):
        result = normalize_text("Café résumé naïve")
        assert "é" in result
        assert "ï" in result

    def test_apostrophe_normalization(self):
        # Curly apostrophe → straight
        result = normalize_text("l\u2019homme")
        assert result == "l'homme"

    def test_collapses_whitespace(self):
        result = normalize_text("  too   many   spaces  ")
        assert result == "too many spaces"


# --- Number conversion ---


class TestNumbers:
    def test_integer(self):
        # Hyphens from num2words ("quarante-deux") become spaces (hyphen not in charset)
        assert normalize_text("J'ai 42 ans") == "j'ai quarante deux ans"

    def test_decimal_dot(self):
        result = normalize_text("Il fait 3.5 degrés")
        assert "trois virgule cinq" in result

    def test_decimal_comma(self):
        result = normalize_text("Il fait 3,5 degrés")
        assert "trois virgule cinq" in result

    def test_large_number(self):
        result = normalize_text("Il y a 1000 personnes")
        assert "mille" in result

    def test_zero(self):
        result = normalize_text("Il a 0 point")
        assert "zéro" in result


# --- Ordinal conversion ---


class TestOrdinals:
    def test_premier(self):
        assert normalize_text("Le 1er mars") == "le premier mars"

    def test_premiere_feminine(self):
        assert normalize_text("La 1ère fois") == "la première fois"

    def test_premiere_short_form(self):
        assert normalize_text("La 1re fois") == "la première fois"

    def test_ordinal_e_suffix(self):
        result = normalize_text("Le 25e jour")
        assert result == "le vingt cinquième jour"

    def test_ordinal_eme_suffix(self):
        result = normalize_text("Le 3ème étage")
        assert result == "le troisième étage"

    def test_ordinal_nd_suffix(self):
        result = normalize_text("Le 2nd rang")
        assert result == "le deuxième rang"

    def test_ordinal_nde_suffix(self):
        result = normalize_text("La 2nde place")
        assert result == "la deuxième place"

    def test_mixed_ordinal_and_cardinal(self):
        result = normalize_text("Le 1er a eu 100 points")
        assert result == "le premier a eu cent points"

    def test_high_ordinal(self):
        result = normalize_text("Le 31e décembre")
        assert "trente et unième" in result


# --- Bracketed annotations ---


class TestBracketedAnnotations:
    def test_square_brackets(self):
        assert normalize_text("bonjour [rires] monde") == "bonjour monde"

    def test_parentheses(self):
        assert normalize_text("oui (musique) non") == "oui non"

    def test_angle_brackets(self):
        assert normalize_text("je <inaudible> pense") == "je pense"

    def test_multiple_brackets(self):
        result = normalize_text("a [rires] b (pause) c <bruit>")
        assert result == "a b c"

    def test_brackets_with_other_normalization(self):
        result = normalize_text("Le 1er [applaudissements] test")
        assert result == "le premier test"


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_punctuation(self):
        assert normalize_text("...!!!???") == ""

    def test_encode_empty(self):
        assert encode_for_ctc("") == []

    def test_real_sps_transcript(self):
        # Real example from the SPS corpus
        text = "Ma préoccupation, c'est qu'on devient trop dépendant des téléphones portables"
        result = normalize_text(text)
        # Should be lowercase, no punctuation, accents preserved
        assert result == "ma préoccupation c'est qu'on devient trop dépendant des téléphones portables"
        # Should encode/decode cleanly
        indices = encode_for_ctc(result)
        assert decode_ctc_indices(indices) == result
