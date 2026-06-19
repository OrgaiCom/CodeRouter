"""Tests for coderouter.language_tax (Phase 1 PoC).

Run standalone against the real repo:

    PYTHONPATH=<repo_root>:<this_src_dir> pytest test_language_tax.py -q

These tests must pass with OR without the optional ``accuracy``
(tokenizers) backend installed — the no-backend path is the common
case and must degrade gracefully (tax_multiplier == 1.0).
"""

from __future__ import annotations

import math

import pytest

from coderouter.language_tax import (
    LanguageTaxBreakdown,
    cjk_char_ratio,
    estimate_language_tax,
    language_tax_usd,
)

# A tiny fake tokenizer.json is hard to ship; instead we exercise the
# accurate path by monkeypatching count_tokens. This keeps the test
# hermetic and backend-independent.
import coderouter.language_tax as lt


# ---------------------------------------------------------------------------
# cjk_char_ratio
# ---------------------------------------------------------------------------


def test_cjk_ratio_pure_ascii_is_zero():
    assert cjk_char_ratio("def main(): return 42") == 0.0


def test_cjk_ratio_pure_japanese_is_one():
    assert cjk_char_ratio("日本語のテキストです") == 1.0


def test_cjk_ratio_empty_is_zero():
    assert cjk_char_ratio("") == 0.0
    assert cjk_char_ratio("   \n\t ") == 0.0


def test_cjk_ratio_whitespace_excluded_from_denominator():
    # "あ a" -> non-ws chars: "あ","a" -> 1/2
    assert cjk_char_ratio("あ a") == pytest.approx(0.5)


def test_cjk_ratio_mixed_code_comment():
    # 6 CJK chars (計算する関数) + 5 ascii (abcde), space excluded -> 6/11
    text = "計算する関数 abcde"
    assert cjk_char_ratio(text) == pytest.approx(6 / 11)


def test_cjk_ratio_korean_and_fullwidth_punct():
    assert cjk_char_ratio("안녕하세요") == 1.0
    assert cjk_char_ratio("、。「」") == 1.0


# ---------------------------------------------------------------------------
# estimate_language_tax — no accurate backend (fallback)
# ---------------------------------------------------------------------------


def test_estimate_empty_returns_default():
    b = estimate_language_tax("")
    assert b == LanguageTaxBreakdown()
    assert b.tax_multiplier == 1.0


def test_estimate_without_tokenizer_has_no_tax():
    # No tokenizer_path -> accurate == heuristic -> multiplier 1.0
    b = estimate_language_tax("日本語のテキストです" * 5)
    assert b.tax_multiplier == 1.0
    assert b.extra_tokens == 0
    assert b.accurate_available is False
    assert b.tokens_heuristic == b.tokens_accurate


def test_heuristic_is_char_over_4():
    text = "x" * 40
    b = estimate_language_tax(text)
    assert b.tokens_heuristic == 10
    assert b.char_count == 40


# ---------------------------------------------------------------------------
# estimate_language_tax — accurate backend (monkeypatched)
# ---------------------------------------------------------------------------


def test_estimate_with_accurate_backend_measures_tax(monkeypatch):
    # Simulate a real tokenizer charging 2.5x the char/4 estimate for CJK.
    text = "日本語" * 20  # 60 chars -> heuristic 15 tokens
    monkeypatch.setattr(lt, "count_tokens", lambda t, *, tokenizer_path=None: 38)
    b = estimate_language_tax(text, tokenizer_path="/fake/tokenizer.json")
    assert b.tokens_heuristic == 15
    assert b.tokens_accurate == 38
    assert b.accurate_available is True
    assert b.extra_tokens == 23
    assert b.tax_multiplier == pytest.approx(38 / 15)


def test_accurate_equal_to_heuristic_reports_no_tax(monkeypatch):
    # English/code: real tokenizer ~= char/4 -> not flagged as tax.
    text = "a" * 40  # heuristic 10
    monkeypatch.setattr(lt, "count_tokens", lambda t, *, tokenizer_path=None: 10)
    b = estimate_language_tax(text, tokenizer_path="/fake/tokenizer.json")
    assert b.accurate_available is False  # equal -> treated as no measurable tax
    assert b.tax_multiplier == 1.0
    assert b.extra_tokens == 0


def test_short_text_does_not_divide_by_zero(monkeypatch):
    # 3 chars -> heuristic 0; must not raise.
    monkeypatch.setattr(lt, "count_tokens", lambda t, *, tokenizer_path=None: 3)
    b = estimate_language_tax("あ", tokenizer_path="/fake/tokenizer.json")
    assert math.isfinite(b.tax_multiplier)
    assert b.tax_multiplier == 1.0
    assert b.extra_tokens == 3


# ---------------------------------------------------------------------------
# language_tax_usd
# ---------------------------------------------------------------------------


def test_language_tax_usd_local_provider_is_zero():
    assert language_tax_usd(1000, input_tokens_per_million=None) == 0.0
    assert language_tax_usd(1000, input_tokens_per_million=0.0) == 0.0


def test_language_tax_usd_negative_or_zero_extra_is_zero():
    assert language_tax_usd(0, input_tokens_per_million=3.0) == 0.0
    assert language_tax_usd(-5, input_tokens_per_million=3.0) == 0.0


def test_language_tax_usd_basic():
    # 1,000,000 extra tokens at $3/M == $3.00
    assert language_tax_usd(1_000_000, input_tokens_per_million=3.0) == pytest.approx(3.0)
    # 23 extra tokens at $3/M
    assert language_tax_usd(23, input_tokens_per_million=3.0) == pytest.approx(
        23 * 3.0 / 1_000_000
    )
