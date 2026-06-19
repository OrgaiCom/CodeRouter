"""Language-tax measurement (Phase 1 PoC, 5-deps invariant).

Why this module exists
======================

Cloud LLM tokenizers charge CJK text far more tokens-per-character
than English. CodeRouter's core router uses a ``char/4`` heuristic
(:mod:`coderouter.token_estimation`) which is *conservative for CJK*
— i.e. it **under-counts** Japanese/Chinese/Korean text. That gap is
the "language tax": a Japanese prompt that the heuristic prices at N
tokens is actually billed at ~1.2-1.5x N by the cloud provider.

Local models are unaffected (no per-token billing), so the tax only
matters on the cloud leg. This module quantifies it so the cost
tracker / dashboard can surface "how much extra am I paying to work
in Japanese?".

Design constraints (mirrors token_estimation_accurate.py)
=========================================================

* **No new core dependency.** CJK detection is pure ``str`` + Unicode
  range checks (stdlib only). The *accurate* token count is delegated
  to :func:`coderouter.token_estimation_accurate.count_tokens`, whose
  precise backend (HuggingFace ``tokenizers``) is the existing
  optional ``accuracy`` extra. When that backend is absent every
  function still returns a sane value — the tax_multiplier simply
  collapses to 1.0 because both legs use char/4.
* **Local only / no network.** No tokenizer is ever downloaded; we
  only pass through a caller-supplied local ``tokenizer.json`` path.
* **Leaf module.** Imports only ``token_estimation`` /
  ``token_estimation_accurate`` (both leaves), never the engine or
  collector — keeps it trivially testable and circular-import-free.

The tax multiplier, defined
===========================

``tax_multiplier = tokens_accurate / tokens_heuristic``

where ``tokens_heuristic`` is the char/4 estimate (CodeRouter's
English-calibrated baseline) and ``tokens_accurate`` is the real
tokenizer count. Reading it:

* English / code text → real tokenizers land near char/4, so the
  multiplier is ~1.0 (no tax).
* Japanese prose → real tokenizers emit ~0.5-1.0 tokens/char vs the
  0.25 the heuristic assumes, so the multiplier lands ~2.0-4.0 on
  *pure* CJK and ~1.2-1.5 on realistic mixed coding prompts (CJK
  comments/instructions + ASCII code/identifiers).

Confidence: **MODERATE.** char/4 is itself an approximation of
English, so the multiplier is "tax relative to CodeRouter's own
English baseline", not a lab-grade JA-vs-EN figure. It is, however,
fully measurable with zero network and no guessing — which is why we
prefer it to a translate-and-compare counterfactual.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coderouter.token_estimation import (
    CHARS_PER_TOKEN_HEURISTIC,
    extract_text_from_anthropic_request,
)
from coderouter.token_estimation_accurate import count_tokens

# ---------------------------------------------------------------------------
# CJK Unicode ranges
# ---------------------------------------------------------------------------
#
# We count a character as "CJK" when it falls in one of the blocks that
# real tokenizers fragment heavily. Latin, digits, punctuation and
# whitespace are excluded so that an ASCII-only prompt scores 0.0 and a
# pure-Japanese prompt scores ~1.0. Half-width katakana and full-width
# forms are included because they tokenize like their full-width kin.
#
# Ranges are (low, high) inclusive code points.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs (common Kanji/Hanzi)
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),  # Half/Full-width forms (full-width punct, half kana)
    (0x3000, 0x303F),  # CJK symbols & punctuation (、。「」etc.)
    (0xAC00, 0xD7A3),  # Hangul syllables (Korean)
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x20000, 0x2A6DF),  # CJK Ext. B (rare ideographs)
)


def _is_cjk(cp: int) -> bool:
    return any(low <= cp <= high for low, high in _CJK_RANGES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cjk_char_ratio(text: str) -> float:
    """Fraction of *non-whitespace* characters in ``text`` that are CJK.

    Whitespace is excluded from the denominator so that indentation /
    blank lines in a code block don't dilute the score. Returns ``0.0``
    for empty or whitespace-only / pure-ASCII text and ``1.0`` for pure
    CJK. The value feeds the Phase-2 ``cjk_ratio_min`` auto-route
    matcher and the Phase-1 reporting below.
    """
    if not text:
        return 0.0
    cjk = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if _is_cjk(ord(ch)):
            cjk += 1
    if total == 0:
        return 0.0
    return cjk / total


@dataclass(frozen=True)
class LanguageTaxBreakdown:
    """Per-text language-tax measurement.

    Fields
        char_count: non-whitespace-inclusive length of the text.
        cjk_ratio: see :func:`cjk_char_ratio` (0.0-1.0).
        tokens_heuristic: char/4 estimate (CodeRouter's English
            baseline). Always available.
        tokens_accurate: real tokenizer count when a ``tokenizer_path``
            was supplied *and* the optional backend is installed;
            otherwise equals ``tokens_heuristic`` (graceful fallback).
        accurate_available: whether ``tokens_accurate`` came from the
            precise backend (True) or fell back to char/4 (False).
        tax_multiplier: ``tokens_accurate / tokens_heuristic``; 1.0
            when no tax is measurable. See module docstring for the
            MODERATE-confidence caveat.
        extra_tokens: ``tokens_accurate - tokens_heuristic`` (>= 0 for
            CJK; the visible "tax" in tokens).
    """

    char_count: int = 0
    cjk_ratio: float = 0.0
    tokens_heuristic: int = 0
    tokens_accurate: int = 0
    accurate_available: bool = False
    tax_multiplier: float = 1.0
    extra_tokens: int = 0


def estimate_language_tax(
    text: str,
    *,
    tokenizer_path: str | Path | None = None,
) -> LanguageTaxBreakdown:
    """Measure the language tax of ``text``.

    With ``tokenizer_path`` pointing at a readable local
    ``tokenizer.json`` (and the ``accuracy`` extra installed), the
    accurate leg uses the real tokenizer and the multiplier reflects
    the true char/4 under-count. Without it, both legs use char/4 and
    the multiplier is 1.0 — the function never raises and never
    touches the network.
    """
    if not text:
        return LanguageTaxBreakdown()

    heuristic = len(text) // CHARS_PER_TOKEN_HEURISTIC
    accurate_raw = count_tokens(text, tokenizer_path=tokenizer_path)

    # When the precise backend is unavailable, count_tokens returns the
    # same char/4 value, so accurate == heuristic and we report no tax.
    accurate_available = tokenizer_path is not None and accurate_raw != heuristic

    # Guard against a zero-heuristic (text shorter than 4 chars) to keep
    # the multiplier finite and meaningful.
    if heuristic <= 0:
        multiplier = 1.0
        extra = max(accurate_raw - 0, 0)
    else:
        multiplier = accurate_raw / heuristic
        extra = accurate_raw - heuristic

    return LanguageTaxBreakdown(
        char_count=len(text),
        cjk_ratio=cjk_char_ratio(text),
        tokens_heuristic=heuristic,
        tokens_accurate=accurate_raw,
        accurate_available=accurate_available,
        tax_multiplier=multiplier,
        extra_tokens=max(extra, 0),
    )


def language_tax_usd(
    extra_tokens: int,
    *,
    input_tokens_per_million: float | None,
) -> float:
    """USD attributable to the language tax for one request leg.

    ``extra_tokens`` is the :attr:`LanguageTaxBreakdown.extra_tokens`
    delta; pricing is the provider's normal input rate. Returns 0.0 for
    a free / unpriced (typically local) provider — mirroring
    :func:`coderouter.cost.compute_cost_for_attempt`'s zero-on-None
    behaviour so callers never special-case local models.
    """
    if not input_tokens_per_million or extra_tokens <= 0:
        return 0.0
    return extra_tokens * (input_tokens_per_million / 1_000_000.0)


def estimate_language_tax_for_request(
    system: Any,
    messages: list[Any],
    *,
    tokenizer_path: str | Path | None = None,
) -> LanguageTaxBreakdown:
    """Measure the language tax of a whole Anthropic-shaped request.

    Convenience wrapper used by the engine's cost-emit path: pulls the
    concatenated request text (system + message text blocks) and runs it
    through :func:`estimate_language_tax`. With no ``tokenizer_path`` the
    multiplier is 1.0 (inert), so calling this on every request is safe
    and cheap — the engine only invokes it when a provider declares a
    local ``tokenizer.json``.
    """
    text = extract_text_from_anthropic_request(system=system, messages=messages)
    return estimate_language_tax(text, tokenizer_path=tokenizer_path)


__all__ = [
    "LanguageTaxBreakdown",
    "cjk_char_ratio",
    "estimate_language_tax",
    "estimate_language_tax_for_request",
    "language_tax_usd",
]
