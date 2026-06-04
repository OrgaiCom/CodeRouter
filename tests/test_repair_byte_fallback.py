"""Unit tests for ``repair_byte_fallback`` — the Gemma/llama.cpp
``<0xNN>`` byte-fallback repair filter.

Mirrors the streaming-boundary style of ``test_output_filters.py``: we
exercise (1) whole-string repair, (2) chunk boundaries inside a single
token, (3) chunk boundaries inside a multi-byte run, (4) passthrough of
ordinary text, and (5) lossless handling of malformed/incomplete tokens.
"""

from __future__ import annotations

import pytest

from coderouter.output_filters import (
    KNOWN_FILTERS,
    RepairByteFallbackFilter,
    apply_output_filters,
)


def _repair(text: str) -> tuple[str, bool]:
    """One-shot repair, returning ``(out, modified)``."""
    f = RepairByteFallbackFilter()
    out = f.feed(text, eof=True)
    return out, f.modified


def _stream(chunks: list[str]) -> str:
    """Feed chunks one at a time, flush at eof, return concatenated output."""
    f = RepairByteFallbackFilter()
    out = []
    for c in chunks:
        out.append(f.feed(c))
    out.append(f.feed("", eof=True))
    return "".join(out)


# ----------------------------------------------------------------------
# registry wiring
# ----------------------------------------------------------------------


def test_registered_in_known_filters() -> None:
    assert KNOWN_FILTERS["repair_byte_fallback"] is RepairByteFallbackFilter


def test_usable_through_chain() -> None:
    out, applied = apply_output_filters(
        ["repair_byte_fallback"], "残酷な<0xE3><0x80><0x80>蹂"
    )
    assert out == "残酷な　蹂"
    assert applied == ["repair_byte_fallback"]


# ----------------------------------------------------------------------
# 1. whole-string repair
# ----------------------------------------------------------------------


def test_full_width_space_single_shot() -> None:
    out, mod = _repair("残酷な<0xE3><0x80><0x80>蹂")
    assert out == "残酷な　蹂"
    assert mod is True


def test_rare_kanji_single_shot() -> None:
    # 躙 = U+8E99 = E8 BA 99
    out, mod = _repair("残酷な蹂<0xE8><0xBA><0x99>だった")
    assert out == "残酷な蹂躙だった"
    assert mod is True


def test_mixed_multiple_runs() -> None:
    out, _ = _repair("A<0xE3><0x80><0x80>B<0xE8><0xBA><0x99>C")
    assert out == "A　B躙C"


# ----------------------------------------------------------------------
# 2. chunk boundary inside a single token
# ----------------------------------------------------------------------


def test_split_inside_token() -> None:
    assert _stream(["残酷な<0x", "E3><0x80><0x80>蹂"]) == "残酷な　蹂"


def test_split_every_char() -> None:
    s = "x<0xE3><0x80><0x80>y"
    assert _stream(list(s)) == "x　y"


# ----------------------------------------------------------------------
# 3. chunk boundary inside a multi-byte run
# ----------------------------------------------------------------------


def test_split_between_run_tokens() -> None:
    assert _stream(["<0xE3>", "<0x80><0x80>"]) == "　"


def test_split_run_three_ways() -> None:
    assert _stream(["pre<0xE8>", "<0xBA>", "<0x99>post"]) == "pre躙post"


# ----------------------------------------------------------------------
# 4. ordinary text passthrough (must not corrupt existing content)
# ----------------------------------------------------------------------


def test_plain_text_untouched() -> None:
    out, mod = _repair("def foo(): return 1  # 普通の日本語")
    assert out == "def foo(): return 1  # 普通の日本語"
    assert mod is False


def test_non_byte_angle_tags_untouched() -> None:
    s = "<think>hmm</think><tool_call>{}</tool_call>"
    out, mod = _repair(s)
    assert out == s
    assert mod is False


def test_literal_less_than_zero_passes() -> None:
    out, _ = _repair("if x <0 and y >1: pass")
    assert out == "if x <0 and y >1: pass"


# ----------------------------------------------------------------------
# 5. lossless handling of malformed / incomplete tokens
# ----------------------------------------------------------------------


def test_malformed_non_hex_kept_literal() -> None:
    out, mod = _repair("a<0xZZ>b")
    assert out == "a<0xZZ>b"
    assert mod is False


def test_incomplete_token_at_eof_kept_literal() -> None:
    out, _ = _repair("tail<0xE3")
    assert out == "tail<0xE3"


def test_lone_invalid_byte_round_trips() -> None:
    out, mod = _repair("x<0xFF>y")
    assert out == "x<0xFF>y"
    assert mod is True  # a token WAS consumed, even though the byte was undecodable


def test_valid_prefix_then_invalid_byte() -> None:
    out, _ = _repair("<0xE3><0x80><0xFF>")
    assert out == "<0xE3><0x80><0xFF>"


# ----------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "s",
    ["", "hello world", "マルチバイト日本語テキスト", "<<<>>>", "a < b and c > d"],
)
def test_no_byte_tokens_means_byte_identical(s: str) -> None:
    out, mod = _repair(s)
    assert out == s
    assert mod is False


def test_modified_flag_semantics() -> None:
    f = RepairByteFallbackFilter()
    f.feed("plain", eof=False)
    assert f.modified is False
    f.feed("<0xE3><0x80><0x80>", eof=True)
    assert f.modified is True
