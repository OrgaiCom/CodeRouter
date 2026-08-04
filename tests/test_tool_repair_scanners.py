"""H-4 regression guards for the bare-JSON brace scanners.

Two things are pinned down here:

* the *contract* of ``_find_balanced_json_objects`` /
  ``_find_candidate_object_spans`` — top-level objects only, nested objects
  surfacing when the outer brace never closes, and a fresh string state at
  every restart. All three are load-bearing for recall and all three are easy
  to lose in a "cleaner" rewrite;
* the *cost* — the scanners used to re-scan to end-of-text once per ``{``
  (``'{' * 49152`` took over two minutes and blocked the whole event loop,
  since ``to_anthropic_response`` is called from the request path).

The timing budgets below are deliberately loose (the fixed implementation is
three to four orders of magnitude under them) so a slow CI box cannot make
them flap; only a return to quadratic behaviour trips them.
"""

from __future__ import annotations

import logging
import time

import pytest

from coderouter.translation.tool_repair import (
    _MAX_BARE_SCAN_CHARS,
    _find_balanced_json_objects,
    _find_candidate_object_spans,
    repair_tool_calls_in_text,
)

# ------------------------------------------------------------------
# Return-value contract
# ------------------------------------------------------------------


def test_toplevel_only_when_outer_closes() -> None:
    """A closed object hides its nesting: only the outer span is returned."""
    text = '{"a": {"b": 1}}'
    assert _find_balanced_json_objects(text) == [(0, len(text), text)]
    assert _find_candidate_object_spans(text) == [(0, len(text), text)]


def test_nested_returned_when_outer_unclosed() -> None:
    """When the outer brace never closes, the inner object still surfaces."""
    text = '{"a": {"b":1}'
    assert _find_balanced_json_objects(text) == [(6, 13, '{"b":1}')]
    assert _find_candidate_object_spans(text) == [(6, 13, '{"b":1}')]

    spaced = '{"a": {"b": 1}'
    assert _find_balanced_json_objects(spaced) == [(6, 14, '{"b": 1}')]


def test_restart_uses_fresh_string_state() -> None:
    """After an unbalanced ``{`` the scan restarts outside any string.

    ``{ "z{}" `` never closes, so the scan drops the leading brace and starts
    over at the ``{`` *inside* the quoted run — with no memory of the quote.
    """
    text = '{ "z{}" '
    assert _find_balanced_json_objects(text) == [(4, 6, "{}")]
    assert _find_candidate_object_spans(text) == [(4, 6, "{}")]


def test_lenient_survives_prose_apostrophe() -> None:
    """The single most important recall guard.

    A global one-pass scanner treats the apostrophe in "I'll" as opening a
    single-quoted string and swallows every object after it. The lenient
    scanner must restart clean and still find the call.
    """
    text = "I'll call it now.\n{'name': 'read_file', 'arguments': {'path': '/a'}}"

    spans = _find_candidate_object_spans(text)
    assert len(spans) == 1
    assert spans[0][2].startswith("{'name'")

    cleaned, calls = repair_tool_calls_in_text(text, ["read_file"])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert "read_file" not in cleaned


def test_escape_sequences_in_strings() -> None:
    """Braces inside escaped quotes must not disturb the balance."""
    text = '{"a": "he said \\"{\\" ok"}'
    assert _find_balanced_json_objects(text) == [(0, len(text), text)]
    assert _find_candidate_object_spans(text) == [(0, len(text), text)]

    # A trailing backslash right before the closing quote leaves the string
    # open, so nothing balances — and the scan must not hang looking.
    unterminated = '{"a": "x\\"}'
    assert _find_balanced_json_objects(unterminated) == []


def test_deeply_nested_object_is_balanced() -> None:
    depth = 500
    text = "{" * depth + "}" * depth
    assert _find_balanced_json_objects(text) == [(0, len(text), text)]


# ------------------------------------------------------------------
# Cost
# ------------------------------------------------------------------


def test_unclosed_brace_run_is_linear() -> None:
    """``'{' * 49152`` took 128-148 s before H-4; it must now be instant."""
    text = "{" * 49152
    started = time.perf_counter()
    _cleaned, calls = repair_tool_calls_in_text(text, ["Bash"])
    elapsed = time.perf_counter() - started
    assert calls == []
    assert elapsed < 1.0, f"bare-brace scan took {elapsed:.2f}s"


def test_prose_with_stray_braces_is_fast() -> None:
    """48 KB of prose sprinkled with unbalanced ``{`` (was ~3.5-22 s)."""
    text = "Lorem ipsum dolor sit amet { consectetur adipiscing elit.\n" * 850
    assert len(text) > 48_000
    started = time.perf_counter()
    repair_tool_calls_in_text(text, ["Bash"])
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"prose scan took {elapsed:.2f}s"


def test_repeated_open_object_prefix_is_fast() -> None:
    """The ``'Here is the data: ' + '{"k": ' * 8000`` shape (was ~22 s)."""
    text = "Here is the data: " + '{"k": ' * 8000
    started = time.perf_counter()
    repair_tool_calls_in_text(text, ["Bash"])
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"open-object scan took {elapsed:.2f}s"


# ------------------------------------------------------------------
# Input-length guard
# ------------------------------------------------------------------

_FENCED_CALL = '```json\n{"name": "Bash", "arguments": {"command": "pwd"}}\n```\n'
_BARE_CALL = '\n{"name": "read_file", "arguments": {"path": "/a"}}\n'


def test_input_length_guard(caplog: pytest.LogCaptureFixture) -> None:
    """Above the ceiling the bare scan is skipped; the fenced path still runs."""
    padding = "lorem ipsum dolor sit amet consectetur " * 8_000
    text = _FENCED_CALL + padding + _BARE_CALL
    assert len(text) > _MAX_BARE_SCAN_CHARS

    with caplog.at_level(logging.WARNING, logger="coderouter.translation.tool_repair"):
        cleaned, calls = repair_tool_calls_in_text(text, ["Bash", "read_file"])

    assert [c["function"]["name"] for c in calls] == ["Bash"]
    # The bare object was never scanned, so it is still sitting in the text.
    assert "read_file" in cleaned
    assert any(rec.message == "tool-repair-input-too-large" for rec in caplog.records)


def test_input_length_guard_not_applied_below_ceiling() -> None:
    """Control: the very same shapes below the ceiling repair both calls."""
    padding = "lorem ipsum dolor sit amet consectetur " * 100
    text = _FENCED_CALL + padding + _BARE_CALL
    assert len(text) < _MAX_BARE_SCAN_CHARS

    cleaned, calls = repair_tool_calls_in_text(text, ["Bash", "read_file"])

    assert sorted(c["function"]["name"] for c in calls) == ["Bash", "read_file"]
    assert "read_file" not in cleaned
