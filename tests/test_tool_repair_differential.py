"""H-4 differential test: the linear brace scanners must match the old ones.

``coderouter.translation.tool_repair`` used to answer "is there a balanced
``{...}`` starting here?" by re-scanning to end-of-text once per ``{``, which
is O(k*n) and blocked the event loop for minutes on pathological assistant
messages. The scanners are now driven by a precomputed close-brace table built
in one right-to-left sweep.

That rewrite is only safe if it is *bit-for-bit* equivalent, and the old
behaviour has three non-obvious properties that a "just use one global stack
pass" rewrite silently breaks:

  1. only top-level objects are returned (a closed object hides its nesting);
  2. nested objects DO surface when the outer brace never closes;
  3. every restart begins from a fresh string state, so a stray quote — an
     ordinary prose apostrophe, most of all — cannot swallow later objects.

This module keeps a verbatim port of the pre-fix scanners as the reference
oracle and compares against it on randomised input plus the whole tool-repair
benchmark corpus (end to end, through ``repair_tool_calls_in_text``).

``test_reference_rejects_naive_global_stack_pass`` pins down that this harness
actually has teeth: it asserts the naive single-pass rewrite *does* diverge
from the reference on the exact inputs that matter.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from coderouter.translation import tool_repair
from coderouter.translation.tool_repair import repair_tool_calls_in_text

CORPUS = Path(__file__).resolve().parents[1] / "benchmarks" / "tool-repair" / "corpus.jsonl"


# ------------------------------------------------------------------
# Reference implementations — verbatim port of the pre-H-4 scanners
# ------------------------------------------------------------------


def ref_find_balanced_json_objects(text: str) -> list[tuple[int, int, str]]:
    """Pre-H-4 ``_find_balanced_json_objects`` (naive per-``{`` rescan)."""
    out: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_str = False
        escape = False
        while j < n:
            c = text[j]
            if escape:
                escape = False
            elif in_str:
                if c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        out.append((i, j + 1, text[i : j + 1]))
                        i = j + 1
                        break
            j += 1
        else:
            i += 1
            continue
    return out


def ref_find_candidate_object_spans(text: str) -> list[tuple[int, int, str]]:
    """Pre-H-4 ``_find_candidate_object_spans`` (naive per-``{`` rescan)."""
    out: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        quote: str | None = None
        escape = False
        while j < n:
            c = text[j]
            if escape:
                escape = False
            elif quote is not None:
                if c == "\\":
                    escape = True
                elif c == quote:
                    quote = None
            else:
                if c == '"' or c == "'":
                    quote = c
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        out.append((i, j + 1, text[i : j + 1]))
                        i = j + 1
                        break
            j += 1
        else:
            i += 1
            continue
    return out


def naive_global_stack_spans(text: str, *, lenient: bool) -> list[tuple[int, int, str]]:
    """The tempting-but-wrong rewrite: ONE stack pass over the whole text.

    Kept here purely as a negative control — see
    ``test_reference_rejects_naive_global_stack_pass``.
    """
    out: list[tuple[int, int, str]] = []
    stack: list[int] = []
    quote: str | None = None
    escape = False
    for j, c in enumerate(text):
        if escape:
            escape = False
        elif quote is not None:
            if c == "\\":
                escape = True
            elif c == quote:
                quote = None
        else:
            if c == '"' or (lenient and c == "'"):
                quote = c
            elif c == "{":
                stack.append(j)
            elif c == "}" and stack:
                start = stack.pop()
                if not stack:
                    out.append((start, j + 1, text[start : j + 1]))
    return out


# ------------------------------------------------------------------
# Randomised differential fuzzing
# ------------------------------------------------------------------

# Five deliberately different generators: the plain full alphabet, a
# quote-free variant (pure brace arithmetic), JSON-ish token soup, a
# brace-dominated variant, and an escape-dominated variant.
_ALPHABETS: dict[str, list[str]] = {
    "full": ["{", "}", '"', "'", "\\", "a", " ", ":", ",", "1", "\n"],
    "no_quotes": ["{", "}", "a", " ", ":", ",", "1", "\n"],
    "json_like": [
        '{"name": ',
        '{"arguments": ',
        '"Bash"',
        "'read_file'",
        "{}",
        "}",
        ", ",
        ": ",
        "\n",
        "prose ",
        "I'll ",
        '\\"',
    ],
    "brace_heavy": ["{", "{", "{", "}", "}", '"', "'", "x"],
    "escape_heavy": ["\\", "\\\\", '\\"', '"', "'", "{", "}", "a"],
}

_SAMPLES_PER_ALPHABET = 2500


def _generate(rng: random.Random, alphabet: list[str], max_tokens: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_tokens)))


@pytest.mark.parametrize("alphabet_name", sorted(_ALPHABETS))
def test_scanners_match_reference_on_random_input(alphabet_name: str) -> None:
    rng = random.Random(0xC0DE_2024 + sum(ord(c) for c in alphabet_name))
    alphabet = _ALPHABETS[alphabet_name]
    strict_mismatch: list[str] = []
    lenient_mismatch: list[str] = []
    for _ in range(_SAMPLES_PER_ALPHABET):
        text = _generate(rng, alphabet, 60)
        if tool_repair._find_balanced_json_objects(text) != ref_find_balanced_json_objects(text):
            strict_mismatch.append(text)
        if tool_repair._find_candidate_object_spans(text) != ref_find_candidate_object_spans(text):
            lenient_mismatch.append(text)
    assert strict_mismatch == [], f"strict scanner diverged: {strict_mismatch[:3]!r}"
    assert lenient_mismatch == [], f"lenient scanner diverged: {lenient_mismatch[:3]!r}"


def test_scanners_match_reference_on_long_random_input() -> None:
    """Same comparison on longer strings, where nesting actually gets deep."""
    rng = random.Random(0x5EED)
    alphabet = _ALPHABETS["full"]
    for _ in range(300):
        text = _generate(rng, alphabet, 600)
        assert tool_repair._find_balanced_json_objects(text) == ref_find_balanced_json_objects(text)
        assert tool_repair._find_candidate_object_spans(text) == ref_find_candidate_object_spans(
            text
        )


def test_reference_rejects_naive_global_stack_pass() -> None:
    """The harness must be able to fail: prove it catches the naive rewrite.

    A single global stack pass carries string state across candidates, so an
    apostrophe in ordinary prose opens a "string" that eats every following
    object. These are the two canonical inputs where that shows up.
    """
    prose = "I'll call the tool now.\n{'name': 'read_file', 'arguments': {'path': '/a'}}"
    quoted_brace = '{ "z{}" '

    assert len(ref_find_candidate_object_spans(prose)) == 1
    assert naive_global_stack_spans(prose, lenient=True) == []

    assert ref_find_balanced_json_objects(quoted_brace) == [(4, 6, "{}")]
    assert naive_global_stack_spans(quoted_brace, lenient=False) == []

    # ... and the shipped scanners side with the reference, not the naive pass.
    assert tool_repair._find_candidate_object_spans(prose) == ref_find_candidate_object_spans(prose)
    assert tool_repair._find_balanced_json_objects(quoted_brace) == [(4, 6, "{}")]


def test_naive_global_stack_diverges_broadly_on_fuzz() -> None:
    """Quantify it: the naive pass is wrong on a large slice of random input."""
    rng = random.Random(0xBADC0DE)
    alphabet = _ALPHABETS["full"]
    strict_diffs = 0
    lenient_diffs = 0
    for _ in range(4000):
        text = _generate(rng, alphabet, 60)
        if naive_global_stack_spans(text, lenient=False) != ref_find_balanced_json_objects(text):
            strict_diffs += 1
        if naive_global_stack_spans(text, lenient=True) != ref_find_candidate_object_spans(text):
            lenient_diffs += 1
    assert strict_diffs > 100, strict_diffs
    assert lenient_diffs > 100, lenient_diffs


# ------------------------------------------------------------------
# End-to-end differential over the benchmark corpus
# ------------------------------------------------------------------


def _load_corpus() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _comparable(result: tuple[str, list[dict[str, Any]]]) -> tuple[str, list[list[Any]]]:
    """Drop the freshly-minted ``id`` — it is random by design."""
    cleaned, calls = result
    return cleaned, [
        [tc.get("function", {}).get("name"), tc.get("function", {}).get("arguments")]
        for tc in calls
    ]


CORPUS_ROWS = _load_corpus()


def test_corpus_is_loaded() -> None:
    assert len(CORPUS_ROWS) >= 50
    assert all("input_text" in row for row in CORPUS_ROWS)


@pytest.mark.parametrize("row", CORPUS_ROWS, ids=[str(r.get("id")) for r in CORPUS_ROWS])
def test_corpus_end_to_end_matches_reference(
    row: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    text = row["input_text"]
    allowed = row.get("allowed_tools")

    new = _comparable(repair_tool_calls_in_text(text, allowed))

    monkeypatch.setattr(tool_repair, "_find_balanced_json_objects", ref_find_balanced_json_objects)
    monkeypatch.setattr(tool_repair, "_find_candidate_object_spans", ref_find_candidate_object_spans)
    old = _comparable(repair_tool_calls_in_text(text, allowed))

    assert new == old


def test_corpus_end_to_end_rejects_naive_global_stack() -> None:
    """The end-to-end comparison also has teeth against the naive rewrite."""
    prose = "I'll call the tool now.\n{'name': 'read_file', 'arguments': {'path': '/a'}}"
    good = _comparable(repair_tool_calls_in_text(prose, ["read_file"]))
    assert [c[0] for c in good[1]] == ["read_file"]

    original_strict = tool_repair._find_balanced_json_objects
    original_lenient = tool_repair._find_candidate_object_spans
    try:
        tool_repair._find_balanced_json_objects = lambda t: naive_global_stack_spans(  # type: ignore[assignment]
            t, lenient=False
        )
        tool_repair._find_candidate_object_spans = lambda t: naive_global_stack_spans(  # type: ignore[assignment]
            t, lenient=True
        )
        broken = _comparable(repair_tool_calls_in_text(prose, ["read_file"]))
    finally:
        tool_repair._find_balanced_json_objects = original_strict  # type: ignore[assignment]
        tool_repair._find_candidate_object_spans = original_lenient  # type: ignore[assignment]

    assert broken != good
    assert broken[1] == []
