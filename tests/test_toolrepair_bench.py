"""CI regression gate for the tool-call repair benchmark (L1 offline).

Motivation: the benchmark corpus (benchmarks/tool-repair/corpus.jsonl) is the
public, reproducible evidence for the repairer's coverage. This test wires it
into the normal pytest run so any regression in
``coderouter/translation/tool_repair.py`` — or any corpus edit that would
weaken the safety guarantee — fails CI.

Gates (aligned with the v2.7.1 acceptance criteria):

- **False positives must be zero.** The ``negative`` category exists so that
  "lenient" can never quietly become "eats your code blocks" again
  (the v2.7.0-era regression class).
- **Recall must stay >= 0.92.** Deliberately below 100% so honest, known-gap
  cases can be added to the corpus as ``missed`` without blocking CI; a real
  regression in the repairer drops recall far below this line (the
  pre-v2.7.1 baseline was 0.806).

Design: the benchmark scripts load ``tool_repair.py`` by file path and are
plain scripts, so this test imports ``run_offline.py`` the same way rather
than through the package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks" / "tool-repair"
TOOL_REPAIR = REPO_ROOT / "coderouter" / "translation" / "tool_repair.py"

MIN_RECALL = 0.92


def _load_bench_module():
    spec = importlib.util.spec_from_file_location(
        "toolrepair_bench_run_offline", BENCH_DIR / "run_offline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_corpus_gate() -> None:
    bench = _load_bench_module()
    repair_mod = bench.load_tool_repair(str(TOOL_REPAIR))
    cases = bench.load_corpus(str(BENCH_DIR / "corpus.jsonl"))
    result = bench.run(
        cases,
        repair_mod.repair_tool_calls_in_text,
        repair_mod.deduplicate_tool_calls,
    )

    overall = result["overall"]
    missed_ids = [
        row["id"] for row in result["cases"] if row["outcome"] == "missed"
    ]
    fp_ids = [
        row["id"] for row in result["cases"] if row["outcome"] == "false_positive"
    ]

    # Sanity: corpus keeps both sides populated.
    assert overall["negatives"] >= 6, "negative corpus shrank below the gate"
    assert overall["positives"] >= 36, "positive corpus shrank unexpectedly"

    # Gate 1: the dangerous direction stays at zero. No exceptions.
    assert overall["false_positive"] == 0, (
        f"repairer fabricated tool calls from negative cases: {fp_ids}"
    )

    # Gate 2: coverage floor.
    assert overall["recall"] is not None
    assert overall["recall"] >= MIN_RECALL, (
        f"recall {overall['recall']:.1%} fell below {MIN_RECALL:.0%} "
        f"(missed: {missed_ids})"
    )

    # Gate 3: dedup stability wherever it was exercised.
    dedup_flags = [
        row["dedup_verified"]
        for row in result["cases"]
        if row["dedup_verified"] is not None
    ]
    assert dedup_flags and all(dedup_flags), "deduplicate_tool_calls unstable"
