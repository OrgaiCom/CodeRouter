#!/usr/bin/env python3
"""L1 offline tool-call repair benchmark.

Applies :func:`repair_tool_calls_in_text` (and, for the ``multiple_calls``
category, :func:`deduplicate_tool_calls`) from
``coderouter.translation.tool_repair`` to every case in ``corpus.jsonl`` and
scores each outcome against the case's declared expectation.

Deterministic: no randomness, no network, stdlib only. The module under test
is loaded directly from its source file so this runs even when the full
``coderouter`` package has import-time side effects.

Outcome classes (per case)
--------------------------
* ``recovered``      expect.repaired=True  and repair produced tool calls
                     that satisfy tool_names / min_calls.
* ``correct_pass``   expect.repaired=False and repair produced NO tool call.
* ``missed``         expect.repaired=True  but repair produced nothing (or too
                     few / wrong names). The visible weakness of the repairer.
* ``false_positive`` expect.repaired=False but repair produced a tool call.
                     The dangerous class — a plain code example turned into an
                     executable tool call.

Usage
-----
    python run_offline.py [--corpus corpus.jsonl] [--out-dir .]
                          [--tool-repair /path/to/tool_repair.py]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import OrderedDict
from datetime import UTC, datetime
from types import ModuleType
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))

# Default location of the module under test inside the shim-b1 worktree.
_HERE_DIR = os.path.dirname(os.path.abspath(__file__))
# The repairer under test. Loaded by file path (not package import) so you
# can point --tool-repair at any branch's tool_repair.py and diff before/
# after on the same corpus. Default: this repo checkout
# (benchmarks/tool-repair/ -> repo root -> coderouter/translation/).
_DEFAULT_TOOL_REPAIR = os.path.join(
    _HERE_DIR, "..", "..", "coderouter", "translation", "tool_repair.py"
)


def load_tool_repair(path: str) -> ModuleType:
    """Load tool_repair.py directly, sidestepping package import cycles."""
    if not os.path.exists(path):
        raise SystemExit(
            f"tool_repair module not found at {path!r}. Pass --tool-repair "
            "with the correct path."
        )
    spec = importlib.util.spec_from_file_location("cr_tool_repair", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"could not build import spec for {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_corpus(path: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                cases.append(json.loads(raw))
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return cases


def tool_names_of(tool_calls: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tc in tool_calls:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.append(fn["name"])
    return names


def classify(
    case: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> tuple[str, str]:
    """Return (outcome, reason)."""
    expect = case.get("expect", {})
    want_repair = bool(expect.get("repaired"))
    got_names = tool_names_of(tool_calls)
    n = len(tool_calls)

    if not want_repair:
        if n == 0:
            return "correct_pass", "no tool call produced, as expected"
        return "false_positive", f"unexpected {n} call(s): {got_names}"

    # want_repair is True
    if n == 0:
        return "missed", "no tool call produced but one was expected"

    min_calls = int(expect.get("min_calls", 1))
    exp_names = list(expect.get("tool_names", []))

    if n < min_calls:
        return "missed", f"produced {n} call(s), expected >= {min_calls}"

    if exp_names:
        # Order-insensitive multiset check of expected names against produced.
        want_sorted = sorted(exp_names)
        got_sorted = sorted(got_names)
        if want_sorted != got_sorted:
            return (
                "missed",
                f"names {got_names} != expected {exp_names}",
            )

    return "recovered", f"produced {n} call(s): {got_names}"


def run(cases: list[dict[str, Any]], repair: Any, dedup: Any) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    for case in cases:
        text = case.get("input_text", "")
        allowed = case.get("allowed_tools")
        cleaned, tool_calls = repair(text, allowed)

        # Re-exercise deduplicate_tool_calls explicitly for multiple_calls so
        # the report can attest the dedup path was verified end-to-end. The
        # repair function already dedups internally; calling it again is
        # idempotent and confirms stability.
        dedup_verified = None
        if case.get("category") == "multiple_calls":
            rededuped = dedup(list(tool_calls))
            dedup_verified = len(rededuped) == len(tool_calls)

        outcome, reason = classify(case, tool_calls)
        per_case.append(
            {
                "id": case.get("id"),
                "category": case.get("category"),
                "outcome": outcome,
                "reason": reason,
                "expect_repaired": bool(case.get("expect", {}).get("repaired")),
                "produced_calls": len(tool_calls),
                "produced_names": tool_names_of(tool_calls),
                "cleaned_text": cleaned,
                "dedup_verified": dedup_verified,
                "note": case.get("note", ""),
            }
        )
    return summarise(per_case)


def summarise(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    cats: OrderedDict[str, dict[str, int]] = OrderedDict()
    totals = {
        "recovered": 0,
        "correct_pass": 0,
        "missed": 0,
        "false_positive": 0,
    }
    for row in per_case:
        cat = row["category"]
        bucket = cats.setdefault(
            cat,
            {"recovered": 0, "correct_pass": 0, "missed": 0, "false_positive": 0},
        )
        bucket[row["outcome"]] += 1
        totals[row["outcome"]] += 1

    def rates(b: dict[str, int]) -> dict[str, Any]:
        positives = b["recovered"] + b["missed"]  # cases expecting repair
        negatives = b["correct_pass"] + b["false_positive"]  # cases expecting pass
        recall = (b["recovered"] / positives) if positives else None
        fp_rate = (b["false_positive"] / negatives) if negatives else None
        return {
            "positives": positives,
            "negatives": negatives,
            "recall": recall,
            "false_positive_rate": fp_rate,
        }

    per_cat = {cat: {**b, **rates(b)} for cat, b in cats.items()}
    overall = {**totals, **rates(totals)}

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_cases": len(per_case),
        "overall": overall,
        "per_category": per_cat,
        "cases": per_case,
    }


def fmt_pct(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def render_md(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# L1 Offline Tool-Repair Benchmark")
    lines.append("")
    lines.append(f"- Generated: {result['generated_at']}")
    lines.append(f"- Total cases: {result['total_cases']}")
    ov = result["overall"]
    lines.append(
        f"- Overall recall (recovered / expect-repair): "
        f"{ov['recovered']}/{ov['positives']} = {fmt_pct(ov['recall'])}"
    )
    lines.append(
        f"- Overall false-positive rate (fp / expect-pass): "
        f"{ov['false_positive']}/{ov['negatives']} = {fmt_pct(ov['false_positive_rate'])}"
    )
    lines.append(
        f"- Missed: {ov['missed']}  |  Correct-pass: {ov['correct_pass']}"
    )
    lines.append("")

    lines.append("## Per-category")
    lines.append("")
    lines.append(
        "| category | cases | recovered | missed | recall | correct_pass | "
        "false_pos | fp_rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for cat, b in result["per_category"].items():
        total = (
            b["recovered"] + b["missed"] + b["correct_pass"] + b["false_positive"]
        )
        lines.append(
            f"| {cat} | {total} | {b['recovered']} | {b['missed']} | "
            f"{fmt_pct(b['recall'])} | {b['correct_pass']} | "
            f"{b['false_positive']} | {fmt_pct(b['false_positive_rate'])} |"
        )
    lines.append("")

    missed = [c for c in result["cases"] if c["outcome"] == "missed"]
    fps = [c for c in result["cases"] if c["outcome"] == "false_positive"]

    lines.append(f"## Missed cases ({len(missed)})")
    lines.append("")
    if missed:
        lines.append("| id | category | reason | note |")
        lines.append("|---|---|---|---|")
        for c in missed:
            note = c["note"].replace("|", "\\|")
            reason = c["reason"].replace("|", "\\|")
            lines.append(f"| {c['id']} | {c['category']} | {reason} | {note} |")
    else:
        lines.append("_none_")
    lines.append("")

    lines.append(f"## False positives ({len(fps)})")
    lines.append("")
    if fps:
        lines.append("| id | category | reason | note |")
        lines.append("|---|---|---|---|")
        for c in fps:
            note = c["note"].replace("|", "\\|")
            reason = c["reason"].replace("|", "\\|")
            lines.append(f"| {c['id']} | {c['category']} | {reason} | {note} |")
    else:
        lines.append("_none — repairer produced no over-eager extractions._")
    lines.append("")

    dd = [
        c
        for c in result["cases"]
        if c["category"] == "multiple_calls" and c["dedup_verified"] is False
    ]
    lines.append("## deduplicate_tool_calls stability")
    lines.append("")
    if dd:
        lines.append(
            "WARNING: re-running dedup changed call count for: "
            + ", ".join(c["id"] for c in dd)
        )
    else:
        lines.append(
            "All multiple_calls cases: re-applying deduplicate_tool_calls is "
            "idempotent (stable)."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", default=os.path.join(HERE, "corpus.jsonl")
    )
    parser.add_argument("--out-dir", default=HERE)
    parser.add_argument("--tool-repair", default=_DEFAULT_TOOL_REPAIR)
    args = parser.parse_args(argv)

    module = load_tool_repair(args.tool_repair)
    repair = module.repair_tool_calls_in_text
    dedup = module.deduplicate_tool_calls

    cases = load_corpus(args.corpus)
    result = run(cases, repair, dedup)

    json_path = os.path.join(args.out_dir, "results_offline.json")
    md_path = os.path.join(args.out_dir, "results_offline.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_md(result))

    ov = result["overall"]
    print(f"cases={result['total_cases']}")
    print(
        f"recovered={ov['recovered']} missed={ov['missed']} "
        f"correct_pass={ov['correct_pass']} false_positive={ov['false_positive']}"
    )
    print(
        f"recall={fmt_pct(ov['recall'])} "
        f"fp_rate={fmt_pct(ov['false_positive_rate'])}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
