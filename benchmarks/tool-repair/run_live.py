#!/usr/bin/env python3
"""L2 live tool-call benchmark.

Sends tool-bearing prompts to a running endpoint (CodeRouter or a backend
directly), then classifies each response with the SAME three-value logic
``coderouter.doctor._probe_tool_calls`` uses:

    native   -> the response carried structured tool_calls / tool_use blocks.
    repair   -> no structured call, but assistant TEXT contained tool JSON that
                ``repair_tool_calls_in_text`` could extract.
    fail     -> nothing tool-shaped at all.

Per model it reports native-rate / repair-rate / fail-rate over ``--reps``
repetitions of each built-in prompt.

Wire formats
------------
* ``--wire openai``    POST {base}/chat/completions, tools=[function spec],
                       read choices[0].message.{tool_calls,content}.
* ``--wire anthropic`` POST {base}/v1/messages, tools=[input_schema spec],
                       read content blocks (tool_use / text).

Self-test
---------
``--dry-run`` skips all HTTP and feeds a fixed bank of canned responses
(covering native / repair / fail for both wires) through the exact same
classifier, then asserts the tallies. Use it to prove the pipeline in a
sandbox with no server.

Dependencies: stdlib + httpx (already in coderouter deps).
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

_HERE_DIR = os.path.dirname(os.path.abspath(__file__))
# The repairer under test. Loaded by file path (not package import) so you
# can point --tool-repair at any branch's tool_repair.py and diff before/
# after on the same corpus. Default: this repo checkout
# (benchmarks/tool-repair/ -> repo root -> coderouter/translation/).
_DEFAULT_TOOL_REPAIR = os.path.join(
    _HERE_DIR, "..", "..", "coderouter", "translation", "tool_repair.py"
)

# The single test tool every prompt is allowed to call. Mirrors doctor's echo
# probe but adds a couple of realistic tools so multi-tool selection is real.
_ECHO_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo back the provided message. Diagnostic only.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
}
_ADD_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
}
_WRITE_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "write_note",
        "description": "Write a note to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["path", "text"],
        },
    },
}

_TOOLS_OPENAI = [_ECHO_TOOL_OPENAI, _ADD_TOOL_OPENAI, _WRITE_TOOL_OPENAI]
_ALLOWED_NAMES = ["echo", "add", "write_note"]


def _to_anthropic_tool(spec: dict[str, Any]) -> dict[str, Any]:
    fn = spec["function"]
    return {
        "name": fn["name"],
        "description": fn["description"],
        "input_schema": fn["parameters"],
    }


_TOOLS_ANTHROPIC = [_to_anthropic_tool(t) for t in _TOOLS_OPENAI]


# Built-in prompt suite (5 kinds). Each stresses a different failure surface.
BUILTIN_PROMPTS: list[dict[str, str]] = [
    {
        "id": "simple_echo",
        "text": (
            "Call the `echo` tool with the message 'probe'. "
            "Do not reply with any text — only the tool call."
        ),
    },
    {
        "id": "complex_args",
        "text": (
            "Use the `write_note` tool to write the text "
            "'line1\\nline2 with \"quotes\" and a comma, plus {braces}' "
            "to the path notes/日本語.txt. Emit only the tool call."
        ),
    },
    {
        "id": "multi_tool_select",
        "text": (
            "You have echo, add and write_note. Add the integers 17 and 25 "
            "using the correct tool. Emit only the tool call."
        ),
    },
    {
        "id": "japanese_instruction",
        "text": (
            "echo ツールを使って、メッセージ『こんにちは世界』を送ってください。"
            "テキストでの返信はせず、ツール呼び出しだけを出力してください。"
        ),
    },
    {
        "id": "no_tool_temptation",
        "text": (
            "What does the echo tool do? Then, to demonstrate, actually call "
            "echo with the message 'demo'. Only the tool call, no prose."
        ),
    },
]


def load_tool_repair(path: str) -> ModuleType:
    if not os.path.exists(path):
        raise SystemExit(f"tool_repair not found at {path!r}; pass --tool-repair")
    spec = importlib.util.spec_from_file_location("cr_tool_repair_live", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise SystemExit(f"could not load spec for {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------
# Classification — identical 3-value logic to doctor._probe_tool_calls
# ------------------------------------------------------------------


def classify_response(
    wire: str, parsed: dict[str, Any], repair: Any
) -> tuple[str, str]:
    """Return (verdict, text_sample). verdict in {native, repair, fail}."""
    native = False
    text_sample = ""

    if wire == "anthropic":
        blocks = parsed.get("content")
        if isinstance(blocks, list):
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    native = True
                    break
            text_sample = " ".join(
                str(b.get("text", ""))
                for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
    else:  # openai
        choices = parsed.get("choices") or []
        msg = {}
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
        if msg.get("tool_calls"):
            native = True
        content = msg.get("content")
        if isinstance(content, str):
            text_sample = content

    if native:
        return "native", text_sample[:200]

    if text_sample:
        _, repaired = repair(text_sample, _ALLOWED_NAMES)
        if repaired:
            return "repair", text_sample[:200]

    return "fail", text_sample[:200]


# ------------------------------------------------------------------
# Live HTTP
# ------------------------------------------------------------------


def build_request(
    wire: str,
    base_url: str,
    model: str,
    prompt_text: str,
    temperature: float | None = 0.0,
) -> tuple[str, dict[str, Any]]:
    base = base_url.rstrip("/")
    if wire == "anthropic":
        url = f"{base}/v1/messages"
        body = {
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt_text}],
            "tools": _TOOLS_ANTHROPIC,
        }
        # Keep sampling identical across wires so a direct-vs-router
        # comparison measures the path, not the sampler.
        if temperature is not None:
            body["temperature"] = temperature
    else:
        url = f"{base}/chat/completions"
        body = {
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt_text}],
            "tools": _TOOLS_OPENAI,
        }
        if temperature is not None:
            body["temperature"] = temperature
    return url, body


def post(
    url: str,
    body: dict[str, Any],
    wire: str,
    api_key: str | None,
    profile: str | None = None,
) -> dict[str, Any]:
    import httpx

    headers = {"Content-Type": "application/json"}
    if profile:
        # CodeRouter extension header; plain backends (Ollama etc.) ignore it.
        headers["X-CodeRouter-Profile"] = profile
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if api_key:
            headers["x-api-key"] = api_key
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# Dry-run canned responses
# ------------------------------------------------------------------


def canned_bank(wire: str) -> list[tuple[str, dict[str, Any], str]]:
    """Return [(label, parsed_response, expected_verdict)]."""
    if wire == "anthropic":
        native = {
            "content": [
                {"type": "text", "text": "Calling echo."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "echo",
                    "input": {"message": "probe"},
                },
            ]
        }
        repair = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I'll call it.\n```json\n"
                        '{"name": "echo", "arguments": {"message": "probe"}}\n```'
                    ),
                }
            ]
        }
        fail = {
            "content": [
                {"type": "text", "text": "The echo tool repeats your message."}
            ]
        }
        malformed = {
            "content": [
                {
                    "type": "text",
                    "text": "{'name': 'echo', 'arguments': {'message': 'probe'}}",
                }
            ]
        }
    else:
        native = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"message": "probe"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        repair = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Sure.\n```json\n"
                            '{"name": "echo", "arguments": {"message": "probe"}}\n```'
                        ),
                    }
                }
            ]
        }
        fail = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "echo just repeats the message you give it.",
                    }
                }
            ]
        }
        malformed = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "{'name': 'echo', 'arguments': {'message': 'x'}}",
                    }
                }
            ]
        }
    return [
        ("native", native, "native"),
        ("repair_from_text", repair, "repair"),
        ("plain_text", fail, "fail"),
        # single-quoted dict: repair can't parse it -> classified as fail.
        # Single-quoted JSON: repairable since the lenient repairer (R1,
        # feature/tool-repair-lenient). With a pre-R1 tool_repair.py this
        # case reports "fail" and the dry-run will flag it — that is the
        # expected signal that you are pointing --tool-repair at an old
        # repairer, not a benchmark bug.
        ("malformed_singlequote", malformed, "repair"),
    ]


def run_dry(wire: str, repair: Any) -> int:
    print(f"[dry-run] wire={wire}")
    failures = 0
    for label, parsed, expected in canned_bank(wire):
        verdict, _sample = classify_response(wire, parsed, repair)
        status = "OK" if verdict == expected else "MISMATCH"
        if verdict != expected:
            failures += 1
        print(f"  {status:9} {label:24} verdict={verdict:7} expected={expected}")
    if failures:
        print(f"[dry-run] FAILED: {failures} mismatch(es)")
    else:
        print("[dry-run] PASS: all canned responses classified as expected")
    return failures


# ------------------------------------------------------------------
# Live run
# ------------------------------------------------------------------


def run_live(args: argparse.Namespace, repair: Any) -> dict[str, Any]:
    prompts = BUILTIN_PROMPTS
    if args.prompts:
        wanted = set(args.prompts.split(","))
        prompts = [p for p in BUILTIN_PROMPTS if p["id"] in wanted]
        if not prompts:
            raise SystemExit(f"no built-in prompts matched {args.prompts!r}")

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None

    per_prompt: OrderedDict[str, dict[str, int]] = OrderedDict()
    tally = {"native": 0, "repair": 0, "fail": 0, "error": 0}
    samples: list[dict[str, Any]] = []

    for prompt in prompts:
        bucket = {"native": 0, "repair": 0, "fail": 0, "error": 0}
        temp = None if str(args.temperature).lower() == "none" else float(args.temperature)
        url, body = build_request(
            args.wire, args.base_url, args.model, prompt["text"], temperature=temp
        )
        for rep in range(args.reps):
            try:
                parsed = post(url, body, args.wire, api_key, profile=args.profile)
            except Exception as exc:  # surface any transport error
                bucket["error"] += 1
                tally["error"] += 1
                if len(samples) < 20:
                    samples.append(
                        {
                            "prompt": prompt["id"],
                            "rep": rep,
                            "verdict": "error",
                            "detail": str(exc)[:200],
                        }
                    )
                continue
            verdict, sample = classify_response(args.wire, parsed, repair)
            bucket[verdict] += 1
            tally[verdict] += 1
            if verdict in ("repair", "fail") and len(samples) < 20:
                samples.append(
                    {
                        "prompt": prompt["id"],
                        "rep": rep,
                        "verdict": verdict,
                        "text_sample": sample,
                    }
                )
        per_prompt[prompt["id"]] = bucket

    total = sum(tally.values())

    def rate(n: int) -> float | None:
        return (n / total) if total else None

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "wire": args.wire,
        "model": args.model,
        "temperature": args.temperature,
        "profile": args.profile or None,
        "reps": args.reps,
        "prompts": [p["id"] for p in prompts],
        "totals": {
            **tally,
            "total": total,
            "native_rate": rate(tally["native"]),
            "repair_rate": rate(tally["repair"]),
            "fail_rate": rate(tally["fail"]),
        },
        "per_prompt": per_prompt,
        "samples": samples,
    }


def fmt_pct(v: Any) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def render_live_md(result: dict[str, Any]) -> str:
    t = result["totals"]
    lines = [
        f"# L2 Live Tool-Call Benchmark — {result['model']}",
        "",
        f"- Generated: {result['generated_at']}",
        f"- Endpoint: {result['base_url']}  (wire={result['wire']})",
        f"- Reps per prompt: {result['reps']}   Prompts: {', '.join(result['prompts'])}",
        f"- Total responses: {t['total']}",
        "",
        "## Totals",
        "",
        "| verdict | count | rate |",
        "|---|---|---|",
        f"| native | {t['native']} | {fmt_pct(t['native_rate'])} |",
        f"| repair | {t['repair']} | {fmt_pct(t['repair_rate'])} |",
        f"| fail | {t['fail']} | {fmt_pct(t['fail_rate'])} |",
        f"| error | {t['error']} | — |",
        "",
        "## Per-prompt",
        "",
        "| prompt | native | repair | fail | error |",
        "|---|---|---|---|---|",
    ]
    for pid, b in result["per_prompt"].items():
        lines.append(
            f"| {pid} | {b['native']} | {b['repair']} | {b['fail']} | {b['error']} |"
        )
    lines.append("")
    lines.append("## Sample non-native / failed responses")
    lines.append("")
    if result["samples"]:
        for s in result["samples"]:
            detail = s.get("text_sample") or s.get("detail") or ""
            detail = detail.replace("\n", " ")[:160]
            lines.append(f"- `{s['prompt']}` [{s['verdict']}]: {detail}")
    else:
        lines.append("_all responses were native tool_calls_")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8787")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--wire", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument(
        "--prompts",
        default="",
        help="comma-separated prompt ids; empty = all built-ins",
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help="env var holding the API key (optional; Ollama needs none)",
    )
    parser.add_argument("--out-dir", default=HERE)
    parser.add_argument("--tool-repair", default=_DEFAULT_TOOL_REPAIR)
    parser.add_argument(
        "--profile",
        default="",
        help="CodeRouter profile to select via X-CodeRouter-Profile header "
        "(ignored by plain backends). Use with providers.bench.yaml.",
    )
    parser.add_argument(
        "--temperature",
        default="0",
        help="sampling temperature: a float, or 'none' to omit the field "
        "(backend default). Default 0 for deterministic path comparison.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="label appended to result filenames (e.g. direct, coderouter)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip HTTP; self-test the classifier on canned responses",
    )
    args = parser.parse_args(argv)

    module = load_tool_repair(args.tool_repair)
    repair = module.repair_tool_calls_in_text

    if args.dry_run:
        failures = run_dry("openai", repair) + run_dry("anthropic", repair)
        return 1 if failures else 0

    result = run_live(args, repair)

    safe_model = result["model"].replace("/", "_").replace(":", "_")
    # Include wire (and optional --tag, e.g. "direct" / "coderouter") in the
    # filename so back-to-back runs against different paths never overwrite
    # each other.
    stem = f"results_live_{safe_model}_{args.wire}"
    if args.tag:
        safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.tag)
        stem += f"_{safe_tag}"
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_live_md(result))

    t = result["totals"]
    print(
        f"native={fmt_pct(t['native_rate'])} repair={fmt_pct(t['repair_rate'])} "
        f"fail={fmt_pct(t['fail_rate'])} (errors={t['error']})"
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
