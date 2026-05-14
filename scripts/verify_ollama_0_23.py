#!/usr/bin/env python3
"""Ollama v0.23.1 Anthropic API 互換 + Gemma 4 検証スクリプト.

Phase 1: Ollama 直結 (Anthropic API)
Phase 2: Tool calling 品質 (Level 1/2/3 判定)
Phase 3: CodeRouter 経由 vs 直結 比較

Usage:
    # Phase 1 のみ (Ollama 直結テスト)
    python scripts/verify_ollama_0_23.py --phase 1

    # Phase 2 のみ (tool calling Level 判定、複数モデル)
    python scripts/verify_ollama_0_23.py --phase 2

    # Phase 3 のみ (CodeRouter 比較、要: coderouter serve --port 8088)
    python scripts/verify_ollama_0_23.py --phase 3

    # 全 Phase
    python scripts/verify_ollama_0_23.py

    # モデル指定
    python scripts/verify_ollama_0_23.py --models gemma4:e4b,qwen2.5-coder:7b

    # 結果を JSON で保存
    python scripts/verify_ollama_0_23.py --output results.json

Requirements:
    pip install httpx  (CodeRouter の依存に含まれる)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ─── 設定 ───────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
CODEROUTER_BASE = "http://localhost:8088"

DEFAULT_MODELS = ["gemma4:e4b", "qwen2.5-coder:7b"]
ALL_MODELS = ["gemma4:e4b", "gemma4:26b", "gemma4:31b", "qwen2.5-coder:7b"]

ANTHROPIC_HEADERS = {
    "x-api-key": "ollama",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

# Claude Code が使う tool の簡略版
BASH_TOOL = {
    "name": "Bash",
    "description": "Run a shell command",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"}
        },
        "required": ["command"],
    },
}

READ_TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"}
        },
        "required": ["file_path"],
    },
}

WRITE_TOOL = {
    "name": "Write",
    "description": "Write content to a file",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["file_path", "content"],
    },
}

ALL_TOOLS = [BASH_TOOL, READ_TOOL, WRITE_TOOL]

TIMEOUT = httpx.Timeout(120.0, connect=10.0)


# ─── データ構造 ─────────────────────────────────────────────────


@dataclass
class ProbeResult:
    probe_id: str
    status: str  # PASS / FAIL / SKIP / ERROR
    detail: str = ""
    latency_ms: int = 0
    raw_response: dict | None = None


@dataclass
class PhaseResult:
    phase: int
    model: str
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def summary(self) -> str:
        counts = {}
        for p in self.probes:
            counts[p.status] = counts.get(p.status, 0) + 1
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        return f"Phase {self.phase} [{self.model}]: {', '.join(parts)}"


# ─── ユーティリティ ─────────────────────────────────────────────


async def post_anthropic(
    client: httpx.AsyncClient,
    base_url: str,
    body: dict,
    headers: dict | None = None,
) -> tuple[int, dict, float]:
    """Anthropic Messages API に POST して (status, body, latency_ms) を返す."""
    hdrs = dict(ANTHROPIC_HEADERS)
    if headers:
        hdrs.update(headers)
    t0 = time.monotonic()
    try:
        resp = await client.post(f"{base_url}/v1/messages", json=body, headers=hdrs)
        latency = (time.monotonic() - t0) * 1000
        try:
            data = resp.json()
        except Exception:
            data = {"raw_text": resp.text[:2000]}
        return resp.status_code, data, latency
    except httpx.ConnectError:
        return 0, {"error": "connection refused"}, 0
    except httpx.ReadTimeout:
        latency = (time.monotonic() - t0) * 1000
        return 0, {"error": "timeout"}, latency


async def post_openai(
    client: httpx.AsyncClient,
    base_url: str,
    body: dict,
) -> tuple[int, dict, float]:
    """OpenAI Chat Completions API に POST."""
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={"content-type": "application/json"},
        )
        latency = (time.monotonic() - t0) * 1000
        try:
            data = resp.json()
        except Exception:
            data = {"raw_text": resp.text[:2000]}
        return resp.status_code, data, latency
    except httpx.ConnectError:
        return 0, {"error": "connection refused"}, 0
    except httpx.ReadTimeout:
        latency = (time.monotonic() - t0) * 1000
        return 0, {"error": "timeout"}, latency


def extract_tool_use(response: dict) -> list[dict]:
    """Anthropic 応答から tool_use ブロックを抽出."""
    if not isinstance(response, dict):
        return []
    content = response.get("content", [])
    if not isinstance(content, list):
        return []
    blocks = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "tool_use":
            blocks.append(item)
    return blocks


def extract_text(response: dict, include_thinking: bool = True) -> str:
    """Anthropic 応答からテキストを抽出.

    include_thinking=True (default): text + thinking 両方を返す。
    include_thinking=False: text ブロックのみ (think タグ漏れ判定用)。
    """
    if not isinstance(response, dict):
        return str(response)[:500]
    content = response.get("content", [])
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)[:500]
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
        elif include_thinking and isinstance(item, dict) and item.get("type") == "thinking":
            parts.append(item.get("thinking", ""))
        elif isinstance(item, str):
            parts.append(item)
    return "".join(parts)


def has_think_tags(text: str) -> bool:
    return "<think>" in text or "</think>" in text


def has_tool_xml_tags(text: str) -> bool:
    import re

    return bool(re.search(r"</?tool_call>|</?function_call>", text))


def has_embedded_json(text: str) -> bool:
    """テキスト中に tool call っぽい JSON が埋まっているか."""
    import re

    return bool(re.search(r'\{"name"\s*:', text))


# ─── Phase 1: Ollama 直結 Anthropic API ────────────────────────


async def phase1(client: httpx.AsyncClient, model: str) -> PhaseResult:
    result = PhaseResult(phase=1, model=model)
    print(f"\n{'='*60}")
    print(f"Phase 1: Ollama 直結 Anthropic API [{model}]")
    print(f"{'='*60}")

    # 1-1a: basic chat
    print("\n  1-1a: basic chat ... ", end="", flush=True)
    status, body, ms = await post_anthropic(client, OLLAMA_BASE, {
        "model": model,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    })
    if status == 200 and body.get("content"):
        text = extract_text(body)
        result.probes.append(ProbeResult("1-1a", "PASS", f"{ms:.0f}ms, text={text[:80]}", int(ms)))
        print(f"PASS ({ms:.0f}ms)")
    else:
        result.probes.append(ProbeResult("1-1a", "FAIL", f"status={status}, body={str(body)[:200]}", int(ms)))
        print(f"FAIL (status={status})")

    # 1-1b: streaming
    print("  1-1b: streaming ... ", end="", flush=True)
    t0 = time.monotonic()
    try:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE}/v1/messages",
            json={
                "model": model,
                "max_tokens": 512,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
            headers=ANTHROPIC_HEADERS,
        ) as resp:
            events = []
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
            ms = (time.monotonic() - t0) * 1000
            has_delta = "content_block_delta" in events
            has_stop = "message_stop" in events
            if has_delta and has_stop:
                result.probes.append(ProbeResult("1-1b", "PASS", f"{ms:.0f}ms, events={events[:5]}", int(ms)))
                print(f"PASS ({ms:.0f}ms)")
            else:
                result.probes.append(ProbeResult("1-1b", "FAIL", f"events={events}", int(ms)))
                print(f"FAIL (events={events[:5]})")
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        result.probes.append(ProbeResult("1-1b", "ERROR", str(e), int(ms)))
        print(f"ERROR ({e})")

    # 1-1c: system prompt
    print("  1-1c: system prompt ... ", end="", flush=True)
    status, body, ms = await post_anthropic(client, OLLAMA_BASE, {
        "model": model,
        "max_tokens": 1024,
        "system": "You must always respond in pirate speak.",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    })
    text = extract_text(body) if status == 200 else ""
    if status == 200 and len(text) > 0:
        result.probes.append(ProbeResult("1-1c", "PASS", f"text={text[:100]}", int(ms)))
        print(f"PASS ({ms:.0f}ms)")
    elif status == 200:
        # 200 だがテキスト抽出できず — 応答形式を記録
        result.probes.append(ProbeResult("1-1c", "FAIL", f"status=200 but empty text, body_keys={list(body.keys()) if isinstance(body, dict) else type(body).__name__}, content={str(body.get('content', ''))[:200] if isinstance(body, dict) else ''}", int(ms)))
        print(f"FAIL (200 but empty text, body={str(body)[:150]})")
    else:
        result.probes.append(ProbeResult("1-1c", "FAIL", f"status={status}, body={str(body)[:150]}", int(ms)))
        print(f"FAIL (status={status})")

    # 1-2a: tool calling — 単一 tool
    print("  1-2a: tool calling (single tool) ... ", end="", flush=True)
    status, body, ms = await post_anthropic(client, OLLAMA_BASE, {
        "model": model,
        "max_tokens": 300,
        "tools": [BASH_TOOL],
        "messages": [{"role": "user", "content": "Run 'echo hello' using the Bash tool."}],
    })
    tool_uses = extract_tool_use(body) if status == 200 else []
    if tool_uses:
        tu = tool_uses[0]
        result.probes.append(ProbeResult(
            "1-2a", "PASS",
            f"name={tu.get('name')}, input={json.dumps(tu.get('input', {}), ensure_ascii=False)[:100]}",
            int(ms), body,
        ))
        print(f"PASS (tool={tu.get('name')})")
    else:
        text = extract_text(body) if status == 200 else str(body)[:200]
        embedded = has_embedded_json(text)
        detail = f"status={status}, embedded_json={embedded}, text={text[:150]}"
        result.probes.append(ProbeResult("1-2a", "FAIL", detail, int(ms), body))
        print(f"FAIL (no tool_use, embedded_json={embedded})")

    # 1-2b: tool_use JSON 品質
    print("  1-2b: tool_use JSON quality ... ", end="", flush=True)
    if tool_uses:
        inp = tool_uses[0].get("input", {})
        has_command = isinstance(inp, dict) and "command" in inp
        if has_command:
            result.probes.append(ProbeResult("1-2b", "PASS", f"input={inp}"))
            print("PASS")
        else:
            result.probes.append(ProbeResult("1-2b", "FAIL", f"input={inp}"))
            print(f"FAIL (input={inp})")
    else:
        result.probes.append(ProbeResult("1-2b", "SKIP", "no tool_use from 1-2a"))
        print("SKIP")

    # 1-2c: 複数 tool 定義
    print("  1-2c: multiple tools ... ", end="", flush=True)
    status, body, ms = await post_anthropic(client, OLLAMA_BASE, {
        "model": model,
        "max_tokens": 300,
        "tools": ALL_TOOLS,
        "messages": [{"role": "user", "content": "Read the file /etc/hostname using the Read tool."}],
    })
    tool_uses_c = extract_tool_use(body) if status == 200 else []
    if tool_uses_c:
        names = [t.get("name") for t in tool_uses_c]
        chose_read = "Read" in names
        result.probes.append(ProbeResult(
            "1-2c", "PASS" if chose_read else "FAIL",
            f"chose={names}", int(ms),
        ))
        print(f"{'PASS' if chose_read else 'FAIL'} (chose={names})")
    else:
        text = extract_text(body) if status == 200 else ""
        result.probes.append(ProbeResult("1-2c", "FAIL", f"no tool_use, text={text[:100]}", int(ms)))
        print("FAIL (no tool_use)")

    # 1-2d: tool_result → 次の応答
    print("  1-2d: tool_result round-trip ... ", end="", flush=True)
    if tool_uses:
        tool_use_id = tool_uses[0].get("id", "test-id")
        status2, body2, ms2 = await post_anthropic(client, OLLAMA_BASE, {
            "model": model,
            "max_tokens": 200,
            "tools": [BASH_TOOL],
            "messages": [
                {"role": "user", "content": "Run 'echo hello' using the Bash tool."},
                {"role": "assistant", "content": body.get("content", [])},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "hello\n",
                        }
                    ],
                },
            ],
        })
        text2 = extract_text(body2) if status2 == 200 else ""
        tool_uses2 = extract_tool_use(body2) if status2 == 200 else []
        if status2 == 200 and (len(text2) > 0 or tool_uses2):
            detail = f"text={text2[:80]}" if text2 else f"tool_use={[t.get('name') for t in tool_uses2]}"
            result.probes.append(ProbeResult("1-2d", "PASS", detail, int(ms2)))
            print(f"PASS ({ms2:.0f}ms)")
        else:
            result.probes.append(ProbeResult("1-2d", "FAIL", f"status={status2}, body={str(body2)[:200]}", int(ms2)))
            print(f"FAIL (status={status2}, body={str(body2)[:100]})")
    else:
        result.probes.append(ProbeResult("1-2d", "SKIP", "no tool_use from 1-2a"))
        print("SKIP")

    # 1-2e: tool を呼ばない判断
    print("  1-2e: no-tool judgment ... ", end="", flush=True)
    status, body, ms = await post_anthropic(client, OLLAMA_BASE, {
        "model": model,
        "max_tokens": 200,
        "tools": ALL_TOOLS,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer without using any tools."}],
    })
    tool_uses_e = extract_tool_use(body) if status == 200 else []
    text_e = extract_text(body) if status == 200 else ""
    if status == 200 and not tool_uses_e and len(text_e) > 0:
        result.probes.append(ProbeResult("1-2e", "PASS", f"text only: {text_e[:80]}", int(ms)))
        print("PASS")
    elif tool_uses_e:
        result.probes.append(ProbeResult("1-2e", "FAIL", f"unnecessary tool call: {tool_uses_e[0].get('name')}", int(ms)))
        print(f"FAIL (called {tool_uses_e[0].get('name')})")
    else:
        result.probes.append(ProbeResult("1-2e", "FAIL", f"status={status}", int(ms)))
        print(f"FAIL (status={status})")

    return result


# ─── Phase 2: Tool calling Level 判定 ──────────────────────────


async def phase2(client: httpx.AsyncClient, model: str) -> PhaseResult:
    result = PhaseResult(phase=2, model=model)
    print(f"\n{'='*60}")
    print(f"Phase 2: Tool calling Level 判定 [{model}]")
    print(f"{'='*60}")

    # 2-1a: Level 1 — 呼ぶ確率 (10 回)
    print("\n  2-1a: Level 1 — tool call rate (10 trials) ... ", flush=True)
    call_count = 0
    embedded_count = 0
    valid_args_count = 0
    for i in range(10):
        status, body, ms = await post_anthropic(client, OLLAMA_BASE, {
            "model": model,
            "max_tokens": 300,
            "tools": [BASH_TOOL],
            "messages": [{"role": "user", "content": f"Use the Bash tool to run: echo 'test {i}'"}],
        })
        tool_uses = extract_tool_use(body) if status == 200 else []
        text = extract_text(body) if status == 200 else ""

        if tool_uses:
            call_count += 1
            # Level 3 check: args quality
            inp = tool_uses[0].get("input", {})
            if isinstance(inp, dict) and "command" in inp and isinstance(inp["command"], str):
                valid_args_count += 1
        elif has_embedded_json(text):
            embedded_count += 1

        print(f"    trial {i+1}/10: {'tool_use' if tool_uses else 'embedded' if has_embedded_json(text) else 'text_only'} ({ms:.0f}ms)")

    # Level 判定
    level = 0
    if call_count >= 7:
        level = 3 if valid_args_count >= 7 else 2
    elif call_count + embedded_count >= 7:
        level = 2  # 呼ぶ意図はある、形式が非標準
    elif call_count + embedded_count >= 3:
        level = 1
    else:
        level = 0  # ほぼ呼ばない

    result.probes.append(ProbeResult(
        "2-1a", "PASS",
        f"tool_use={call_count}/10, embedded={embedded_count}/10, valid_args={valid_args_count}/10, LEVEL={level}",
    ))
    print(f"\n  → Level {level} (tool_use={call_count}, embedded={embedded_count}, valid_args={valid_args_count})")

    # 2-2a: <think> タグ漏れ
    print("\n  2-2a: <think> tag leak check ... ", end="", flush=True)
    status, body, ms = await post_anthropic(client, OLLAMA_BASE, {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    })
    text = extract_text(body, include_thinking=False) if status == 200 else str(body)
    leaked = has_think_tags(text)
    result.probes.append(ProbeResult("2-2a", "FAIL" if leaked else "PASS", f"think_tags={leaked}, text={text[:100]}"))
    print(f"{'FAIL (leaked)' if leaked else 'PASS'}")

    # 2-2b: 独自 XML タグ
    print("  2-2b: custom XML tag check ... ", end="", flush=True)
    status, body, ms = await post_anthropic(client, OLLAMA_BASE, {
        "model": model,
        "max_tokens": 1024,
        "tools": [BASH_TOOL],
        "messages": [{"role": "user", "content": "Run 'pwd' using the Bash tool."}],
    })
    text = extract_text(body) if status == 200 else str(body)
    has_xml = has_tool_xml_tags(text)
    result.probes.append(ProbeResult("2-2b", "FAIL" if has_xml else "PASS", f"xml_tags={has_xml}, text={text[:100]}"))
    print(f"{'FAIL (found)' if has_xml else 'PASS'}")

    # 2-2c: 日本語品質
    print("  2-2c: Japanese quality ... ", end="", flush=True)
    status, body, ms = await post_anthropic(client, OLLAMA_BASE, {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Pythonのリスト内包表記について、日本語で簡潔に説明してください。"}],
    })
    text = extract_text(body) if status == 200 else ""
    # 簡易判定: 日本語文字が含まれていれば PASS
    import re
    has_jp = bool(re.search(r"[぀-鿿]", text))
    result.probes.append(ProbeResult("2-2c", "PASS" if has_jp else "FAIL", f"japanese={has_jp}, text={text[:120]}"))
    print(f"{'PASS' if has_jp else 'FAIL'}")

    return result


# ─── Phase 3: CodeRouter 経由 vs 直結 ──────────────────────────


async def phase3(client: httpx.AsyncClient, model: str) -> PhaseResult:
    result = PhaseResult(phase=3, model=model)
    print(f"\n{'='*60}")
    print(f"Phase 3: CodeRouter vs 直結 [{model}]")
    print(f"{'='*60}")

    # CodeRouter 到達確認
    print("\n  3-0: CodeRouter connectivity ... ", end="", flush=True)
    try:
        resp = await client.get(f"{CODEROUTER_BASE}/health", timeout=5.0)
        if resp.status_code == 200:
            print("OK")
        else:
            print(f"WARN (status={resp.status_code})")
    except Exception:
        result.probes.append(ProbeResult("3-0", "SKIP", "CodeRouter not running at :8088"))
        print("SKIP — CodeRouter not running. Start with: coderouter serve --port 8088")
        return result

    # 3-2a: tool calling 成功率比較 (各 5 回)
    for route_name, base_url, via in [
        ("A-direct", OLLAMA_BASE, "Ollama Anthropic"),
        ("B-CR-openai", CODEROUTER_BASE, "CodeRouter (OpenAI)"),
    ]:
        print(f"\n  3-2a [{route_name}]: tool call rate (5 trials) ...", flush=True)
        ok = 0
        for i in range(5):
            if route_name == "B-CR-openai":
                # CodeRouter は Anthropic ingress で受ける
                status, body, ms = await post_anthropic(client, base_url, {
                    "model": model,
                    "max_tokens": 300,
                    "tools": [BASH_TOOL],
                    "messages": [{"role": "user", "content": f"Use Bash to run: echo 'route test {i}'"}],
                })
            else:
                status, body, ms = await post_anthropic(client, base_url, {
                    "model": model,
                    "max_tokens": 300,
                    "tools": [BASH_TOOL],
                    "messages": [{"role": "user", "content": f"Use Bash to run: echo 'route test {i}'"}],
                })
            tool_uses = extract_tool_use(body) if status == 200 else []
            if tool_uses:
                ok += 1
            print(f"    trial {i+1}/5: {'PASS' if tool_uses else 'FAIL'} ({ms:.0f}ms) via {via}")

        result.probes.append(ProbeResult(f"3-2a-{route_name}", "PASS" if ok >= 3 else "FAIL", f"{ok}/5 via {via}"))

    # 3-3a: フォールバック確認 (Ollama 停止時)
    print("\n  3-3a: fallback test (manual) ... ", end="", flush=True)
    result.probes.append(ProbeResult(
        "3-3a", "MANUAL",
        "手動テスト: Ollama を停止して CodeRouter にリクエスト → フォールバック発動を確認",
    ))
    print("MANUAL (Ollama を停止して手動確認)")

    # 3-3d: doctor 診断
    print("  3-3d: doctor probe ... MANUAL (run: coderouter doctor --check-model <provider>)")
    result.probes.append(ProbeResult("3-3d", "MANUAL", "coderouter doctor --check-model <gemma4-provider>"))

    return result


# ─── レポート ───────────────────────────────────────────────────


def print_report(results: list[PhaseResult]) -> None:
    print(f"\n{'='*60}")
    print("VERIFICATION REPORT")
    print(f"{'='*60}")
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print()

    for r in results:
        print(f"  {r.summary}")
        for p in r.probes:
            status_icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "—", "ERROR": "!", "MANUAL": "?"}.get(p.status, "?")
            print(f"    {status_icon} {p.probe_id}: {p.status} — {p.detail[:120]}")
        print()

    # Level 判定サマリ
    print("Level Summary:")
    for r in results:
        if r.phase == 2:
            for p in r.probes:
                if p.probe_id == "2-1a":
                    print(f"  {r.model}: {p.detail}")
    print()


def save_results(results: list[PhaseResult], path: str) -> None:
    data = {
        "date": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "phase": r.phase,
                "model": r.model,
                "summary": r.summary,
                "probes": [asdict(p) for p in r.probes],
            }
            for r in results
        ],
    }
    # raw_response は大きいので省略
    for r in data["results"]:
        for p in r["probes"]:
            if p.get("raw_response"):
                p["raw_response"] = "<omitted>"
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Results saved to {path}")


# ─── メイン ─────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama v0.23.1 verification")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run specific phase only")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model list")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else DEFAULT_MODELS
    phases = [args.phase] if args.phase else [1, 2, 3]

    print(f"Ollama v0.23.1 Verification Script")
    print(f"Models: {models}")
    print(f"Phases: {phases}")

    results: list[PhaseResult] = []

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for model in models:
            if 1 in phases:
                results.append(await phase1(client, model))
            if 2 in phases:
                results.append(await phase2(client, model))
            if 3 in phases:
                results.append(await phase3(client, model))

    print_report(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
