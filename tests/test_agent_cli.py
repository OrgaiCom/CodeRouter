"""Tests for the agent_cli adapter (Phase 1a claude + Phase 1d grok).

Implements the design's §8 test plan for the Phase 1a/1d scope:

* T1  argv construction (model / max-turns / permission-mode mapping,
      allow_file_writes clamp) — for both claude and grok (grok adds the
      --prompt-file / --no-memory / --sandbox shape).
* T2  claude / grok JSON parsing (success / is_error / malformed / missing
      fields; grok's zero-usage + sessionId meta).
* T3  stubbed subprocess via ``asyncio.create_subprocess_exec`` monkeypatch
      (grok additionally checks the prompt-file lifecycle: 0600 file exists
      and contains the prompt DURING the exec, deleted afterwards, also on
      failure / timeout paths).
* T4  TestClient E2E through ``POST /v1/chat/completions``.
* T6  timeout → process-group kill + retryable AdapterError.
* T7  child-env isolation (no ANTHROPIC_API_KEY leak, NO_COLOR present,
      passthrough allowlist incl. GROK_CODE_XAI_API_KEY, depth +1).
* T8  recursion depth limit → non-retryable AdapterError.

Plus config-schema validation (agent_cli required, base_url optionality,
sandbox / allow_file_writes conflict) and healthcheck / retryable semantics.

Subprocess stubbing follows the ``_FakeProc`` shape used in
``tests/test_launcher_mtp.py``; the TestClient scaffolding mirrors
``tests/test_fix_h8_launcher_auth.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from coderouter.adapters.agent_cli import AgentCliAdapter
from coderouter.adapters.base import AdapterError, ChatRequest, Message
from coderouter.config.schemas import (
    AgentCliConfig,
    Capabilities,
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.ingress.app import create_app
from coderouter.metrics import uninstall_collector

# ---------------------------------------------------------------------------
# Canned claude `--output-format json` result (design §5.1.6 shape)
# ---------------------------------------------------------------------------

CLAUDE_RESULT = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "duration_ms": 1234,
    "num_turns": 3,
    "result": "Hello from claude",
    "session_id": "sess-abc123",
    "total_cost_usd": 0.0123,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 5,
    },
}
CLAUDE_JSON = json.dumps(CLAUDE_RESULT).encode("utf-8")

# ---------------------------------------------------------------------------
# Canned grok `--output-format json` result (verified on grok CLI v0.2.93 —
# NO token usage / cost fields exist; "thought" may or may not be present)
# ---------------------------------------------------------------------------

GROK_RESULT = {
    "text": "Hello from grok",
    "stopReason": "EndTurn",
    "sessionId": "0f9b6a3c-1d2e-4f56-8a7b-9c0d1e2f3a4b",
    "requestId": "7a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d",
    "thought": "The user greeted me; respond briefly.",
}
GROK_JSON = json.dumps(GROK_RESULT).encode("utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(**agent_cli_overrides: object) -> AgentCliAdapter:
    """Build an AgentCliAdapter with a claude sub-config + overrides."""
    acfg: dict[str, object] = {"agent": "claude"}
    acfg.update(agent_cli_overrides)
    provider = ProviderConfig(
        name="agent-claude",
        kind="agent_cli",
        model="opus",
        paid=True,
        capabilities=Capabilities(streaming=False),
        agent_cli=AgentCliConfig(**acfg),  # type: ignore[arg-type]
    )
    return AgentCliAdapter(provider)


def _make_grok_adapter(**agent_cli_overrides: object) -> AgentCliAdapter:
    """Build an AgentCliAdapter with a grok sub-config + overrides."""
    acfg: dict[str, object] = {"agent": "grok"}
    acfg.update(agent_cli_overrides)
    provider = ProviderConfig(
        name="agent-grok",
        kind="agent_cli",
        model="grok-4.5",
        paid=True,
        capabilities=Capabilities(streaming=False),
        agent_cli=AgentCliConfig(**acfg),  # type: ignore[arg-type]
    )
    return AgentCliAdapter(provider)


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="hi there")])


class _FakeProc:
    """Minimal stand-in for an asyncio subprocess (mirrors test_launcher_mtp)."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        hang: bool = False,
    ) -> None:
        # A pid that cannot exist, so the timeout kill path's os.getpgid()
        # raises ProcessLookupError instead of signalling a real process.
        self.pid = 2_147_483_647
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False
        self.stdin_received: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.stdin_received = input
        if self._hang:
            await asyncio.sleep(3600)
        return self._stdout, self._stderr

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class _GrokFakeProc(_FakeProc):
    """_FakeProc that snapshots the ``--prompt-file`` contents at
    ``communicate()`` time — i.e. while the real CLI would be running — so
    tests can prove the file existed, held the prompt, and was mode 0600
    DURING the exec (it is deleted again before generate() returns)."""

    def __init__(self, captured: list[list[str]], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._captured = captured
        self.prompt_file_path: str | None = None
        self.prompt_file_content: str | None = None
        self.prompt_file_mode: int | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        argv = self._captured[0]
        path = argv[argv.index("--prompt-file") + 1]
        self.prompt_file_path = path
        self.prompt_file_mode = stat.S_IMODE(os.stat(path).st_mode)
        self.prompt_file_content = Path(path).read_text(encoding="utf-8")
        return await super().communicate(input)


def _patch_exec(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> list[list[str]]:
    """Patch asyncio.create_subprocess_exec to return ``proc``; capture argv."""
    captured: list[list[str]] = []

    async def fake_exec(*argv: str, **kwargs: object) -> _FakeProc:
        captured.append(list(argv))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return captured


# ---------------------------------------------------------------------------
# T1 — argv construction
# ---------------------------------------------------------------------------


def test_argv_read_only_default() -> None:
    adapter = _make_adapter(workdir="/tmp/wd")
    argv = adapter._build_claude_argv("/tmp/wd")
    assert argv == [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--model",
        "opus",
        "--max-turns",
        "8",
        "--permission-mode",
        "plan",
        "--add-dir",
        "/tmp/wd",
    ]


def test_argv_model_override_and_max_turns() -> None:
    adapter = _make_adapter(model="sonnet", max_turns=3)
    argv = adapter._build_claude_argv("/w")
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--max-turns") + 1] == "3"


def test_argv_omits_max_turns_when_none() -> None:
    adapter = _make_adapter(max_turns=None)
    argv = adapter._build_claude_argv("/w")
    assert "--max-turns" not in argv


def test_argv_permission_mode_edit_maps_accept_edits() -> None:
    adapter = _make_adapter(sandbox_mode="edit", allow_file_writes=True)
    argv = adapter._build_claude_argv("/w")
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"


def test_argv_permission_mode_full_auto_maps_accept_edits() -> None:
    adapter = _make_adapter(sandbox_mode="full_auto", allow_file_writes=True)
    argv = adapter._build_claude_argv("/w")
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"


def test_argv_read_only_clamp_when_writes_disabled() -> None:
    # sandbox_mode=edit but allow_file_writes=False → clamp to read-only (plan).
    adapter = _make_adapter(sandbox_mode="edit", allow_file_writes=False)
    argv = adapter._build_claude_argv("/w")
    assert argv[argv.index("--permission-mode") + 1] == "plan"


def test_argv_never_includes_bare_flag() -> None:
    adapter = _make_adapter()
    assert "--bare" not in adapter._build_claude_argv("/w")


# ---------------------------------------------------------------------------
# T1 (grok) — argv construction
# ---------------------------------------------------------------------------


def test_grok_argv_read_only_default_full_snapshot() -> None:
    adapter = _make_grok_adapter(workdir="/tmp/wd")
    argv = adapter._build_grok_argv("/tmp/wd", "/tmp/wd/.coderouter-prompt-x.txt")
    assert argv == [
        "grok",
        "--prompt-file",
        "/tmp/wd/.coderouter-prompt-x.txt",
        "--output-format",
        "json",
        "-m",
        "grok-4.5",
        "--cwd",
        "/tmp/wd",
        "--max-turns",
        "8",
        "--no-memory",
        "--sandbox",
        "read-only",
        "--permission-mode",
        "plan",
    ]


def test_grok_argv_edit_maps_workspace_accept_edits() -> None:
    adapter = _make_grok_adapter(sandbox_mode="edit", allow_file_writes=True)
    argv = adapter._build_grok_argv("/w", "/w/p.txt")
    assert argv[argv.index("--sandbox") + 1] == "workspace"
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"


def test_grok_argv_full_auto_maps_always_approve() -> None:
    adapter = _make_grok_adapter(sandbox_mode="full_auto", allow_file_writes=True)
    argv = adapter._build_grok_argv("/w", "/w/p.txt")
    assert argv[argv.index("--sandbox") + 1] == "workspace"
    assert "--always-approve" in argv
    # full_auto uses auto-approval INSTEAD of a permission mode.
    assert "--permission-mode" not in argv


def test_grok_argv_read_only_clamp_when_writes_disabled() -> None:
    # sandbox_mode=edit but allow_file_writes=False → clamp to read-only/plan.
    adapter = _make_grok_adapter(sandbox_mode="edit", allow_file_writes=False)
    argv = adapter._build_grok_argv("/w", "/w/p.txt")
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("--permission-mode") + 1] == "plan"


def test_grok_argv_omits_max_turns_when_none() -> None:
    adapter = _make_grok_adapter(max_turns=None)
    argv = adapter._build_grok_argv("/w", "/w/p.txt")
    assert "--max-turns" not in argv


def test_grok_argv_model_override() -> None:
    adapter = _make_grok_adapter(model="grok-composer-2.5-fast")
    argv = adapter._build_grok_argv("/w", "/w/p.txt")
    assert argv[argv.index("-m") + 1] == "grok-composer-2.5-fast"


def test_grok_argv_no_memory_and_no_prompt_in_argv() -> None:
    adapter = _make_grok_adapter()
    argv = adapter._build_grok_argv("/w", "/w/p.txt")
    # --no-memory enforces one-request-one-transformation statelessness.
    assert "--no-memory" in argv
    # The builder only ever receives the prompt-FILE path — grok's -p/--single
    # (prompt as argv value) is never used, so prompt text can't reach argv.
    assert "-p" not in argv
    assert "--single" not in argv


# ---------------------------------------------------------------------------
# T7 — child-env isolation
# ---------------------------------------------------------------------------


def test_env_excludes_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    adapter = _make_adapter()
    env = adapter._build_child_env()
    assert "ANTHROPIC_API_KEY" not in env


def test_env_has_no_color_and_dumb_term() -> None:
    adapter = _make_adapter()
    env = adapter._build_child_env()
    assert env["NO_COLOR"] == "1"
    assert env["TERM"] == "dumb"


def test_env_inherits_user_and_logname(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS Keychain lookup needs USER — without it claude -p reports
    'Not logged in' (field-verified on Claude Code 2.1.x)."""
    monkeypatch.setenv("USER", "hyamamoto")
    monkeypatch.setenv("LOGNAME", "hyamamoto")
    adapter = _make_adapter()
    env = adapter._build_child_env()
    assert env["USER"] == "hyamamoto"
    assert env["LOGNAME"] == "hyamamoto"


def test_error_detail_prefers_is_error_stdout_json() -> None:
    from coderouter.adapters.agent_cli import AgentCliAdapter

    stdout = (
        b'{"type":"result","is_error":true,"result":"Not logged in \xc2\xb7 Please run /login"}'
    )
    detail = AgentCliAdapter._error_detail(stdout, b"")
    assert "Not logged in" in detail

    # Non-JSON stdout with empty stderr falls back to the stdout tail.
    detail = AgentCliAdapter._error_detail(b"boom", b"")
    assert "boom" in detail

    # stderr wins over non-error stdout.
    detail = AgentCliAdapter._error_detail(b'{"is_error":false}', b"trace")
    assert "trace" in detail


def test_env_passthrough_allowlist_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-nope")
    adapter = _make_adapter(passthrough_env=["CLAUDE_CODE_OAUTH_TOKEN"])
    env = adapter._build_child_env()
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok-123"
    assert "ANTHROPIC_API_KEY" not in env


def test_env_grok_api_key_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """GROK_CODE_XAI_API_KEY (grok's CI key env — NOT XAI_API_KEY) is
    forwarded only when allowlisted; OAuth under ~/.grok rides on HOME."""
    monkeypatch.setenv("GROK_CODE_XAI_API_KEY", "xai-tok-123")
    monkeypatch.setenv("XAI_API_KEY", "wrong-var-should-not-leak")
    adapter = _make_grok_adapter(passthrough_env=["GROK_CODE_XAI_API_KEY"])
    env = adapter._build_child_env()
    assert env["GROK_CODE_XAI_API_KEY"] == "xai-tok-123"
    assert "XAI_API_KEY" not in env


def test_env_depth_incremented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEROUTER_AGENT_DEPTH", "1")
    adapter = _make_adapter()
    env = adapter._build_child_env()
    assert env["CODEROUTER_AGENT_DEPTH"] == "2"


def test_env_depth_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEROUTER_AGENT_DEPTH", raising=False)
    adapter = _make_adapter()
    env = adapter._build_child_env()
    assert env["CODEROUTER_AGENT_DEPTH"] == "1"


# ---------------------------------------------------------------------------
# T2 — claude JSON parsing
# ---------------------------------------------------------------------------


def test_parse_success() -> None:
    adapter = _make_adapter()
    text, usage, meta = adapter._parse_claude(CLAUDE_JSON, b"")
    assert text == "Hello from claude"
    # 100 input + 10 cache_read + 5 cache_creation = 115 prompt tokens.
    assert usage["prompt_tokens"] == 115
    assert usage["completion_tokens"] == 50
    assert usage["total_tokens"] == 165
    assert usage["prompt_tokens_details"]["cached_tokens"] == 10
    assert usage["num_turns"] == 3
    assert meta["coderouter_cost_usd"] == pytest.approx(0.0123)
    assert meta["coderouter_session_id"] == "sess-abc123"


def test_parse_is_error_raises_retryable() -> None:
    adapter = _make_adapter()
    payload = json.dumps({"is_error": True, "result": "boom"}).encode()
    with pytest.raises(AdapterError) as info:
        adapter._parse_claude(payload, b"")
    assert info.value.retryable is True


def test_parse_malformed_json_raises_retryable() -> None:
    adapter = _make_adapter()
    with pytest.raises(AdapterError) as info:
        adapter._parse_claude(b"not json at all {", b"")
    assert info.value.retryable is True


def test_parse_missing_result_raises_retryable() -> None:
    adapter = _make_adapter()
    payload = json.dumps({"is_error": False, "usage": {}}).encode()
    with pytest.raises(AdapterError) as info:
        adapter._parse_claude(payload, b"")
    assert info.value.retryable is True


def test_parse_empty_stdout_raises_retryable() -> None:
    adapter = _make_adapter()
    with pytest.raises(AdapterError) as info:
        adapter._parse_claude(b"   ", b"")
    assert info.value.retryable is True


# ---------------------------------------------------------------------------
# T2 (grok) — grok JSON parsing
# ---------------------------------------------------------------------------


def test_grok_parse_success() -> None:
    adapter = _make_grok_adapter()
    text, usage, meta = adapter._parse_grok(GROK_JSON, b"")
    assert text == "Hello from grok"
    # grok emits NO token counts — usage is all zeros (cost stays 0 unless
    # the operator sets ProviderConfig.cost, design §5.1.6).
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # sessionId is the only surfaced meta; thought / requestId / stopReason
    # are deliberately ignored.
    assert meta == {"coderouter_session_id": GROK_RESULT["sessionId"]}


def test_grok_parse_missing_text_raises_retryable() -> None:
    adapter = _make_grok_adapter()
    payload = json.dumps({"stopReason": "EndTurn", "sessionId": "s"}).encode()
    with pytest.raises(AdapterError) as info:
        adapter._parse_grok(payload, b"")
    assert info.value.retryable is True
    assert "missing string 'text'" in str(info.value)


def test_grok_parse_non_json_raises_retryable() -> None:
    adapter = _make_grok_adapter()
    with pytest.raises(AdapterError) as info:
        adapter._parse_grok(b"grok exploded {", b"")
    assert info.value.retryable is True


def test_grok_parse_non_object_raises_retryable() -> None:
    adapter = _make_grok_adapter()
    with pytest.raises(AdapterError) as info:
        adapter._parse_grok(b'["not", "an", "object"]', b"")
    assert info.value.retryable is True


def test_grok_parse_empty_stdout_raises_retryable() -> None:
    adapter = _make_grok_adapter()
    with pytest.raises(AdapterError) as info:
        adapter._parse_grok(b"   \n", b"")
    assert info.value.retryable is True


def test_grok_parse_thought_field_ignored() -> None:
    # "thought" may or may not be present (early beta) — absence is fine and
    # presence never leaks into text/usage/meta.
    adapter = _make_grok_adapter()
    payload = json.dumps({"text": "ok", "sessionId": "s-1"}).encode()
    text, _usage, meta = adapter._parse_grok(payload, b"")
    assert text == "ok"
    assert meta == {"coderouter_session_id": "s-1"}


# ---------------------------------------------------------------------------
# T3 — generate() with a stubbed subprocess
# ---------------------------------------------------------------------------


async def test_generate_success(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    proc = _FakeProc(stdout=CLAUDE_JSON, returncode=0)
    captured = _patch_exec(monkeypatch, proc)
    adapter = _make_adapter(workdir=str(tmp_path))

    resp = await adapter.generate(_request())

    assert resp.choices[0]["message"]["content"] == "Hello from claude"
    assert resp.coderouter_provider == "agent-claude"
    assert resp.model == "opus"
    # Cost propagated as response metadata.
    assert resp.coderouter_cost_usd == pytest.approx(0.0123)
    # Prompt was fed on stdin, not argv.
    assert proc.stdin_received is not None
    assert b"hi there" in proc.stdin_received
    assert all("hi there" not in arg for arg in captured[0])


async def test_generate_nonzero_exit_is_retryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    proc = _FakeProc(stdout=b"", stderr=b"auth failed", returncode=1)
    _patch_exec(monkeypatch, proc)
    adapter = _make_adapter(workdir=str(tmp_path))
    with pytest.raises(AdapterError) as info:
        await adapter.generate(_request())
    assert info.value.retryable is True


# ---------------------------------------------------------------------------
# T3 (grok) — generate() with a stubbed subprocess + prompt-file lifecycle
# ---------------------------------------------------------------------------


async def test_grok_generate_success_prompt_file_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[list[str]] = []
    proc = _GrokFakeProc(captured, stdout=GROK_JSON, returncode=0)

    async def fake_exec(*argv: str, **kwargs: object) -> _GrokFakeProc:
        captured.append(list(argv))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    adapter = _make_grok_adapter(workdir=str(tmp_path))

    resp = await adapter.generate(_request())

    assert resp.choices[0]["message"]["content"] == "Hello from grok"
    assert resp.coderouter_provider == "agent-grok"
    assert resp.model == "grok-4.5"
    assert resp.coderouter_session_id == GROK_RESULT["sessionId"]
    # grok takes the prompt via --prompt-file — never stdin, never argv.
    assert proc.stdin_received is None
    argv = captured[0]
    assert "--prompt-file" in argv
    assert all("hi there" not in arg for arg in argv)
    # DURING the exec the private prompt file existed in the workdir with
    # mode 0600 and held the rendered prompt (_GrokFakeProc snapshots it
    # inside communicate()).
    assert proc.prompt_file_path is not None
    assert proc.prompt_file_path.startswith(str(tmp_path))
    assert Path(proc.prompt_file_path).name.startswith(".coderouter-prompt-")
    assert proc.prompt_file_mode == 0o600
    assert proc.prompt_file_content is not None
    assert "hi there" in proc.prompt_file_content
    # ...and it is deleted once generate() returns.
    assert not os.path.exists(proc.prompt_file_path)
    assert list(tmp_path.glob(".coderouter-prompt-*")) == []


async def test_grok_generate_nonzero_exit_cleans_prompt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proc = _FakeProc(stdout=b"", stderr=b"auth failed", returncode=1)
    captured = _patch_exec(monkeypatch, proc)
    adapter = _make_grok_adapter(workdir=str(tmp_path))
    with pytest.raises(AdapterError) as info:
        await adapter.generate(_request())
    assert info.value.retryable is True
    # The prompt file is cleaned up on the failure path too.
    argv = captured[0]
    path = argv[argv.index("--prompt-file") + 1]
    assert not os.path.exists(path)
    assert list(tmp_path.glob(".coderouter-prompt-*")) == []


# ---------------------------------------------------------------------------
# T6 — timeout → process-group kill
# ---------------------------------------------------------------------------


async def test_generate_timeout_kills_and_raises_retryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    proc = _FakeProc(hang=True)
    _patch_exec(monkeypatch, proc)
    adapter = _make_adapter(workdir=str(tmp_path), exec_timeout_s=1.0)

    # Force an immediate timeout regardless of wall-clock so the test is fast.
    async def instant_timeout(coro: object, timeout: float) -> None:
        # Close the coroutine we were handed, then raise as if it timed out.
        if hasattr(coro, "close"):
            coro.close()  # type: ignore[attr-defined]
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", instant_timeout)

    with pytest.raises(AdapterError) as info:
        await adapter.generate(_request())
    assert info.value.retryable is True
    assert "timed out" in str(info.value)
    # The fake pid can't be group-killed, so the fallback direct kill fired.
    assert proc.killed is True


async def test_grok_generate_timeout_cleans_prompt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proc = _FakeProc(hang=True)
    captured = _patch_exec(monkeypatch, proc)
    adapter = _make_grok_adapter(workdir=str(tmp_path), exec_timeout_s=1.0)

    # Force an immediate timeout regardless of wall-clock so the test is fast.
    async def instant_timeout(coro: object, timeout: float) -> None:
        if hasattr(coro, "close"):
            coro.close()  # type: ignore[attr-defined]
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", instant_timeout)

    with pytest.raises(AdapterError) as info:
        await adapter.generate(_request())
    assert info.value.retryable is True
    assert "timed out" in str(info.value)
    # Even on the timeout path the finally block removed the prompt file.
    argv = captured[0]
    path = argv[argv.index("--prompt-file") + 1]
    assert not os.path.exists(path)
    assert list(tmp_path.glob(".coderouter-prompt-*")) == []


# ---------------------------------------------------------------------------
# T8 — recursion depth limit
# ---------------------------------------------------------------------------


async def test_generate_depth_limit_non_retryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    # Default agent_depth_limit is 2; set current depth at the limit.
    monkeypatch.setenv("CODEROUTER_AGENT_DEPTH", "2")
    called = _patch_exec(monkeypatch, _FakeProc(stdout=CLAUDE_JSON))
    adapter = _make_adapter(workdir=str(tmp_path))
    with pytest.raises(AdapterError) as info:
        await adapter.generate(_request())
    assert info.value.retryable is False
    # Guard fired before any subprocess was launched.
    assert called == []


# ---------------------------------------------------------------------------
# healthcheck + unsupported-agent + retryable semantics
# ---------------------------------------------------------------------------


async def test_healthcheck_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("coderouter.adapters.agent_cli.shutil.which", lambda _c: None)
    adapter = _make_adapter()
    assert await adapter.healthcheck() is False


async def test_healthcheck_present_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("coderouter.adapters.agent_cli.shutil.which", lambda _c: "/usr/bin/claude")
    adapter = _make_adapter()
    assert await adapter.healthcheck() is True


def test_unsupported_agent_raises_non_retryable() -> None:
    provider = ProviderConfig(
        name="agent-codex",
        kind="agent_cli",
        model="gpt-5.5",
        paid=True,
        agent_cli=AgentCliConfig(agent="codex"),
    )
    with pytest.raises(AdapterError) as info:
        AgentCliAdapter(provider)
    assert info.value.retryable is False
    assert "not implemented yet" in str(info.value)
    assert "implemented: claude, grok" in str(info.value)


# ---------------------------------------------------------------------------
# config schema validation
# ---------------------------------------------------------------------------


def test_schema_agent_cli_required_for_kind() -> None:
    with pytest.raises(ValidationError, match="agent_cli sub-config is required"):
        ProviderConfig(name="x", kind="agent_cli", model="opus")


def test_schema_base_url_optional_for_agent_cli() -> None:
    p = ProviderConfig(
        name="x",
        kind="agent_cli",
        model="opus",
        agent_cli=AgentCliConfig(agent="claude"),
    )
    assert p.base_url is None


def test_schema_base_url_required_for_openai_compat() -> None:
    with pytest.raises(ValidationError, match="base_url is required"):
        ProviderConfig(name="x", kind="openai_compat", model="m")


def test_schema_command_defaults_to_agent() -> None:
    cfg = AgentCliConfig(agent="claude")
    assert cfg.command == "claude"


def test_schema_write_conflict_rejected() -> None:
    with pytest.raises(ValidationError, match="conflicts with"):
        AgentCliConfig(agent="claude", allow_file_writes=True, sandbox_mode="read_only")


# ---------------------------------------------------------------------------
# T4 — TestClient E2E through /v1/chat/completions
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_config(tmp_path: object) -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=True,
        default_profile="claude-agent",
        providers=[
            ProviderConfig(
                name="agent-claude",
                kind="agent_cli",
                model="opus",
                paid=True,
                capabilities=Capabilities(streaming=False),
                agent_cli=AgentCliConfig(agent="claude", workdir=str(tmp_path)),
            ),
        ],
        profiles=[FallbackChain(name="claude-agent", providers=["agent-claude"])],
    )


@pytest.fixture
def e2e_client(
    e2e_config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    proc = _FakeProc(stdout=CLAUDE_JSON, returncode=0)

    async def fake_exec(*argv: str, **kwargs: object) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr("coderouter.ingress.app.load_config", lambda path=None: e2e_config)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


def test_e2e_chat_completions(e2e_client: TestClient) -> None:
    resp = e2e_client.post(
        "/v1/chat/completions",
        json={"model": "opus", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-CodeRouter-Profile": "claude-agent"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello from claude"
    assert body["coderouter_provider"] == "agent-claude"
    # Cost from claude's total_cost_usd propagated to the client-visible body.
    assert body["coderouter_cost_usd"] == pytest.approx(0.0123)


# ---------------------------------------------------------------------------
# T4 (grok) — TestClient E2E through /v1/chat/completions
# ---------------------------------------------------------------------------


@pytest.fixture
def grok_e2e_config(tmp_path: Path) -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=True,
        default_profile="grok-agent",
        providers=[
            ProviderConfig(
                name="agent-grok",
                kind="agent_cli",
                model="grok-4.5",
                paid=True,
                capabilities=Capabilities(streaming=False),
                agent_cli=AgentCliConfig(agent="grok", workdir=str(tmp_path)),
            ),
        ],
        profiles=[FallbackChain(name="grok-agent", providers=["agent-grok"])],
    )


@pytest.fixture
def grok_e2e_client(
    grok_e2e_config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    proc = _FakeProc(stdout=GROK_JSON, returncode=0)

    async def fake_exec(*argv: str, **kwargs: object) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr("coderouter.ingress.app.load_config", lambda path=None: grok_e2e_config)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


def test_e2e_grok_chat_completions(grok_e2e_client: TestClient) -> None:
    resp = grok_e2e_client.post(
        "/v1/chat/completions",
        json={"model": "grok-4.5", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-CodeRouter-Profile": "grok-agent"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello from grok"
    assert body["coderouter_provider"] == "agent-grok"
    # grok emits no token counts — usage is reported as zeros end-to-end.
    assert body["usage"]["total_tokens"] == 0
