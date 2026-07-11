"""External coding-agent CLI adapter (``kind="agent_cli"``).

This adapter invokes an external coding-agent CLI (Claude Code / Codex /
Antigravity / Grok) as a single one-shot ``exec`` and returns the agent's
final answer as one ``prompt in → text out`` transformation. It is the
in-core implementation of the external-agents-adapter design
(``docs/designs/external-agents-adapter.md``).

Design in one paragraph
=======================

A coding-agent CLI is normally a stateful, multi-turn, filesystem-editing
control loop — at odds with CodeRouter's "one request = one stateless
transformation" ethos. Restricting it to a single non-interactive
one-shot ``exec`` collapses it back into a single transformation, which
keeps the ethos intact: orchestration stays on the *client* side, and
CodeRouter merely performs the one conversion. Following the
``openai_compat`` precedent, a *single* adapter class fronts *multiple*
target agents, dispatched on the ``agent`` field.

Implemented agents (Phase 1 complete: 1a + 1b + 1c + 1d)
==========================================================

Four targets are implemented here:

* ``claude`` (Claude Code CLI, Phase 1a) — the most stable CLI, the most
  fine-grained safety controls, and the only one that emits
  ``total_cost_usd`` directly (making it the reference implementation for
  the parser / cost path).
* ``codex`` (codex CLI, Phase 1b) — headless one-shot via ``codex exec
  --json``, prompt on stdin (like claude). The CLI is pre-1.0 and emits
  defensively-parsed JSONL (one event per line); usage comes from
  ``turn.completed`` events, normalized per design §5.1.6 with
  ``cached_input_tokens`` kept as a *subset* of ``input_tokens`` (unlike
  claude, which folds its cache buckets into ``prompt_tokens``).
  ``--skip-git-repo-check`` and ``--ephemeral`` are always passed — the
  isolated workdir is not a git repo, and the adapter never wants
  session persistence.
* ``grok`` (grok CLI, Phase 1d) — headless one-shot via ``--prompt-file`` +
  ``--output-format json``. The CLI emits no token/cost figures, so usage
  is reported as zeros (cost stays 0 unless the operator sets
  ``ProviderConfig.cost``, design §5.1.6). ``--no-memory`` is always passed
  so a user-level cross-session memory setting cannot leak state between
  requests.
* ``antigravity`` (Antigravity CLI, command ``agy``, Phase 1c, in lieu of
  ``gemini``) — headless one-shot via ``agy -p <prompt> --mode ...``.
  Google discontinued the legacy Gemini CLI's OAuth for individual
  accounts in June 2026 (field-verified ``IneligibleTierError`` /
  ``UNSUPPORTED_CLIENT``); its successor, the Antigravity CLI, is a
  separate Go implementation (not a gemini-cli fork) fulfilling the design's
  "gemini" slot. It has no stdin or ``--prompt-file`` channel — piped stdin
  hangs the CLI (field-verified on agy 1.1.1) — so the prompt rides argv
  (see the security note below). Output is plain text with no
  ``--output-format`` flag, no token/cost figures, and no session id, so
  usage is reported as zeros and meta is empty, mirroring grok's rationale.
  It is also the only agent with a CLI-side self-termination flag
  (``--print-timeout``), layered *underneath* the adapter's own
  ``asyncio.wait_for`` + PGID SIGKILL rather than replacing it.

``gemini`` itself is declared in the config schema for backward-compatible
config parsing, but constructing an adapter for it raises a clear
``AdapterError`` with a migration pointer to ``agent="antigravity"``.

Security (design §6, non-negotiable)
====================================

* **allowlist argv only** — the child is launched with
  :func:`asyncio.create_subprocess_exec` and a list argv. ``shell=True`` is
  never used. Prompt delivery is one of three mechanisms depending on the
  agent: claude and codex read it from stdin (codex's argv carries a
  trailing ``-`` sentinel making that explicit); grok reads it from a
  private ``0600`` prompt file inside the resolved workdir (its ``-p``
  requires the prompt as an argv value, and argv would both hit Linux's
  ~128KiB ``MAX_ARG_STRLEN`` on huge prompts and leak the text into ``ps``
  output); antigravity has neither a stdin nor a ``--prompt-file`` channel
  (piped stdin hangs the CLI, field-verified), so its prompt is carried on
  argv — accepting the same ``MAX_ARG_STRLEN`` cap and local ``ps``
  visibility that grok's file delivery was specifically designed to avoid.
  No shell is involved in any case (list argv, never shell text), and the
  documented threat model (an isolated, single-operator workstation) treats
  local ``ps`` visibility as an accepted, documented limitation for
  antigravity rather than a defect.
* **default read-only** — ``allow_file_writes=False`` /
  ``sandbox_mode="read_only"`` are the defaults, mapped to claude's
  ``--permission-mode plan``. Writes require explicit opt-in and the sandbox
  mapping is clamped to read-only whenever ``allow_file_writes`` is False.
* **workdir boundary** — the working directory is expanded, resolved to an
  absolute path and created; a literal ``..`` escape is rejected.
* **timeout with process-group kill** — ``exec_timeout_s`` is enforced with
  :func:`asyncio.wait_for`; on expiry the whole process *group* is
  ``SIGKILL``ed (the CLI hangs a real LLM call off a child, so killing only
  the parent would orphan it).
* **env allowlist** — the child does NOT inherit the parent environment. A
  minimal env is built explicitly; ``ANTHROPIC_API_KEY`` is never forwarded
  unless the operator lists it in ``passthrough_env`` (this prevents a
  stray key from silently overriding subscription OAuth).
* **recursion cap** — ``CODEROUTER_AGENT_DEPTH`` is propagated (incremented)
  into the child and refused at or above ``agent_depth_limit``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from coderouter.adapters.base import (
    AdapterError,
    BaseAdapter,
    ChatRequest,
    ChatResponse,
    ProviderCallOverrides,
    StreamChunk,
)
from coderouter.config.schemas import AgentCliConfig, ProviderConfig
from coderouter.logging import get_logger

logger = get_logger(__name__)

# Fixed, minimal PATH injected into the child (design §5.3.1). The adapter
# resolves the CLI executable to an absolute path itself (see ``generate``),
# so this PATH only governs any helper binaries the CLI spawns.
_SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Environment variable that carries the recursion depth across nested agent
# invocations (design §5.5).
_DEPTH_ENV = "CODEROUTER_AGENT_DEPTH"

# How many trailing stderr bytes to attach to a non-zero-exit error message.
_MAX_STDERR_TAIL = 2000

# Coarse chunk size (characters) for the pseudo-stream splitter (design §5.1.4).
_STREAM_CHUNK_CHARS = 512

# sandbox_mode → claude ``--permission-mode`` value (design §5.4). ``edit``
# and ``full_auto`` both map to ``acceptEdits`` in Phase 1a.
_CLAUDE_PERMISSION_MODE = {
    "read_only": "plan",
    "edit": "acceptEdits",
    "full_auto": "acceptEdits",
}

# sandbox_mode → grok sandbox/approval flags (design §5.4, grok CLI v0.2.93).
# grok's ``--sandbox`` takes a built-in profile VALUE (off|workspace|
# read-only|strict) and its ``--permission-mode`` shares Claude Code's value
# set; ``full_auto`` swaps the permission mode for ``--always-approve``
# (auto-approve all tool executions).
_GROK_SANDBOX_ARGS = {
    "read_only": ["--sandbox", "read-only", "--permission-mode", "plan"],
    "edit": ["--sandbox", "workspace", "--permission-mode", "acceptEdits"],
    "full_auto": ["--sandbox", "workspace", "--always-approve"],
}

# sandbox_mode → codex ``-s/--sandbox`` value (design §5.4, codex-cli
# 0.144.1, verified via facts-codex.md). ``codex exec`` has NO approval
# flag at all (non-interactive, so there is no prompt to approve/skip), so
# ``full_auto`` collapses onto the same ``workspace-write`` value as
# ``edit`` — there is nothing further to "auto" beyond granting writes.
_CODEX_SANDBOX_ARGS = {
    "read_only": ["-s", "read-only"],
    "edit": ["-s", "workspace-write"],
    "full_auto": ["-s", "workspace-write"],
}

# sandbox_mode → antigravity ``--mode`` flags (design §5.4, Antigravity CLI
# 1.1.1, verified via facts-antigravity.md). ``agy --help`` only enumerates
# two ``--mode`` values (``plan`` / ``accept-edits``), so ``full_auto`` maps
# onto ``accept-edits`` plus the separate ``--dangerously-skip-permissions``
# flag (auto-approves all tool executions) rather than a third ``--mode``
# value. ``--sandbox`` is deliberately never used here: it has a known bypass
# bug when combined with ``--dangerously-skip-permissions`` (agy issue #36).
_ANTIGRAVITY_MODE_ARGS = {
    "read_only": ["--mode", "plan"],
    "edit": ["--mode", "accept-edits"],
    "full_auto": ["--mode", "accept-edits", "--dangerously-skip-permissions"],
}

# Compiled ANSI escape sequence stripper for antigravity's plain-text output
# (design §5.1.6). Covers CSI sequences (``ESC [ ... final-byte``) plus bare
# OSC-style ``ESC ]`` sequences terminated by BEL or ``ESC \`` (kept simple —
# antigravity's TUI chrome is not fully specified, so this is a defensive
# best-effort strip, not a full ANSI parser).
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI ... final byte
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC ... BEL or ST
)


def _chunk_text(text: str, size: int = _STREAM_CHUNK_CHARS) -> Iterator[str]:
    """Split ``text`` into ``size``-char pieces for the pseudo-stream."""
    for start in range(0, len(text), size):
        yield text[start : start + size]


class AgentCliAdapter(BaseAdapter):
    """Invoke an external coding-agent CLI one-shot (claude + codex + grok +
    antigravity).

    The ``agent`` field selects the argv builder / output parser via the
    dispatch tables built in :meth:`__init__`, mirroring how
    ``openai_compat`` fronts many HTTP backends from one class.
    """

    _IMPLEMENTED_AGENTS = ("claude", "codex", "grok", "antigravity")

    def __init__(self, config: ProviderConfig) -> None:
        """Bind to a ``ProviderConfig`` and reject unsupported agents.

        ``gemini`` gets its own rejection message (non-retryable): Google
        discontinued the Gemini CLI's OAuth for individual accounts in June
        2026, so this adapter points the operator at ``antigravity``
        instead of a generic "not implemented" message. Any other
        unimplemented value (future-proofing; currently none, since the
        schema's ``Literal`` only allows the five known agents) falls back
        to a generic message listing what IS implemented.
        """
        super().__init__(config)
        if config.agent_cli is None:  # pragma: no cover - schema enforces this
            raise AdapterError(
                "agent_cli provider is missing its agent_cli sub-config",
                provider=config.name,
                retryable=False,
            )
        self.acfg: AgentCliConfig = config.agent_cli
        if self.acfg.agent not in self._IMPLEMENTED_AGENTS:
            if self.acfg.agent == "gemini":
                raise AdapterError(
                    "agent 'gemini' is not supported: Google discontinued "
                    "the Gemini CLI for individual accounts (June 2026; "
                    "IneligibleTierError). Use agent='antigravity' "
                    "(Antigravity CLI, command 'agy') instead.",
                    provider=config.name,
                    retryable=False,
                )
            raise AdapterError(
                f"agent {self.acfg.agent!r} is not implemented "
                f"(implemented: {', '.join(self._IMPLEMENTED_AGENTS)}).",
                provider=config.name,
                retryable=False,
            )
        # agent → argv builder / output parser dispatch tables. claude landed
        # in Phase 1a, codex in Phase 1b, grok in Phase 1d, antigravity in
        # Phase 1c — all four (design §9) are now implemented.
        self._builders = {
            "claude": self._build_claude_argv,
            "codex": self._build_codex_argv,
            "grok": self._build_grok_argv,
            "antigravity": self._build_antigravity_argv,
        }
        self._parsers = {
            "claude": self._parse_claude,
            "codex": self._parse_codex,
            "grok": self._parse_grok,
            "antigravity": self._parse_antigravity,
        }
        # Prompt delivery is per-agent, one of three mechanisms: claude and
        # codex read their prompt from stdin (10MB cap; codex's argv carries
        # a trailing "-" sentinel making that explicit), which keeps argv
        # free of the (potentially huge) prompt text. grok's ``-p``
        # REQUIRES the prompt as its argv value (piped stdin is only
        # appended as extra context, verified on v0.2.93), so grok gets the
        # prompt via ``--prompt-file`` instead — see ``_write_prompt_file``
        # for the rationale. antigravity has NEITHER a stdin nor a
        # ``--prompt-file`` channel — piped stdin hangs the CLI outright
        # (field-verified on agy 1.1.1) — so its prompt rides argv instead;
        # see ``_build_antigravity_argv`` for the tradeoff this accepts.
        self._uses_stdin = self.acfg.agent in ("claude", "codex")
        self._uses_argv = self.acfg.agent == "antigravity"

    # ------------------------------------------------------------------
    # BaseAdapter contract
    # ------------------------------------------------------------------

    async def healthcheck(self) -> bool:
        """Lightweight check: the CLI binary exists on PATH (design §5.1.2)."""
        return shutil.which(self.acfg.command) is not None

    async def generate(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> ChatResponse:
        """Run the CLI once and shape the final answer into a ChatResponse.

        Raises :class:`AdapterError` on every failure path, with
        ``retryable`` set so the fallback engine can decide whether to try
        the next provider (transient failures) or stop (config / recursion
        errors).
        """
        # Recursion guard (design §5.5): refuse when we are already nested at
        # or beyond the configured depth. Non-retryable — retrying the same
        # chain would recurse again.
        depth = self._current_depth()
        if depth >= self.acfg.agent_depth_limit:
            raise AdapterError(
                f"agent recursion depth {depth} >= limit {self.acfg.agent_depth_limit}",
                provider=self.name,
                retryable=False,
            )

        # exec_timeout_s is the base; a profile-level override wins when set.
        # NOTE: the inherited ``effective_timeout`` falls back to
        # ``ProviderConfig.timeout_s``, which is the wrong knob here, so we
        # resolve against ``exec_timeout_s`` explicitly (design §5.1.3).
        timeout = (
            overrides.timeout_s
            if overrides is not None and overrides.timeout_s is not None
            else self.acfg.exec_timeout_s
        )

        prompt = self._render_prompt(request, overrides)
        workdir = self._resolve_workdir()

        # Agents that cannot take the prompt on stdin (grok) get it through a
        # private temp file inside the workdir; it is ALWAYS removed in the
        # ``finally`` below, including on timeout / exception paths. Argv
        # agents (antigravity) get neither a file nor stdin bytes — the
        # prompt text is handed straight to the builder instead.
        prompt_file = (
            None
            if self._uses_stdin or self._uses_argv
            else self._write_prompt_file(prompt, workdir)
        )
        try:
            argv = self._builders[self.acfg.agent](workdir, prompt_file, prompt)
            # Resolve the executable to an absolute path so argv[0] is a
            # concrete binary independent of the child's minimal PATH
            # (design §6 allowlist).
            resolved = shutil.which(argv[0])
            if resolved is not None:
                argv = [resolved, *argv[1:]]
            env = self._build_child_env()

            logger.info(
                "agent-cli-exec",
                extra={
                    "provider": self.name,
                    "agent": self.acfg.agent,
                    "argv0": argv[0],
                    "timeout_s": timeout,
                },
            )

            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workdir,
                    env=env,
                    # New session/process group so a timeout can SIGKILL the
                    # whole group (the CLI hangs its real LLM call off a
                    # child).
                    start_new_session=True,
                )
            except (FileNotFoundError, OSError) as exc:
                raise AdapterError(
                    f"failed to launch {self.acfg.command!r}: {exc}",
                    provider=self.name,
                    retryable=False,
                ) from exc

            # Non-stdin agents (grok's file delivery, antigravity's argv
            # delivery) get None here, so ``communicate()`` closes stdin
            # immediately without writing to it — verified required for
            # antigravity, whose CLI hangs if anything is piped to stdin.
            stdin_bytes = prompt.encode("utf-8") if self._uses_stdin else None
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin_bytes), timeout=timeout
                )
            except TimeoutError as exc:
                self._kill_process_group(proc)
                with contextlib.suppress(Exception):
                    await proc.wait()
                raise AdapterError(
                    f"{self.acfg.agent} exec timed out after {timeout}s",
                    provider=self.name,
                    retryable=True,
                ) from exc

            if proc.returncode != 0:
                detail = self._error_detail(stdout, stderr)
                raise AdapterError(
                    f"{self.acfg.agent} exited {proc.returncode}: {detail}",
                    provider=self.name,
                    status_code=None,
                    retryable=self._is_retryable_exit(proc.returncode),
                )

            final_text, usage, meta = self._parsers[self.acfg.agent](stdout, stderr)
        finally:
            if prompt_file is not None:
                # Best-effort cleanup — the child may already have exited and
                # a vanished file is not an error worth surfacing.
                with contextlib.suppress(OSError):
                    os.unlink(prompt_file)
        return self._to_chat_response(final_text, usage, meta)

    async def stream(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Pseudo-stream (design §5.1.4): run once, then chunk the answer.

        No CLI in Phase 1a exposes a stable token stream, so the final text
        from :meth:`generate` is split into content chunks followed by a
        terminal ``finish_reason="stop"`` chunk carrying usage.
        """
        resp = await self.generate(request, overrides=overrides)
        try:
            content = resp.choices[0]["message"]["content"] or ""
        except (IndexError, KeyError, TypeError):  # pragma: no cover - defensive
            content = ""
        for piece in _chunk_text(content):
            yield StreamChunk(
                id=resp.id,
                created=resp.created,
                model=resp.model,
                choices=[{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            )
        yield StreamChunk(
            id=resp.id,
            created=resp.created,
            model=resp.model,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage=resp.usage,
        )

    # ------------------------------------------------------------------
    # claude argv builder + output parser
    # ------------------------------------------------------------------

    def _build_claude_argv(
        self, workdir: str, prompt_file: str | None = None, prompt: str | None = None
    ) -> list[str]:
        """Assemble the ``claude -p`` argv (design §5.1.5 / §5.4).

        Shape::

            claude -p --output-format json --model <m> --max-turns <n>
                   --permission-mode <plan|acceptEdits> --add-dir <workdir>

        The prompt is fed on stdin (not argv), so it never appears here and
        both ``prompt_file`` and ``prompt`` are ignored (they exist only to
        keep the builder signature uniform across agents — see
        ``_build_antigravity_argv`` for the one agent that needs ``prompt``).
        ``--bare`` is deliberately NOT added — it would skip OAuth/keychain
        reads and break subscription auth (design §5.3.4).
        """
        del prompt_file, prompt  # claude takes the prompt on stdin.
        model = self.acfg.model or self.config.model
        argv = [self.acfg.command, "-p", "--output-format", "json", "--model", model]
        if self.acfg.max_turns is not None:
            argv += ["--max-turns", str(self.acfg.max_turns)]
        argv += ["--permission-mode", self._claude_permission_mode()]
        argv += ["--add-dir", workdir]
        return argv

    def _claude_permission_mode(self) -> str:
        """Map ``sandbox_mode`` → claude ``--permission-mode``, clamped.

        When ``allow_file_writes`` is False the effective mode is clamped to
        ``read_only`` regardless of ``sandbox_mode`` — defense in depth so an
        ``edit`` / ``full_auto`` request cannot grant writes without the
        explicit ``allow_file_writes`` opt-in (design §5.4).
        """
        mode = self.acfg.sandbox_mode if self.acfg.allow_file_writes else "read_only"
        return _CLAUDE_PERMISSION_MODE[mode]

    def _parse_claude(
        self, stdout: bytes, stderr: bytes
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Parse claude ``--output-format json`` output (design §5.1.6).

        Returns ``(final_text, usage, meta)``. Parsing is deliberately
        defensive: any missing/blank field or a reported ``is_error`` raises a
        retryable :class:`AdapterError` so the chain can fall through.
        """
        text = stdout.decode("utf-8", "replace").strip()
        if not text:
            raise AdapterError(
                "claude produced no stdout to parse",
                provider=self.name,
                retryable=True,
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AdapterError(
                f"claude emitted non-JSON output: {exc}",
                provider=self.name,
                retryable=True,
            ) from exc
        if not isinstance(data, dict):
            raise AdapterError(
                "claude JSON output was not an object",
                provider=self.name,
                retryable=True,
            )
        if data.get("is_error"):
            raise AdapterError(
                f"claude reported is_error=true: {str(data.get('result'))[:500]!r}",
                provider=self.name,
                retryable=True,
            )
        result = data.get("result")
        if not isinstance(result, str):
            raise AdapterError(
                "claude JSON output missing string 'result' field",
                provider=self.name,
                retryable=True,
            )

        usage = self._claude_usage(data)
        meta: dict[str, Any] = {}
        cost = data.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            # claude is the only CLI that emits a dollar figure directly;
            # surface it as response metadata for the cost dashboard.
            meta["coderouter_cost_usd"] = float(cost)
        session_id = data.get("session_id")
        if isinstance(session_id, str):
            meta["coderouter_session_id"] = session_id
        return result, usage, meta

    def _claude_usage(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize claude token usage into the OpenAI usage shape.

        ``input_tokens`` plus the two cache buckets fold into
        ``prompt_tokens``; ``cache_read_input_tokens`` is preserved under
        ``prompt_tokens_details.cached_tokens``. ``num_turns`` / ``duration_ms``
        ride along as extra keys (design §5.1.6).
        """
        raw = data.get("usage")
        raw = raw if isinstance(raw, dict) else {}

        def _int(key: str) -> int:
            value = raw.get(key)
            return int(value) if isinstance(value, (int, float)) else 0

        input_tokens = _int("input_tokens")
        output_tokens = _int("output_tokens")
        cache_read = _int("cache_read_input_tokens")
        cache_creation = _int("cache_creation_input_tokens")
        prompt_tokens = input_tokens + cache_read + cache_creation

        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        }
        if cache_read:
            usage["prompt_tokens_details"] = {"cached_tokens": cache_read}
        if isinstance(data.get("num_turns"), int):
            usage["num_turns"] = data["num_turns"]
        if isinstance(data.get("duration_ms"), (int, float)):
            usage["duration_ms"] = data["duration_ms"]
        return usage

    # ------------------------------------------------------------------
    # codex argv builder + output parser (Phase 1b)
    # ------------------------------------------------------------------

    def _build_codex_argv(
        self, workdir: str, prompt_file: str | None = None, prompt: str | None = None
    ) -> list[str]:
        """Assemble the ``codex exec`` argv (design §5.1.5 / §5.4, verified
        against codex-cli 0.144.1 — see ``_codex/facts-codex.md``).

        Shape::

            codex exec --json --skip-git-repo-check --ephemeral
                       -m <model> -C <workdir> -s <read-only|workspace-write> -

        The prompt is fed on stdin (not argv), so it never appears here and
        both ``prompt_file`` and ``prompt`` are ignored (they exist only to
        keep the builder signature uniform across agents) — the trailing
        ``-`` makes the stdin intent explicit to ``codex exec``, which
        otherwise treats a bare invocation with no PROMPT arg the same way
        but reads more ambiguously in a fixed argv list. ``--skip-git-repo-
        check`` is ALWAYS passed because the isolated workdir is not a git
        repository (without it the CLI exits 1). ``--ephemeral`` is ALWAYS
        passed so no session state persists to disk, matching the adapter's
        stateless one-shot ethos (the same rationale as grok's
        ``--no-memory``). codex has no ``--max-turns`` equivalent, so
        ``AgentCliConfig.max_turns`` is silently ignored here (documented in
        the schema).
        """
        del prompt_file, prompt  # codex takes the prompt on stdin.
        model = self.acfg.model or self.config.model
        argv = [
            self.acfg.command,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "-m",
            model,
            "-C",
            workdir,
        ]
        argv += self._codex_sandbox_args()
        argv += ["-"]
        return argv

    def _codex_sandbox_args(self) -> list[str]:
        """Map ``sandbox_mode`` → codex ``-s/--sandbox`` flags, clamped.

        Same clamp as claude/grok (design §5.4): when ``allow_file_writes``
        is False the effective mode is forced to ``read_only`` regardless of
        ``sandbox_mode``, so writes always require the explicit opt-in.
        ``codex exec`` has no approval flag in 0.144.1 (non-interactive, so
        there is nothing to approve), so ``full_auto`` maps onto the same
        ``workspace-write`` value as ``edit`` —
        ``--dangerously-bypass-approvals-and-sandbox`` is never used.
        """
        mode = self.acfg.sandbox_mode if self.acfg.allow_file_writes else "read_only"
        return list(_CODEX_SANDBOX_ARGS[mode])

    def _parse_codex(
        self, stdout: bytes, stderr: bytes
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Parse codex ``exec --json`` JSONL output (verified codex-cli
        0.144.1, ``_codex/facts-codex.md``).

        Real-run shape (one JSON object per line, newline-delimited)::

            {"type":"thread.started","thread_id":"<uuid>"}
            {"type":"turn.started"}
            {"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"2"}}
            {"type":"turn.completed","usage":{"input_tokens":13810,"cached_input_tokens":9984,"output_tokens":5,"reasoning_output_tokens":0}}

        The CLI is pre-1.0 and its JSON schema is not frozen, so parsing is
        deliberately defensive at every level: individual lines that fail to
        parse (or parse to something other than a JSON object) are SKIPPED
        rather than aborting the whole parse — stray non-JSON noise on
        stdout should not sink an otherwise-valid answer. The final answer
        is the LAST ``item.completed`` event whose ``item`` is an
        ``agent_message`` with a string ``text``. If a completed answer was
        found, it is returned even when a later ``error`` / ``turn.failed``
        event also appears (a completed answer beats a trailing error); if
        no answer was found, an ``error`` / ``turn.failed`` event (or the
        total absence of any agent_message) raises a retryable
        :class:`AdapterError`.
        """
        text = stdout.decode("utf-8", "replace")
        if not text.strip():
            raise AdapterError(
                "codex produced no stdout to parse",
                provider=self.name,
                retryable=True,
            )

        final_text: str | None = None
        thread_id: str | None = None
        failure_event: dict[str, Any] | None = None
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Defensive against stray non-JSON noise (progress text that
                # leaked onto stdout, partial writes, etc.) — skip the line.
                continue
            if not isinstance(event, dict):
                continue

            etype = event.get("type")
            if etype == "thread.started":
                tid = event.get("thread_id")
                if isinstance(tid, str):
                    thread_id = tid
            elif etype == "item.completed":
                item = event.get("item")
                if (
                    isinstance(item, dict)
                    and item.get("type") == "agent_message"
                    and isinstance(item.get("text"), str)
                ):
                    final_text = item["text"]
            elif etype == "turn.completed":
                usage = event.get("usage")
                usage = usage if isinstance(usage, dict) else {}

                def _int(key: str, _usage: dict[str, Any] = usage) -> int:
                    value = _usage.get(key)
                    return int(value) if isinstance(value, (int, float)) else 0

                prompt_tokens += _int("input_tokens")
                completion_tokens += _int("output_tokens")
                cached_tokens += _int("cached_input_tokens")
                reasoning_tokens += _int("reasoning_output_tokens")
            elif etype in ("error", "turn.failed"):
                failure_event = event

        if final_text is None:
            if failure_event is not None:
                raise AdapterError(
                    f"codex reported {failure_event.get('type')}: {failure_event!r}"[:500],
                    provider=self.name,
                    retryable=True,
                )
            raise AdapterError(
                "codex JSONL output contained no agent_message",
                provider=self.name,
                retryable=True,
            )

        # cached_input_tokens is a SUBSET of input_tokens (not additive) —
        # verified sample: input 13810 ⊇ cached 9984 — so it is preserved
        # under prompt_tokens_details rather than folded into prompt_tokens
        # (this differs from claude's normalization, design §5.1.6).
        usage_out: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cached_tokens > 0:
            usage_out["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
        if reasoning_tokens > 0:
            usage_out["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}

        meta: dict[str, Any] = {}
        if thread_id is not None:
            meta["coderouter_session_id"] = thread_id
        return final_text, usage_out, meta

    # ------------------------------------------------------------------
    # grok argv builder + output parser (Phase 1d)
    # ------------------------------------------------------------------

    def _build_grok_argv(
        self, workdir: str, prompt_file: str | None = None, prompt: str | None = None
    ) -> list[str]:
        """Assemble the grok headless argv (design §5, grok CLI v0.2.93).

        Shape::

            grok --prompt-file <f> --output-format json -m <m> --cwd <w>
                 --max-turns <n> --no-memory --sandbox <profile>
                 [--permission-mode <mode> | --always-approve]

        The prompt travels via ``--prompt-file`` (never argv / stdin): grok's
        ``-p`` requires the prompt as its argv value, and putting it there
        would hit Linux's ~128KiB ``MAX_ARG_STRLEN`` on large prompts and
        leak the text into ``ps`` output. ``--no-memory`` is deliberate: it
        enforces the one-request-one-transformation statelessness even if
        the user's grok config enables cross-session memory. ``prompt`` is
        ignored (it exists only to keep the builder signature uniform across
        agents — grok never takes it on argv).
        """
        del prompt
        if prompt_file is None:  # pragma: no cover - generate() always supplies it
            raise AdapterError(
                "grok argv requires a prompt file",
                provider=self.name,
                retryable=False,
            )
        model = self.acfg.model or self.config.model
        argv = [
            self.acfg.command,
            "--prompt-file",
            prompt_file,
            "--output-format",
            "json",
            "-m",
            model,
            "--cwd",
            workdir,
        ]
        if self.acfg.max_turns is not None:
            argv += ["--max-turns", str(self.acfg.max_turns)]
        argv += ["--no-memory"]
        argv += self._grok_sandbox_args()
        return argv

    def _grok_sandbox_args(self) -> list[str]:
        """Map ``sandbox_mode`` → grok sandbox/approval flags, clamped.

        Same clamp as claude (design §5.4): when ``allow_file_writes`` is
        False the effective mode is forced to ``read_only`` regardless of
        ``sandbox_mode``, so writes always require the explicit opt-in.
        """
        mode = self.acfg.sandbox_mode if self.acfg.allow_file_writes else "read_only"
        return list(_GROK_SANDBOX_ARGS[mode])

    def _parse_grok(
        self, stdout: bytes, stderr: bytes
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Parse grok ``--output-format json`` output (verified v0.2.93).

        Real-run shape::

            {"text": "...", "stopReason": "EndTurn", "sessionId": "<uuid>",
             "requestId": "<uuid>", "thought": "..."}

        The CLI is early beta, so parsing is deliberately defensive: empty /
        non-JSON / non-object stdout and a missing ``text`` field all raise
        a retryable :class:`AdapterError` so the chain can fall through.
        ``thought`` is ignored — only the final ``text`` is the answer.
        """
        text = stdout.decode("utf-8", "replace").strip()
        if not text:
            raise AdapterError(
                "grok produced no stdout to parse",
                provider=self.name,
                retryable=True,
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AdapterError(
                f"grok emitted non-JSON output: {exc}",
                provider=self.name,
                retryable=True,
            ) from exc
        if not isinstance(data, dict):
            raise AdapterError(
                "grok JSON output was not an object",
                provider=self.name,
                retryable=True,
            )
        result = data.get("text")
        if not isinstance(result, str):
            raise AdapterError(
                "grok JSON output missing string 'text' field",
                provider=self.name,
                retryable=True,
            )

        # grok emits NO token usage / cost fields (verified), so usage is
        # all-zeros; the cost dashboard shows 0 for this provider unless the
        # operator sets ``ProviderConfig.cost`` rates (design §5.1.6).
        usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        meta: dict[str, Any] = {}
        session_id = data.get("sessionId")
        if isinstance(session_id, str):
            meta["coderouter_session_id"] = session_id
        return result, usage, meta

    # ------------------------------------------------------------------
    # antigravity argv builder + output parser (Phase 1c, in lieu of gemini)
    # ------------------------------------------------------------------

    def _build_antigravity_argv(
        self, workdir: str, prompt_file: str | None = None, prompt: str | None = None
    ) -> list[str]:
        """Assemble the ``agy -p`` argv (design §5.1.5 / §5.4, verified
        against Antigravity CLI 1.1.1 — see ``_codex/facts-antigravity.md``).

        Shape::

            agy -p <prompt> --model <m> --mode <plan|accept-edits>
                [--dangerously-skip-permissions] --print-timeout <n>s

        ``workdir`` is unused here — antigravity picks up its working
        directory from the child process's ``cwd`` (set by ``generate()``),
        and this adapter deliberately never passes ``--add-dir`` (design
        keeps the argv minimal; only claude's multi-root model needs it).
        ``prompt_file`` is unused (antigravity has no such flag).

        Prompt-delivery tradeoff (read this before touching the ``-p``
        line): agy has no stdin channel — piping content to stdin makes the
        real CLI hang waiting for a response that never comes (field-
        verified on 1.1.1) — and no ``--prompt-file`` equivalent either. The
        prompt therefore rides argv as the ``-p`` value, which is the *only*
        delivery mechanism the CLI offers. This caps practical prompt size
        at Linux's ~128KiB ``MAX_ARG_STRLEN`` and exposes the prompt text to
        ``ps`` on the local host — exactly the two costs grok's
        ``--prompt-file`` delivery was built to avoid (see the module
        docstring's security section). No shell is involved (list argv, not
        shell text), and the adapter's threat model — an isolated,
        single-operator workstation — accepts local ``ps`` visibility as a
        documented limitation rather than a defect; there is no safer
        channel to fall back to.

        ``--print-timeout`` is antigravity's own self-termination clock,
        derived from ``exec_timeout_s`` — it is the CLI's *first* wall
        against a hung call; the adapter's outer ``asyncio.wait_for`` +
        process-group ``SIGKILL`` remains the second, unconditional wall
        (design §6). ``max_turns`` is never emitted: agy has no ``--max-
        turns``-equivalent flag (like codex), so ``AgentCliConfig.max_turns``
        is silently ignored here (documented in the schema). No ``--sandbox``
        (known bypass bug alongside ``--dangerously-skip-permissions``,
        agy issue #36) and no ``--add-dir`` are ever passed.
        """
        del prompt_file  # antigravity has no prompt-file flag.
        if prompt is None:  # pragma: no cover - generate() always supplies it
            raise AdapterError(
                "antigravity argv requires the prompt text",
                provider=self.name,
                retryable=False,
            )
        model = self.acfg.model or self.config.model
        argv = [self.acfg.command, "-p", prompt, "--model", model]
        argv += self._antigravity_mode_args()
        argv += ["--print-timeout", f"{int(self.acfg.exec_timeout_s)}s"]
        return argv

    def _antigravity_mode_args(self) -> list[str]:
        """Map ``sandbox_mode`` → antigravity ``--mode`` flags, clamped.

        Same clamp as claude/codex/grok (design §5.4): when
        ``allow_file_writes`` is False the effective mode is forced to
        ``read_only`` regardless of ``sandbox_mode``, so writes always
        require the explicit opt-in.
        """
        mode = self.acfg.sandbox_mode if self.acfg.allow_file_writes else "read_only"
        return list(_ANTIGRAVITY_MODE_ARGS[mode])

    def _parse_antigravity(
        self, stdout: bytes, stderr: bytes
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Parse antigravity's plain-text ``-p`` output (verified agy 1.1.1).

        There is no ``--output-format`` flag at all (agy's ``--help`` does
        not list one) — output is whatever the model printed, decorated
        with whatever terminal styling the CLI applied even in non-TTY runs.
        Parsing is therefore: UTF-8 decode (defensively, replacing invalid
        bytes) → strip ANSI escape sequences (``_ANSI_RE``, best-effort, see
        its definition) → ``.strip()`` surrounding whitespace. Empty output
        raises a retryable :class:`AdapterError` so the chain can fall
        through. There is no token/cost figure and no session id anywhere
        in agy's output, so ``usage`` is all-zeros (cost stays 0 unless the
        operator sets ``ProviderConfig.cost``, same rationale as grok,
        design §5.1.6) and ``meta`` is empty.
        """
        text = _ANSI_RE.sub("", stdout.decode("utf-8", "replace")).strip()
        if not text:
            raise AdapterError(
                "antigravity produced no stdout",
                provider=self.name,
                retryable=True,
            )
        usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        meta: dict[str, Any] = {}
        return text, usage, meta

    # ------------------------------------------------------------------
    # helpers: prompt rendering, response shaping, env, workdir, kill
    # ------------------------------------------------------------------

    def _write_prompt_file(self, prompt: str, workdir: str) -> str:
        """Write the prompt to a private ``0600`` temp file in ``workdir``.

        Rationale: grok's ``-p`` requires the prompt as an argv value, but a
        huge prompt would hit Linux's ~128KiB ``MAX_ARG_STRLEN`` and argv
        leaks into ``ps`` output; file delivery keeps argv small and private,
        and ``0600`` + the isolated workdir bounds exposure. ``O_EXCL`` with
        a uuid4 name makes creation race-free; the caller (``generate``)
        deletes the file in a ``finally`` block on every path.
        """
        path = os.path.join(workdir, f".coderouter-prompt-{uuid.uuid4().hex}.txt")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            raise AdapterError(
                f"failed to create prompt file in {workdir}: {exc}",
                provider=self.name,
                retryable=False,
            ) from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(prompt)
        except OSError as exc:
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise AdapterError(
                f"failed to write prompt file {path}: {exc}",
                provider=self.name,
                retryable=False,
            ) from exc
        return path

    def _to_chat_response(
        self, final_text: str, usage: dict[str, Any], meta: dict[str, Any]
    ) -> ChatResponse:
        """Wrap the agent's final text in an OpenAI ChatResponse."""
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=self.config.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_text},
                    "finish_reason": "stop",
                }
            ],
            usage=usage,
            coderouter_provider=self.name,
            **meta,
        )

    def _render_prompt(self, request: ChatRequest, overrides: ProviderCallOverrides | None) -> str:
        """Flatten the chat messages into a single role-tagged prompt string.

        The profile-level ``append_system_prompt`` (if any) is prepended as a
        leading system block, matching the openai_compat directive semantics.
        """
        parts: list[str] = []
        directive = self.effective_append_system_prompt(overrides)
        if directive:
            parts.append(f"[system]\n{directive}")
        for message in request.messages:
            text = self._message_text(message.content)
            if text:
                parts.append(f"[{message.role}]\n{text}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _message_text(content: str | list[dict[str, Any]] | None) -> str:
        """Extract plain text from a message's ``content`` (str or blocks)."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)

    def _current_depth(self) -> int:
        """Read the current recursion depth from the environment (0 default)."""
        raw = os.environ.get(_DEPTH_ENV, "0")
        try:
            return int(raw)
        except ValueError:
            return 0

    def _build_child_env(self) -> dict[str, str]:
        """Build the minimal child environment (design §5.3).

        The child does NOT inherit the parent environment. Only a fixed base
        (PATH / NO_COLOR / TERM), the inherited HOME / USER / LOGNAME (for
        credential discovery — on macOS the Claude Code CLI resolves its
        Keychain entry via ``USER``; without it headless runs fail with
        "Not logged in"), the incremented recursion depth, and the operator's
        ``passthrough_env`` allowlist are injected. ``ANTHROPIC_API_KEY`` is
        therefore excluded unless explicitly allowlisted.
        """
        parent = os.environ
        env: dict[str, str] = {
            "PATH": _SAFE_PATH,
            "NO_COLOR": "1",
            "TERM": "dumb",
            _DEPTH_ENV: str(self._current_depth() + 1),
        }
        for name in ("HOME", "USER", "LOGNAME"):
            value = parent.get(name)
            if value:
                env[name] = value
        for name in self.acfg.passthrough_env:
            value = parent.get(name)
            if value is not None:
                env[name] = value
        return env

    @staticmethod
    def _error_detail(stdout: bytes, stderr: bytes) -> str:
        """Extract the most useful error text from a failed CLI run.

        The Claude Code CLI reports auth / API failures as an ``is_error:
        true`` result JSON on **stdout** with exit code 1 (stderr stays
        empty), so a stderr-only tail hides the actual cause (e.g. ``Not
        logged in · Please run /login``). Preference order: the ``result``
        field of an ``is_error`` stdout JSON → stderr tail → stdout tail.
        """
        if stdout:
            with contextlib.suppress(ValueError, TypeError):
                doc = json.loads(stdout)
                if isinstance(doc, dict) and doc.get("is_error"):
                    result = doc.get("result")
                    if isinstance(result, str) and result.strip():
                        return result.strip()[:_MAX_STDERR_TAIL]
        if stderr:
            return repr(stderr[-_MAX_STDERR_TAIL:].decode("utf-8", "replace"))
        if stdout:
            return repr(stdout[-_MAX_STDERR_TAIL:].decode("utf-8", "replace"))
        return "''"

    def _resolve_workdir(self) -> str:
        """Resolve + create the working directory, rejecting ``..`` escapes.

        A configured ``workdir`` is ``~`` / env-var expanded and resolved to
        an absolute path; a literal ``..`` component is rejected as an escape
        attempt. When unset, a dedicated isolated directory
        (``~/.coderouter/agents/<name>``) is used (design §6).
        """
        raw = self.acfg.workdir
        if raw:
            if ".." in Path(raw).parts:
                raise AdapterError(
                    f"workdir {raw!r} must not contain '..' (escape attempt)",
                    provider=self.name,
                    retryable=False,
                )
            expanded = os.path.expanduser(os.path.expandvars(raw))
            path = Path(expanded).resolve()
        else:
            path = (Path.home() / ".coderouter" / "agents" / self.name).resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AdapterError(
                f"failed to prepare workdir {path}: {exc}",
                provider=self.name,
                retryable=False,
            ) from exc
        return str(path)

    def _kill_process_group(self, proc: asyncio.subprocess.Process) -> None:
        """SIGKILL the child's whole process group (best effort)."""
        pid = proc.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # Group already gone / no permission — fall back to a direct kill.
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()

    def _is_retryable_exit(self, returncode: int | None) -> bool:
        """Whether a non-zero exit should let the chain fall through.

        Phase 1a treats any non-zero exit as transient (rate-limit / OAuth
        expiry / network), so the fallback engine advances to the next
        provider rather than surfacing a terminal failure.
        """
        return True

    # ``BaseAdapter`` (HTTP-oriented) lazily builds an httpx client; the agent
    # adapter never touches it, but ``aclose`` on the inherited base is a safe
    # no-op, so nothing extra is needed here.


__all__ = ["AgentCliAdapter"]
