"""External coding-agent CLI adapter (``kind="agent_cli"``).

This adapter invokes an external coding-agent CLI (Claude Code / Codex /
Gemini / Grok) as a single one-shot ``exec`` and returns the agent's final
answer as one ``prompt in → text out`` transformation. It is the in-core
implementation of the external-agents-adapter design
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

Implemented agents (Phase 1a + 1d)
==================================

Two targets are implemented here:

* ``claude`` (Claude Code CLI, Phase 1a) — the most stable CLI, the most
  fine-grained safety controls, and the only one that emits
  ``total_cost_usd`` directly (making it the reference implementation for
  the parser / cost path).
* ``grok`` (grok CLI, Phase 1d) — headless one-shot via ``--prompt-file`` +
  ``--output-format json``. The CLI emits no token/cost figures, so usage
  is reported as zeros (cost stays 0 unless the operator sets
  ``ProviderConfig.cost``, design §5.1.6). ``--no-memory`` is always passed
  so a user-level cross-session memory setting cannot leak state between
  requests.

The remaining agents (codex / gemini) are declared in the config schema so
providers.yaml is forward-compatible, but constructing an adapter for them
raises a clear ``AdapterError`` until their phase (1b / 1c) lands.

Security (design §6, non-negotiable)
====================================

* **allowlist argv only** — the child is launched with
  :func:`asyncio.create_subprocess_exec` and a list argv. ``shell=True`` is
  never used; the prompt is never subject to shell interpretation (claude
  reads it from stdin; grok reads it from a private ``0600`` prompt file
  inside the resolved workdir — its ``-p`` requires the prompt as an argv
  value, and argv would both hit Linux's ~128KiB ``MAX_ARG_STRLEN`` on huge
  prompts and leak the text into ``ps`` output).
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


def _chunk_text(text: str, size: int = _STREAM_CHUNK_CHARS) -> Iterator[str]:
    """Split ``text`` into ``size``-char pieces for the pseudo-stream."""
    for start in range(0, len(text), size):
        yield text[start : start + size]


class AgentCliAdapter(BaseAdapter):
    """Invoke an external coding-agent CLI one-shot (claude + grok).

    The ``agent`` field selects the argv builder / output parser via the
    dispatch tables built in :meth:`__init__`, mirroring how
    ``openai_compat`` fronts many HTTP backends from one class.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Bind to a ``ProviderConfig`` and reject unsupported agents.

        Constructing an adapter for an agent other than ``claude`` / ``grok``
        raises a non-retryable :class:`AdapterError` — the other targets are
        declared in the schema but not implemented until their phase
        (design §9).
        """
        super().__init__(config)
        if config.agent_cli is None:  # pragma: no cover - schema enforces this
            raise AdapterError(
                "agent_cli provider is missing its agent_cli sub-config",
                provider=config.name,
                retryable=False,
            )
        self.acfg: AgentCliConfig = config.agent_cli
        if self.acfg.agent not in ("claude", "grok"):
            raise AdapterError(
                f"agent {self.acfg.agent!r} is not implemented yet "
                f"(implemented: claude, grok). Wait for Phase 1b/1c.",
                provider=config.name,
                retryable=False,
            )
        # agent → argv builder / output parser dispatch tables. claude landed
        # in Phase 1a, grok in Phase 1d; Phase 1b/1c add codex / gemini.
        self._builders = {
            "claude": self._build_claude_argv,
            "grok": self._build_grok_argv,
        }
        self._parsers = {
            "claude": self._parse_claude,
            "grok": self._parse_grok,
        }
        # Prompt delivery is per-agent: claude reads its print-mode prompt
        # from stdin (10MB cap), which keeps argv free of the (potentially
        # huge) prompt text. grok's ``-p`` REQUIRES the prompt as its argv
        # value (piped stdin is only appended as extra context, verified on
        # v0.2.93), so grok gets the prompt via ``--prompt-file`` instead —
        # see ``_write_prompt_file`` for the rationale.
        self._uses_stdin = self.acfg.agent == "claude"

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
        # ``finally`` below, including on timeout / exception paths.
        prompt_file = None if self._uses_stdin else self._write_prompt_file(prompt, workdir)
        try:
            argv = self._builders[self.acfg.agent](workdir, prompt_file)
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

    def _build_claude_argv(self, workdir: str, prompt_file: str | None = None) -> list[str]:
        """Assemble the ``claude -p`` argv (design §5.1.5 / §5.4).

        Shape::

            claude -p --output-format json --model <m> --max-turns <n>
                   --permission-mode <plan|acceptEdits> --add-dir <workdir>

        The prompt is fed on stdin (not argv), so it never appears here and
        ``prompt_file`` is ignored (it exists only to keep the builder
        signature uniform across agents). ``--bare`` is deliberately NOT
        added — it would skip OAuth/keychain reads and break subscription
        auth (design §5.3.4).
        """
        del prompt_file  # claude takes the prompt on stdin, not from a file.
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
    # grok argv builder + output parser (Phase 1d)
    # ------------------------------------------------------------------

    def _build_grok_argv(self, workdir: str, prompt_file: str | None = None) -> list[str]:
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
        the user's grok config enables cross-session memory.
        """
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
