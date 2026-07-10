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

Phase 1a scope
==============

Only the ``claude`` (Claude Code CLI) target is implemented here — it is
the most stable CLI, has the most fine-grained safety controls, and is the
only one that emits ``total_cost_usd`` directly (making it the reference
implementation for the parser / cost path). The other agents
(codex / gemini / grok) are declared in the config schema so providers.yaml
is forward-compatible, but constructing an adapter for them raises a clear
``AdapterError`` until their phase lands.

Security (design §6, non-negotiable)
====================================

* **allowlist argv only** — the child is launched with
  :func:`asyncio.create_subprocess_exec` and a list argv. ``shell=True`` is
  never used; the prompt is fed on stdin, so it is never subject to shell
  interpretation.
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


def _chunk_text(text: str, size: int = _STREAM_CHUNK_CHARS) -> Iterator[str]:
    """Split ``text`` into ``size``-char pieces for the pseudo-stream."""
    for start in range(0, len(text), size):
        yield text[start : start + size]


class AgentCliAdapter(BaseAdapter):
    """Invoke an external coding-agent CLI one-shot (Phase 1a: claude only).

    The ``agent`` field selects the argv builder / output parser via the
    dispatch tables built in :meth:`__init__`, mirroring how
    ``openai_compat`` fronts many HTTP backends from one class.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Bind to a ``ProviderConfig`` and reject unsupported agents.

        Constructing an adapter for an agent other than ``claude`` raises a
        non-retryable :class:`AdapterError` — the other targets are declared
        in the schema but not implemented until their phase (design §9).
        """
        super().__init__(config)
        if config.agent_cli is None:  # pragma: no cover - schema enforces this
            raise AdapterError(
                "agent_cli provider is missing its agent_cli sub-config",
                provider=config.name,
                retryable=False,
            )
        self.acfg: AgentCliConfig = config.agent_cli
        if self.acfg.agent != "claude":
            raise AdapterError(
                f"agent {self.acfg.agent!r} is not implemented in Phase 1a "
                f"(claude only). Configure agent='claude' or wait for the "
                f"agent's phase.",
                provider=config.name,
                retryable=False,
            )
        # agent → argv builder / output parser dispatch tables. Phase 1a
        # registers only claude; later phases add codex / gemini / grok.
        self._builders = {"claude": self._build_claude_argv}
        self._parsers = {"claude": self._parse_claude}
        # claude reads its print-mode prompt from stdin (10MB cap), which
        # keeps argv free of the (potentially huge) prompt text.
        self._uses_stdin = True

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
                f"agent recursion depth {depth} >= limit "
                f"{self.acfg.agent_depth_limit}",
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
        argv = self._builders[self.acfg.agent](workdir)
        # Resolve the executable to an absolute path so argv[0] is a concrete
        # binary independent of the child's minimal PATH (design §6 allowlist).
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
                # New session/process group so a timeout can SIGKILL the whole
                # group (the CLI hangs its real LLM call off a child).
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
            tail = (stderr[-_MAX_STDERR_TAIL:].decode("utf-8", "replace") if stderr else "")
            raise AdapterError(
                f"{self.acfg.agent} exited {proc.returncode}: {tail!r}",
                provider=self.name,
                status_code=None,
                retryable=self._is_retryable_exit(proc.returncode),
            )

        final_text, usage, meta = self._parsers[self.acfg.agent](stdout, stderr)
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
                choices=[
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                ],
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

    def _build_claude_argv(self, workdir: str) -> list[str]:
        """Assemble the ``claude -p`` argv (design §5.1.5 / §5.4).

        Shape::

            claude -p --output-format json --model <m> --max-turns <n>
                   --permission-mode <plan|acceptEdits> --add-dir <workdir>

        The prompt is fed on stdin (not argv), so it never appears here.
        ``--bare`` is deliberately NOT added — it would skip OAuth/keychain
        reads and break subscription auth (design §5.3.4).
        """
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
    # helpers: prompt rendering, response shaping, env, workdir, kill
    # ------------------------------------------------------------------

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

    def _render_prompt(
        self, request: ChatRequest, overrides: ProviderCallOverrides | None
    ) -> str:
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
        (PATH / NO_COLOR / TERM), the inherited HOME (for credential-dir
        discovery), the incremented recursion depth, and the operator's
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
        home = parent.get("HOME")
        if home:
            env["HOME"] = home
        for name in self.acfg.passthrough_env:
            value = parent.get(name)
            if value is not None:
                env[name] = value
        return env

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
