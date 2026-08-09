"""Request journal for replay analysis (v2.0-K Replay framework).

Records per-request metadata (provider, model, profile, token counts,
latency, cost, stop_reason) as append-only JSONL.  The request/response
body is NOT recorded (privacy + size).

Architecture
============

Implements ``logging.Handler`` and captures ``cache-observed`` events
(one per successful request, carrying token counts + cost) paired with
timing data computed from ``try-provider`` → ``provider-ok`` deltas.

The engine emits these events in sequence::

    try-provider  → provider (name), stream (bool)
    provider-ok   → provider (name), stream (bool)
    cache-observed→ provider, input_tokens, output_tokens, cost_usd, ...

The handler captures ``cache-observed`` as the authoritative "request
completed" signal (it fires exactly once per successful response and
carries the richest payload).

File rotation
=============

Same single-backup rotation as :class:`AuditLogHandler`: when the
active file exceeds ``max_bytes``, rename to ``.1`` and start fresh.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from coderouter.secret_redaction import install_secret_filter

# Events to capture for the request journal.
_JOURNAL_EVENTS: frozenset[str] = frozenset({"cache-observed"})

# M12(2) buffering defaults. A request journal line is written per
# successful request; fsync-on-every-write is wasteful under load. We
# buffer in-process and flush on whichever comes first — a count
# threshold or a wall-clock interval — plus an unconditional flush on
# close()/atexit so no completed request is ever lost on clean shutdown.
_DEFAULT_FLUSH_EVERY_N: int = 20
_DEFAULT_FLUSH_INTERVAL_S: float = 2.0


class RequestLogHandler(logging.Handler):
    """Append-only JSONL handler for request metadata.

    Each line records one successful request's metadata (provider,
    tokens, cost, streaming flag).  Failed requests are not recorded
    — only requests that produced a response.

    Write buffering (M12(2))
        Lines are buffered and flushed to disk when either
        ``flush_every_n`` lines have accumulated or ``flush_interval_s``
        seconds have elapsed since the last flush — whichever comes
        first. :meth:`close` (registered with :mod:`atexit`) always
        flushes, so a clean shutdown never drops a buffered line. Pass
        ``flush_every_n=1`` for the pre-M12 write-through behavior (tests
        that assert immediate on-disk visibility use this).
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        max_bytes: int = 52_428_800,  # 50 MiB default
        flush_every_n: int = _DEFAULT_FLUSH_EVERY_N,
        flush_interval_s: float = _DEFAULT_FLUSH_INTERVAL_S,
    ) -> None:
        super().__init__(level=logging.DEBUG)
        # v2.14.0: scrub registered credentials on the way to disk. This
        # sink is the one that persists, so an unscrubbed key here outlives
        # the process that leaked it.
        install_secret_filter(self)
        self._log_path = Path(log_path)
        self._max_bytes = max_bytes
        self._flush_every_n = max(1, flush_every_n)
        self._flush_interval_s = flush_interval_s
        self._buffer: list[str] = []
        self._last_flush_monotonic = time.monotonic()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115
        # Guarantee a final flush even if close() is never called
        # explicitly (e.g. an ungraceful lifespan teardown).
        atexit.register(self._atexit_flush)

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer a journal line for cache-observed events; flush lazily."""
        if record.msg not in _JOURNAL_EVENTS:
            return
        try:
            self.acquire()
            try:
                self._buffer.append(self._format_line(record))
                if self._should_flush():
                    self._flush_buffer()
            finally:
                self.release()
        except Exception:
            self.handleError(record)

    def _should_flush(self) -> bool:
        """True when the count threshold or time interval has been reached."""
        if len(self._buffer) >= self._flush_every_n:
            return True
        return (
            time.monotonic() - self._last_flush_monotonic
        ) >= self._flush_interval_s

    def _flush_buffer(self) -> None:
        """Write all buffered lines to disk, fsync, and maybe rotate.

        Caller must hold the handler lock. No-op when the buffer is empty.
        """
        if not self._buffer:
            return
        self._file.write("".join(self._buffer))
        self._buffer.clear()
        self._file.flush()
        self._last_flush_monotonic = time.monotonic()
        self._maybe_rotate()

    def _atexit_flush(self) -> None:
        """Best-effort flush of any buffered lines at interpreter exit."""
        try:
            self.acquire()
            try:
                if self._file and not self._file.closed:
                    self._flush_buffer()
            finally:
                self.release()
        except Exception:  # pragma: no cover - defensive at shutdown
            pass

    def close(self) -> None:
        """Flush buffered lines and close the underlying file."""
        self.acquire()
        try:
            if self._file and not self._file.closed:
                self._flush_buffer()
                self._file.close()
        finally:
            self.release()
        with contextlib.suppress(Exception):
            atexit.unregister(self._atexit_flush)
        super().close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_line(self, record: logging.LogRecord) -> str:
        """Build a journal JSONL line from a cache-observed log record."""
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "provider": getattr(record, "provider", None),
            "input_tokens": getattr(record, "input_tokens", 0),
            "output_tokens": getattr(record, "output_tokens", 0),
            "cost_usd": getattr(record, "cost_usd", 0.0),
            "cost_savings_usd": getattr(record, "cost_savings_usd", 0.0),
            "streaming": getattr(record, "streaming", False),
            "cache_read_input_tokens": getattr(record, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(record, "cache_creation_input_tokens", 0),
            "outcome": getattr(record, "outcome", "unknown"),
        }
        return json.dumps(payload, default=str) + "\n"

    def _maybe_rotate(self) -> None:
        """Rotate if the current file exceeds max_bytes."""
        try:
            size = self._file.tell()
            if size < self._max_bytes:
                return
            self._file.close()
            backup = self._log_path.with_suffix(".jsonl.1")
            if backup.exists():
                backup.unlink()
            self._log_path.rename(backup)
            self._file = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115
        except OSError:
            if self._file.closed:
                self._file = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115


def read_request_log(
    log_path: str | Path,
    *,
    tail: int | None = None,
    provider_filter: str | None = None,
    since: str | None = None,
) -> list[dict[str, object]]:
    """Read and filter request journal entries.

    Parameters:

    - ``tail`` — return only the last N entries.
    - ``provider_filter`` — only entries matching this provider name
      (exact, case-insensitive).
    - ``since`` — only entries with ``ts >= since`` (ISO 8601 prefix).

    Returns a list of parsed dicts, oldest first.
    """
    path = Path(log_path)
    if not path.exists():
        return []

    entries: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if provider_filter and str(entry.get("provider", "")).lower() != provider_filter.lower():
                continue

            if since:
                ts = str(entry.get("ts", ""))
                if ts < since:
                    continue

            entries.append(entry)

    if tail is not None and tail > 0:
        entries = entries[-tail:]

    return entries


__all__ = [
    "RequestLogHandler",
    "read_request_log",
]
