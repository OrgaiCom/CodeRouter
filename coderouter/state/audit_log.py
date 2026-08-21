"""Structured JSONL audit log (v2.0-K).

Captures guard activations, chain fallbacks, budget warnings,
self-healing events, and drift transitions as append-only JSONL
records.  Implements ``logging.Handler`` so it taps the same
structured log stream that :class:`MetricsCollector` observes — no
second instrumentation path needed.

Architecture
============

::

    logger.info("backend-health-changed", extra={...})
        │
        ├─ MetricsCollector.emit()   → in-memory counters
        └─ AuditLogHandler.emit()    → append JSONL line to disk

Only *audit-worthy* events are written (guard state changes, chain
decisions, cost/budget events, self-healing lifecycle).  High-frequency
per-request events (``try-provider``, ``provider-ok``) are excluded to
keep the log small.

File rotation
=============

Simple single-backup rotation: when the active file exceeds
``max_bytes``, it is renamed to ``audit.jsonl.1`` (overwriting any
existing backup) and a fresh ``audit.jsonl`` is started.  One backup
is enough for the typical use case (reviewing yesterday's events while
today's stream runs).

Thread safety
=============

Inherits ``logging.Handler``'s built-in lock (``self.lock``) via the
``acquire()``/``release()`` protocol.  File writes are atomic single
lines (no partial-line interleaving) because Python's stdlib file
``write()`` of a single string ≤ PIPE_BUF is POSIX-atomic on Linux.
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

# M12(2) buffering defaults. Audit lines are lower-volume than request
# journal lines, but still fsync'd once per event pre-M12. We buffer and
# flush on whichever comes first — a count threshold or a wall-clock
# interval — with an unconditional flush on close()/atexit so no
# audit-worthy event is lost on clean shutdown.
_DEFAULT_FLUSH_EVERY_N: int = 20
_DEFAULT_FLUSH_INTERVAL_S: float = 2.0

# Events that are audit-worthy: guard state changes, chain decisions,
# cost/budget, self-healing lifecycle, drift, probing milestones.
_AUDIT_EVENTS: frozenset[str] = frozenset(
    {
        # Backend health
        "backend-health-changed",
        "demote-unhealthy-provider",
        # Self-healing (v2.0-J)
        "self-healing-exclude",
        "self-healing-restore",
        "self-healing-restart",
        "self-healing-recovery-probe",
        # Budget / cost
        "skip-budget-exceeded",
        "chain-budget-exceeded",
        # Chain gate events
        "chain-paid-gate-blocked",
        "chain-memory-pressure-blocked",
        "chain-uniform-auth-failure",
        # Fallback reason trail (v2.15.0) — one line per chain transition.
        # Low volume by construction: nothing is emitted when the first
        # provider serves the request.
        "fallback-occurred",
        # Memory pressure
        "memory-pressure-detected",
        # Drift (v2.0-G)
        "drift-detected",
        "drift-promoted",
        "drift-reload-attempted",
        "drift-recovered",
        # Context budget (v2.0-F)
        "context-budget-warning",
        "context-budget-trimmed",
        # Tool loop (L3)
        "tool-loop-detected",
        # Probe milestones (v2.0-I)
        "probe-capabilities-drift",
        # Startup / shutdown
        "coderouter-startup",
        "coderouter-shutdown",
    }
)


class AuditLogHandler(logging.Handler):
    """Append-only JSONL handler for audit-worthy events.

    Public API:

    - Constructor: ``AuditLogHandler(log_path, max_bytes=10_485_760)``
    - Inherited ``emit()`` is called automatically by the logging
      framework for every log record.
    - :meth:`close()` — flush and close the file handle.

    Only events whose ``record.msg`` is in :data:`_AUDIT_EVENTS` are
    written.  Everything else is silently ignored (zero I/O cost for
    non-audit log lines).
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        max_bytes: int = 10_485_760,
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
        """Buffer an audit line if the event is audit-worthy; flush lazily.

        Write buffering (M12(2)): lines are flushed when either
        ``flush_every_n`` lines have accumulated or ``flush_interval_s``
        seconds have elapsed since the last flush. :meth:`close`
        (registered with :mod:`atexit`) always flushes so a clean
        shutdown never drops a buffered audit line. Pass
        ``flush_every_n=1`` for the pre-M12 write-through behavior.
        """
        if record.msg not in _AUDIT_EVENTS:
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
        """Build a single JSONL line from a log record."""
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": record.msg,
            "level": record.levelname,
        }
        # Merge structured extras (skip stdlib internal fields).
        _stdlib_keys = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _stdlib_keys:
                continue
            payload[key] = value
        return json.dumps(payload, default=str) + "\n"

    def _maybe_rotate(self) -> None:
        """Rotate if the current file exceeds max_bytes."""
        try:
            size = self._file.tell()
            if size < self._max_bytes:
                return
            self._file.close()
            backup = self._log_path.with_suffix(".jsonl.1")
            # Overwrite any existing backup.
            if backup.exists():
                backup.unlink()
            self._log_path.rename(backup)
            self._file = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115
        except OSError:
            # If rotation fails, just keep writing to the current file.
            if self._file.closed:
                self._file = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115


def read_audit_log(
    log_path: str | Path,
    *,
    tail: int | None = None,
    event_filter: str | None = None,
    since: str | None = None,
) -> list[dict[str, object]]:
    """Read and filter audit log entries.

    Parameters:

    - ``tail`` — return only the last N entries.
    - ``event_filter`` — only entries whose ``event`` field contains
      this substring (case-insensitive).
    - ``since`` — only entries with ``ts >= since`` (ISO 8601 prefix
      match).

    Returns a list of parsed dicts, newest last.
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

            if event_filter and event_filter.lower() not in str(
                entry.get("event", "")
            ).lower():
                continue

            if since:
                ts = str(entry.get("ts", ""))
                if ts < since:
                    continue

            entries.append(entry)

    if tail is not None and tail > 0:
        entries = entries[-tail:]

    return entries


def summarize_audit_log(entries: list[dict[str, object]]) -> dict[str, int]:
    """Return event type → count summary from a list of audit entries."""
    summary: dict[str, int] = {}
    for entry in entries:
        event = str(entry.get("event", "unknown"))
        summary[event] = summary.get(event, 0) + 1
    return dict(sorted(summary.items(), key=lambda x: -x[1]))


__all__ = [
    "AuditLogHandler",
    "read_audit_log",
    "summarize_audit_log",
]
