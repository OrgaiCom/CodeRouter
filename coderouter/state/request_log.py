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

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

# Events to capture for the request journal.
_JOURNAL_EVENTS: frozenset[str] = frozenset({"cache-observed"})


class RequestLogHandler(logging.Handler):
    """Append-only JSONL handler for request metadata.

    Each line records one successful request's metadata (provider,
    tokens, cost, streaming flag).  Failed requests are not recorded
    — only requests that produced a response.
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        max_bytes: int = 52_428_800,  # 50 MiB default
    ) -> None:
        super().__init__(level=logging.DEBUG)
        self._log_path = Path(log_path)
        self._max_bytes = max_bytes
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115

    def emit(self, record: logging.LogRecord) -> None:
        """Write a journal line for cache-observed events."""
        if record.msg not in _JOURNAL_EVENTS:
            return
        try:
            self.acquire()
            try:
                line = self._format_line(record)
                self._file.write(line)
                self._file.flush()
                self._maybe_rotate()
            finally:
                self.release()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Flush and close the underlying file."""
        self.acquire()
        try:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()
        finally:
            self.release()
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
