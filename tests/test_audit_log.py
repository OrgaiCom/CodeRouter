"""v2.0-K: AuditLog tests.

Test groups:

- **AuditLogHandler**: write audit-worthy events, ignore non-audit events.
- **File rotation**: rotation when max_bytes is exceeded.
- **Reader**: read_audit_log filtering (tail, event_filter, since).
- **Summarizer**: summarize_audit_log event counts.
- **CLI audit command**: basic argument parsing and output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from coderouter.state.audit_log import (
    AuditLogHandler,
    read_audit_log,
    summarize_audit_log,
)

# ----------------------------------------------------------------------
# Group 1: AuditLogHandler — writes audit events, ignores others
# ----------------------------------------------------------------------


def test_audit_handler_writes_audit_event(tmp_path: Path) -> None:
    """Audit-worthy events are written to the JSONL file."""
    log_path = tmp_path / "audit.jsonl"
    handler = AuditLogHandler(log_path)

    logger = logging.getLogger("test_audit_handler_writes")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.warning(
        "backend-health-changed",
        extra={"provider": "p1", "new_state": "UNHEALTHY"},
    )

    handler.close()
    logger.removeHandler(handler)

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "backend-health-changed"
    assert entry["provider"] == "p1"


def test_audit_handler_ignores_non_audit_event(tmp_path: Path) -> None:
    """Non-audit events (e.g. try-provider) are NOT written."""
    log_path = tmp_path / "audit.jsonl"
    handler = AuditLogHandler(log_path)

    logger = logging.getLogger("test_audit_handler_ignores")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("try-provider", extra={"provider": "p1"})
    logger.info("provider-ok", extra={"provider": "p1"})

    handler.close()
    logger.removeHandler(handler)

    # File should be empty (or not exist).
    if log_path.exists():
        assert log_path.read_text().strip() == ""


def test_audit_handler_multiple_events(tmp_path: Path) -> None:
    """Multiple audit events produce multiple JSONL lines."""
    log_path = tmp_path / "audit.jsonl"
    handler = AuditLogHandler(log_path)

    logger = logging.getLogger("test_audit_multi")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.warning("self-healing-exclude", extra={"provider": "p1"})
    logger.info("self-healing-restore", extra={"provider": "p1"})
    logger.info("drift-detected", extra={"provider": "p2"})

    handler.close()
    logger.removeHandler(handler)

    lines = [line for line in log_path.read_text().strip().split("\n") if line]
    assert len(lines) == 3


# ----------------------------------------------------------------------
# Group 2: File rotation
# ----------------------------------------------------------------------


def test_rotation_on_max_bytes(tmp_path: Path) -> None:
    """File is rotated when max_bytes is exceeded."""
    log_path = tmp_path / "audit.jsonl"
    # Very small max to trigger rotation quickly.
    handler = AuditLogHandler(log_path, max_bytes=200)

    logger = logging.getLogger("test_rotation")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # Write enough events to exceed 200 bytes.
    for i in range(10):
        logger.warning("backend-health-changed", extra={"i": i})

    handler.close()
    logger.removeHandler(handler)

    backup = tmp_path / "audit.jsonl.1"
    assert backup.exists(), "Backup file should exist after rotation"
    assert log_path.exists(), "Active file should still exist"


# ----------------------------------------------------------------------
# Group 3: Reader — read_audit_log filtering
# ----------------------------------------------------------------------


def _write_test_log(path: Path) -> None:
    """Write a small test audit log."""
    entries = [
        {"ts": "2026-05-06T10:00:00Z", "event": "backend-health-changed", "level": "WARNING", "provider": "p1"},
        {"ts": "2026-05-06T10:05:00Z", "event": "self-healing-exclude", "level": "WARNING", "provider": "p1"},
        {"ts": "2026-05-06T10:10:00Z", "event": "drift-detected", "level": "INFO", "provider": "p2"},
        {"ts": "2026-05-06T10:15:00Z", "event": "self-healing-restore", "level": "INFO", "provider": "p1"},
        {"ts": "2026-05-06T10:20:00Z", "event": "backend-health-changed", "level": "WARNING", "provider": "p2"},
    ]
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_read_all(tmp_path: Path) -> None:
    """read_audit_log with no filters returns all entries."""
    log_path = tmp_path / "audit.jsonl"
    _write_test_log(log_path)
    entries = read_audit_log(log_path)
    assert len(entries) == 5


def test_read_tail(tmp_path: Path) -> None:
    """--tail N returns only the last N entries."""
    log_path = tmp_path / "audit.jsonl"
    _write_test_log(log_path)
    entries = read_audit_log(log_path, tail=2)
    assert len(entries) == 2
    assert entries[0]["event"] == "self-healing-restore"
    assert entries[1]["event"] == "backend-health-changed"


def test_read_event_filter(tmp_path: Path) -> None:
    """--filter narrows by event name substring."""
    log_path = tmp_path / "audit.jsonl"
    _write_test_log(log_path)
    entries = read_audit_log(log_path, event_filter="self-healing")
    assert len(entries) == 2
    assert all("self-healing" in str(e["event"]) for e in entries)


def test_read_since(tmp_path: Path) -> None:
    """--since filters by timestamp prefix."""
    log_path = tmp_path / "audit.jsonl"
    _write_test_log(log_path)
    entries = read_audit_log(log_path, since="2026-05-06T10:10")
    assert len(entries) == 3  # 10:10, 10:15, 10:20


def test_read_combined_filters(tmp_path: Path) -> None:
    """Multiple filters stack (AND logic)."""
    log_path = tmp_path / "audit.jsonl"
    _write_test_log(log_path)
    entries = read_audit_log(
        log_path,
        event_filter="backend-health",
        since="2026-05-06T10:15",
    )
    assert len(entries) == 1
    assert entries[0]["event"] == "backend-health-changed"
    assert entries[0]["provider"] == "p2"


def test_read_nonexistent_file(tmp_path: Path) -> None:
    """Reading a nonexistent file returns empty list."""
    entries = read_audit_log(tmp_path / "nonexistent.jsonl")
    assert entries == []


# ----------------------------------------------------------------------
# Group 4: Summarizer
# ----------------------------------------------------------------------


def test_summarize(tmp_path: Path) -> None:
    """summarize_audit_log counts events by type."""
    log_path = tmp_path / "audit.jsonl"
    _write_test_log(log_path)
    entries = read_audit_log(log_path)
    summary = summarize_audit_log(entries)
    assert summary["backend-health-changed"] == 2
    assert summary["self-healing-exclude"] == 1
    assert summary["drift-detected"] == 1
    assert summary["self-healing-restore"] == 1


# ----------------------------------------------------------------------
# Group 5: CLI audit command
# ----------------------------------------------------------------------


def test_cli_audit_no_file(tmp_path: Path) -> None:
    """coderouter audit with no log file returns exit 1."""
    from coderouter.cli import main

    exit_code = main(["audit", "--state-dir", str(tmp_path / "nonexistent")])
    assert exit_code == 1


def test_cli_audit_summary(tmp_path: Path) -> None:
    """coderouter audit --summary produces output."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_test_log(state_dir / "audit.jsonl")

    from coderouter.cli import main

    exit_code = main(["audit", "--state-dir", str(state_dir), "--summary"])
    assert exit_code == 0


def test_cli_audit_tail(tmp_path: Path) -> None:
    """coderouter audit --tail 2 shows last 2 entries."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_test_log(state_dir / "audit.jsonl")

    from coderouter.cli import main

    exit_code = main(["audit", "--state-dir", str(state_dir), "--tail", "2"])
    assert exit_code == 0
