"""Regression tests for the M-batch logging / metrics fixes.

Coverage:

- **M4**: ``configure_logging`` called twice in one process must not
  detach an installed :class:`MetricsCollector` — metrics keep flowing
  after the second call. Also covers the ``install_collector`` re-attach
  path when the collector is orphaned from the root logger.
- **M5**: ``MetricsCollector.save_state`` → ``load_state`` round-trip
  preserves every persisted key, including the v2.6 language-tax pair.
  Guards against future save/load key-set drift.
- **M12(1)**: unrecognized log events take the lock-free early-exit path
  (``_KNOWN_EVENTS`` membership) and never mutate collector state; the
  dispatch table stays in lockstep with ``_KNOWN_EVENTS``. Behavioral
  equivalence of the recognized-event path is covered by the existing
  ``test_metrics_*`` suite.
- **M12(2)**: buffered audit / request-log handlers flush on ``close()``
  so no buffered line is lost on clean shutdown; ``flush_every_n=1``
  restores immediate write-through.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from coderouter.logging import (
    _CODEROUTER_LOG_HANDLER_MARKER,
    configure_logging,
)
from coderouter.metrics.collector import (
    _KNOWN_EVENTS,
    MetricsCollector,
    install_collector,
    uninstall_collector,
)
from coderouter.state.audit_log import AuditLogHandler
from coderouter.state.request_log import RequestLogHandler

# ----------------------------------------------------------------------
# M4: configure_logging must not orphan the collector
# ----------------------------------------------------------------------


def test_m4_collector_survives_second_configure_logging() -> None:
    """A collector installed once keeps receiving events after a second
    ``configure_logging`` call (simulates a 2nd ``create_app`` in-process)."""
    uninstall_collector()
    try:
        configure_logging()
        collector = install_collector()
        collector.reset()

        # First app path: an event is counted.
        logger = logging.getLogger("test_m4")
        logger.setLevel(logging.DEBUG)
        logger.info("try-provider", extra={"provider": "p1"})
        assert collector.snapshot()["counters"]["requests_total"] == 1

        # Second create_app() re-runs configure_logging. Pre-M4 this
        # removed *all* root handlers, silently detaching the collector.
        configure_logging()
        # install_collector is idempotent + re-attaches if orphaned.
        again = install_collector()
        assert again is collector

        # The collector must still be tapping the root logger.
        root = logging.getLogger()
        assert collector in root.handlers

        logger.info("try-provider", extra={"provider": "p2"})
        assert collector.snapshot()["counters"]["requests_total"] == 2
    finally:
        uninstall_collector()


def test_m4_configure_logging_only_removes_own_handler() -> None:
    """configure_logging removes only its marked handler, leaving foreign
    handlers (e.g. the collector) attached."""
    uninstall_collector()
    try:
        configure_logging()
        collector = install_collector()
        root = logging.getLogger()

        marked = [
            h
            for h in root.handlers
            if getattr(h, _CODEROUTER_LOG_HANDLER_MARKER, False)
        ]
        assert len(marked) == 1

        configure_logging()

        # Exactly one marked handler again (old one swapped for new).
        marked = [
            h
            for h in root.handlers
            if getattr(h, _CODEROUTER_LOG_HANDLER_MARKER, False)
        ]
        assert len(marked) == 1
        # The collector (unmarked, foreign) is untouched.
        assert collector in root.handlers
    finally:
        uninstall_collector()


def test_m4_install_collector_reattaches_when_orphaned() -> None:
    """If the collector is detached from the root logger but the singleton
    still exists, install_collector re-attaches the same instance."""
    uninstall_collector()
    try:
        collector = install_collector()
        root = logging.getLogger()
        assert collector in root.handlers

        # Simulate a manual clear of the root handlers.
        root.removeHandler(collector)
        assert collector not in root.handlers

        again = install_collector()
        assert again is collector
        assert collector in root.handlers
    finally:
        uninstall_collector()


# ----------------------------------------------------------------------
# M5: save_state / load_state round-trip preserves every key
# ----------------------------------------------------------------------


def _seed_collector(c: MetricsCollector) -> None:
    """Drive a variety of events so every persisted counter is non-zero."""
    logger = logging.getLogger("test_m5_seed")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(c)
    try:
        logger.info("try-provider", extra={"provider": "p1"})
        logger.info("provider-ok", extra={"provider": "p1"})
        logger.info(
            "cache-observed",
            extra={
                "provider": "p1",
                "outcome": "cache_hit",
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 10,
                "input_tokens": 500,
                "output_tokens": 50,
                "cost_usd": 0.01,
                "cost_savings_usd": 0.002,
                "language_tax_usd": 0.003,
            },
        )
        logger.warning("chain-paid-gate-blocked", extra={})
        logger.warning("chain-budget-exceeded", extra={})
        logger.warning("chain-memory-pressure-blocked", extra={})
        logger.warning("chain-uniform-auth-failure", extra={})
        logger.info(
            "context-budget-trimmed",
            extra={
                "profile": "default",
                "estimated_tokens_before": 1000,
                "estimated_tokens_after": 600,
            },
        )
        logger.info("probe-round-completed", extra={})
    finally:
        logger.removeHandler(c)


def test_m5_save_load_roundtrip_all_keys() -> None:
    """Every key produced by save_state is restored by load_state, and the
    v2.6 language-tax pair (previously missing from save_state) is present."""
    src = MetricsCollector()
    _seed_collector(src)
    state = src.save_state()

    # Guard against future save/load key-set drift: the language-tax keys
    # read by load_state must be emitted by save_state.
    assert "language_tax_usd" in state
    assert "language_tax_usd_aggregate" in state
    assert state["language_tax_usd"] == {"p1": 0.003}
    assert state["language_tax_usd_aggregate"] == 0.003

    # Round-trip into a fresh collector; snapshots of the persisted subset
    # must match.
    dst = MetricsCollector()
    dst.load_state(state)

    src_snap = src.snapshot()["counters"]
    dst_snap = dst.snapshot()["counters"]

    # The keys that save_state persists must round-trip identically.
    persisted_keys = [
        "requests_total",
        "provider_attempts",
        "provider_outcomes",
        "cost_total_usd",
        "cost_savings_usd",
        "cost_total_usd_aggregate",
        "cost_savings_usd_aggregate",
        "language_tax_usd",
        "language_tax_usd_aggregate",
        "chain_paid_gate_blocked_total",
        "chain_budget_exceeded_total",
        "chain_memory_pressure_blocked_total",
        "chain_uniform_auth_failure_total",
        "tokens_saved_total",
        "tokens_saved_by_mechanism",
        "probe_rounds_total",
    ]
    for key in persisted_keys:
        assert dst_snap[key] == src_snap[key], f"key mismatch: {key}"

    # Specifically confirm the M5 fix: language-tax aggregate survived
    # the restart round-trip instead of zero-resetting.
    assert dst._language_tax_usd_aggregate == 0.003
    assert dst._language_tax_usd == {"p1": 0.003}


def test_m5_load_state_reads_every_key_save_state_writes() -> None:
    """load_state consumes every key save_state emits (no orphan keys that
    would silently drop on restart)."""
    src = MetricsCollector()
    _seed_collector(src)
    state = src.save_state()

    # Apply the full saved dict and a second fresh collector fed the same
    # events, then compare the persisted subset. If load_state ignored a
    # key that save_state wrote, the restored value would diverge from the
    # live-fed value.
    restored = MetricsCollector()
    restored.load_state(state)

    live = MetricsCollector()
    _seed_collector(live)

    for key, value in state.items():
        # Both restored and live must reflect this key. We compare the
        # relevant private attribute via save_state again — the restored
        # collector's re-export must equal the original.
        assert restored.save_state()[key] == value, f"lost on load: {key}"


# ----------------------------------------------------------------------
# M12(1): early-exit + dispatch-table lockstep
# ----------------------------------------------------------------------


def test_m12_dispatch_table_matches_known_events() -> None:
    """The per-instance dispatch table and _KNOWN_EVENTS never drift."""
    c = MetricsCollector()
    assert set(c._dispatch_table) == set(_KNOWN_EVENTS)


def test_m12_unknown_event_is_early_exit_noop() -> None:
    """An unrecognized event mutates no state (and takes the lock-free
    early-exit path in _dispatch)."""
    c = MetricsCollector()
    before = c.snapshot()

    rec = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="this-is-not-a-metrics-event",
        args=(),
        exc_info=None,
    )
    rec.provider = "p1"  # would be counted if this were a known event
    c.emit(rec)

    after = c.snapshot()
    # Only volatile timing fields (uptime) may differ; the counters and
    # provider rows must be byte-identical.
    assert after["counters"] == before["counters"]
    assert after["providers"] == before["providers"]
    assert after["recent"] == before["recent"]


def test_m12_non_string_msg_is_ignored() -> None:
    """A record whose msg is not a str is ignored without error."""
    c = MetricsCollector()
    rec = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg={"not": "a string"},
        args=(),
        exc_info=None,
    )
    c.emit(rec)  # must not raise
    assert c.snapshot()["counters"]["requests_total"] == 0


def test_m12_known_event_still_counted() -> None:
    """Recognized events still flow through the dispatch table correctly."""
    c = MetricsCollector()
    logger = logging.getLogger("test_m12_known")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(c)
    try:
        logger.info("try-provider", extra={"provider": "p1"})
        logger.info("provider-ok", extra={"provider": "p1"})
    finally:
        logger.removeHandler(c)
    snap = c.snapshot()
    assert snap["counters"]["requests_total"] == 1
    assert snap["counters"]["provider_outcomes"]["p1"]["ok"] == 1


# ----------------------------------------------------------------------
# M12(2): buffered handlers flush on close; flush_every_n=1 is immediate
# ----------------------------------------------------------------------


def test_m12_audit_buffer_flushed_on_close(tmp_path: Path) -> None:
    """Buffered audit lines (default flush_every_n) are written on close()."""
    log_path = tmp_path / "audit.jsonl"
    # Large N + interval so nothing flushes mid-run; only close() flushes.
    handler = AuditLogHandler(
        log_path, flush_every_n=1000, flush_interval_s=3600.0
    )
    logger = logging.getLogger("test_m12_audit_buffer")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.warning("self-healing-exclude", extra={"provider": "p1"})
        logger.info("drift-detected", extra={"provider": "p2"})
    finally:
        logger.removeHandler(handler)

    # Before close, the buffer holds the lines (nothing flushed yet).
    assert not log_path.exists() or log_path.read_text().strip() == ""

    handler.close()

    lines = [ln for ln in log_path.read_text().strip().split("\n") if ln]
    assert len(lines) == 2
    events = {json.loads(ln)["event"] for ln in lines}
    assert events == {"self-healing-exclude", "drift-detected"}


def test_m12_audit_flush_every_n_one_is_immediate(tmp_path: Path) -> None:
    """flush_every_n=1 restores pre-M12 write-through behavior."""
    log_path = tmp_path / "audit.jsonl"
    handler = AuditLogHandler(log_path, flush_every_n=1)
    logger = logging.getLogger("test_m12_audit_immediate")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.warning("self-healing-exclude", extra={"provider": "p1"})
        # Visible on disk immediately, before close().
        lines = [
            ln for ln in log_path.read_text().strip().split("\n") if ln
        ]
        assert len(lines) == 1
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_m12_request_log_buffer_flushed_on_close(tmp_path: Path) -> None:
    """Buffered request-log lines are written on close()."""
    log_path = tmp_path / "requests.jsonl"
    handler = RequestLogHandler(
        log_path, flush_every_n=1000, flush_interval_s=3600.0
    )
    logger = logging.getLogger("test_m12_reqlog_buffer")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        for i in range(3):
            logger.info(
                "cache-observed",
                extra={"provider": f"p{i}", "input_tokens": i * 10},
            )
    finally:
        logger.removeHandler(handler)

    assert not log_path.exists() or log_path.read_text().strip() == ""

    handler.close()

    lines = [ln for ln in log_path.read_text().strip().split("\n") if ln]
    assert len(lines) == 3


def test_m12_request_log_flush_every_n_one_is_immediate(
    tmp_path: Path,
) -> None:
    """flush_every_n=1 restores pre-M12 write-through behavior."""
    log_path = tmp_path / "requests.jsonl"
    handler = RequestLogHandler(log_path, flush_every_n=1)
    logger = logging.getLogger("test_m12_reqlog_immediate")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.info("cache-observed", extra={"provider": "p1"})
        lines = [
            ln for ln in log_path.read_text().strip().split("\n") if ln
        ]
        assert len(lines) == 1
    finally:
        logger.removeHandler(handler)
        handler.close()
