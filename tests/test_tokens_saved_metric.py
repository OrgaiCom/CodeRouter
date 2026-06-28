"""Token-savings accounting in the MetricsCollector.

Covers the two feeds (trim via ``context-budget-trimmed`` and the neutral
``tokens-saved`` event used by the compress plugin), negative-delta
clamping, snapshot shape, persistence round-trip, and reset.
"""
from __future__ import annotations

import logging

from coderouter.metrics.collector import MetricsCollector


def _rec(event: str, **extra: object) -> logging.LogRecord:
    r = logging.makeLogRecord({"msg": event})
    r.__dict__.update(extra)
    return r


def test_trim_event_records_token_savings() -> None:
    c = MetricsCollector()
    c.emit(
        _rec(
            "context-budget-trimmed",
            profile="p1",
            estimated_tokens_before=1000,
            estimated_tokens_after=600,
        )
    )
    counters = c.snapshot()["counters"]
    assert counters["tokens_saved_total"] == 400
    assert counters["tokens_saved_by_mechanism"] == {"trim": 400}


def test_tokens_saved_event_compress_and_clamp() -> None:
    c = MetricsCollector()
    c.emit(_rec("tokens-saved", mechanism="compress", tokens_saved=250))
    c.emit(_rec("tokens-saved", mechanism="compress", tokens_before=100, tokens_after=70))
    c.emit(_rec("tokens-saved", mechanism="compress", tokens_saved=-5))  # clamps to 0
    counters = c.snapshot()["counters"]
    assert counters["tokens_saved_total"] == 280
    assert counters["tokens_saved_by_mechanism"] == {"compress": 280}


def test_trim_and_compress_aggregate_together() -> None:
    c = MetricsCollector()
    c.emit(
        _rec(
            "context-budget-trimmed",
            profile="p1",
            estimated_tokens_before=1000,
            estimated_tokens_after=600,
        )
    )
    c.emit(_rec("tokens-saved", mechanism="compress", tokens_saved=280))
    counters = c.snapshot()["counters"]
    assert counters["tokens_saved_total"] == 680
    assert counters["tokens_saved_by_mechanism"] == {"trim": 400, "compress": 280}


def test_save_load_round_trip() -> None:
    c = MetricsCollector()
    c.emit(_rec("tokens-saved", mechanism="compress", tokens_saved=42))
    c.emit(
        _rec(
            "context-budget-trimmed",
            profile="p",
            estimated_tokens_before=100,
            estimated_tokens_after=90,
        )
    )
    state = c.save_state()

    restored = MetricsCollector()
    restored.load_state(state)
    counters = restored.snapshot()["counters"]
    assert counters["tokens_saved_total"] == 52
    assert counters["tokens_saved_by_mechanism"] == {"compress": 42, "trim": 10}


def test_reset_clears_token_savings() -> None:
    c = MetricsCollector()
    c.emit(_rec("tokens-saved", mechanism="compress", tokens_saved=5))
    c.reset()
    counters = c.snapshot()["counters"]
    assert counters["tokens_saved_total"] == 0
    assert counters["tokens_saved_by_mechanism"] == {}
