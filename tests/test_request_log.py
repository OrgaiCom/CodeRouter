"""v2.0-K (Replay): RequestLog + Replay engine tests.

Test groups:

- **RequestLogHandler**: write cache-observed events, ignore others.
- **File rotation**: rotation when max_bytes is exceeded.
- **Reader**: read_request_log filtering (tail, provider_filter, since).
- **Replay summarize_window**: per-provider aggregation.
- **Replay compare_providers**: A/B comparison with deltas.
- **Replay formatting**: CLI table output.
- **CLI replay command**: argument parsing and output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from coderouter.state.replay import (
    compare_providers,
    format_comparison_table,
    format_summary_table,
    summarize_window,
)
from coderouter.state.request_log import (
    RequestLogHandler,
    read_request_log,
)

# ----------------------------------------------------------------------
# Group 1: RequestLogHandler — writes cache-observed events, ignores others
# ----------------------------------------------------------------------


def test_handler_writes_cache_observed(tmp_path: Path) -> None:
    """cache-observed events are written to the JSONL file."""
    log_path = tmp_path / "requests.jsonl"
    handler = RequestLogHandler(log_path)

    logger = logging.getLogger("test_reqlog_writes")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info(
        "cache-observed",
        extra={
            "provider": "anthropic-api",
            "input_tokens": 1000,
            "output_tokens": 200,
            "cost_usd": 0.005,
            "cost_savings_usd": 0.001,
            "streaming": True,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 100,
            "outcome": "cache_hit",
        },
    )

    handler.close()
    logger.removeHandler(handler)

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["provider"] == "anthropic-api"
    assert entry["input_tokens"] == 1000
    assert entry["output_tokens"] == 200
    assert entry["cost_usd"] == 0.005
    assert entry["streaming"] is True
    assert entry["cache_read_input_tokens"] == 500
    assert entry["outcome"] == "cache_hit"
    assert "ts" in entry


def test_handler_ignores_non_cache_observed(tmp_path: Path) -> None:
    """Non-cache-observed events are NOT written."""
    log_path = tmp_path / "requests.jsonl"
    handler = RequestLogHandler(log_path)

    logger = logging.getLogger("test_reqlog_ignores")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("try-provider", extra={"provider": "p1"})
    logger.info("provider-ok", extra={"provider": "p1"})
    logger.info("backend-health-changed", extra={"provider": "p1"})

    handler.close()
    logger.removeHandler(handler)

    if log_path.exists():
        assert log_path.read_text().strip() == ""


def test_handler_multiple_events(tmp_path: Path) -> None:
    """Multiple cache-observed events produce multiple JSONL lines."""
    log_path = tmp_path / "requests.jsonl"
    handler = RequestLogHandler(log_path)

    logger = logging.getLogger("test_reqlog_multi")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    for i in range(5):
        logger.info(
            "cache-observed",
            extra={"provider": f"p{i}", "input_tokens": i * 100},
        )

    handler.close()
    logger.removeHandler(handler)

    lines = [line for line in log_path.read_text().strip().split("\n") if line]
    assert len(lines) == 5


# ----------------------------------------------------------------------
# Group 2: File rotation
# ----------------------------------------------------------------------


def test_rotation_on_max_bytes(tmp_path: Path) -> None:
    """File is rotated when max_bytes is exceeded."""
    log_path = tmp_path / "requests.jsonl"
    handler = RequestLogHandler(log_path, max_bytes=200)

    logger = logging.getLogger("test_reqlog_rotation")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    for i in range(10):
        logger.info("cache-observed", extra={"provider": f"provider-{i}"})

    handler.close()
    logger.removeHandler(handler)

    backup = tmp_path / "requests.jsonl.1"
    assert backup.exists(), "Backup file should exist after rotation"
    assert log_path.exists(), "Active file should still exist"


# ----------------------------------------------------------------------
# Group 3: Reader — read_request_log filtering
# ----------------------------------------------------------------------


def _write_test_request_log(path: Path) -> None:
    """Write a small test request journal."""
    entries = [
        {"ts": "2026-05-01T10:00:00Z", "provider": "anthropic-api", "input_tokens": 1000, "output_tokens": 200, "cost_usd": 0.005},
        {"ts": "2026-05-01T11:00:00Z", "provider": "openrouter-free", "input_tokens": 800, "output_tokens": 150, "cost_usd": 0.0},
        {"ts": "2026-05-02T09:00:00Z", "provider": "anthropic-api", "input_tokens": 1200, "output_tokens": 300, "cost_usd": 0.008},
        {"ts": "2026-05-02T14:00:00Z", "provider": "openrouter-free", "input_tokens": 600, "output_tokens": 100, "cost_usd": 0.0},
        {"ts": "2026-05-03T08:00:00Z", "provider": "anthropic-api", "input_tokens": 900, "output_tokens": 250, "cost_usd": 0.006},
    ]
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_read_all(tmp_path: Path) -> None:
    """read_request_log with no filters returns all entries."""
    log_path = tmp_path / "requests.jsonl"
    _write_test_request_log(log_path)
    entries = read_request_log(log_path)
    assert len(entries) == 5


def test_read_tail(tmp_path: Path) -> None:
    """--tail N returns only the last N entries."""
    log_path = tmp_path / "requests.jsonl"
    _write_test_request_log(log_path)
    entries = read_request_log(log_path, tail=2)
    assert len(entries) == 2
    assert entries[0]["provider"] == "openrouter-free"
    assert entries[1]["provider"] == "anthropic-api"


def test_read_provider_filter(tmp_path: Path) -> None:
    """--provider filters by provider name (case-insensitive)."""
    log_path = tmp_path / "requests.jsonl"
    _write_test_request_log(log_path)
    entries = read_request_log(log_path, provider_filter="anthropic-api")
    assert len(entries) == 3
    assert all(e["provider"] == "anthropic-api" for e in entries)


def test_read_since(tmp_path: Path) -> None:
    """--since filters by timestamp prefix."""
    log_path = tmp_path / "requests.jsonl"
    _write_test_request_log(log_path)
    entries = read_request_log(log_path, since="2026-05-02")
    assert len(entries) == 3


def test_read_combined_filters(tmp_path: Path) -> None:
    """Multiple filters stack (AND logic)."""
    log_path = tmp_path / "requests.jsonl"
    _write_test_request_log(log_path)
    entries = read_request_log(
        log_path,
        provider_filter="anthropic-api",
        since="2026-05-02",
    )
    assert len(entries) == 2


def test_read_nonexistent_file(tmp_path: Path) -> None:
    """Reading a nonexistent file returns empty list."""
    entries = read_request_log(tmp_path / "nonexistent.jsonl")
    assert entries == []


# ----------------------------------------------------------------------
# Group 4: Replay — summarize_window
# ----------------------------------------------------------------------


def test_summarize_window(tmp_path: Path) -> None:
    """summarize_window aggregates per-provider statistics."""
    log_path = tmp_path / "requests.jsonl"
    _write_test_request_log(log_path)
    entries = read_request_log(log_path)
    summary = summarize_window(entries)

    assert summary.total_requests == 5
    assert len(summary.providers) == 2
    assert "anthropic-api" in summary.providers
    assert "openrouter-free" in summary.providers

    api = summary.providers["anthropic-api"]
    assert api.request_count == 3
    assert api.total_input_tokens == 3100  # 1000 + 1200 + 900
    assert api.total_output_tokens == 750  # 200 + 300 + 250
    assert api.total_cost_usd == pytest.approx(0.019)

    free = summary.providers["openrouter-free"]
    assert free.request_count == 2
    assert free.total_cost_usd == 0.0


def test_summarize_window_empty() -> None:
    """Empty entries produce a zeroed summary."""
    summary = summarize_window([])
    assert summary.total_requests == 0
    assert len(summary.providers) == 0


def test_summarize_window_averages() -> None:
    """Averages and ratios are computed correctly."""
    entries = [
        {"provider": "p1", "input_tokens": 100, "output_tokens": 50,
         "cost_usd": 0.01, "streaming": True, "cache_read_input_tokens": 30},
        {"provider": "p1", "input_tokens": 200, "output_tokens": 100,
         "cost_usd": 0.02, "streaming": False, "cache_read_input_tokens": 70},
    ]
    summary = summarize_window(entries)
    p1 = summary.providers["p1"]

    assert p1.avg_input_tokens == 150.0
    assert p1.avg_output_tokens == 75.0
    assert p1.avg_cost_usd == pytest.approx(0.015)
    assert p1.streaming_ratio == 0.5
    # cache_hit_ratio = 100 / (300 + 100)  [total_cache_read / (total_input + total_cache_read)]
    assert p1.cache_hit_ratio == pytest.approx(100 / 400)


# ----------------------------------------------------------------------
# Group 5: Replay — compare_providers
# ----------------------------------------------------------------------


def test_compare_providers(tmp_path: Path) -> None:
    """compare_providers produces correct deltas."""
    log_path = tmp_path / "requests.jsonl"
    _write_test_request_log(log_path)
    entries = read_request_log(log_path)

    comp = compare_providers(entries, "anthropic-api", "openrouter-free")
    assert comp.provider_a.request_count == 3
    assert comp.provider_b.request_count == 2
    assert comp.delta_total_cost_usd == pytest.approx(-0.019)  # 0.0 - 0.019


def test_compare_missing_provider() -> None:
    """Comparing with a nonexistent provider produces zero stats."""
    entries = [
        {"provider": "p1", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.01},
    ]
    comp = compare_providers(entries, "p1", "nonexistent")
    assert comp.provider_a.request_count == 1
    assert comp.provider_b.request_count == 0


# ----------------------------------------------------------------------
# Group 6: Replay — CLI table formatting
# ----------------------------------------------------------------------


def test_format_summary_table() -> None:
    """format_summary_table produces readable text output."""
    entries = [
        {"ts": "2026-05-01T10:00:00Z", "provider": "p1", "input_tokens": 1000,
         "output_tokens": 200, "cost_usd": 0.005, "streaming": True},
        {"ts": "2026-05-01T11:00:00Z", "provider": "p2", "input_tokens": 800,
         "output_tokens": 150, "cost_usd": 0.0, "streaming": False},
    ]
    summary = summarize_window(entries)
    table = format_summary_table(summary)

    assert "Window:" in table
    assert "p1" in table
    assert "p2" in table
    assert "Total:" in table


def test_format_comparison_table() -> None:
    """format_comparison_table produces readable text output."""
    entries = [
        {"provider": "p1", "input_tokens": 1000, "output_tokens": 200, "cost_usd": 0.01},
        {"provider": "p2", "input_tokens": 800, "output_tokens": 150, "cost_usd": 0.005},
    ]
    comp = compare_providers(entries, "p1", "p2")
    table = format_comparison_table(comp)

    assert "p1" in table
    assert "p2" in table
    assert "Requests" in table
    assert "Total cost" in table


# ----------------------------------------------------------------------
# Group 7: CLI replay command
# ----------------------------------------------------------------------


def test_cli_replay_no_file(tmp_path: Path) -> None:
    """coderouter replay with no journal file returns exit 1."""
    from coderouter.cli import main

    exit_code = main(["replay", "--state-dir", str(tmp_path / "nonexistent")])
    assert exit_code == 1


def test_cli_replay_summary(tmp_path: Path) -> None:
    """coderouter replay produces summary output."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_test_request_log(state_dir / "requests.jsonl")

    from coderouter.cli import main

    exit_code = main(["replay", "--state-dir", str(state_dir)])
    assert exit_code == 0


def test_cli_replay_compare(tmp_path: Path) -> None:
    """coderouter replay --compare produces comparison output."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_test_request_log(state_dir / "requests.jsonl")

    from coderouter.cli import main

    exit_code = main([
        "replay", "--state-dir", str(state_dir),
        "--compare", "anthropic-api", "openrouter-free",
    ])
    assert exit_code == 0


def test_cli_replay_with_limit(tmp_path: Path) -> None:
    """coderouter replay --limit 2 uses only last 2 entries."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_test_request_log(state_dir / "requests.jsonl")

    from coderouter.cli import main

    exit_code = main(["replay", "--state-dir", str(state_dir), "--limit", "2"])
    assert exit_code == 0


def test_cli_replay_direct_log(tmp_path: Path) -> None:
    """coderouter replay --log PATH reads from a direct path."""
    log_path = tmp_path / "custom.jsonl"
    _write_test_request_log(log_path)

    from coderouter.cli import main

    exit_code = main(["replay", "--log", str(log_path)])
    assert exit_code == 0
