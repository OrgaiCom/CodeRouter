"""Regression tests for H1: drift counter exposition (:mod:`coderouter.metrics`).

The drift promote/reload counters are unlabeled scalars. A malformed
label shape ``(((),), value)`` (a one-element tuple *containing* an empty
tuple) used to slip through ``_counter`` and blow up in ``_fmt_labels``
with a ``ValueError`` at ``for k, v in pairs`` — turning ``GET /metrics``
into a 500 whenever any drift counter went non-zero. The correct shape is
an empty labels tuple ``((), value)`` (see ``partial_stitch_surfaced``).

These tests feed canned snapshots with non-zero drift counters and assert
the formatter succeeds and emits the expected unlabeled counter lines.
"""

from __future__ import annotations

from typing import Any

from coderouter.metrics import format_prometheus


def _snapshot_with_counters(extra: dict[str, Any]) -> dict[str, Any]:
    """Minimal snapshot shape with ``extra`` merged into ``counters``."""
    counters: dict[str, Any] = {
        "requests_total": 0,
        "provider_attempts": {},
        "provider_outcomes": {},
    }
    counters.update(extra)
    return {
        "uptime_s": 0.0,
        "started_at": "1970-01-01T00:00:00",
        "startup": {},
        "counters": counters,
        "providers": [],
        "recent": [],
    }


# ---------------------------------------------------------------------------
# H1: non-zero drift counters must not crash exposition
# ---------------------------------------------------------------------------


def test_drift_promoted_nonzero_emits_unlabeled_sample() -> None:
    out = format_prometheus(
        _snapshot_with_counters({"drift_promoted_total": 3})
    )
    assert "# TYPE coderouter_drift_promoted_total counter" in out
    # Unlabeled scalar: name directly followed by a space + value, no ``{}``.
    assert "coderouter_drift_promoted_total 3" in out
    assert "coderouter_drift_promoted_total{" not in out


def test_drift_reload_nonzero_emits_unlabeled_sample() -> None:
    out = format_prometheus(
        _snapshot_with_counters({"drift_reload_total": 5})
    )
    assert "# TYPE coderouter_drift_reload_total counter" in out
    assert "coderouter_drift_reload_total 5" in out
    assert "coderouter_drift_reload_total{" not in out


def test_drift_reload_success_nonzero_emits_unlabeled_sample() -> None:
    out = format_prometheus(
        _snapshot_with_counters({"drift_reload_success_total": 7})
    )
    assert "# TYPE coderouter_drift_reload_success_total counter" in out
    assert "coderouter_drift_reload_success_total 7" in out
    assert "coderouter_drift_reload_success_total{" not in out


def test_all_drift_counters_nonzero_render_without_error() -> None:
    """The failure mode was a raised ValueError, not a wrong string."""
    out = format_prometheus(
        _snapshot_with_counters(
            {
                "drift_promoted_total": 1,
                "drift_reload_total": 2,
                "drift_reload_success_total": 1,
            }
        )
    )
    assert "coderouter_drift_promoted_total 1" in out
    assert "coderouter_drift_reload_total 2" in out
    assert "coderouter_drift_reload_success_total 1" in out
    assert out.endswith("\n")


def test_zero_drift_counters_are_omitted() -> None:
    """Zero-valued drift counters produce no sample line (guarded by ``if``)."""
    out = format_prometheus(
        _snapshot_with_counters(
            {
                "drift_promoted_total": 0,
                "drift_reload_total": 0,
                "drift_reload_success_total": 0,
            }
        )
    )
    assert "coderouter_drift_promoted_total 0" not in out
    assert "coderouter_drift_reload_total 0" not in out
    assert "coderouter_drift_reload_success_total 0" not in out
