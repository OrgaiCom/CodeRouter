"""Drift detection guard (v2.0-G, L4).

Detects gradual quality degradation in model responses during long-running
agent sessions. Unlike L5 (binary crash) and adaptive routing (latency),
drift detection targets the "succeeds but quality is decaying" pattern:
empty responses, shrinking output length, tool-call silence, anomalous
stop reasons.

Architecture
============

Three layers:

  1. **Observation model** — :class:`ResponseObservation` captures the
     quality-relevant fields from each successful provider response.
  2. **Detector** — :func:`detect_drift` is a pure function that takes
     a window of observations and thresholds, returns a
     :class:`DriftVerdict`.
  3. **Window manager** — :class:`DriftWindow` maintains per-provider
     rolling deques of observations, thread-safe for the async engine.

The engine calls :meth:`DriftWindow.record` after each provider-ok/failed
event, then calls :func:`detect_drift` to check whether corrective action
is needed.

Signals
=======

  * ``empty_response_rate`` — fraction of responses with output_tokens == 0
  * ``length_collapse`` — median output_tokens in the recent half vs. the
    earlier half of the window; ratio below threshold = collapse
  * ``tool_silence_rate`` — fraction of responses missing tool_use blocks
    (only meaningful when the request contained tools)
  * ``stop_anomaly_rate`` — fraction of responses with unexpected stop_reason
    (not "end_turn" / "tool_use" / "max_tokens")
  * ``error_rate`` — fraction of attempts that ended in failure
  * ``goal_progress_stall`` (P1-4) — fraction of fingerprinted responses
    whose fingerprint matches a previously-seen fingerprint in the window,
    indicating the model is repeating itself without making progress.
    Only fires when ``response_fingerprint`` is populated on observations.

Thresholds are bundled as :class:`DriftThresholds` with three presets
(``low`` / ``normal`` / ``high`` sensitivity).
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResponseObservation:
    """Quality-relevant snapshot of a single provider response.

    Captured post-response (after adapter translation), before returning
    to the client. Fields are intentionally minimal — only what the
    detector needs.
    """

    provider: str
    output_tokens: int
    has_tool_use: bool
    """Whether the response contained at least one tool_use block."""
    request_had_tools: bool
    """Whether the request included a tools[] array (context for tool_silence)."""
    stop_reason: str | None
    """Anthropic stop_reason: 'end_turn' / 'tool_use' / 'max_tokens' / None."""
    is_error: bool = False
    """True if the attempt ended in provider-failed / provider-failed-midstream."""
    stream: bool = False
    response_fingerprint: str | None = None
    """P1-4: compact content fingerprint of the response text.

    When set, used by the ``goal_progress_stall`` signal to detect
    repetition: the same fingerprint appearing multiple times in the
    window indicates the model is not making progress. Computed by
    :func:`coderouter.guards._fingerprint.fingerprint_response`.
    Pass ``None`` (default) to opt-out — the signal is silently skipped.
    """


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriftThresholds:
    """Threshold set for drift detection.

    Each field is the value above which (or below which for length_collapse)
    the corresponding signal is considered anomalous.
    """

    # Rate thresholds (signal > threshold → anomaly)
    empty_response_rate: float = 0.3
    """Fraction of responses with output_tokens == 0 to trigger."""
    stop_anomaly_rate: float = 0.4
    """Fraction of responses with unexpected stop_reason."""
    error_rate: float = 0.25
    """Fraction of failed attempts."""
    tool_silence_rate: float = 0.7
    """Fraction of tool-eligible responses missing tool_use."""

    # Ratio threshold (recent_median / earlier_median < threshold → collapse)
    length_collapse_ratio: float = 0.5
    """If recent half median is < 50% of earlier half median → collapse."""

    # P1-4: repetition/stall threshold
    repetition_rate_threshold: float = 0.4
    """P1-4: fraction of fingerprinted responses whose fingerprint has
    appeared before in the window. Above this rate → goal_progress_stall
    signal fires (mild). Default 0.4 = 2 out of 5 responses are repeats."""

    # Minimum observations before detection fires
    min_window_fill: int = 6
    """Don't trigger until at least this many observations in the window."""


# Presets
THRESHOLDS_LOW = DriftThresholds(
    empty_response_rate=0.5,
    length_collapse_ratio=0.3,
    tool_silence_rate=0.8,
    stop_anomaly_rate=0.6,
    error_rate=0.4,
    repetition_rate_threshold=0.6,
    min_window_fill=10,
)

THRESHOLDS_NORMAL = DriftThresholds()  # defaults

THRESHOLDS_HIGH = DriftThresholds(
    empty_response_rate=0.2,
    length_collapse_ratio=0.7,
    tool_silence_rate=0.5,
    stop_anomaly_rate=0.3,
    error_rate=0.15,
    repetition_rate_threshold=0.25,
    min_window_fill=4,
)

# P1-5: goal-mode preset — tighter thresholds + lower min_window_fill.
# Applied automatically when the profile has goal_mode=True.
THRESHOLDS_GOAL = DriftThresholds(
    empty_response_rate=0.2,
    length_collapse_ratio=0.6,
    tool_silence_rate=0.5,
    stop_anomaly_rate=0.3,
    error_rate=0.15,
    repetition_rate_threshold=0.2,
    min_window_fill=4,
)

SENSITIVITY_PRESETS: dict[str, DriftThresholds] = {
    "low": THRESHOLDS_LOW,
    "normal": THRESHOLDS_NORMAL,
    "high": THRESHOLDS_HIGH,
    "goal": THRESHOLDS_GOAL,
}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriftVerdict:
    """Result of drift detection for a single provider window."""

    drifted: bool
    severity: Literal["none", "mild", "severe"]
    signals: dict[str, float]
    """Signal name → computed value for observability."""
    reason: str
    """Human-readable explanation of why drift was detected (empty if none)."""


_NO_DRIFT = DriftVerdict(drifted=False, severity="none", signals={}, reason="")


# ---------------------------------------------------------------------------
# Detector (pure function)
# ---------------------------------------------------------------------------


def detect_drift(
    window: list[ResponseObservation],
    thresholds: DriftThresholds | None = None,
) -> DriftVerdict:
    """Analyze a window of observations and return a drift verdict.

    Pure function — no I/O, no side effects. Safe to call from any context.

    Parameters
    ----------
    window:
        List of recent :class:`ResponseObservation` for a single provider,
        ordered oldest-first.
    thresholds:
        Detection thresholds. Defaults to ``THRESHOLDS_NORMAL``.

    Returns
    -------
    DriftVerdict with severity ``none`` / ``mild`` / ``severe``.
    """
    if thresholds is None:
        thresholds = THRESHOLDS_NORMAL

    if len(window) < thresholds.min_window_fill:
        return _NO_DRIFT

    signals: dict[str, float] = {}
    mild_flags: list[str] = []
    severe_flags: list[str] = []

    total = len(window)

    # --- Signal 1: Empty response rate ---
    empty_count = sum(1 for obs in window if obs.output_tokens == 0 and not obs.is_error)
    non_error_count = sum(1 for obs in window if not obs.is_error)
    if non_error_count > 0:
        empty_rate = empty_count / non_error_count
        signals["empty_response_rate"] = round(empty_rate, 3)
        if empty_rate > thresholds.empty_response_rate:
            severe_flags.append(f"empty_response_rate={empty_rate:.2f}")

    # --- Signal 2: Length collapse (median comparison) ---
    non_error_lengths = [obs.output_tokens for obs in window if not obs.is_error]
    if len(non_error_lengths) >= 4:
        mid = len(non_error_lengths) // 2
        earlier_half = non_error_lengths[:mid]
        recent_half = non_error_lengths[mid:]
        earlier_median = statistics.median(earlier_half)
        recent_median = statistics.median(recent_half)
        if earlier_median > 0:
            collapse_ratio = recent_median / earlier_median
            signals["length_collapse_ratio"] = round(collapse_ratio, 3)
            if collapse_ratio < thresholds.length_collapse_ratio:
                severe_flags.append(
                    f"length_collapse={collapse_ratio:.2f}"
                    f" (recent_median={recent_median:.0f}, earlier={earlier_median:.0f})"
                )

    # --- Signal 3: Tool silence rate ---
    tool_eligible = [obs for obs in window if obs.request_had_tools and not obs.is_error]
    if len(tool_eligible) >= 3:
        tool_silent_count = sum(1 for obs in tool_eligible if not obs.has_tool_use)
        tool_silence_rate = tool_silent_count / len(tool_eligible)
        signals["tool_silence_rate"] = round(tool_silence_rate, 3)
        if tool_silence_rate > thresholds.tool_silence_rate:
            mild_flags.append(f"tool_silence_rate={tool_silence_rate:.2f}")

    # --- Signal 4: Stop reason anomaly rate ---
    # H-6: `stop_sequence` (a normal, expected terminator when the caller
    # configures a stop sequence) was missing here, so every stop-sequence
    # terminated response counted as an anomaly. `pause_turn` and `refusal`
    # are legitimate Anthropic stop reasons too (see translation/anthropic.py
    # for the forward-compat rationale) and are expected, not anomalous.
    _EXPECTED_STOP = {
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "pause_turn",
        "refusal",
    }
    non_error_obs = [obs for obs in window if not obs.is_error]
    if non_error_obs:
        anomaly_count = sum(
            1 for obs in non_error_obs if obs.stop_reason not in _EXPECTED_STOP
        )
        stop_anomaly_rate = anomaly_count / len(non_error_obs)
        signals["stop_anomaly_rate"] = round(stop_anomaly_rate, 3)
        if stop_anomaly_rate > thresholds.stop_anomaly_rate:
            mild_flags.append(f"stop_anomaly_rate={stop_anomaly_rate:.2f}")

    # --- Signal 5: Error rate ---
    error_count = sum(1 for obs in window if obs.is_error)
    error_rate = error_count / total
    signals["error_rate"] = round(error_rate, 3)
    if error_rate > thresholds.error_rate:
        mild_flags.append(f"error_rate={error_rate:.2f}")

    # --- Signal 6: Goal progress stall (P1-4) ---
    # Only active when at least some observations have a fingerprint.
    # Computes: how many fingerprinted responses repeat a fingerprint
    # already seen earlier in the window.  High repetition → stall.
    fingerprinted = [
        obs for obs in window if obs.response_fingerprint  # excludes None and ""
    ]
    if len(fingerprinted) >= 3:
        seen: set[str] = set()
        repeat_count = 0
        for obs in fingerprinted:
            fp = obs.response_fingerprint  # guaranteed non-empty by filter above
            if fp in seen:
                repeat_count += 1
            else:
                seen.add(fp)
        repetition_rate = repeat_count / len(fingerprinted)
        signals["goal_progress_stall"] = round(repetition_rate, 3)
        if repetition_rate > thresholds.repetition_rate_threshold:
            mild_flags.append(f"goal_progress_stall={repetition_rate:.2f}")

    # --- Severity synthesis ---
    if severe_flags:
        severity: Literal["none", "mild", "severe"] = "severe"
    elif len(mild_flags) >= 2:
        severity = "severe"
    elif mild_flags:
        severity = "mild"
    else:
        return DriftVerdict(drifted=False, severity="none", signals=signals, reason="")

    reason_parts = severe_flags + mild_flags
    return DriftVerdict(
        drifted=True,
        severity=severity,
        signals=signals,
        reason=", ".join(reason_parts),
    )


# ---------------------------------------------------------------------------
# Window manager
# ---------------------------------------------------------------------------


@dataclass
class DriftWindow:
    """Per-provider rolling window of response observations.

    Thread-safe for the single-threaded async event loop (no locking needed
    since all access happens on the same asyncio loop). If CodeRouter ever
    goes multi-threaded, add a Lock.
    """

    max_size: int = 20
    _windows: dict[str, deque[ResponseObservation]] = field(default_factory=dict)

    def record(self, obs: ResponseObservation) -> None:
        """Append an observation to the provider's window."""
        dq = self._windows.get(obs.provider)
        if dq is None:
            dq = deque(maxlen=self.max_size)
            self._windows[obs.provider] = dq
        dq.append(obs)

    def get_window(self, provider: str) -> list[ResponseObservation]:
        """Return a snapshot of the provider's window (oldest-first)."""
        dq = self._windows.get(provider)
        if dq is None:
            return []
        return list(dq)

    def clear(self, provider: str) -> None:
        """Clear a provider's window (e.g. after recovery)."""
        self._windows.pop(provider, None)

    def clear_all(self) -> None:
        """Reset all windows."""
        self._windows.clear()

    def __len__(self) -> int:
        """Total observations across all providers."""
        return sum(len(dq) for dq in self._windows.values())
