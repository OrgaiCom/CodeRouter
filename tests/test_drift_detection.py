"""v2.0-G: Drift detection guard tests.

Pure-function tests for the drift detector — no network, no I/O.
Tests are grouped by signal, severity synthesis, and window management.
"""

from __future__ import annotations

from coderouter.guards.drift_detection import (
    THRESHOLDS_HIGH,
    THRESHOLDS_LOW,
    THRESHOLDS_NORMAL,
    DriftThresholds,
    DriftWindow,
    ResponseObservation,
    detect_drift,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(
    *,
    provider: str = "local",
    output_tokens: int = 100,
    has_tool_use: bool = False,
    request_had_tools: bool = False,
    stop_reason: str | None = "end_turn",
    is_error: bool = False,
    stream: bool = False,
) -> ResponseObservation:
    return ResponseObservation(
        provider=provider,
        output_tokens=output_tokens,
        has_tool_use=has_tool_use,
        request_had_tools=request_had_tools,
        stop_reason=stop_reason,
        is_error=is_error,
        stream=stream,
    )


def _window_of(n: int, **kwargs) -> list[ResponseObservation]:
    """Create a uniform window of N observations."""
    return [_obs(**kwargs) for _ in range(n)]


# ---------------------------------------------------------------------------
# Signal 1: Empty response rate
# ---------------------------------------------------------------------------


class TestEmptyResponseRate:
    def test_no_drift_when_all_normal(self):
        window = _window_of(10, output_tokens=100)
        v = detect_drift(window)
        assert not v.drifted
        assert v.severity == "none"

    def test_severe_when_many_empty(self):
        # 5 normal + 5 empty = 50% empty rate (> 0.3 threshold)
        window = _window_of(5, output_tokens=100) + _window_of(5, output_tokens=0)
        v = detect_drift(window)
        assert v.drifted
        assert v.severity == "severe"
        assert v.signals["empty_response_rate"] == 0.5

    def test_below_threshold_no_drift(self):
        # 8 normal + 2 empty = 20% (< 0.3)
        window = _window_of(8, output_tokens=100) + _window_of(2, output_tokens=0)
        v = detect_drift(window)
        # empty_response_rate alone doesn't trigger
        assert v.signals.get("empty_response_rate", 0) <= 0.3

    def test_errors_excluded_from_empty_rate(self):
        # 5 normal + 5 errors (output=0 but is_error=True)
        window = _window_of(5, output_tokens=100) + _window_of(5, output_tokens=0, is_error=True)
        v = detect_drift(window)
        # empty rate should be 0/5 = 0 (errors don't count as empty)
        assert v.signals.get("empty_response_rate", 0) == 0.0


# ---------------------------------------------------------------------------
# Signal 2: Length collapse
# ---------------------------------------------------------------------------


class TestLengthCollapse:
    def test_severe_when_length_halves(self):
        # Earlier: 200 tokens, Recent: 50 tokens → ratio 0.25 (< 0.5)
        earlier = _window_of(5, output_tokens=200)
        recent = _window_of(5, output_tokens=50)
        window = earlier + recent
        v = detect_drift(window)
        assert v.drifted
        assert v.severity == "severe"
        assert v.signals["length_collapse_ratio"] == 0.25

    def test_no_collapse_when_stable(self):
        window = _window_of(10, output_tokens=100)
        v = detect_drift(window)
        assert v.signals.get("length_collapse_ratio", 1.0) == 1.0

    def test_no_collapse_when_length_increases(self):
        earlier = _window_of(5, output_tokens=50)
        recent = _window_of(5, output_tokens=150)
        window = earlier + recent
        v = detect_drift(window)
        assert v.signals["length_collapse_ratio"] == 3.0
        assert not v.drifted

    def test_earlier_median_zero_no_crash(self):
        # All zeros in earlier half — division by zero guard
        earlier = _window_of(5, output_tokens=0)
        recent = _window_of(5, output_tokens=100)
        window = earlier + recent
        v = detect_drift(window)
        # Should not crash, length_collapse_ratio not computed
        assert "length_collapse_ratio" not in v.signals or v.signals["length_collapse_ratio"] >= 0


# ---------------------------------------------------------------------------
# Signal 3: Tool silence rate
# ---------------------------------------------------------------------------


class TestToolSilence:
    def test_mild_when_tools_go_silent(self):
        # All requests have tools, but responses don't use them
        window = _window_of(10, request_had_tools=True, has_tool_use=False, output_tokens=100)
        v = detect_drift(window)
        assert v.drifted
        assert "tool_silence_rate" in v.signals
        assert v.signals["tool_silence_rate"] == 1.0

    def test_no_signal_when_no_tools_in_request(self):
        # Requests don't have tools → tool silence not measured
        window = _window_of(10, request_had_tools=False, has_tool_use=False, output_tokens=100)
        v = detect_drift(window)
        assert "tool_silence_rate" not in v.signals

    def test_below_threshold(self):
        # 6 use tools, 4 don't → silence rate 0.4 (< 0.7)
        using = [_obs(request_had_tools=True, has_tool_use=True, output_tokens=100) for _ in range(6)]
        silent = [_obs(request_had_tools=True, has_tool_use=False, output_tokens=100) for _ in range(4)]
        window = using + silent
        v = detect_drift(window)
        assert v.signals["tool_silence_rate"] == 0.4
        # 0.4 < 0.7 threshold → not flagged
        assert "tool_silence" not in v.reason


# ---------------------------------------------------------------------------
# Signal 4: Stop reason anomaly
# ---------------------------------------------------------------------------


class TestStopReasonAnomaly:
    def test_mild_when_many_anomalous_stops(self):
        # 5 normal stops + 5 weird stops = 50% (> 0.4)
        normal = _window_of(5, stop_reason="end_turn", output_tokens=100)
        weird = _window_of(5, stop_reason="unknown_stop", output_tokens=100)
        window = normal + weird
        v = detect_drift(window)
        assert v.drifted
        assert v.signals["stop_anomaly_rate"] == 0.5

    def test_tool_use_stop_is_normal(self):
        window = _window_of(10, stop_reason="tool_use", output_tokens=100)
        v = detect_drift(window)
        assert v.signals.get("stop_anomaly_rate", 0) == 0.0

    def test_max_tokens_is_normal(self):
        window = _window_of(10, stop_reason="max_tokens", output_tokens=100)
        v = detect_drift(window)
        assert v.signals.get("stop_anomaly_rate", 0) == 0.0

    def test_none_stop_reason_is_anomalous(self):
        window = _window_of(10, stop_reason=None, output_tokens=100)
        v = detect_drift(window)
        assert v.signals["stop_anomaly_rate"] == 1.0


# ---------------------------------------------------------------------------
# Signal 5: Error rate
# ---------------------------------------------------------------------------


class TestErrorRate:
    def test_mild_when_errors_above_threshold(self):
        # 7 ok + 3 errors = 30% (> 0.25)
        ok = _window_of(7, output_tokens=100)
        err = _window_of(3, is_error=True, output_tokens=0)
        window = ok + err
        v = detect_drift(window)
        assert v.signals["error_rate"] == 0.3

    def test_below_threshold(self):
        # 9 ok + 1 error = 10% (< 0.25)
        ok = _window_of(9, output_tokens=100)
        err = _window_of(1, is_error=True, output_tokens=0)
        window = ok + err
        v = detect_drift(window)
        assert v.signals["error_rate"] == 0.1
        assert not v.drifted


# ---------------------------------------------------------------------------
# Severity synthesis
# ---------------------------------------------------------------------------


class TestSeveritySynthesis:
    def test_single_severe_signal(self):
        # Empty rate alone → severe
        window = _window_of(4, output_tokens=100) + _window_of(6, output_tokens=0)
        v = detect_drift(window)
        assert v.severity == "severe"

    def test_two_mild_signals_become_severe(self):
        # tool silence (mild) + stop anomaly (mild) → severe
        window = [
            _obs(
                output_tokens=100,
                request_had_tools=True,
                has_tool_use=False,
                stop_reason="weird",
            )
            for _ in range(10)
        ]
        v = detect_drift(window)
        assert v.severity == "severe"
        assert "tool_silence" in v.reason
        assert "stop_anomaly" in v.reason

    def test_single_mild_stays_mild(self):
        # Only error rate mild, everything else fine
        ok = _window_of(7, output_tokens=100)
        err = _window_of(3, is_error=True, output_tokens=0)
        window = ok + err
        v = detect_drift(window)
        # error_rate=0.3 > 0.25 → mild, but only 1 mild flag
        if v.drifted:
            assert v.severity == "mild"


# ---------------------------------------------------------------------------
# Window too small
# ---------------------------------------------------------------------------


class TestMinWindowFill:
    def test_no_detection_below_min_fill(self):
        # Only 3 observations (min_window_fill default = 6)
        window = _window_of(3, output_tokens=0)
        v = detect_drift(window)
        assert not v.drifted
        assert v.severity == "none"

    def test_detection_at_min_fill(self):
        window = _window_of(6, output_tokens=0)
        v = detect_drift(window)
        assert v.drifted

    def test_custom_min_fill(self):
        thresholds = DriftThresholds(min_window_fill=10)
        window = _window_of(8, output_tokens=0)
        v = detect_drift(window, thresholds=thresholds)
        assert not v.drifted  # 8 < 10


# ---------------------------------------------------------------------------
# Threshold presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_low_sensitivity_harder_to_trigger(self):
        # 40% empty → triggers normal (0.3) but not low (0.5)
        # Interleave empty responses so length_collapse doesn't also fire
        # (grouping all empties at the end causes recent_median=0 → collapse).
        window = [
            _obs(output_tokens=100),
            _obs(output_tokens=0),
            _obs(output_tokens=100),
            _obs(output_tokens=0),
            _obs(output_tokens=100),
            _obs(output_tokens=0),
            _obs(output_tokens=100),
            _obs(output_tokens=0),
            _obs(output_tokens=100),
            _obs(output_tokens=100),
        ]
        v_normal = detect_drift(window, THRESHOLDS_NORMAL)
        v_low = detect_drift(window, THRESHOLDS_LOW)
        assert v_normal.drifted
        assert not v_low.drifted

    def test_high_sensitivity_easier_to_trigger(self):
        # 25% empty → triggers high (0.2) but not normal (0.3)
        window = _window_of(9, output_tokens=100) + _window_of(3, output_tokens=0)
        v_high = detect_drift(window, THRESHOLDS_HIGH)
        v_normal = detect_drift(window, THRESHOLDS_NORMAL)
        # high threshold is 0.2, 3/12 = 0.25 > 0.2
        assert v_high.drifted
        assert not v_normal.drifted


# ---------------------------------------------------------------------------
# DriftWindow manager
# ---------------------------------------------------------------------------


class TestDriftWindow:
    def test_record_and_get(self):
        dw = DriftWindow(max_size=5)
        for i in range(3):
            dw.record(_obs(provider="p1", output_tokens=i * 10))
        window = dw.get_window("p1")
        assert len(window) == 3
        assert window[0].output_tokens == 0
        assert window[2].output_tokens == 20

    def test_max_size_evicts_oldest(self):
        dw = DriftWindow(max_size=3)
        for i in range(5):
            dw.record(_obs(provider="p1", output_tokens=i))
        window = dw.get_window("p1")
        assert len(window) == 3
        assert window[0].output_tokens == 2  # oldest surviving

    def test_providers_independent(self):
        dw = DriftWindow(max_size=10)
        dw.record(_obs(provider="p1", output_tokens=10))
        dw.record(_obs(provider="p2", output_tokens=20))
        assert len(dw.get_window("p1")) == 1
        assert len(dw.get_window("p2")) == 1

    def test_clear_single_provider(self):
        dw = DriftWindow(max_size=10)
        dw.record(_obs(provider="p1", output_tokens=10))
        dw.record(_obs(provider="p2", output_tokens=20))
        dw.clear("p1")
        assert len(dw.get_window("p1")) == 0
        assert len(dw.get_window("p2")) == 1

    def test_clear_all(self):
        dw = DriftWindow(max_size=10)
        dw.record(_obs(provider="p1", output_tokens=10))
        dw.record(_obs(provider="p2", output_tokens=20))
        dw.clear_all()
        assert len(dw) == 0

    def test_get_window_unknown_provider(self):
        dw = DriftWindow(max_size=10)
        assert dw.get_window("unknown") == []

    def test_len_total(self):
        dw = DriftWindow(max_size=10)
        dw.record(_obs(provider="p1", output_tokens=10))
        dw.record(_obs(provider="p1", output_tokens=20))
        dw.record(_obs(provider="p2", output_tokens=30))
        assert len(dw) == 3
