"""Unit tests for v2.0-I continuous probe (coderouter.guards.continuous_probe).

Coverage:
    - probe_one() success/failure/timeout (httpx mock)
    - probe_loop() shutdown, interval timing, record_fn integration
    - check_probe_drift() model mismatch / match / no model
    - MetricsCollector dispatch for probe-completed / probe-round-completed
    - log functions emit correct event names
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from coderouter.guards.continuous_probe import (
    DriftReport,
    ProbeResult,
    check_probe_drift,
    probe_loop,
    probe_one,
)
from coderouter.logging import (
    log_probe_capabilities_drift,
    log_probe_completed,
    log_probe_round_completed,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal ProviderConfig stub
# ---------------------------------------------------------------------------


@dataclass
class _StubProvider:
    name: str = "test-ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:32b"
    kind: str = "openai_compat"
    api_key_env: str | None = None
    paid: bool = False


# ---------------------------------------------------------------------------
# probe_one tests
# ---------------------------------------------------------------------------


class TestProbeOne:
    """Tests for the probe_one() function."""

    @pytest.mark.asyncio
    async def test_success_openai_compat(self):
        """Successful probe returns success=True with latency."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={"model": "qwen3:32b", "choices": [{"message": {"content": "h"}}]},
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await probe_one(_StubProvider())

        assert result.success is True
        assert result.provider == "test-ollama"
        assert result.model_name == "qwen3:32b"
        assert result.latency_ms > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_success_anthropic(self):
        """Successful anthropic probe."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={"model": "claude-sonnet-4-20250514", "content": [{"text": "h"}]},
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await probe_one(
                _StubProvider(
                    name="anthropic-direct",
                    kind="anthropic",
                    base_url="https://api.anthropic.com",
                    model="claude-sonnet-4-20250514",
                )
            )

        assert result.success is True
        assert result.model_name == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_http_error(self):
        """HTTP 500 returns success=False."""
        import httpx

        mock_response = httpx.Response(500, text="Internal Server Error")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await probe_one(_StubProvider())

        assert result.success is False
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Timeout returns success=False."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await probe_one(_StubProvider(), timeout_s=1.0)

        assert result.success is False
        assert "timeout" in result.error

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Connection error returns success=False."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await probe_one(_StubProvider())

        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# probe_loop tests
# ---------------------------------------------------------------------------


class TestProbeLoop:
    """Tests for the probe_loop() background task."""

    @pytest.mark.asyncio
    async def test_shutdown_during_initial_delay(self):
        """Setting shutdown_event during initial delay exits immediately."""
        shutdown = asyncio.Event()
        shutdown.set()  # already signaled

        await probe_loop(
            [_StubProvider()],
            interval_s=0.1,
            timeout_s=1.0,
            shutdown_event=shutdown,
        )
        # Should return immediately — no probes executed

    @pytest.mark.asyncio
    async def test_probes_providers_and_calls_record_fn(self):
        """probe_loop calls record_fn for each provider probe result."""
        records: list[tuple[str, bool]] = []

        def mock_record(name, *, success, threshold):
            records.append((name, success))

        shutdown = asyncio.Event()

        # Mock probe_one to return immediately
        async def mock_probe(provider, *, timeout_s=10.0):
            return ProbeResult(
                provider=provider.name,
                success=True,
                latency_ms=5.0,
                model_name=provider.model,
            )

        with patch(
            "coderouter.guards.continuous_probe.probe_one", side_effect=mock_probe
        ):
            # Run for a very short interval then shutdown
            async def stop_after_delay():
                await asyncio.sleep(0.15)
                shutdown.set()

            task = asyncio.create_task(
                probe_loop(
                    [_StubProvider(name="p1"), _StubProvider(name="p2")],
                    record_fn=mock_record,
                    interval_s=0.05,
                    timeout_s=1.0,
                    shutdown_event=shutdown,
                )
            )
            stopper = asyncio.create_task(stop_after_delay())
            await asyncio.gather(task, stopper)

        # Should have probed at least once
        assert len(records) >= 2
        assert all(success for _, success in records)

    @pytest.mark.asyncio
    async def test_skips_paid_providers_when_probe_paid_false(self):
        """Paid providers are skipped when probe_paid=False."""
        records: list[str] = []

        def mock_record(name, *, success, threshold):
            records.append(name)

        shutdown = asyncio.Event()

        async def mock_probe(provider, *, timeout_s=10.0):
            return ProbeResult(
                provider=provider.name, success=True, latency_ms=1.0
            )

        with patch(
            "coderouter.guards.continuous_probe.probe_one", side_effect=mock_probe
        ):

            async def stop():
                await asyncio.sleep(0.15)
                shutdown.set()

            providers = [
                _StubProvider(name="free-one", paid=False),
                _StubProvider(name="paid-one", paid=True),
            ]
            task = asyncio.create_task(
                probe_loop(
                    providers,
                    record_fn=mock_record,
                    interval_s=0.05,
                    timeout_s=1.0,
                    probe_paid=False,
                    shutdown_event=shutdown,
                )
            )
            stopper = asyncio.create_task(stop())
            await asyncio.gather(task, stopper)

        assert "free-one" in records
        assert "paid-one" not in records


# ---------------------------------------------------------------------------
# check_probe_drift tests
# ---------------------------------------------------------------------------


class TestCheckProbeDrift:
    """Tests for model-capabilities drift detection."""

    def test_no_drift_when_models_match(self):
        """No drift when observed model equals configured model."""
        result = check_probe_drift(
            _StubProvider(model="qwen3:32b"), "qwen3:32b"
        )
        assert result is None

    def test_drift_when_models_differ(self):
        """Drift detected when observed model != configured model."""
        result = check_probe_drift(
            _StubProvider(model="qwen3:32b"), "qwen3:14b"
        )
        assert result is not None
        assert isinstance(result, DriftReport)
        assert result.configured_model == "qwen3:32b"
        assert result.observed_model == "qwen3:14b"
        assert result.provider == "test-ollama"

    def test_no_drift_when_no_model_name(self):
        """No drift when probe didn't return a model name."""
        result = check_probe_drift(_StubProvider(), None)
        assert result is None

        result = check_probe_drift(_StubProvider(), "")
        assert result is None

    def test_drift_with_registry_unknown_model(self):
        """Drift with registry that doesn't know the observed model."""

        class _MockResolved:
            thinking = None
            reasoning_passthrough = None
            tools = None
            max_context_tokens = None
            claude_code_suitability = None
            cache_control = None

        class _MockRegistry:
            def lookup(self, *, kind, model):
                return _MockResolved()

        result = check_probe_drift(
            _StubProvider(model="qwen3:32b"),
            "unknown-model:latest",
            registry=_MockRegistry(),
        )
        assert result is not None
        assert result.in_registry is False

    def test_drift_with_registry_known_model(self):
        """Drift with registry that knows the observed model."""

        class _MockResolved:
            thinking = None
            reasoning_passthrough = None
            tools = True  # known model has at least one flag
            max_context_tokens = None
            claude_code_suitability = None
            cache_control = None

        class _MockRegistry:
            def lookup(self, *, kind, model):
                return _MockResolved()

        result = check_probe_drift(
            _StubProvider(model="qwen3:32b"),
            "qwen3:14b",
            registry=_MockRegistry(),
        )
        assert result is not None
        assert result.in_registry is True

    def test_whitespace_normalization(self):
        """Whitespace differences are ignored."""
        result = check_probe_drift(
            _StubProvider(model="qwen3:32b"), " qwen3:32b "
        )
        assert result is None


# ---------------------------------------------------------------------------
# Logging function tests
# ---------------------------------------------------------------------------


class TestProbeLogging:
    """Tests for the formal probe log functions."""

    def test_log_probe_completed(self, caplog):
        """log_probe_completed emits the correct event."""
        test_logger = logging.getLogger("test.probe")
        with caplog.at_level(logging.INFO):
            log_probe_completed(
                test_logger,
                provider="ollama-local",
                success=True,
                latency_ms=42.5,
                error=None,
                model_name="qwen3:32b",
            )
        assert "probe-completed" in caplog.text

    def test_log_probe_round_completed(self, caplog):
        """log_probe_round_completed emits the correct event."""
        test_logger = logging.getLogger("test.probe")
        with caplog.at_level(logging.INFO):
            log_probe_round_completed(
                test_logger,
                providers_probed=3,
                failures=1,
            )
        assert "probe-round-completed" in caplog.text

    def test_log_probe_capabilities_drift(self, caplog):
        """log_probe_capabilities_drift emits the correct event."""
        test_logger = logging.getLogger("test.probe")
        with caplog.at_level(logging.WARNING):
            log_probe_capabilities_drift(
                test_logger,
                provider="ollama-local",
                configured_model="qwen3:32b",
                observed_model="qwen3:14b",
                in_registry=False,
            )
        assert "probe-capabilities-drift" in caplog.text


# ---------------------------------------------------------------------------
# MetricsCollector dispatch tests
# ---------------------------------------------------------------------------


class TestProbeMetrics:
    """Tests for MetricsCollector probe event dispatch."""

    def test_probe_completed_increments_counters(self):
        """probe-completed event increments per-provider counters."""
        from coderouter.metrics.collector import MetricsCollector

        collector = MetricsCollector(ring_size=16)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="probe-completed",
            args=None,
            exc_info=None,
        )
        record.provider = "ollama-local"
        record.success = True
        record.latency_ms = 25.3
        record.error = None
        record.model_name = "qwen3:32b"
        collector.emit(record)

        snap = collector.snapshot()
        assert snap["counters"]["probe_total"] == {"ollama-local": 1}
        assert snap["counters"]["probe_success"] == {"ollama-local": 1}
        assert snap["counters"]["probe_failure"] == {}
        assert snap["counters"]["probe_latency_ms"] == {"ollama-local": 25.3}

    def test_probe_completed_failure(self):
        """probe-completed with success=False increments failure counter."""
        from coderouter.metrics.collector import MetricsCollector

        collector = MetricsCollector(ring_size=16)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="probe-completed",
            args=None,
            exc_info=None,
        )
        record.provider = "ollama-local"
        record.success = False
        record.latency_ms = 10000.0
        record.error = "timeout"
        record.model_name = None
        collector.emit(record)

        snap = collector.snapshot()
        assert snap["counters"]["probe_total"] == {"ollama-local": 1}
        assert snap["counters"]["probe_success"] == {}
        assert snap["counters"]["probe_failure"] == {"ollama-local": 1}

    def test_probe_round_completed_increments(self):
        """probe-round-completed increments the round counter."""
        from coderouter.metrics.collector import MetricsCollector

        collector = MetricsCollector(ring_size=16)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="probe-round-completed",
            args=None,
            exc_info=None,
        )
        record.providers_probed = 3
        record.failures = 1
        collector.emit(record)

        snap = collector.snapshot()
        assert snap["counters"]["probe_rounds_total"] == 1

    def test_probe_drift_detected_increments(self):
        """probe-capabilities-drift increments the drift counter."""
        from coderouter.metrics.collector import MetricsCollector

        collector = MetricsCollector(ring_size=16)
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="probe-capabilities-drift",
            args=None,
            exc_info=None,
        )
        record.provider = "ollama-local"
        record.configured_model = "qwen3:32b"
        record.observed_model = "qwen3:14b"
        record.in_registry = False
        collector.emit(record)

        snap = collector.snapshot()
        assert snap["counters"]["probe_drift_detected"] == {"ollama-local": 1}
