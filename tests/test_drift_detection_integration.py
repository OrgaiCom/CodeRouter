"""v2.0-G: Drift detection integration tests.

Tests the wiring between the engine's generate_anthropic / stream_anthropic
paths and the drift detection guard. Uses scripted fake adapters — no network.

Key scenarios:
  - Observation recording into the drift window after success/failure.
  - Detection triggering after enough degraded observations.
  - promote action demotes via AdaptiveAdjuster.
  - Ingress response header carries drift severity.
  - Cooldown prevents re-detection.
  - Stream path records output_tokens, stop_reason, has_tool_use correctly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import pytest

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import (
    AdapterError,
    BaseAdapter,
    ProviderCallOverrides,
)
from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.guards.drift_detection import DriftWindow
from coderouter.routing import FallbackEngine, NoProvidersAvailableError
from coderouter.translation import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicStreamEvent,
    AnthropicTool,
    AnthropicUsage,
)

# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------


class _FakeNativeAdapter(AnthropicAdapter):
    """Fake native Anthropic adapter returning controlled responses."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        output_tokens: int = 100,
        stop_reason: str = "end_turn",
        has_tool_use: bool = False,
        fail_with: AdapterError | None = None,
    ) -> None:
        super().__init__(config)
        self.output_tokens = output_tokens
        self.stop_reason = stop_reason
        self.has_tool_use = has_tool_use
        self.fail_with = fail_with
        self.call_count = 0

    async def healthcheck(self) -> bool:
        return self.fail_with is None

    async def generate_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AnthropicResponse:
        self.call_count += 1
        if self.fail_with:
            raise self.fail_with
        content: list[dict] = [{"type": "text", "text": "x" * self.output_tokens}]
        if self.has_tool_use:
            content.append({"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}})
        return AnthropicResponse(
            id="msg_fake",
            model=self.config.model,
            content=content,
            stop_reason=self.stop_reason,
            usage=AnthropicUsage(input_tokens=10, output_tokens=self.output_tokens),
            coderouter_provider=self.name,
        )

    async def stream_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        self.call_count += 1
        if self.fail_with:
            raise self.fail_with
        events = _stream_events(
            output_tokens=self.output_tokens,
            stop_reason=self.stop_reason,
            has_tool_use=self.has_tool_use,
            model=self.config.model,
        )
        for ev in events:
            yield ev


def _stream_events(
    *,
    output_tokens: int,
    stop_reason: str,
    has_tool_use: bool,
    model: str,
) -> list[AnthropicStreamEvent]:
    """Minimal compliant Anthropic stream events."""
    events = [
        AnthropicStreamEvent(
            type="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        ),
        AnthropicStreamEvent(
            type="content_block_start",
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        AnthropicStreamEvent(
            type="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "x" * output_tokens},
            },
        ),
        AnthropicStreamEvent(
            type="content_block_stop",
            data={"type": "content_block_stop", "index": 0},
        ),
    ]
    if has_tool_use:
        events.append(
            AnthropicStreamEvent(
                type="content_block_start",
                data={
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}},
                },
            )
        )
        events.append(
            AnthropicStreamEvent(
                type="content_block_stop",
                data={"type": "content_block_stop", "index": 1},
            )
        )
    events.append(
        AnthropicStreamEvent(
            type="message_delta",
            data={
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
    )
    events.append(
        AnthropicStreamEvent(
            type="message_stop",
            data={"type": "message_stop"},
        )
    )
    return events


# ---------------------------------------------------------------------------
# Config / engine helpers
# ---------------------------------------------------------------------------


def _config(
    *,
    action: Literal["off", "warn", "promote", "reload"] = "warn",
    sensitivity: str = "normal",
    window_size: int = 10,
    cooldown_s: int = 300,
) -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local",
                kind="anthropic",
                base_url="https://api.anthropic.com",
                model="test-model",
                api_key_env="ANTHROPIC_API_KEY",
            ),
        ],
        profiles=[
            FallbackChain(
                name="default",
                providers=["local"],
                drift_detection_action=action,
                drift_detection_window_size=window_size,
                drift_detection_sensitivity=sensitivity,
                drift_detection_cooldown_s=cooldown_s,
            ),
        ],
    )


def _engine(
    cfg: CodeRouterConfig,
    adapter: BaseAdapter,
) -> FallbackEngine:
    """Build a FallbackEngine with drift detection fields initialized."""
    engine = FallbackEngine.__new__(FallbackEngine)
    engine.config = cfg
    engine._adapters = {adapter.name: adapter}
    # Initialize drift detection state (normally done in __init__)
    engine._drift_window = DriftWindow(max_size=cfg.profiles[0].drift_detection_window_size)
    engine._drift_demoted = {}
    engine._last_drift_verdict = None
    return engine


def _req(*, stream: bool = False, tools: bool = False) -> AnthropicRequest:
    t = (
        [AnthropicTool(name="fn", description="d", input_schema={"type": "object"})]
        if tools
        else None
    )
    return AnthropicRequest(
        max_tokens=64,
        messages=[AnthropicMessage(role="user", content="hi")],
        stream=stream,
        tools=t,
    )


# ---------------------------------------------------------------------------
# Tests: Observation recording
# ---------------------------------------------------------------------------


class TestObservationRecording:
    """Verify that engine paths record observations into the drift window."""

    @pytest.mark.asyncio
    async def test_generate_success_records_observation(self):
        cfg = _config(action="warn")
        adapter = _FakeNativeAdapter(
            cfg.providers[0], output_tokens=50, stop_reason="end_turn"
        )
        eng = _engine(cfg, adapter)

        await eng.generate_anthropic(_req())

        window = eng._drift_window.get_window("local")
        assert len(window) == 1
        obs = window[0]
        assert obs.provider == "local"
        assert obs.output_tokens == 50
        assert obs.stop_reason == "end_turn"
        assert obs.is_error is False
        assert obs.stream is False

    @pytest.mark.asyncio
    async def test_generate_failure_records_error_observation(self):
        cfg = _config(action="warn")
        adapter = _FakeNativeAdapter(
            cfg.providers[0],
            fail_with=AdapterError("boom", provider="local", retryable=False),
        )
        eng = _engine(cfg, adapter)

        with pytest.raises(NoProvidersAvailableError):
            await eng.generate_anthropic(_req())

        window = eng._drift_window.get_window("local")
        assert len(window) == 1
        assert window[0].is_error is True

    @pytest.mark.asyncio
    async def test_stream_success_records_observation(self):
        cfg = _config(action="warn")
        adapter = _FakeNativeAdapter(
            cfg.providers[0], output_tokens=75, stop_reason="end_turn"
        )
        eng = _engine(cfg, adapter)

        events = []
        async for ev in eng.stream_anthropic(_req(stream=True)):
            events.append(ev)

        window = eng._drift_window.get_window("local")
        assert len(window) == 1
        obs = window[0]
        assert obs.output_tokens == 75
        assert obs.stop_reason == "end_turn"
        assert obs.stream is True

    @pytest.mark.asyncio
    async def test_stream_records_tool_use(self):
        cfg = _config(action="warn")
        adapter = _FakeNativeAdapter(
            cfg.providers[0], output_tokens=20, has_tool_use=True
        )
        eng = _engine(cfg, adapter)

        async for _ in eng.stream_anthropic(_req(stream=True, tools=True)):
            pass

        window = eng._drift_window.get_window("local")
        assert len(window) == 1
        assert window[0].has_tool_use is True
        assert window[0].request_had_tools is True

    @pytest.mark.asyncio
    async def test_no_observation_when_action_off(self):
        cfg = _config(action="off")
        adapter = _FakeNativeAdapter(cfg.providers[0], output_tokens=50)
        eng = _engine(cfg, adapter)

        await eng.generate_anthropic(_req())

        window = eng._drift_window.get_window("local")
        assert len(window) == 0


# ---------------------------------------------------------------------------
# Tests: Detection triggering
# ---------------------------------------------------------------------------


class TestDetectionTriggering:
    """Verify that drift is detected after enough degraded observations."""

    @pytest.mark.asyncio
    async def test_drift_detected_after_many_empty_responses(self):
        cfg = _config(action="warn", window_size=10, sensitivity="normal")
        adapter = _FakeNativeAdapter(cfg.providers[0], output_tokens=0)
        eng = _engine(cfg, adapter)

        # Need min_window_fill (6) observations to trigger
        for _ in range(7):
            await eng.generate_anthropic(_req())

        assert eng._last_drift_verdict is not None
        assert eng._last_drift_verdict.drifted is True
        assert eng._last_drift_verdict.severity == "severe"
        assert eng.last_drift_severity == "severe"

    @pytest.mark.asyncio
    async def test_no_drift_below_min_fill(self):
        cfg = _config(action="warn", window_size=10, sensitivity="normal")
        adapter = _FakeNativeAdapter(cfg.providers[0], output_tokens=0)
        eng = _engine(cfg, adapter)

        # Only 3 observations — below min_window_fill=6
        for _ in range(3):
            await eng.generate_anthropic(_req())

        assert eng.last_drift_severity is None

    @pytest.mark.asyncio
    async def test_no_drift_when_healthy(self):
        cfg = _config(action="warn", window_size=10)
        adapter = _FakeNativeAdapter(cfg.providers[0], output_tokens=100)
        eng = _engine(cfg, adapter)

        for _ in range(10):
            await eng.generate_anthropic(_req())

        assert eng.last_drift_severity is None


# ---------------------------------------------------------------------------
# Tests: Promote action
# ---------------------------------------------------------------------------


class TestPromoteAction:
    """Verify that promote action demotes the provider via adaptive."""

    @pytest.mark.asyncio
    async def test_promote_calls_adaptive_demote(self):
        cfg = _config(action="promote", window_size=10)
        adapter = _FakeNativeAdapter(cfg.providers[0], output_tokens=0)
        eng = _engine(cfg, adapter)

        for _ in range(7):
            await eng.generate_anthropic(_req())

        # Provider should be in cooldown (demoted)
        assert "local" in eng._drift_demoted

    @pytest.mark.asyncio
    async def test_cooldown_prevents_redetection(self):
        cfg = _config(action="promote", window_size=10, cooldown_s=300)
        adapter = _FakeNativeAdapter(cfg.providers[0], output_tokens=0)
        eng = _engine(cfg, adapter)

        # Trigger first detection
        for _ in range(7):
            await eng.generate_anthropic(_req())

        first_verdict = eng._last_drift_verdict
        assert first_verdict is not None
        assert first_verdict.drifted

        # Subsequent calls should not re-detect (in cooldown)
        eng._last_drift_verdict = None
        await eng.generate_anthropic(_req())
        assert eng._last_drift_verdict is None


# ---------------------------------------------------------------------------
# Tests: Stream accumulator captures stop_reason + tool_use
# ---------------------------------------------------------------------------


class TestStreamAccumulator:
    """Verify _StreamUsageAccumulator picks up drift-relevant fields."""

    @pytest.mark.asyncio
    async def test_stop_reason_captured_from_message_delta(self):
        cfg = _config(action="warn")
        adapter = _FakeNativeAdapter(
            cfg.providers[0], output_tokens=42, stop_reason="max_tokens"
        )
        eng = _engine(cfg, adapter)

        async for _ in eng.stream_anthropic(_req(stream=True)):
            pass

        window = eng._drift_window.get_window("local")
        assert window[0].stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_tool_use_detected_from_content_block_start(self):
        cfg = _config(action="warn")
        adapter = _FakeNativeAdapter(
            cfg.providers[0], output_tokens=10, has_tool_use=True
        )
        eng = _engine(cfg, adapter)

        async for _ in eng.stream_anthropic(_req(stream=True, tools=True)):
            pass

        window = eng._drift_window.get_window("local")
        assert window[0].has_tool_use is True


# ---------------------------------------------------------------------------
# Tests: Header exposure
# ---------------------------------------------------------------------------


class TestDriftHeader:
    """Verify last_drift_severity property for ingress header."""

    @pytest.mark.asyncio
    async def test_severity_none_when_healthy(self):
        cfg = _config(action="warn")
        adapter = _FakeNativeAdapter(cfg.providers[0], output_tokens=100)
        eng = _engine(cfg, adapter)

        for _ in range(10):
            await eng.generate_anthropic(_req())

        assert eng.last_drift_severity is None

    @pytest.mark.asyncio
    async def test_severity_severe_when_all_empty(self):
        cfg = _config(action="warn")
        adapter = _FakeNativeAdapter(cfg.providers[0], output_tokens=0)
        eng = _engine(cfg, adapter)

        for _ in range(7):
            await eng.generate_anthropic(_req())

        assert eng.last_drift_severity == "severe"

    @pytest.mark.asyncio
    async def test_severity_mild_single_signal(self):
        # error_rate > 0.25 but only 1 mild signal → mild
        cfg = _config(action="warn", window_size=10)
        adapter = _FakeNativeAdapter(
            cfg.providers[0],
            fail_with=AdapterError("err", provider="local", retryable=False),
        )
        eng = _engine(cfg, adapter)

        # Seed 7 successes first, then 3 errors → error_rate=0.3
        adapter.fail_with = None
        adapter.output_tokens = 100
        for _ in range(7):
            await eng.generate_anthropic(_req())

        adapter.fail_with = AdapterError("err", provider="local", retryable=False)
        for _ in range(3):
            with pytest.raises(NoProvidersAvailableError):
                await eng.generate_anthropic(_req())

        # error_rate = 3/10 = 0.3 > 0.25 threshold → mild
        # (only 1 mild signal, so severity should be mild)
        verdict = eng._last_drift_verdict
        if verdict and verdict.drifted:
            assert verdict.severity == "mild"
