"""v2.0-H: Partial stitch tests.

Tests for mid-stream partial content accumulation and the ingress surface
mode that delivers accumulated text on failure.

Groups:
    1. _StreamUsageAccumulator.partial_content accumulation
    2. MidStreamError partial_content propagation
    3. Ingress surface mode integration (SSE output on mid-stream failure)
    4. Off mode regression (legacy error event)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from coderouter.routing.fallback import MidStreamError, _StreamUsageAccumulator
from coderouter.translation import AnthropicStreamEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev(event_type: str, data: dict[str, Any]) -> AnthropicStreamEvent:
    return AnthropicStreamEvent(type=event_type, data=data)


def _text_block_start(index: int = 0) -> AnthropicStreamEvent:
    return _ev("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    })


def _text_delta(text: str) -> AnthropicStreamEvent:
    return _ev("content_block_delta", {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": text},
    })


def _block_stop() -> AnthropicStreamEvent:
    return _ev("content_block_stop", {"type": "content_block_stop"})


def _tool_use_start(index: int = 1) -> AnthropicStreamEvent:
    return _ev("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "fn", "input": {}},
    })


def _tool_delta(json_str: str) -> AnthropicStreamEvent:
    return _ev("content_block_delta", {
        "type": "content_block_delta",
        "delta": {"type": "input_json_delta", "partial_json": json_str},
    })


# ---------------------------------------------------------------------------
# 1. _StreamUsageAccumulator.partial_content
# ---------------------------------------------------------------------------


class TestAccumulatorPartialContent:
    def test_empty_when_no_events(self):
        acc = _StreamUsageAccumulator()
        assert acc.partial_content == []

    def test_single_completed_block(self):
        acc = _StreamUsageAccumulator()
        acc.observe(_text_block_start())
        acc.observe(_text_delta("Hello "))
        acc.observe(_text_delta("world"))
        acc.observe(_block_stop())
        assert acc.partial_content == [{"type": "text", "text": "Hello world"}]

    def test_multiple_text_blocks(self):
        acc = _StreamUsageAccumulator()
        # Block 0
        acc.observe(_text_block_start(0))
        acc.observe(_text_delta("First"))
        acc.observe(_block_stop())
        # Block 1
        acc.observe(_text_block_start(1))
        acc.observe(_text_delta("Second"))
        acc.observe(_block_stop())
        assert acc.partial_content == [
            {"type": "text", "text": "First"},
            {"type": "text", "text": "Second"},
        ]

    def test_in_progress_block_included(self):
        """When stream interrupts mid-block, partial text is still surfaced."""
        acc = _StreamUsageAccumulator()
        acc.observe(_text_block_start())
        acc.observe(_text_delta("partial te"))
        # No block_stop — stream interrupted
        assert acc.partial_content == [{"type": "text", "text": "partial te"}]

    def test_completed_plus_in_progress(self):
        acc = _StreamUsageAccumulator()
        # Completed block
        acc.observe(_text_block_start(0))
        acc.observe(_text_delta("Done."))
        acc.observe(_block_stop())
        # In-progress block
        acc.observe(_text_block_start(1))
        acc.observe(_text_delta("In prog"))
        assert acc.partial_content == [
            {"type": "text", "text": "Done."},
            {"type": "text", "text": "In prog"},
        ]

    def test_tool_use_blocks_excluded(self):
        """tool_use blocks are not accumulated (partial JSON is unusable)."""
        acc = _StreamUsageAccumulator()
        acc.observe(_text_block_start(0))
        acc.observe(_text_delta("text here"))
        acc.observe(_block_stop())
        # tool_use block
        acc.observe(_tool_use_start(1))
        acc.observe(_tool_delta('{"key":'))
        acc.observe(_block_stop())
        # Only text block in output
        assert acc.partial_content == [{"type": "text", "text": "text here"}]

    def test_empty_text_block_excluded(self):
        """A text block with no deltas produces no output."""
        acc = _StreamUsageAccumulator()
        acc.observe(_text_block_start())
        acc.observe(_block_stop())
        assert acc.partial_content == []

    def test_empty_in_progress_text_excluded(self):
        """An in-progress text block with no deltas produces no output."""
        acc = _StreamUsageAccumulator()
        acc.observe(_text_block_start())
        # No deltas, no stop
        assert acc.partial_content == []


# ---------------------------------------------------------------------------
# 2. MidStreamError partial_content propagation
# ---------------------------------------------------------------------------


class TestMidStreamErrorPartialContent:
    def test_default_empty(self):
        from coderouter.adapters.base import AdapterError

        exc = MidStreamError("prov1", AdapterError("oops", provider="prov1", status_code=500))
        assert exc.partial_content == []

    def test_carries_content(self):
        from coderouter.adapters.base import AdapterError

        content = [{"type": "text", "text": "hello"}]
        exc = MidStreamError("prov1", AdapterError("oops", provider="prov1", status_code=500), partial_content=content)
        assert exc.partial_content == content
        assert exc.provider == "prov1"

    def test_none_becomes_empty_list(self):
        from coderouter.adapters.base import AdapterError

        exc = MidStreamError("prov1", AdapterError("oops", provider="prov1", status_code=500), partial_content=None)
        assert exc.partial_content == []


# ---------------------------------------------------------------------------
# 3. Ingress surface mode integration
# ---------------------------------------------------------------------------


class TestIngressSurfaceMode:
    """Test the SSE output of _anthropic_sse_iterator when partial_stitch_action=surface."""

    @pytest.mark.asyncio
    async def test_surface_mode_emits_partial_events(self):
        """When surface mode is on and partial_content exists, emit graceful termination."""
        from coderouter.adapters.base import AdapterError
        from coderouter.ingress.anthropic_routes import _anthropic_sse_iterator
        from coderouter.translation import AnthropicRequest

        partial = [{"type": "text", "text": "Accumulated response text"}]
        mid_err = MidStreamError("ollama-local", AdapterError("connection reset", provider="ollama-local", status_code=502), partial_content=partial)

        # Mock the engine
        engine = MagicMock()

        async def _raise_midstream(req):
            yield _ev("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 10}}})
            raise mid_err

        engine.stream_anthropic = _raise_midstream

        # Mock config with surface action
        profile_cfg = MagicMock()
        profile_cfg.partial_stitch_action = "surface"
        engine.config = MagicMock()
        engine.config.default_profile = "default"
        engine.config.profile_by_name.return_value = profile_cfg

        req = AnthropicRequest(
            model="test-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

        events = []
        async for chunk in _anthropic_sse_iterator(engine, req):
            events.append(chunk)

        # Should have: message_start (yielded before error) + message_delta + message_stop + coderouter_partial
        assert len(events) == 4

        # First event is the normal message_start
        assert "event: message_start" in events[0]

        # Second: synthesized message_delta
        assert "event: message_delta" in events[1]
        delta_data = json.loads(events[1].split("data: ")[1].strip())
        assert delta_data["type"] == "message_delta"

        # Third: message_stop
        assert "event: message_stop" in events[2]

        # Fourth: coderouter_partial metadata
        assert "event: coderouter_partial" in events[3]
        partial_data = json.loads(events[3].split("data: ")[1].strip())
        assert partial_data["type"] == "coderouter_partial"
        assert partial_data["partial_content"] == partial
        assert partial_data["provider"] == "ollama-local"
        assert partial_data["reason"] == "mid_stream_failure"

    @pytest.mark.asyncio
    async def test_surface_mode_no_partial_content_falls_to_error(self):
        """When surface mode is on but partial_content is empty, emit legacy error."""
        from coderouter.adapters.base import AdapterError
        from coderouter.ingress.anthropic_routes import _anthropic_sse_iterator
        from coderouter.translation import AnthropicRequest

        mid_err = MidStreamError("ollama-local", AdapterError("reset", provider="ollama-local", status_code=502), partial_content=[])

        engine = MagicMock()

        async def _raise_midstream(req):
            yield _ev("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 10}}})
            raise mid_err

        engine.stream_anthropic = _raise_midstream

        profile_cfg = MagicMock()
        profile_cfg.partial_stitch_action = "surface"
        engine.config = MagicMock()
        engine.config.default_profile = "default"
        engine.config.profile_by_name.return_value = profile_cfg

        req = AnthropicRequest(
            model="test-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

        events = []
        async for chunk in _anthropic_sse_iterator(engine, req):
            events.append(chunk)

        # message_start + error event (no partial content to surface)
        assert len(events) == 2
        assert "event: error" in events[1]


# ---------------------------------------------------------------------------
# 4. Off mode regression
# ---------------------------------------------------------------------------


class TestIngressOffMode:
    """When partial_stitch_action=off, mid-stream errors emit the legacy error event."""

    @pytest.mark.asyncio
    async def test_off_mode_emits_error_event(self):
        from coderouter.adapters.base import AdapterError
        from coderouter.ingress.anthropic_routes import _anthropic_sse_iterator
        from coderouter.translation import AnthropicRequest

        partial = [{"type": "text", "text": "some text"}]
        mid_err = MidStreamError("prov", AdapterError("fail", provider="prov", status_code=500), partial_content=partial)

        engine = MagicMock()

        async def _raise_midstream(req):
            yield _ev("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 5}}})
            raise mid_err

        engine.stream_anthropic = _raise_midstream

        profile_cfg = MagicMock()
        profile_cfg.partial_stitch_action = "off"
        engine.config = MagicMock()
        engine.config.default_profile = "default"
        engine.config.profile_by_name.return_value = profile_cfg

        req = AnthropicRequest(
            model="test-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

        events = []
        async for chunk in _anthropic_sse_iterator(engine, req):
            events.append(chunk)

        # message_start + error event (surface mode is off)
        assert len(events) == 2
        assert "event: error" in events[1]
        err_data = json.loads(events[1].split("data: ")[1].strip())
        assert err_data["error"]["type"] == "api_error"

    @pytest.mark.asyncio
    async def test_default_config_is_off(self):
        """When profile lookup fails (KeyError), fall back to 'off' behavior."""
        from coderouter.adapters.base import AdapterError
        from coderouter.ingress.anthropic_routes import _anthropic_sse_iterator
        from coderouter.translation import AnthropicRequest

        partial = [{"type": "text", "text": "partial"}]
        mid_err = MidStreamError("prov", AdapterError("fail", provider="prov", status_code=500), partial_content=partial)

        engine = MagicMock()

        async def _raise_midstream(req):
            yield _ev("message_start", {"type": "message_start", "message": {}})
            raise mid_err

        engine.stream_anthropic = _raise_midstream

        engine.config = MagicMock()
        engine.config.default_profile = "default"
        engine.config.profile_by_name.side_effect = KeyError("no such profile")

        req = AnthropicRequest(
            model="test-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

        events = []
        async for chunk in _anthropic_sse_iterator(engine, req):
            events.append(chunk)

        # Falls back to error event
        assert any("event: error" in e for e in events)


# ---------------------------------------------------------------------------
# 5. MetricsCollector dispatch
# ---------------------------------------------------------------------------


class TestMetricsCollectorPartialStitch:
    def test_counter_increments(self):
        import logging

        from coderouter.metrics.collector import MetricsCollector

        mc = MetricsCollector(ring_size=10)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="partial-stitch-surfaced",
            args=None,
            exc_info=None,
        )
        record.provider = "ollama"
        record.profile = "default"
        record.text_blocks = 2
        record.text_length = 150
        mc.emit(record)

        snap = mc.snapshot()
        assert snap["counters"]["partial_stitch_surfaced_total"] == 1

    def test_appears_in_recent(self):
        import logging

        from coderouter.metrics.collector import MetricsCollector

        mc = MetricsCollector(ring_size=10)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="partial-stitch-surfaced",
            args=None,
            exc_info=None,
        )
        record.provider = "ollama"
        mc.emit(record)

        snap = mc.snapshot()
        assert any(e["event"] == "partial-stitch-surfaced" for e in snap["recent"])
