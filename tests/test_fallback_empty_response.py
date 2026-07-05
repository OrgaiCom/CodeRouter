"""Engine-level tests for ⑧ the per-request empty-response fallback.

These exercise ``FallbackChain.empty_response_action`` on the Anthropic
non-streaming (``generate_anthropic``) and streaming (``stream_anthropic``)
paths:

    - the ``_anthropic_response_is_empty`` content judgement
      (whitespace-only text = empty, thinking-only = empty, tool_use =
      non-empty, non-whitespace text = non-empty);
    - non-streaming ``fallback``: an empty 200 is swapped to the next
      provider; a fully-empty chain returns the last empty response verbatim;
    - non-streaming ``warn``: the empty response is logged but returned;
    - non-streaming ``off``: byte-for-byte legacy behavior (no detection);
    - streaming ``fallback``: an empty stream is withheld from the client
      and swapped provider-side; a stream with real content is delivered in
      full, in order; a fully-empty chain flushes the last buffered preamble
      and terminates normally;
    - the ``empty-response-detected`` log line + the metrics counter fire.

The engine fakes and helpers are reused from ``test_fallback_anthropic``;
this module only adds an empty-response config helper and empty-content
adapters.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import pytest

from coderouter.adapters.base import (
    AdapterError,
    ProviderCallOverrides,
)
from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.metrics.collector import MetricsCollector
from coderouter.routing import FallbackEngine, NoProvidersAvailableError
from coderouter.routing.fallback import (
    _anthropic_response_is_empty,
    _stream_event_is_real_content,
)
from coderouter.translation import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from tests.test_fallback_anthropic import (
    FakeAnthropicAdapter,
    _default_native_events,
    _engine_with_adapters,
)

# ----------------------------------------------------------------------
# Empty-content fakes
# ----------------------------------------------------------------------


class EmptyAnthropicAdapter(FakeAnthropicAdapter):
    """Native fake that returns a *content-empty* response / stream.

    ``content`` is fully configurable so a single fake can model an empty
    list, whitespace-only text, thinking-only, etc. ``empty_stream_events``
    supplies a stream that never emits real content (preamble + terminator).
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        content: list[dict] | None = None,
        empty_stream_events: list[AnthropicStreamEvent] | None = None,
    ) -> None:
        super().__init__(config)
        # default: an empty content list
        self._empty_content = content if content is not None else []
        self._empty_stream_events = empty_stream_events

    async def generate_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AnthropicResponse:
        self.generate_calls.append(request)
        self.last_overrides = overrides
        return AnthropicResponse(
            id="msg_empty",
            model=self.config.model,
            content=list(self._empty_content),
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=1, output_tokens=0),
            coderouter_provider=self.name,
        )

    async def stream_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        self.stream_calls.append(request)
        self.last_overrides = overrides
        events = self._empty_stream_events or _empty_native_events(self.config.model)
        for ev in events:
            yield ev


def _empty_native_events(model: str) -> list[AnthropicStreamEvent]:
    """A compliant Anthropic stream that carries *no* real content.

    message_start → empty-text content_block_start → content_block_stop →
    message_delta → message_stop. No content_block_delta with text, so
    ``_stream_event_is_real_content`` is False for every event.
    """
    return [
        AnthropicStreamEvent(
            type="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": "msg_empty",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
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
            type="content_block_stop",
            data={"type": "content_block_stop", "index": 0},
        ),
        AnthropicStreamEvent(
            type="message_delta",
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        ),
        AnthropicStreamEvent(
            type="message_stop",
            data={"type": "message_stop"},
        ),
    ]


# ----------------------------------------------------------------------
# Config helper
# ----------------------------------------------------------------------


def _empty_config(action: str) -> CodeRouterConfig:
    """Two native providers with the given ``empty_response_action``."""
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="first",
                kind="anthropic",
                base_url="https://api.anthropic.com",
                model="first-model",
                api_key_env="ANTHROPIC_API_KEY",
            ),
            ProviderConfig(
                name="second",
                kind="anthropic",
                base_url="https://api.anthropic.com",
                model="second-model",
                api_key_env="ANTHROPIC_API_KEY",
            ),
        ],
        profiles=[
            FallbackChain(
                name="default",
                providers=["first", "second"],
                empty_response_action=action,  # type: ignore[arg-type]
            )
        ],
    )


def _req(stream: bool = False) -> AnthropicRequest:
    return AnthropicRequest(
        max_tokens=64,
        messages=[AnthropicMessage(role="user", content="hi")],
        stream=stream,
    )


# ----------------------------------------------------------------------
# _anthropic_response_is_empty unit tests
# ----------------------------------------------------------------------


def _resp(content: list[dict]) -> AnthropicResponse:
    return AnthropicResponse(
        id="m",
        model="x",
        content=content,
        stop_reason="end_turn",
        usage=AnthropicUsage(input_tokens=1, output_tokens=0),
        coderouter_provider="p",
    )


def test_is_empty_empty_content_list() -> None:
    assert _anthropic_response_is_empty(_resp([])) is True


def test_is_empty_whitespace_only_text() -> None:
    assert _anthropic_response_is_empty(_resp([{"type": "text", "text": "   \n\t"}])) is True


def test_is_empty_thinking_only() -> None:
    assert _anthropic_response_is_empty(
        _resp([{"type": "thinking", "thinking": "hmm..."}])
    ) is True


def test_is_not_empty_real_text() -> None:
    assert _anthropic_response_is_empty(_resp([{"type": "text", "text": "hello"}])) is False


def test_is_not_empty_tool_use() -> None:
    assert _anthropic_response_is_empty(
        _resp([{"type": "tool_use", "id": "t1", "name": "run", "input": {}}])
    ) is False


def test_is_not_empty_mixed_thinking_and_text() -> None:
    # thinking + a real text block → non-empty
    assert _anthropic_response_is_empty(
        _resp(
            [
                {"type": "thinking", "thinking": "..."},
                {"type": "text", "text": "answer"},
            ]
        )
    ) is False


def test_stream_event_real_content_detection() -> None:
    text_delta = AnthropicStreamEvent(
        type="content_block_delta",
        data={
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
    )
    ws_delta = AnthropicStreamEvent(
        type="content_block_delta",
        data={
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "  "},
        },
    )
    tool_start = AnthropicStreamEvent(
        type="content_block_start",
        data={
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "run", "input": {}},
        },
    )
    msg_start = AnthropicStreamEvent(
        type="message_start", data={"type": "message_start", "message": {}}
    )
    assert _stream_event_is_real_content(text_delta) is True
    assert _stream_event_is_real_content(tool_start) is True
    assert _stream_event_is_real_content(ws_delta) is False
    assert _stream_event_is_real_content(msg_start) is False


# ----------------------------------------------------------------------
# Non-streaming behavior
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nonstream_fallback_empty_recovered_by_next() -> None:
    """fallback: first returns empty → second's real response is returned."""
    config = _empty_config("fallback")
    first = EmptyAnthropicAdapter(config.provider_by_name("first"))
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="recovered")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    resp = await engine.generate_anthropic(_req())

    assert resp.coderouter_provider == "second"
    assert resp.content == [{"type": "text", "text": "recovered"}]
    assert first.generate_calls  # first was tried
    assert second.generate_calls  # second recovered


@pytest.mark.asyncio
async def test_nonstream_fallback_chain_exhausted_returns_last_empty() -> None:
    """fallback: both empty → the last empty response is returned as-is."""
    config = _empty_config("fallback")
    first = EmptyAnthropicAdapter(config.provider_by_name("first"))
    second = EmptyAnthropicAdapter(config.provider_by_name("second"))
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    resp = await engine.generate_anthropic(_req())

    # last empty (from "second") returned verbatim, no exception
    assert resp.coderouter_provider == "second"
    assert resp.content == []


@pytest.mark.asyncio
async def test_nonstream_warn_returns_empty_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """warn: the empty response from the first provider is returned; second
    is never tried; a log line is emitted."""
    config = _empty_config("warn")
    first = EmptyAnthropicAdapter(config.provider_by_name("first"))
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="unused")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    with caplog.at_level(logging.WARNING):
        resp = await engine.generate_anthropic(_req())

    assert resp.coderouter_provider == "first"
    assert resp.content == []
    assert second.generate_calls == []  # warn does not fall through
    assert any(r.message == "empty-response-detected" for r in caplog.records)


@pytest.mark.asyncio
async def test_nonstream_off_is_unchanged() -> None:
    """off: the empty response is returned immediately, no detection."""
    config = _empty_config("off")
    first = EmptyAnthropicAdapter(config.provider_by_name("first"))
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="unused")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    resp = await engine.generate_anthropic(_req())

    assert resp.coderouter_provider == "first"
    assert resp.content == []
    assert second.generate_calls == []


@pytest.mark.asyncio
async def test_nonstream_whitespace_text_treated_empty() -> None:
    """fallback: a whitespace-only text response is treated as empty."""
    config = _empty_config("fallback")
    first = EmptyAnthropicAdapter(
        config.provider_by_name("first"),
        content=[{"type": "text", "text": "   \n"}],
    )
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="real")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    resp = await engine.generate_anthropic(_req())

    assert resp.coderouter_provider == "second"


@pytest.mark.asyncio
async def test_nonstream_tool_use_is_not_empty() -> None:
    """fallback: a tool_use-only response is NOT empty → returned as-is."""
    config = _empty_config("fallback")
    first = EmptyAnthropicAdapter(
        config.provider_by_name("first"),
        content=[{"type": "tool_use", "id": "t", "name": "run", "input": {}}],
    )
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="unused")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    resp = await engine.generate_anthropic(_req())

    assert resp.coderouter_provider == "first"
    assert second.generate_calls == []


@pytest.mark.asyncio
async def test_nonstream_thinking_only_treated_empty() -> None:
    """fallback: a thinking-only response is empty → next provider used."""
    config = _empty_config("fallback")
    first = EmptyAnthropicAdapter(
        config.provider_by_name("first"),
        content=[{"type": "thinking", "thinking": "let me think"}],
    )
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="real")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    resp = await engine.generate_anthropic(_req())

    assert resp.coderouter_provider == "second"


@pytest.mark.asyncio
async def test_nonstream_fallback_real_first_short_circuits() -> None:
    """fallback with a non-empty first provider is a pure no-op."""
    config = _empty_config("fallback")
    first = FakeAnthropicAdapter(config.provider_by_name("first"), text="fine")
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="unused")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    resp = await engine.generate_anthropic(_req())

    assert resp.coderouter_provider == "first"
    assert second.generate_calls == []


# ----------------------------------------------------------------------
# Streaming behavior
# ----------------------------------------------------------------------


async def _collect(engine: FallbackEngine, req: AnthropicRequest) -> list[AnthropicStreamEvent]:
    return [ev async for ev in engine.stream_anthropic(req)]


@pytest.mark.asyncio
async def test_stream_fallback_empty_swapped_to_next() -> None:
    """fallback: an empty stream is withheld; the second provider's real
    stream is delivered in full and unchanged."""
    config = _empty_config("fallback")
    first = EmptyAnthropicAdapter(config.provider_by_name("first"))
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="hello")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    events = await _collect(engine, _req(stream=True))

    # The client sees exactly the second provider's stream, in order —
    # none of the first provider's (empty) preamble leaks through.
    expected = _default_native_events("hello", "second-model")
    assert [e.type for e in events] == [e.type for e in expected]
    # real text delta present
    assert any(
        e.data.get("delta", {}).get("text") == "hello"
        for e in events
        if e.type == "content_block_delta"
    )
    assert first.stream_calls  # first was tried
    assert second.stream_calls  # second recovered


@pytest.mark.asyncio
async def test_stream_fallback_real_content_delivered_in_order() -> None:
    """fallback with a non-empty first stream: buffering is transparent —
    every event arrives in the original order."""
    config = _empty_config("fallback")
    first = FakeAnthropicAdapter(config.provider_by_name("first"), text="content")
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="unused")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    events = await _collect(engine, _req(stream=True))

    expected = _default_native_events("content", "first-model")
    assert [e.type for e in events] == [e.type for e in expected]
    assert second.stream_calls == []  # second never tried


@pytest.mark.asyncio
async def test_stream_fallback_chain_exhausted_terminates_normally() -> None:
    """fallback: both streams empty → the last buffered (empty) stream is
    flushed and the SSE terminates normally (no exception)."""
    config = _empty_config("fallback")
    first = EmptyAnthropicAdapter(config.provider_by_name("first"))
    second = EmptyAnthropicAdapter(config.provider_by_name("second"))
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    events = await _collect(engine, _req(stream=True))

    # a well-formed, terminating empty stream reaches the client
    assert events[0].type == "message_start"
    assert events[-1].type == "message_stop"
    assert first.stream_calls and second.stream_calls


@pytest.mark.asyncio
async def test_stream_off_is_unchanged() -> None:
    """off: the empty stream is streamed through immediately (legacy)."""
    config = _empty_config("off")
    first = EmptyAnthropicAdapter(config.provider_by_name("first"))
    second = FakeAnthropicAdapter(config.provider_by_name("second"), text="unused")
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    events = await _collect(engine, _req(stream=True))

    # first provider's empty stream is delivered verbatim; second untouched
    assert events[0].type == "message_start"
    assert events[-1].type == "message_stop"
    assert second.stream_calls == []


@pytest.mark.asyncio
async def test_stream_fallback_all_empty_then_error_raises_no_providers() -> None:
    """fallback: first empty (swapped), second raises before content →
    the chain is exhausted with no buffered stream to flush AND no successful
    stream, so NoProvidersAvailableError surfaces."""
    config = _empty_config("fallback")
    first = FakeAnthropicAdapter(
        config.provider_by_name("first"),
        stream_fail_with=AdapterError("boom", provider="first", retryable=True),
        stream_fail_after=0,
    )
    second = FakeAnthropicAdapter(
        config.provider_by_name("second"),
        stream_fail_with=AdapterError("boom2", provider="second", retryable=True),
        stream_fail_after=0,
    )
    engine = _engine_with_adapters(config, {"first": first, "second": second})

    with pytest.raises(NoProvidersAvailableError):
        await _collect(engine, _req(stream=True))


# ----------------------------------------------------------------------
# Metrics integration
# ----------------------------------------------------------------------


def test_metrics_empty_response_counter() -> None:
    """The collector counts ``empty-response-detected`` events per-provider."""
    collector = MetricsCollector()
    rec = logging.LogRecord(
        name="coderouter",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="empty-response-detected",
        args=(),
        exc_info=None,
    )
    rec.provider = "gemma4"
    rec.action = "fallback"
    rec.stream = False
    rec.chain_exhausted = False
    collector.emit(rec)
    collector.emit(rec)

    snap = collector.snapshot()
    assert snap["counters"]["empty_responses_total"] == 2
    assert snap["counters"]["empty_responses_by_provider"]["gemma4"] == 2
