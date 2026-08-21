"""v2.15.0: stream-truncation detection tests.

An upstream can end its HTTP body *cleanly* while the LLM protocol carried
inside it is still mid-message: no ``message_stop`` on the Anthropic wire, no
``data: [DONE]`` and no ``finish_reason`` on the OpenAI wire. Before v2.15.0
that was indistinguishable from a complete stream — the adapters never
recorded whether a terminator arrived, and the translation layer then
synthesized ``stop_reason: end_turn`` / ``finish_reason: "stop"`` so the
client would not hang, erasing the evidence.

Groups:
    1. AnthropicAdapter.stream_anthropic — terminator tracking over real SSE
       (off / warn / error, plus the terminator-leniency false-positive
       guards and the tool_use-in-flight flag).
    2. OpenAICompatAdapter.stream — the same, on the OpenAI wire.
    3. Engine integration — a real adapter behind httpx_mock, driven through
       FallbackEngine: fallback on truncation, MidStreamError once bytes are
       out, and the ``off`` regression that proves byte-for-byte v2.14.0
       compatibility.
    4. Ingress — ``partial_stitch_action: surface`` labels the
       ``coderouter_partial`` event ``stream_truncated``.
    5. Vocabulary, config plumbing, logging and metrics wiring.

Every test that does not explicitly opt in runs with the default
``stream_truncation_action: off``; group 3's regression test is the explicit
proof that the default path is unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_httpx import HTTPXMock

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import (
    AdapterError,
    ChatRequest,
    Message,
    ProviderCallOverrides,
    StreamTruncatedError,
)
from coderouter.adapters.openai_compat import OpenAICompatAdapter
from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.metrics.collector import MetricsCollector
from coderouter.metrics.prometheus import format_prometheus
from coderouter.routing.fallback import MidStreamError
from coderouter.routing.fallback_trace import (
    ATTEMPT_FAILURE_REASONS,
    FALLBACK_REASONS,
    REASON_STREAM_TRUNCATED,
    classify_adapter_error,
    describe_adapter_error,
)
from coderouter.translation import AnthropicMessage, AnthropicRequest
from tests.test_fallback_anthropic import _engine_with_adapters

_ANTHROPIC_URL = "https://anthropic.test/v1/messages"
_SECOND_ANTHROPIC_URL = "https://anthropic2.test/v1/messages"
_OPENAI_URL = "https://openai.test/v1/chat/completions"

# ----------------------------------------------------------------------
# SSE fixtures
# ----------------------------------------------------------------------

# A stream cut after two text deltas: no message_delta, no message_stop.
_ANTHROPIC_TRUNCATED = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_1","type":"message",'
    '"role":"assistant","content":[],"model":"m","stop_reason":null,'
    '"usage":{"input_tokens":7,"output_tokens":0}}}\n'
    "\n"
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n'
    "\n"
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"half a "}}\n'
    "\n"
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"sentence"}}\n'
    "\n"
)

# The same stream, terminated properly.
_ANTHROPIC_COMPLETE = _ANTHROPIC_TRUNCATED + (
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n'
    "\n"
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
    '"stop_sequence":null},"usage":{"output_tokens":4}}\n'
    "\n"
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n'
    "\n"
)

# A stream cut before any *real* content: the preamble arrived, then silence.
# This is the shape that a local backend dying at slot-acquisition time
# produces, and — under ``empty_response_action: fallback``, which withholds
# the preamble from the client — the only shape where a clean provider swap is
# still possible (see ``_stream_event_is_real_content``).
_ANTHROPIC_TRUNCATED_PREAMBLE = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_3","type":"message",'
    '"role":"assistant","content":[],"model":"m","stop_reason":null,'
    '"usage":{"input_tokens":7,"output_tokens":0}}}\n'
    "\n"
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n'
    "\n"
)

# A provider that ends on message_delta (stop_reason present) but omits the
# message_stop terminator. Semantically complete — must NOT be flagged.
_ANTHROPIC_NO_MESSAGE_STOP = _ANTHROPIC_TRUNCATED + (
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n'
    "\n"
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
    '"stop_sequence":null},"usage":{"output_tokens":4}}\n'
    "\n"
)

# Cut in the middle of a tool_use block's input_json_delta.
_ANTHROPIC_TRUNCATED_IN_TOOL_USE = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_2","type":"message",'
    '"role":"assistant","content":[],"model":"m","stop_reason":null,'
    '"usage":{"input_tokens":7,"output_tokens":0}}}\n'
    "\n"
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"tool_use","id":"toolu_1","name":"fn","input":{}}}\n'
    "\n"
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"{\\"path\\": \\"/et"}}\n'
    "\n"
)


def _openai_chunk(text: str, finish: str | None = None) -> str:
    delta: dict[str, Any] = {"content": text} if text else {}
    choice: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": finish}
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [choice],
    }
    return f"data: {json.dumps(body)}\n\n"


_OPENAI_TRUNCATED = _openai_chunk("half a ") + _openai_chunk("sentence")
_OPENAI_COMPLETE = _OPENAI_TRUNCATED + _openai_chunk("", "stop") + "data: [DONE]\n\n"
# finish_reason without the [DONE] sentinel — complete, must not be flagged.
_OPENAI_NO_DONE = _OPENAI_TRUNCATED + _openai_chunk("", "stop")

# One chunk with an empty delta: enough to open the Anthropic message_start
# preamble, but no *real* content — the OpenAI-wire counterpart of
# ``_ANTHROPIC_TRUNCATED_PREAMBLE``.
_OPENAI_TRUNCATED_PREAMBLE = _openai_chunk("")


# ----------------------------------------------------------------------
# Config / helper builders
# ----------------------------------------------------------------------


def _anthropic_provider(name: str = "trunc", url: str = _ANTHROPIC_URL) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="anthropic",
        base_url=url.removesuffix("/v1/messages"),
        model="m",
    )


def _openai_provider(name: str = "trunc-oai") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="openai_compat",
        base_url=_OPENAI_URL.removesuffix("/chat/completions"),
        model="m",
    )


def _overrides(action: str) -> ProviderCallOverrides:
    return ProviderCallOverrides(stream_truncation_action=action)  # type: ignore[arg-type]


def _anth_req(stream: bool = True) -> AnthropicRequest:
    return AnthropicRequest(
        max_tokens=64,
        messages=[AnthropicMessage(role="user", content="hi")],
        stream=stream,
    )


def _chat_req(stream: bool = True) -> ChatRequest:
    return ChatRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
        stream=stream,
    )


def _sse(body: str) -> dict[str, Any]:
    return {
        "content": body.encode("utf-8"),
        "headers": {"content-type": "text/event-stream"},
    }


def _truncation_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.msg == "stream-truncation-detected"]


# ======================================================================
# 1. AnthropicAdapter.stream_anthropic
# ======================================================================


class TestAnthropicAdapterDetection:
    async def test_off_is_silent_and_yields_every_event(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Default action: a truncated stream is delivered exactly as before.

        This is the byte-compat anchor for the native path — no exception,
        no log line, and the same event sequence v2.14.0 produced.
        """
        httpx_mock.add_response(
            url=_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_TRUNCATED)
        )
        adapter = AnthropicAdapter(_anthropic_provider())
        with caplog.at_level(logging.WARNING):
            events = [e async for e in adapter.stream_anthropic(_anth_req())]

        assert [e.type for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
        ]
        assert _truncation_records(caplog) == []

    async def test_off_is_the_default_when_no_overrides_passed(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``overrides=None`` (legacy call sites / direct unit tests) is off."""
        httpx_mock.add_response(
            url=_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_TRUNCATED)
        )
        adapter = AnthropicAdapter(_anthropic_provider())
        assert adapter.effective_stream_truncation_action(None) == "off"
        with caplog.at_level(logging.WARNING):
            events = [
                e async for e in adapter.stream_anthropic(_anth_req(), overrides=None)
            ]
        assert len(events) == 4
        assert _truncation_records(caplog) == []

    async def test_warn_logs_but_still_yields_every_event(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        httpx_mock.add_response(
            url=_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_TRUNCATED)
        )
        adapter = AnthropicAdapter(_anthropic_provider())
        with caplog.at_level(logging.WARNING):
            events = [
                e
                async for e in adapter.stream_anthropic(
                    _anth_req(), overrides=_overrides("warn")
                )
            ]

        assert len(events) == 4  # nothing withheld under warn
        (rec,) = _truncation_records(caplog)
        assert rec.provider == "trunc"
        assert rec.action == "warn"
        assert rec.wire == "anthropic"
        assert rec.events_forwarded == 4
        assert rec.saw_stream_start is True
        assert rec.tool_call_in_flight is False

    async def test_error_raises_after_yielding_what_arrived(
        self, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_TRUNCATED)
        )
        adapter = AnthropicAdapter(_anthropic_provider())
        seen = []
        with pytest.raises(StreamTruncatedError) as excinfo:
            async for ev in adapter.stream_anthropic(
                _anth_req(), overrides=_overrides("error")
            ):
                seen.append(ev)

        # Everything the upstream managed to send still reached the caller;
        # only the missing tail became an error.
        assert len(seen) == 4
        exc = excinfo.value
        assert isinstance(exc, AdapterError)  # existing engine branches catch it
        assert exc.retryable is True
        assert exc.status_code is None
        assert exc.provider == "trunc"
        assert exc.tool_call_in_flight is False

    async def test_complete_stream_never_raises(self, httpx_mock: HTTPXMock) -> None:
        """False-positive guard: a well-terminated stream is untouched."""
        httpx_mock.add_response(
            url=_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_COMPLETE)
        )
        adapter = AnthropicAdapter(_anthropic_provider())
        events = [
            e
            async for e in adapter.stream_anthropic(
                _anth_req(), overrides=_overrides("error")
            )
        ]
        assert [e.type for e in events][-1] == "message_stop"

    async def test_message_delta_stop_reason_counts_as_terminator(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """False-positive guard: a provider that omits ``message_stop``.

        ``convert.py``'s M9 comment explicitly anticipates "a provider that
        omits the terminator". Such a stream is semantically complete —
        ``message_delta`` already carried the stop_reason — so flagging it
        would be a false positive.
        """
        httpx_mock.add_response(
            url=_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_NO_MESSAGE_STOP)
        )
        adapter = AnthropicAdapter(_anthropic_provider())
        with caplog.at_level(logging.WARNING):
            events = [
                e
                async for e in adapter.stream_anthropic(
                    _anth_req(), overrides=_overrides("error")
                )
            ]
        assert [e.type for e in events][-1] == "message_delta"
        assert _truncation_records(caplog) == []

    async def test_tool_use_in_flight_is_reported(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A cut inside ``input_json_delta`` is flagged as such.

        ``_close_current_block`` in ``translation/convert.py`` emits a bare
        ``content_block_stop`` — it does not repair or validate the partial
        argument JSON — so this is the case where the synthesized terminator
        hands the client a structurally valid tool_use block whose input is
        provably incomplete.
        """
        httpx_mock.add_response(
            url=_ANTHROPIC_URL,
            method="POST",
            **_sse(_ANTHROPIC_TRUNCATED_IN_TOOL_USE),
        )
        adapter = AnthropicAdapter(_anthropic_provider())
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(StreamTruncatedError) as excinfo,
        ):
            async for _ in adapter.stream_anthropic(
                _anth_req(), overrides=_overrides("error")
            ):
                pass

        assert excinfo.value.tool_call_in_flight is True
        (rec,) = _truncation_records(caplog)
        assert rec.tool_call_in_flight is True

    async def test_completely_empty_body_reports_no_stream_start(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        httpx_mock.add_response(url=_ANTHROPIC_URL, method="POST", **_sse(""))
        adapter = AnthropicAdapter(_anthropic_provider())
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(StreamTruncatedError),
        ):
            async for _ in adapter.stream_anthropic(
                _anth_req(), overrides=_overrides("error")
            ):
                pass
        (rec,) = _truncation_records(caplog)
        assert rec.saw_stream_start is False
        assert rec.events_forwarded == 0


# ======================================================================
# 2. OpenAICompatAdapter.stream
# ======================================================================


class TestOpenAICompatAdapterDetection:
    async def test_off_is_silent_and_yields_every_chunk(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Byte-compat anchor for the openai_compat path."""
        httpx_mock.add_response(
            url=_OPENAI_URL, method="POST", **_sse(_OPENAI_TRUNCATED)
        )
        adapter = OpenAICompatAdapter(_openai_provider())
        with caplog.at_level(logging.WARNING):
            chunks = [c async for c in adapter.stream(_chat_req())]
        assert len(chunks) == 2
        assert _truncation_records(caplog) == []

    async def test_warn_logs_but_still_yields_every_chunk(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        httpx_mock.add_response(
            url=_OPENAI_URL, method="POST", **_sse(_OPENAI_TRUNCATED)
        )
        adapter = OpenAICompatAdapter(_openai_provider())
        with caplog.at_level(logging.WARNING):
            chunks = [
                c async for c in adapter.stream(_chat_req(), overrides=_overrides("warn"))
            ]
        assert len(chunks) == 2
        (rec,) = _truncation_records(caplog)
        assert rec.provider == "trunc-oai"
        assert rec.action == "warn"
        assert rec.wire == "openai"
        assert rec.events_forwarded == 2

    async def test_error_raises_after_yielding_what_arrived(
        self, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=_OPENAI_URL, method="POST", **_sse(_OPENAI_TRUNCATED)
        )
        adapter = OpenAICompatAdapter(_openai_provider())
        seen = []
        with pytest.raises(StreamTruncatedError) as excinfo:
            async for c in adapter.stream(_chat_req(), overrides=_overrides("error")):
                seen.append(c)
        assert len(seen) == 2
        assert excinfo.value.retryable is True
        assert excinfo.value.provider == "trunc-oai"

    async def test_done_sentinel_never_raises(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=_OPENAI_URL, method="POST", **_sse(_OPENAI_COMPLETE)
        )
        adapter = OpenAICompatAdapter(_openai_provider())
        chunks = [
            c async for c in adapter.stream(_chat_req(), overrides=_overrides("error"))
        ]
        assert len(chunks) == 3

    async def test_finish_reason_without_done_never_raises(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """False-positive guard: ``[DONE]`` is optional when a finish_reason
        arrived — the message is complete either way."""
        httpx_mock.add_response(
            url=_OPENAI_URL, method="POST", **_sse(_OPENAI_NO_DONE)
        )
        adapter = OpenAICompatAdapter(_openai_provider())
        with caplog.at_level(logging.WARNING):
            chunks = [
                c
                async for c in adapter.stream(
                    _chat_req(), overrides=_overrides("error")
                )
            ]
        assert len(chunks) == 3
        assert _truncation_records(caplog) == []

    async def test_tool_call_in_flight_is_reported(
        self, httpx_mock: HTTPXMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        body = (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-2",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "fn",
                                            "arguments": '{"path": "/et',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
            + "\n\n"
        )
        httpx_mock.add_response(url=_OPENAI_URL, method="POST", **_sse(body))
        adapter = OpenAICompatAdapter(_openai_provider())
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(StreamTruncatedError) as excinfo,
        ):
            async for _ in adapter.stream(_chat_req(), overrides=_overrides("error")):
                pass
        assert excinfo.value.tool_call_in_flight is True
        (rec,) = _truncation_records(caplog)
        assert rec.tool_call_in_flight is True


# ======================================================================
# 3. Engine integration (real adapters behind httpx_mock)
# ======================================================================


def _two_provider_config(
    *,
    truncation_action: str,
    empty_response_action: str = "off",
    partial_stitch_action: str = "off",
    first_kind: str = "anthropic",
) -> CodeRouterConfig:
    """A `first` provider that truncates and a healthy `second`."""
    if first_kind == "anthropic":
        first = _anthropic_provider("first", _ANTHROPIC_URL)
    else:
        first = _openai_provider("first")
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            first,
            _anthropic_provider("second", _SECOND_ANTHROPIC_URL),
        ],
        profiles=[
            FallbackChain(
                name="default",
                providers=["first", "second"],
                stream_truncation_action=truncation_action,  # type: ignore[arg-type]
                empty_response_action=empty_response_action,  # type: ignore[arg-type]
                partial_stitch_action=partial_stitch_action,  # type: ignore[arg-type]
            )
        ],
    )


def _build_engine(config: CodeRouterConfig):
    adapters: dict[str, Any] = {}
    for prov in config.providers:
        if prov.kind == "anthropic":
            adapters[prov.name] = AnthropicAdapter(prov)
        else:
            adapters[prov.name] = OpenAICompatAdapter(prov)
    return _engine_with_adapters(config, adapters)


async def _collect_stream(engine, req: AnthropicRequest) -> list[Any]:
    return [ev async for ev in engine.stream_anthropic(req)]


class TestEngineIntegration:
    async def test_off_delivers_the_truncated_stream_unchanged(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """(e) The v2.14.0 regression proof, end to end.

        With the default ``off`` the truncated first provider still "wins":
        the chain is not advanced, the client receives the partial events,
        and the second provider is never contacted.
        """
        httpx_mock.add_response(
            url=_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_TRUNCATED)
        )
        config = _two_provider_config(truncation_action="off")
        engine = _build_engine(config)

        events = await _collect_stream(engine, _anth_req())

        assert [e.type for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
        ]
        # The second provider was never called — no request recorded for it.
        assert all(
            str(r.url) != _SECOND_ANTHROPIC_URL for r in httpx_mock.get_requests()
        )

    async def test_error_falls_back_when_nothing_reached_the_client(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """(a)+(b) Native passthrough: truncation → next provider.

        ``empty_response_action: fallback`` is what withholds the opening
        events from the client, which is the precondition for a clean
        provider swap; ``stream_truncation_action: error`` is what turns the
        silent cut into a failure the engine can act on.
        """
        httpx_mock.add_response(
            url=_ANTHROPIC_URL,
            method="POST",
            **_sse(_ANTHROPIC_TRUNCATED_PREAMBLE),
        )
        httpx_mock.add_response(
            url=_SECOND_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_COMPLETE)
        )
        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        engine = _build_engine(config)

        events = await _collect_stream(engine, _anth_req())

        assert [e.type for e in events][-1] == "message_stop"
        # The first provider's withheld preamble never reached the client:
        # exactly one message_start, and it carries the second provider's.
        assert sum(1 for e in events if e.type == "message_start") == 1
        urls = [str(r.url) for r in httpx_mock.get_requests()]
        assert _ANTHROPIC_URL in urls
        assert _SECOND_ANTHROPIC_URL in urls

    async def test_error_records_the_stream_truncated_reason(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """The fallback hop is labelled ``stream-truncated``, not
        ``empty-stream`` — the upstream *did* produce events."""
        from coderouter.routing.fallback_trace import current_fallback_trace

        httpx_mock.add_response(
            url=_ANTHROPIC_URL,
            method="POST",
            **_sse(_ANTHROPIC_TRUNCATED_PREAMBLE),
        )
        httpx_mock.add_response(
            url=_SECOND_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_COMPLETE)
        )
        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        engine = _build_engine(config)

        await _collect_stream(engine, _anth_req())

        trace = current_fallback_trace()
        assert trace is not None
        assert trace.occurred
        payload = trace.as_event_payload()
        reasons = [hop["reason"] for hop in payload["hops"]]
        assert REASON_STREAM_TRUNCATED in reasons
        hop = next(h for h in payload["hops"] if h["reason"] == REASON_STREAM_TRUNCATED)
        assert hop["from"] == "first"
        assert hop["to"] == "second"
        assert hop["detail"] == "no-terminator"

    async def test_openai_compat_path_falls_back(self, httpx_mock: HTTPXMock) -> None:
        """(c) The translated path: OpenAI SSE in, Anthropic events out."""
        httpx_mock.add_response(
            url=_OPENAI_URL, method="POST", **_sse(_OPENAI_TRUNCATED_PREAMBLE)
        )
        httpx_mock.add_response(
            url=_SECOND_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_COMPLETE)
        )
        config = _two_provider_config(
            truncation_action="error",
            empty_response_action="fallback",
            first_kind="openai_compat",
        )
        engine = _build_engine(config)

        events = await _collect_stream(engine, _anth_req())

        assert [e.type for e in events][-1] == "message_stop"
        urls = [str(r.url) for r in httpx_mock.get_requests()]
        assert _OPENAI_URL in urls
        assert _SECOND_ANTHROPIC_URL in urls

    async def test_openai_compat_off_synthesizes_the_terminator(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """(e) The H6 guard still fabricates ``end_turn`` under ``off``.

        This is the behavior the memo calls "active concealment" — it stays,
        because removing it hangs the client. The point of this release is
        that the engine can now be *told* about it, not that it stops.
        """
        httpx_mock.add_response(
            url=_OPENAI_URL, method="POST", **_sse(_OPENAI_TRUNCATED)
        )
        config = _two_provider_config(
            truncation_action="off", first_kind="openai_compat"
        )
        engine = _build_engine(config)

        events = await _collect_stream(engine, _anth_req())

        assert [e.type for e in events][-1] == "message_stop"
        delta = next(e for e in events if e.type == "message_delta")
        assert delta.data["delta"]["stop_reason"] == "end_turn"
        assert all(
            str(r.url) != _SECOND_ANTHROPIC_URL for r in httpx_mock.get_requests()
        )

    async def test_midstream_when_bytes_already_forwarded(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """(d) Once the client has bytes, truncation is terminal.

        Without ``empty_response_action: fallback`` the opening events are
        forwarded immediately, so by the time the cut is detected a swap
        would corrupt the stream. The engine raises ``MidStreamError``,
        carrying the partial text for ``partial_stitch_action: surface``.
        """
        httpx_mock.add_response(
            url=_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_TRUNCATED)
        )
        config = _two_provider_config(truncation_action="error")
        engine = _build_engine(config)

        seen = []
        with pytest.raises(MidStreamError) as excinfo:
            async for ev in engine.stream_anthropic(_anth_req()):
                seen.append(ev)

        assert len(seen) == 4
        exc = excinfo.value
        assert exc.provider == "first"
        assert isinstance(exc.original, StreamTruncatedError)
        assert exc.partial_content == [{"type": "text", "text": "half a sentence"}]
        # No fallback was attempted.
        assert all(
            str(r.url) != _SECOND_ANTHROPIC_URL for r in httpx_mock.get_requests()
        )

    async def test_complete_stream_under_error_never_falls_back(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """(f) False-positive guard at engine level."""
        httpx_mock.add_response(
            url=_ANTHROPIC_URL, method="POST", **_sse(_ANTHROPIC_COMPLETE)
        )
        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        engine = _build_engine(config)

        events = await _collect_stream(engine, _anth_req())

        assert [e.type for e in events][-1] == "message_stop"
        assert all(
            str(r.url) != _SECOND_ANTHROPIC_URL for r in httpx_mock.get_requests()
        )


# ======================================================================
# 4. Ingress: partial stitch labelling
# ======================================================================


class TestIngressPartialReason:
    async def test_surface_labels_truncation(self) -> None:
        from coderouter.ingress.anthropic_routes import _anthropic_sse_iterator
        from coderouter.translation import AnthropicStreamEvent

        partial = [{"type": "text", "text": "half a sentence"}]
        mid = MidStreamError(
            "first",
            StreamTruncatedError("no terminator", provider="first"),
            partial_content=partial,
        )

        async def _raise(req):
            yield AnthropicStreamEvent(
                type="message_start",
                data={"type": "message_start", "message": {"usage": {}}},
            )
            raise mid

        engine = MagicMock()
        engine.stream_anthropic = _raise
        profile_cfg = MagicMock()
        profile_cfg.partial_stitch_action = "surface"
        engine.config = MagicMock()
        engine.config.default_profile = "default"
        engine.config.profile_by_name.return_value = profile_cfg

        chunks = [c async for c in _anthropic_sse_iterator(engine, _anth_req())]
        partial_frame = next(c for c in chunks if "event: coderouter_partial" in c)
        data = json.loads(partial_frame.split("data: ")[1].strip())
        assert data["reason"] == "stream_truncated"
        assert data["partial_content"] == partial

    async def test_surface_keeps_legacy_reason_for_other_failures(self) -> None:
        """Regression: an ordinary mid-stream failure is still labelled
        ``mid_stream_failure``."""
        from coderouter.ingress.anthropic_routes import _anthropic_sse_iterator
        from coderouter.translation import AnthropicStreamEvent

        mid = MidStreamError(
            "first",
            AdapterError("boom", provider="first", status_code=502),
            partial_content=[{"type": "text", "text": "x"}],
        )

        async def _raise(req):
            yield AnthropicStreamEvent(
                type="message_start",
                data={"type": "message_start", "message": {"usage": {}}},
            )
            raise mid

        engine = MagicMock()
        engine.stream_anthropic = _raise
        profile_cfg = MagicMock()
        profile_cfg.partial_stitch_action = "surface"
        engine.config = MagicMock()
        engine.config.default_profile = "default"
        engine.config.profile_by_name.return_value = profile_cfg

        chunks = [c async for c in _anthropic_sse_iterator(engine, _anth_req())]
        partial_frame = next(c for c in chunks if "event: coderouter_partial" in c)
        data = json.loads(partial_frame.split("data: ")[1].strip())
        assert data["reason"] == "mid_stream_failure"


# ======================================================================
# 5. Vocabulary / config / metrics wiring
# ======================================================================


class TestVocabulary:
    def test_reason_is_an_attempt_failure(self) -> None:
        assert REASON_STREAM_TRUNCATED == "stream-truncated"
        assert REASON_STREAM_TRUNCATED in ATTEMPT_FAILURE_REASONS
        assert REASON_STREAM_TRUNCATED in FALLBACK_REASONS

    def test_classify_prefers_the_truncation_reason(self) -> None:
        exc = StreamTruncatedError("cut", provider="p")
        assert classify_adapter_error(exc) == REASON_STREAM_TRUNCATED
        # …and does not fall through to the generic transport bucket.
        assert describe_adapter_error(exc) == "no-terminator"

    def test_plain_adapter_errors_are_unaffected(self) -> None:
        assert classify_adapter_error(
            AdapterError("transport error: boom", provider="p")
        ) == "connection"
        assert (
            describe_adapter_error(AdapterError("x", provider="p", status_code=500))
            == "status=500"
        )


class TestConfigPlumbing:
    def test_profile_default_is_off(self) -> None:
        chain = FallbackChain(name="default", providers=["a"])
        assert chain.stream_truncation_action == "off"

    def test_overrides_default_is_off(self) -> None:
        assert ProviderCallOverrides().stream_truncation_action == "off"

    def test_engine_threads_the_profile_value_into_overrides(self) -> None:
        config = _two_provider_config(truncation_action="warn")
        engine = _build_engine(config)
        overrides = engine._resolve_profile_overrides("default")
        assert overrides.stream_truncation_action == "warn"

    def test_invalid_action_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            FallbackChain(
                name="default",
                providers=["a"],
                stream_truncation_action="fallback",  # type: ignore[arg-type]
            )


class TestMetrics:
    def _record(self, provider: str, action: str) -> logging.LogRecord:
        rec = logging.LogRecord(
            name="coderouter",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="stream-truncation-detected",
            args=(),
            exc_info=None,
        )
        rec.provider = provider
        rec.action = action
        rec.wire = "anthropic"
        rec.events_forwarded = 4
        rec.saw_stream_start = True
        rec.tool_call_in_flight = False
        return rec

    def test_collector_counts_per_provider_and_action(self) -> None:
        collector = MetricsCollector()
        collector.emit(self._record("llama-local", "warn"))
        collector.emit(self._record("llama-local", "warn"))
        collector.emit(self._record("ollama", "error"))

        counters = collector.snapshot()["counters"]
        assert counters["stream_truncated_total"] == 3
        assert counters["stream_truncated_by_provider"]["llama-local"] == 2
        assert counters["stream_truncated_by_provider"]["ollama"] == 1
        assert counters["stream_truncated_by_action"]["warn"] == 2
        assert counters["stream_truncated_by_action"]["error"] == 1

    def test_collector_reset_clears_the_counters(self) -> None:
        collector = MetricsCollector()
        collector.emit(self._record("llama-local", "warn"))
        collector.reset()
        counters = collector.snapshot()["counters"]
        assert counters["stream_truncated_total"] == 0
        assert counters["stream_truncated_by_provider"] == {}
        assert counters["stream_truncated_by_action"] == {}

    def test_prometheus_export(self) -> None:
        collector = MetricsCollector()
        collector.emit(self._record("llama-local", "warn"))
        text = format_prometheus(collector.snapshot())
        assert 'stream_truncated_total{provider="llama-local"} 1' in text
        assert 'stream_truncated_by_action_total{action="warn"} 1' in text

    def test_prometheus_omits_the_metric_when_unused(self) -> None:
        """Zero-cost when nobody enabled the knob."""
        text = format_prometheus(MetricsCollector().snapshot())
        assert "stream_truncated_total" not in text


# ======================================================================
# 6. Post-review hardening (v2.15.0)
#
# Three defects found by adversarial review of the initial implementation:
#
#   (1) the "no bytes forwarded yet" fallback branch recorded an adaptive
#       failure but never told the L2 / L4 / L5 self-healing guards, while
#       ``_observe_provider_success`` had *already* fired at first event —
#       so a backend that kept going quiet stayed "healthy";
#   (2) a chain where every provider truncated flushed the last (unterminated)
#       buffer and hung the client, and mislabelled it ``empty-response``;
#   (3) the ``partial-stitch-surfaced`` log gained an unconditional ``reason``
#       key, changing the JSON log shape even under the default ``off``.
# ======================================================================


def _fake_pair(
    config: CodeRouterConfig,
    first: Any,
    second: Any,
) -> Any:
    return _engine_with_adapters(config, {"first": first, "second": second})


def _spy_guards(monkeypatch: pytest.MonkeyPatch, engine: Any) -> dict[str, list[Any]]:
    """Wrap the L2/L5 and L4 observation hooks, keeping real behavior."""
    seen: dict[str, list[Any]] = {"failure": [], "drift": []}
    orig_failure = engine._observe_provider_failure
    orig_drift = engine._observe_drift_signal

    def spy_failure(provider: str, exc: Any, **kwargs: Any) -> Any:
        seen["failure"].append((provider, exc))
        return orig_failure(provider, exc, **kwargs)

    def spy_drift(provider: str, **kwargs: Any) -> Any:
        seen["drift"].append((provider, kwargs))
        return orig_drift(provider, **kwargs)

    monkeypatch.setattr(engine, "_observe_provider_failure", spy_failure)
    monkeypatch.setattr(engine, "_observe_drift_signal", spy_drift)
    return seen


class TestSelfHealingLearnsPreContentFailures:
    """(1) The guards must see a failure that never reached the client."""

    async def test_truncation_before_any_byte_notifies_the_guards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The headline path: first provider opens a stream, goes quiet,
        engine swaps to the second — and L2/L4/L5 learn about it.

        Without this, ``_observe_provider_success`` (fired the moment the
        first event landed) is the *only* health signal the silent backend
        ever produces, so adaptive routing keeps preferring it for its fast
        first byte. That is precisely what the release notes promise not to
        happen.
        """
        from tests.test_fallback_anthropic import FakeAnthropicAdapter

        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        first = FakeAnthropicAdapter(
            config.provider_by_name("first"),
            stream_fail_with=StreamTruncatedError(
                "upstream stream truncated (no message_stop terminator)",
                provider="first",
            ),
            stream_fail_after=1,  # message_start out, then silence
        )
        second = FakeAnthropicAdapter(config.provider_by_name("second"), text="ok")
        engine = _fake_pair(config, first, second)
        seen = _spy_guards(monkeypatch, engine)

        events = await _collect_stream(engine, _anth_req())

        assert [e.type for e in events][-1] == "message_stop"
        # L2 / L5
        assert [p for p, _ in seen["failure"]] == ["first"]
        assert isinstance(seen["failure"][0][1], StreamTruncatedError)
        # L4 — recorded as an error observation on the streaming wire.
        drift_errors = [
            (p, kw) for p, kw in seen["drift"] if kw.get("is_error") is True
        ]
        assert [p for p, _ in drift_errors] == ["first"]
        assert drift_errors[0][1]["stream"] is True

    async def test_plain_adapter_error_before_any_byte_notifies_the_guards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same branch also carries the pre-existing empty-stream case.

        The observations are deliberately NOT gated on
        ``StreamTruncatedError``: an ``AdapterError`` raised here means the
        provider failed, whatever the label. Before this fix, whether a
        provider's failure reached the health guards depended on
        ``empty_response_action`` — an unrelated knob — because the identical
        failure under ``off`` takes the mid-stream path and *is* observed.
        """
        from tests.test_fallback_anthropic import FakeAnthropicAdapter

        config = _two_provider_config(
            truncation_action="off", empty_response_action="fallback"
        )
        first = FakeAnthropicAdapter(
            config.provider_by_name("first"),
            stream_fail_with=AdapterError("boom", provider="first", retryable=True),
            stream_fail_after=1,
        )
        second = FakeAnthropicAdapter(config.provider_by_name("second"), text="ok")
        engine = _fake_pair(config, first, second)
        seen = _spy_guards(monkeypatch, engine)

        events = await _collect_stream(engine, _anth_req())

        assert [e.type for e in events][-1] == "message_stop"
        assert [p for p, _ in seen["failure"]] == ["first"]
        assert any(kw.get("is_error") is True for _, kw in seen["drift"])

    async def test_a_clean_empty_stream_is_still_not_a_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard on the *other* side of the line.

        A stream that terminates properly but carries no content is a 200
        blank, not a provider failure — it exits the loop normally and must
        not reach the failure hooks. Only the ``except AdapterError`` branch
        got the new observations.
        """
        from tests.test_fallback_anthropic import FakeAnthropicAdapter
        from tests.test_fallback_empty_response import _empty_native_events

        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        first = FakeAnthropicAdapter(
            config.provider_by_name("first"),
            events=_empty_native_events("m"),
        )
        second = FakeAnthropicAdapter(config.provider_by_name("second"), text="ok")
        engine = _fake_pair(config, first, second)
        seen = _spy_guards(monkeypatch, engine)

        events = await _collect_stream(engine, _anth_req())

        assert [e.type for e in events][-1] == "message_stop"
        assert first.stream_calls and second.stream_calls
        assert seen["failure"] == []
        assert not any(kw.get("is_error") for _, kw in seen["drift"])


class TestChainExhaustedByTruncation:
    """(2) A fully-truncated chain must not hand the client a dead stream."""

    @staticmethod
    def _truncating(config: CodeRouterConfig, name: str) -> Any:
        from tests.test_fallback_anthropic import FakeAnthropicAdapter

        return FakeAnthropicAdapter(
            config.provider_by_name(name),
            stream_fail_with=StreamTruncatedError(
                "upstream stream truncated (no message_stop terminator)",
                provider=name,
            ),
            stream_fail_after=1,
        )

    async def test_every_provider_truncated_raises_no_providers(self) -> None:
        """The buffered preamble of a truncated stream has no
        ``message_stop`` — flushing it would emit an SSE sequence that never
        ends, with no exception raised and nothing logged as an error. The
        engine raises instead."""
        from coderouter.routing import NoProvidersAvailableError

        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        engine = _fake_pair(
            config,
            self._truncating(config, "first"),
            self._truncating(config, "second"),
        )

        seen: list[Any] = []
        with pytest.raises(NoProvidersAvailableError) as excinfo:
            async for ev in engine.stream_anthropic(_anth_req()):
                seen.append(ev)

        # Nothing was ever forwarded, so raising is safe.
        assert seen == []
        # (1) put the per-hop errors on the exception — this is where they
        # become visible to the operator.
        assert len(excinfo.value.errors) == 2
        assert all(isinstance(e, StreamTruncatedError) for e in excinfo.value.errors)

    async def test_no_empty_response_metric_is_charged_for_a_truncation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``empty_responses_total`` counted truncations as empty responses.
        A truncated stream is not empty — the provider produced events."""
        from coderouter.routing import NoProvidersAvailableError

        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        engine = _fake_pair(
            config,
            self._truncating(config, "first"),
            self._truncating(config, "second"),
        )

        with caplog.at_level(logging.INFO), pytest.raises(NoProvidersAvailableError):
            await _collect_stream(engine, _anth_req())

        assert [r for r in caplog.records if r.msg == "empty-response-detected"] == []

    async def test_the_client_still_gets_a_terminating_sse_stream(self) -> None:
        """Constraint (iii): whatever we do, the client must not hang.

        ``NoProvidersAvailableError`` is raised before any byte is
        forwarded, so the ingress converts it into an ``event: error``
        frame — a terminal frame the client acts on.
        """
        from coderouter.ingress.anthropic_routes import _anthropic_sse_iterator

        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        engine = _fake_pair(
            config,
            self._truncating(config, "first"),
            self._truncating(config, "second"),
        )

        chunks = [c async for c in _anthropic_sse_iterator(engine, _anth_req())]

        err = next(c for c in chunks if c.startswith("event: error"))
        payload = json.loads(err.split("data: ")[1].strip())
        assert payload["error"]["type"] == "overloaded_error"
        # No half-open message was emitted ahead of it.
        assert not any(c.startswith("event: message_start") for c in chunks)

    async def test_all_empty_still_returns_a_blank_200(self) -> None:
        """Constraint (i), the load-bearing back-compat guard: a chain that
        is genuinely *empty* (every provider terminated cleanly with no
        content) still flushes the last buffer and ends on ``message_stop``
        — even with ``stream_truncation_action: error`` switched on."""
        from tests.test_fallback_anthropic import FakeAnthropicAdapter
        from tests.test_fallback_empty_response import _empty_native_events

        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        engine = _fake_pair(
            config,
            FakeAnthropicAdapter(
                config.provider_by_name("first"), events=_empty_native_events("m")
            ),
            FakeAnthropicAdapter(
                config.provider_by_name("second"), events=_empty_native_events("m")
            ),
        )

        events = await _collect_stream(engine, _anth_req())

        types = [e.type for e in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"

    async def test_a_clean_empty_last_hop_is_flushed_even_after_a_truncation(
        self,
    ) -> None:
        """The gate tracks the buffer we are *holding*, not "did a truncation
        ever happen". A truncated first hop followed by a provider that
        answered with a well-formed empty message still yields that 200
        blank — its buffer terminates."""
        from tests.test_fallback_anthropic import FakeAnthropicAdapter
        from tests.test_fallback_empty_response import _empty_native_events

        config = _two_provider_config(
            truncation_action="error", empty_response_action="fallback"
        )
        engine = _fake_pair(
            config,
            self._truncating(config, "first"),
            FakeAnthropicAdapter(
                config.provider_by_name("second"), events=_empty_native_events("m")
            ),
        )

        events = await _collect_stream(engine, _anth_req())

        assert [e.type for e in events][-1] == "message_stop"


class TestPartialStitchLogShape:
    """(3) ``off`` must not change the shape of an existing log line."""

    @staticmethod
    async def _stitch_records(
        original: AdapterError, caplog: pytest.LogCaptureFixture
    ) -> list[logging.LogRecord]:
        from coderouter.ingress.anthropic_routes import _anthropic_sse_iterator
        from coderouter.translation import AnthropicStreamEvent

        mid = MidStreamError(
            "first", original, partial_content=[{"type": "text", "text": "x"}]
        )

        async def _raise(req):
            yield AnthropicStreamEvent(
                type="message_start",
                data={"type": "message_start", "message": {"usage": {}}},
            )
            raise mid

        engine = MagicMock()
        engine.stream_anthropic = _raise
        profile_cfg = MagicMock()
        profile_cfg.partial_stitch_action = "surface"
        engine.config = MagicMock()
        engine.config.default_profile = "default"
        engine.config.profile_by_name.return_value = profile_cfg

        with caplog.at_level(logging.INFO):
            async for _ in _anthropic_sse_iterator(engine, _anth_req()):
                pass
        return [r for r in caplog.records if r.msg == "partial-stitch-surfaced"]

    async def test_off_keeps_the_v2_14_0_log_keys(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An ordinary mid-stream failure — the only kind that can occur
        under the default ``off`` — logs no ``reason`` key. The JSON
        formatter emits whatever ``extra`` carries, so an unconditional key
        would change every such line."""
        (rec,) = await self._stitch_records(
            AdapterError("boom", provider="first", status_code=502), caplog
        )
        assert not hasattr(rec, "reason")
        assert rec.provider == "first"
        assert rec.text_blocks == 1

    async def test_truncation_still_labels_the_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The key is not removed — it is conditional. A truncation, which
        can only happen when the operator opted in, still carries it."""
        (rec,) = await self._stitch_records(
            StreamTruncatedError("no terminator", provider="first"), caplog
        )
        assert rec.reason == "stream_truncated"
