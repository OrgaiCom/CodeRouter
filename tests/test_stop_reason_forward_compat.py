"""Regression tests for H-6: closed `stop_reason` Literal rejected valid
Anthropic responses.

Before the fix, `AnthropicResponse.stop_reason` was a closed
`Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]`. Anthropic
has since shipped additional stop reasons (`pause_turn`, `refusal`,
`model_context_window_exceeded`); a response using any of them failed
pydantic validation in `AnthropicResponse.model_validate(...)`
(`adapters/anthropic_native.py`), which the adapter turns into a
*retryable* `AdapterError` — silently discarding an already-billed, valid
response and falling over to the next provider in the fallback chain (or
raising `NoProvidersAvailableError` if it was the last one). Streaming was
never affected (`AnthropicStreamEvent` carries its payload as a raw dict).
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import AdapterError, ChatRequest, Message
from coderouter.config.schemas import ProviderConfig
from coderouter.translation.anthropic import AnthropicMessage, AnthropicRequest, AnthropicResponse
from coderouter.translation.convert import _REVERSE_FINISH_REASON_MAP


def _base_response_payload(stop_reason: str) -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


# ----------------------------------------------------------------------
# AnthropicResponse.stop_reason accepts forward-compat values
# ----------------------------------------------------------------------


def test_response_accepts_pause_turn() -> None:
    resp = AnthropicResponse.model_validate(_base_response_payload("pause_turn"))
    assert resp.stop_reason == "pause_turn"


def test_response_accepts_refusal() -> None:
    resp = AnthropicResponse.model_validate(_base_response_payload("refusal"))
    assert resp.stop_reason == "refusal"


def test_response_accepts_model_context_window_exceeded() -> None:
    resp = AnthropicResponse.model_validate(
        _base_response_payload("model_context_window_exceeded")
    )
    assert resp.stop_reason == "model_context_window_exceeded"


def test_response_accepts_unknown_future_stop_reason() -> None:
    """The whole point of `str | None` over a closed Literal: values that
    don't exist yet must not break validation."""
    resp = AnthropicResponse.model_validate(_base_response_payload("some_future_reason"))
    assert resp.stop_reason == "some_future_reason"


def test_response_still_accepts_known_values() -> None:
    for value in ("end_turn", "max_tokens", "stop_sequence", "tool_use"):
        resp = AnthropicResponse.model_validate(_base_response_payload(value))
        assert resp.stop_reason == value


def test_response_accepts_null_stop_reason() -> None:
    resp = AnthropicResponse.model_validate(_base_response_payload(None))
    assert resp.stop_reason is None


# ----------------------------------------------------------------------
# The bug in its natural habitat: non-streaming adapter call must not
# fall back / raise AdapterError just because upstream used pause_turn.
# ----------------------------------------------------------------------


def _provider(**overrides) -> ProviderConfig:
    defaults = dict(
        name="anthropic-native",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
    )
    defaults.update(overrides)
    return ProviderConfig(**defaults)


@pytest.mark.asyncio
async def test_native_adapter_does_not_fallback_on_pause_turn(httpx_mock: HTTPXMock) -> None:
    """This is the bug: a 200 response with stop_reason=pause_turn used to
    raise AdapterError(retryable=True) out of `generate_anthropic`, which
    looks to the fallback engine exactly like an upstream failure."""
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json=_base_response_payload("pause_turn"),
    )
    adapter = AnthropicAdapter(_provider())
    anth_req = AnthropicRequest(
        max_tokens=32,
        messages=[AnthropicMessage(role="user", content="hi")],
    )

    # Must not raise AdapterError — that's the H-6 failure mode.
    resp = await adapter.generate_anthropic(anth_req)
    assert resp.stop_reason == "pause_turn"
    assert resp.content[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_openai_shaped_generate_does_not_fallback_on_refusal(
    httpx_mock: HTTPXMock,
) -> None:
    """Same bug via the OpenAI-shaped `generate()` reverse-translation path."""
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json=_base_response_payload("refusal"),
    )
    adapter = AnthropicAdapter(_provider())
    req = ChatRequest(messages=[Message(role="user", content="hi")])

    try:
        chat_resp = await adapter.generate(req)
    except AdapterError as exc:  # pragma: no cover - failure path documentation
        pytest.fail(f"generate() incorrectly raised AdapterError for a valid response: {exc}")

    assert chat_resp.choices[0]["finish_reason"] == "content_filter"


# ----------------------------------------------------------------------
# Reverse finish_reason map (Anthropic stop_reason -> OpenAI finish_reason)
# ----------------------------------------------------------------------


def test_reverse_map_pause_turn_and_refusal() -> None:
    assert _REVERSE_FINISH_REASON_MAP["pause_turn"] == "stop"
    assert _REVERSE_FINISH_REASON_MAP["refusal"] == "content_filter"
    assert _REVERSE_FINISH_REASON_MAP["model_context_window_exceeded"] == "length"


def test_reverse_map_unknown_value_falls_back_to_stop() -> None:
    """Values not yet in the map must still degrade gracefully (existing
    `.get(..., "stop")` fallback at call sites), not raise KeyError."""
    assert _REVERSE_FINISH_REASON_MAP.get("brand_new_value", "stop") == "stop"


# ----------------------------------------------------------------------
# drift_detection: stop_sequence / pause_turn / refusal must not count as
# anomalies (see coderouter/guards/drift_detection.py _EXPECTED_STOP).
# ----------------------------------------------------------------------


def test_drift_expected_stop_covers_stop_sequence() -> None:
    from coderouter.guards.drift_detection import detect_drift
    from tests.test_drift_detection import _window_of

    window = _window_of(10, stop_reason="stop_sequence", output_tokens=100)
    v = detect_drift(window)
    assert v.signals.get("stop_anomaly_rate", 0) == 0.0


def test_drift_expected_stop_covers_pause_turn_and_refusal() -> None:
    from coderouter.guards.drift_detection import detect_drift
    from tests.test_drift_detection import _window_of

    for stop_reason in ("pause_turn", "refusal"):
        window = _window_of(10, stop_reason=stop_reason, output_tokens=100)
        v = detect_drift(window)
        assert v.signals.get("stop_anomaly_rate", 0) == 0.0
