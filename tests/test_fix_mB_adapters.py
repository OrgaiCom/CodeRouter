"""Fix-batch M (M6 + M10) regression tests.

M6 — adapter response-shape validation:
    A 200 response whose JSON body is not a valid Chat Completions /
    Anthropic Messages shape used to raise a bare pydantic
    ``ValidationError`` that escaped the engine's AdapterError-based
    retry/fallback logic. The adapters now convert it to a retryable
    ``AdapterError`` (non-streaming) or skip the bad chunk and continue
    (streaming), logging once per stream.

M10 — memory-pressure "oom" false positives:
    ``is_memory_pressure_error`` matched a bare ``"oom"`` substring, so
    "room" / "zoom" / "bloom" in an unrelated error body wrongly tripped
    the OOM cooldown. The check now uses a word-boundary regex for the
    standalone token while keeping the compound phrases intact.
"""

from __future__ import annotations

import logging

import pytest
from pytest_httpx import HTTPXMock

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import (
    AdapterError,
    ChatRequest,
    Message,
)
from coderouter.adapters.openai_compat import OpenAICompatAdapter
from coderouter.config.schemas import Capabilities, ProviderConfig
from coderouter.guards.memory_pressure import is_memory_pressure_error

# ======================================================================
# M6 — OpenAI-compat adapter: malformed response shape
# ======================================================================


def _oa_provider() -> ProviderConfig:
    return ProviderConfig(
        name="ollama-local",
        base_url="http://localhost:11434/v1",
        model="qwen2.5-coder:14b",
        api_key_env=None,
        capabilities=Capabilities(),
    )


def _oa_request(*, stream: bool = False) -> ChatRequest:
    req = ChatRequest(messages=[Message(role="user", content="hi")])
    req.stream = stream
    return req


@pytest.mark.asyncio
async def test_m6_openai_missing_choices_raises_retryable_adapter_error(
    httpx_mock: HTTPXMock,
) -> None:
    """A 200 body missing ``choices`` must surface as a retryable AdapterError.

    Regression: previously ``ChatResponse(**data)`` raised a bare
    pydantic ValidationError that bypassed the fallback engine.
    """
    # Valid JSON, but not a valid Chat Completions response: an error
    # envelope returned with HTTP 200 (some OpenAI-compat servers do this).
    httpx_mock.add_response(
        url="http://localhost:11434/v1/chat/completions",
        method="POST",
        status_code=200,
        json={"error": {"message": "context length exceeded"}},
    )

    adapter = OpenAICompatAdapter(_oa_provider())
    with pytest.raises(AdapterError) as info:
        await adapter.generate(_oa_request())
    assert info.value.retryable is True
    assert info.value.provider == "ollama-local"
    # Not misclassified as the invalid-JSON (non-retryable) branch.
    assert "malformed response shape" in str(info.value)


@pytest.mark.asyncio
async def test_m6_openai_stream_skips_bad_chunk_and_continues(
    httpx_mock: HTTPXMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed SSE chunk is skipped; well-formed chunks still stream.

    The bad chunk is valid JSON but not a valid StreamChunk (missing the
    required ``id``/``created``/``model`` fields). It must not abort the
    whole stream, and a single ``malformed-stream-chunk`` warning is
    emitted (deduped even when several bad chunks arrive).
    """
    good = (
        '{"id":"x","object":"chat.completion.chunk","created":0,'
        '"model":"qwen2.5-coder:14b","choices":[{"index":0,'
        '"delta":{"content":"hi"}}]}'
    )
    # Two bad chunks to exercise the once-per-stream log dedupe.
    bad = '{"totally":"wrong"}'
    sse_body = (
        f"data: {bad}\n\n"
        f"data: {good}\n\n"
        f"data: {bad}\n\n"
        "data: [DONE]\n\n"
    )
    httpx_mock.add_response(
        url="http://localhost:11434/v1/chat/completions",
        method="POST",
        status_code=200,
        content=sse_body.encode("utf-8"),
        headers={"content-type": "text/event-stream"},
    )

    adapter = OpenAICompatAdapter(_oa_provider())
    with caplog.at_level(logging.WARNING, logger="coderouter"):
        chunks = [c async for c in adapter.stream(_oa_request(stream=True))]

    # Only the well-formed chunk survives.
    assert len(chunks) == 1
    assert chunks[0].id == "x"
    assert chunks[0].choices[0]["delta"]["content"] == "hi"

    # Warning logged exactly once despite two malformed chunks.
    warns = [r for r in caplog.records if r.msg == "malformed-stream-chunk"]
    assert len(warns) == 1
    assert warns[0].provider == "ollama-local"


# ======================================================================
# M6 — Anthropic native adapter: malformed response shape
# ======================================================================


def _anth_provider() -> ProviderConfig:
    return ProviderConfig(
        name="anthropic-direct",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        api_key_env=None,
    )


@pytest.mark.asyncio
async def test_m6_anthropic_missing_content_raises_retryable_adapter_error(
    httpx_mock: HTTPXMock,
) -> None:
    """An Anthropic 200 body missing ``content`` → retryable AdapterError.

    Regression: ``AnthropicResponse.model_validate(data)`` used to raise a
    bare ValidationError that escaped the engine.
    """
    from coderouter.translation.anthropic import AnthropicMessage, AnthropicRequest

    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        status_code=200,
        # Valid JSON, but an error envelope rather than a Messages response.
        json={"type": "error", "error": {"message": "overloaded"}},
    )

    adapter = AnthropicAdapter(_anth_provider())
    req = AnthropicRequest(
        max_tokens=16,
        messages=[AnthropicMessage(role="user", content="hi")],
    )
    with pytest.raises(AdapterError) as info:
        await adapter.generate_anthropic(req)
    assert info.value.retryable is True
    assert info.value.provider == "anthropic-direct"
    assert "malformed response shape" in str(info.value)


# ======================================================================
# M10 — memory-pressure "oom" word-boundary
# ======================================================================


@pytest.mark.parametrize(
    "message",
    [
        "500 from upstream: no space left in room",
        "zoom meeting webhook failed",
        "bloom filter rebuild error",
        "groom the index before retry",
    ],
)
def test_m10_oom_substring_does_not_false_fire(message: str) -> None:
    """"room" / "zoom" / "bloom" / "groom" must NOT count as OOM.

    Regression: a bare ``"oom" in text`` matched all of these, wrongly
    putting a healthy provider into memory-pressure cooldown.
    """
    exc = AdapterError(message, provider="ollama-local", retryable=True)
    assert is_memory_pressure_error(exc) is False


@pytest.mark.parametrize(
    "message",
    [
        "OOM",
        "process killed: OOM",
        "500 from upstream: OOM",
        "out of memory",
        "CUDA out of memory",
        "OOM killed by oom_killer",  # standalone leading "OOM" still matches
    ],
)
def test_m10_standalone_oom_and_phrases_still_fire(message: str) -> None:
    """Standalone "OOM" and the existing compound phrases still detect."""
    exc = AdapterError(message, provider="ollama-local", retryable=True)
    assert is_memory_pressure_error(exc) is True
