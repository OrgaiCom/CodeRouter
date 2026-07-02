"""H3 regression: adapters must reuse a single shared httpx.AsyncClient.

Before the fix each adapter call built a fresh ``httpx.AsyncClient`` in an
``async with`` block, so the connection pool / keep-alive / TLS session were
never reused across requests. The fix moves the client to ``BaseAdapter`` as a
lazily-created shared instance (``BaseAdapter.client()``) with an idempotent
``BaseAdapter.aclose()`` closing it on shutdown.

These tests assert the *identity* contract (same instance across calls) and the
close/re-create lifecycle. Upstream HTTP is mocked with pytest-httpx, matching
the mocking style used by the sibling adapter test modules.
"""

from __future__ import annotations

import httpx
import pytest
from pytest_httpx import HTTPXMock

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import ChatRequest, Message
from coderouter.adapters.openai_compat import OpenAICompatAdapter
from coderouter.config.schemas import Capabilities, ProviderConfig
from coderouter.translation import AnthropicMessage, AnthropicRequest


def _openai_provider() -> ProviderConfig:
    return ProviderConfig(
        name="ollama-local",
        base_url="http://localhost:11434/v1",
        model="qwen2.5-coder:14b",
        api_key_env=None,
        capabilities=Capabilities(),
    )


def _anthropic_provider() -> ProviderConfig:
    return ProviderConfig(
        name="anthropic-native",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        api_key_env=None,
    )


def _openai_response_body(model: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }


def _anthropic_response_body() -> dict[str, object]:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


# ----------------------------------------------------------------------
# Lazy creation + identity of the shared client
# ----------------------------------------------------------------------


def test_client_is_none_until_first_use() -> None:
    """Constructing an adapter must not build a client (no I/O at build time)."""
    adapter = OpenAICompatAdapter(_openai_provider())
    assert adapter._client is None


def test_client_lazily_created_and_cached() -> None:
    """``client()`` builds a client on first use and returns the same one after."""
    adapter = OpenAICompatAdapter(_openai_provider())
    first = adapter.client()
    second = adapter.client()
    assert isinstance(first, httpx.AsyncClient)
    assert first is second
    assert adapter._client is first


@pytest.mark.asyncio
async def test_openai_generate_reuses_same_client(httpx_mock: HTTPXMock) -> None:
    """Two ``generate`` calls on one adapter must share one AsyncClient."""
    httpx_mock.add_response(
        url="http://localhost:11434/v1/chat/completions",
        method="POST",
        json=_openai_response_body("qwen2.5-coder:14b"),
    )
    httpx_mock.add_response(
        url="http://localhost:11434/v1/chat/completions",
        method="POST",
        json=_openai_response_body("qwen2.5-coder:14b"),
    )

    adapter = OpenAICompatAdapter(_openai_provider())
    req = ChatRequest(messages=[Message(role="user", content="hi")])

    await adapter.generate(req)
    client_after_first = adapter._client
    await adapter.generate(req)
    client_after_second = adapter._client

    assert client_after_first is not None
    assert client_after_first is client_after_second


@pytest.mark.asyncio
async def test_anthropic_generate_reuses_same_client(httpx_mock: HTTPXMock) -> None:
    """Two ``generate_anthropic`` calls on one adapter share one AsyncClient."""
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json=_anthropic_response_body(),
    )
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        json=_anthropic_response_body(),
    )

    adapter = AnthropicAdapter(_anthropic_provider())
    req = AnthropicRequest(
        max_tokens=8,
        messages=[AnthropicMessage(role="user", content="hi")],
    )

    await adapter.generate_anthropic(req)
    client_after_first = adapter._client
    await adapter.generate_anthropic(req)
    client_after_second = adapter._client

    assert client_after_first is not None
    assert client_after_first is client_after_second


# ----------------------------------------------------------------------
# aclose lifecycle
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_closes_and_drops_client() -> None:
    """After ``aclose`` the client is closed and the reference is cleared."""
    adapter = OpenAICompatAdapter(_openai_provider())
    client = adapter.client()
    assert client.is_closed is False

    await adapter.aclose()

    assert adapter._client is None
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_aclose_is_idempotent_without_client() -> None:
    """``aclose`` is a safe no-op when no client was ever created."""
    adapter = OpenAICompatAdapter(_openai_provider())
    assert adapter._client is None
    # Should not raise.
    await adapter.aclose()
    await adapter.aclose()
    assert adapter._client is None


@pytest.mark.asyncio
async def test_client_recreated_after_aclose() -> None:
    """A fresh client is created on demand after a previous one was closed."""
    adapter = OpenAICompatAdapter(_openai_provider())
    first = adapter.client()
    await adapter.aclose()
    second = adapter.client()
    assert second is not first
    assert second.is_closed is False
