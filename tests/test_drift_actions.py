"""v2.0-G: Tests for drift_actions.py (reload action).

Tests the Ollama KV cache flush logic with mocked httpx calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from coderouter.config.schemas import ProviderConfig
from coderouter.guards.drift_actions import (
    _is_ollama_shape,
    _ollama_base_url,
    attempt_reload,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ollama_provider(
    *,
    name: str = "local",
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen3:30b",
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="openai_compat",
        base_url=base_url,
        model=model,
    )


def _non_ollama_provider() -> ProviderConfig:
    return ProviderConfig(
        name="openrouter",
        kind="openai_compat",
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-4-scout",
        api_key_env="OPENROUTER_API_KEY",
    )


def _anthropic_provider() -> ProviderConfig:
    return ProviderConfig(
        name="claude",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
    )


# ---------------------------------------------------------------------------
# _is_ollama_shape
# ---------------------------------------------------------------------------


class TestIsOllamaShape:
    def test_port_11434(self):
        assert _is_ollama_shape(_ollama_provider()) is True

    def test_non_standard_port_with_num_ctx(self):
        p = ProviderConfig(
            name="local",
            kind="openai_compat",
            base_url="http://localhost:8080/v1",
            model="qwen3:30b",
            extra_body={"options": {"num_ctx": 32768}},
        )
        assert _is_ollama_shape(p) is True

    def test_non_ollama_openai_compat(self):
        assert _is_ollama_shape(_non_ollama_provider()) is False

    def test_anthropic_kind(self):
        assert _is_ollama_shape(_anthropic_provider()) is False


# ---------------------------------------------------------------------------
# _ollama_base_url
# ---------------------------------------------------------------------------


class TestOllamaBaseUrl:
    def test_strips_v1_suffix(self):
        p = _ollama_provider(base_url="http://localhost:11434/v1")
        assert _ollama_base_url(p) == "http://localhost:11434"

    def test_strips_v1_trailing_slash(self):
        p = _ollama_provider(base_url="http://localhost:11434/v1/")
        assert _ollama_base_url(p) == "http://localhost:11434"

    def test_no_v1_suffix(self):
        p = _ollama_provider(base_url="http://localhost:11434")
        assert _ollama_base_url(p) == "http://localhost:11434"


# ---------------------------------------------------------------------------
# attempt_reload
# ---------------------------------------------------------------------------


class TestAttemptReload:
    @pytest.mark.asyncio
    async def test_success(self):
        provider = _ollama_provider(model="qwen3:30b")
        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("coderouter.guards.drift_actions.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await attempt_reload(provider)

        assert result is True
        mock_client.post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={"model": "qwen3:30b", "keep_alive": 0},
        )

    @pytest.mark.asyncio
    async def test_http_error(self):
        import httpx as _httpx

        provider = _ollama_provider()

        with patch("coderouter.guards.drift_actions.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=_httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await attempt_reload(provider)

        assert result is False

    @pytest.mark.asyncio
    async def test_non_200_response(self):
        provider = _ollama_provider()
        mock_response = AsyncMock()
        mock_response.status_code = 404

        with patch("coderouter.guards.drift_actions.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await attempt_reload(provider)

        assert result is False

    @pytest.mark.asyncio
    async def test_non_ollama_returns_false(self):
        """Non-Ollama providers skip reload entirely."""
        result = await attempt_reload(_non_ollama_provider())
        assert result is False

    @pytest.mark.asyncio
    async def test_anthropic_returns_false(self):
        """Anthropic-kind providers skip reload."""
        result = await attempt_reload(_anthropic_provider())
        assert result is False
