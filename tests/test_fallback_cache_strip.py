"""Engine-level tests for the S3 cache_control strip shim.

Focus: FallbackEngine.generate_anthropic / stream_anthropic with a
cache_control request routed through a provider that does not support it,
under ``cache_control_action``.

  * ``strip`` → the adapter receives a request with the cache_control keys
    removed; a ``cache-control-stripped`` log fires with the marker count;
    the original request is untouched; a supporting provider is not stripped.
  * ``off``   → legacy behavior — the request is passed through unchanged
    and only the existing v0.5-B ``capability-degraded`` log fires (no
    ``cache-control-stripped`` line).

Reuses the FakeAdapter scaffolding from tests/test_fallback_anthropic.py.
"""

from __future__ import annotations

import logging

import pytest

from coderouter.adapters.base import BaseAdapter
from coderouter.config.schemas import (
    Capabilities,
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.routing import FallbackEngine
from coderouter.translation.anthropic import AnthropicRequest
from tests.test_fallback_anthropic import FakeAnthropicAdapter, FakeOpenAIAdapter

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _anthropic_provider(name: str) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
    )


def _openai_provider(name: str, *, prompt_cache: bool = False) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="qwen-coder",
        capabilities=Capabilities(prompt_cache=prompt_cache),
    )


def _config(
    providers: list[ProviderConfig],
    chain: list[str],
    *,
    action: str = "off",
) -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=providers,
        profiles=[
            FallbackChain(name="default", providers=chain, cache_control_action=action)
        ],
    )


def _engine(config: CodeRouterConfig, adapters: dict[str, BaseAdapter]) -> FallbackEngine:
    engine = FallbackEngine.__new__(FallbackEngine)
    engine.config = config
    engine._adapters = adapters  # type: ignore[attr-defined]
    return engine


def _cache_request() -> AnthropicRequest:
    return AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 64,
            "system": [
                {
                    "type": "text",
                    "text": "long reusable system prompt",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }
    )


def _stripped_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.msg == "cache-control-stripped"]


def _tokens_saved_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.msg == "tokens-saved"]


# ----------------------------------------------------------------------
# strip — non-streaming
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strip_removes_cache_control_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    compat_cfg = _openai_provider("ollama")
    config = _config([compat_cfg], chain=["ollama"], action="strip")
    compat = FakeOpenAIAdapter(compat_cfg, text="ok")
    engine = _engine(config, {"ollama": compat})

    original = _cache_request()
    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(original)

    logs = _stripped_logs(caplog)
    assert len(logs) == 1
    assert logs[0].provider == "ollama"
    assert logs[0].markers_removed == 2

    # No tokens-saved event — this is a wire-compat strip, not savings.
    assert _tokens_saved_logs(caplog) == []

    # Original request untouched (markers preserved for a later provider).
    assert original.system[0]["cache_control"] == {"type": "ephemeral"}
    assert original.messages[0].content[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_strip_preserves_other_fields(caplog: pytest.LogCaptureFixture) -> None:
    """The strip removes only cache_control; the system text and messages
    survive so the request is otherwise byte-for-byte equivalent."""
    compat_cfg = _openai_provider("ollama")
    config = _config([compat_cfg], chain=["ollama"], action="strip")
    compat = FakeOpenAIAdapter(compat_cfg, text="ok")
    engine = _engine(config, {"ollama": compat})

    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(_cache_request())

    # The translated ChatRequest carries the system text intact.
    chat_req = compat.generate_calls[0]
    system_text = " ".join(
        m.content or "" for m in chat_req.messages if m.role == "system"
    )
    assert "long reusable system prompt" in system_text


@pytest.mark.asyncio
async def test_off_passes_through_no_strip_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """action=off → the request passes through untouched; only the existing
    v0.5-B capability-degraded log fires (no cache-control-stripped)."""
    compat_cfg = _openai_provider("ollama")
    config = _config([compat_cfg], chain=["ollama"], action="off")
    compat = FakeOpenAIAdapter(compat_cfg, text="ok")
    engine = _engine(config, {"ollama": compat})

    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(_cache_request())

    assert _stripped_logs(caplog) == []
    # The legacy translation-lossy observability log still fires.
    degraded = [
        r
        for r in caplog.records
        if r.msg == "capability-degraded"
        and getattr(r, "reason", None) == "translation-lossy"
    ]
    assert len(degraded) == 1


@pytest.mark.asyncio
async def test_supporting_provider_not_stripped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An anthropic-kind provider preserves cache_control end-to-end →
    even with action=strip the markers are left in place, no strip log."""
    anth_cfg = _anthropic_provider("sonnet")
    config = _config([anth_cfg], chain=["sonnet"], action="strip")
    anth = FakeAnthropicAdapter(anth_cfg, text="ok")
    engine = _engine(config, {"sonnet": anth})

    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(_cache_request())

    assert _stripped_logs(caplog) == []
    # The native adapter received the request WITH cache_control intact.
    received = anth.generate_calls[0]
    assert received.system[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_explicit_prompt_cache_flag_suppresses_strip(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """capabilities.prompt_cache: true declares the openai_compat upstream
    preserves the marker → treated as capable, no strip."""
    compat_cfg = _openai_provider("future-compat", prompt_cache=True)
    config = _config([compat_cfg], chain=["future-compat"], action="strip")
    compat = FakeOpenAIAdapter(compat_cfg, text="ok")
    engine = _engine(config, {"future-compat": compat})

    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(_cache_request())

    assert _stripped_logs(caplog) == []


# ----------------------------------------------------------------------
# strip — streaming
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_strip_removes_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    compat_cfg = _openai_provider("ollama")
    config = _config([compat_cfg], chain=["ollama"], action="strip")
    compat = FakeOpenAIAdapter(compat_cfg, text="streamed")
    engine = _engine(config, {"ollama": compat})

    original = _cache_request()
    with caplog.at_level(logging.INFO, logger="coderouter"):
        events = [ev async for ev in engine.stream_anthropic(original)]

    assert events
    logs = _stripped_logs(caplog)
    assert len(logs) == 1
    assert logs[0].markers_removed == 2
    # Original untouched.
    assert original.system[0]["cache_control"] == {"type": "ephemeral"}
