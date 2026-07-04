"""Engine-level tests for the S2 tool_choice shim (warn / emulate).

Focus: FallbackEngine.generate_anthropic / stream_anthropic with a forced
``tool_choice`` request routed through a provider that does not support it.

  * ``emulate`` → the adapter receives a request with NO tool_choice and a
    system-prompt directive injected; the ORIGINAL request is unchanged;
    a ``tool-choice-emulated`` log fires.
  * ``warn``    → the adapter receives the request UNCHANGED; only a
    ``capability-degraded`` (reason=unsupported-backend) log fires.
  * ``off``     → no mutation, no shim log.
  * A supporting (anthropic-kind) provider is never mutated.

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


def _anthropic_provider(name: str, *, tool_choice: bool = False) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
        capabilities=Capabilities(tool_choice=tool_choice),
    )


def _openai_provider(name: str) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="qwen-coder",
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
            FallbackChain(name="default", providers=chain, tool_choice_action=action)
        ],
    )


def _engine(config: CodeRouterConfig, adapters: dict[str, BaseAdapter]) -> FallbackEngine:
    engine = FallbackEngine.__new__(FallbackEngine)
    engine.config = config
    engine._adapters = adapters  # type: ignore[attr-defined]
    return engine


def _forced_request(mode: str = "tool", name: str = "get_weather") -> AnthropicRequest:
    tc: dict = {"type": mode}
    if mode == "tool":
        tc["name"] = name
    return AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 64,
            "system": "base system",
            "messages": [{"role": "user", "content": "what's the weather?"}],
            "tools": [{"name": name, "input_schema": {}}],
            "tool_choice": tc,
        }
    )


def _emulated_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.msg == "tool-choice-emulated"]


def _degraded_logs(
    caplog: pytest.LogCaptureFixture, *, reason: str | None = None
) -> list[logging.LogRecord]:
    records = [r for r in caplog.records if r.msg == "capability-degraded"]
    if reason is not None:
        records = [r for r in records if getattr(r, "reason", None) == reason]
    return records


# ----------------------------------------------------------------------
# emulate — non-streaming
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emulate_strips_tool_choice_and_injects_system(
    caplog: pytest.LogCaptureFixture,
) -> None:
    compat_cfg = _openai_provider("ollama")
    config = _config([compat_cfg], chain=["ollama"], action="emulate")
    compat = FakeOpenAIAdapter(compat_cfg, text="ok")
    engine = _engine(config, {"ollama": compat})

    original = _forced_request()
    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(original)

    # The openai_compat adapter records the ChatRequest it received; the
    # translated system prompt must carry the injected directive and the
    # request must not force a tool anymore. We assert via the translated
    # ChatRequest messages (system role).
    assert compat.generate_calls, "adapter was called"
    chat_req = compat.generate_calls[0]
    system_text = " ".join(
        m.content or "" for m in chat_req.messages if m.role == "system"
    )
    assert 'the tool named "get_weather"' in system_text
    # tool_choice forcing must not survive into the ChatRequest.
    assert getattr(chat_req, "tool_choice", None) in (None, "auto", "none")

    # Original request object is untouched (fallback to a capable provider
    # later in the chain must still see the real tool_choice).
    assert original.tool_choice == {"type": "tool", "name": "get_weather"}
    assert original.system == "base system"

    # Structured log fired with the tool name + mode.
    logs = _emulated_logs(caplog)
    assert len(logs) == 1
    assert logs[0].tool_name == "get_weather"
    assert logs[0].mode == "tool"
    assert logs[0].provider == "ollama"


@pytest.mark.asyncio
async def test_warn_leaves_request_unchanged_logs_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    compat_cfg = _openai_provider("ollama")
    config = _config([compat_cfg], chain=["ollama"], action="warn")
    compat = FakeOpenAIAdapter(compat_cfg, text="ok")
    engine = _engine(config, {"ollama": compat})

    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(_forced_request())

    # No emulation log; a capability-degraded/unsupported-backend log fires.
    assert _emulated_logs(caplog) == []
    degraded = _degraded_logs(caplog, reason="unsupported-backend")
    assert len(degraded) == 1
    assert degraded[0].dropped == ["tool_choice"]
    assert degraded[0].provider == "ollama"

    # The system prompt was NOT rewritten (no directive injected).
    chat_req = compat.generate_calls[0]
    system_text = " ".join(
        m.content or "" for m in chat_req.messages if m.role == "system"
    )
    assert "Do not respond with plain text." not in system_text


@pytest.mark.asyncio
async def test_off_does_not_mutate_or_log(caplog: pytest.LogCaptureFixture) -> None:
    compat_cfg = _openai_provider("ollama")
    config = _config([compat_cfg], chain=["ollama"], action="off")
    compat = FakeOpenAIAdapter(compat_cfg, text="ok")
    engine = _engine(config, {"ollama": compat})

    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(_forced_request())

    assert _emulated_logs(caplog) == []
    assert _degraded_logs(caplog, reason="unsupported-backend") == []


@pytest.mark.asyncio
async def test_supporting_provider_not_emulated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An anthropic-kind provider honors tool_choice natively → even with
    action=emulate the request is passed through untouched, no shim log."""
    anth_cfg = _anthropic_provider("sonnet")
    config = _config([anth_cfg], chain=["sonnet"], action="emulate")
    anth = FakeAnthropicAdapter(anth_cfg, text="ok")
    engine = _engine(config, {"sonnet": anth})

    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(_forced_request())

    assert _emulated_logs(caplog) == []
    # Native adapter received the real, unmutated request.
    assert anth.generate_calls
    received = anth.generate_calls[0]
    assert received.tool_choice == {"type": "tool", "name": "get_weather"}


@pytest.mark.asyncio
async def test_auto_tool_choice_not_emulated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-forced ``tool_choice: auto`` request is never emulated even on
    a non-supporting provider — only forcing modes matter."""
    compat_cfg = _openai_provider("ollama")
    config = _config([compat_cfg], chain=["ollama"], action="emulate")
    compat = FakeOpenAIAdapter(compat_cfg, text="ok")
    engine = _engine(config, {"ollama": compat})

    req = AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "auto"},
        }
    )
    with caplog.at_level(logging.INFO, logger="coderouter"):
        await engine.generate_anthropic(req)

    assert _emulated_logs(caplog) == []


# ----------------------------------------------------------------------
# emulate — streaming
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emulate_injects_system(
    caplog: pytest.LogCaptureFixture,
) -> None:
    compat_cfg = _openai_provider("ollama")
    config = _config([compat_cfg], chain=["ollama"], action="emulate")
    compat = FakeOpenAIAdapter(compat_cfg, text="streamed")
    engine = _engine(config, {"ollama": compat})

    original = _forced_request(mode="any")
    with caplog.at_level(logging.INFO, logger="coderouter"):
        events = [ev async for ev in engine.stream_anthropic(original)]

    assert events
    logs = _emulated_logs(caplog)
    assert len(logs) == 1
    assert logs[0].mode == "any"
    # any-mode → tool_name is None.
    assert logs[0].tool_name is None
    # Original unchanged.
    assert original.tool_choice == {"type": "any"}
