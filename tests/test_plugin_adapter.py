"""Tests for the Adapter plugin hook wiring (Phase 2a-2c, v2.8.0-v2.9.0).

Covers docs/designs/agent-cli-plugin-extraction.md §7's Phase 2a test
plan: a fake ``kind="fake_agent"`` adapter plugin is registered
through the same paths a real ``coderouter.adapter`` entry point would
use, and we verify:

* ``build_adapter`` resolves plugin-provided kinds (§3.2 step 2).
* in-core kinds always win over a plugin claiming the same kind — no
  shadowing (§3.2 "in-core を先に見る").
* the unknown-kind error lists both in-core and plugin kinds (§3.3).
* the two-stage gate (installed but not ``plugins.enabled`` ->
  unavailable) holds for the ``adapter`` group exactly like it does
  for ``input_filter`` / ``observer`` (§3.4).
* a provider using a plugin kind flows end to end through the fallback
  engine to serve a real ``/v1/chat/completions`` request (§4.5's "fake
  adapter plugin wiring E2E" replacement for the agent_cli-specific E2E
  tests moved to the plugin package in Phase 2b).
* the *runtime* ``register_provider`` path (fallback.py:1224, the second
  of §3.5's two ``build_adapter`` call sites) also passes the plugin
  registry, so a plugin kind registered after startup resolves too.

Phase 2c (§7 "2c" row) removed the in-core ``AgentCliAdapter`` and its
``build_adapter`` branch entirely, along with the Phase 2b
"in-core wins, plugin shadowed" resolution order and its
``agent-cli-in-core-deprecated`` once-per-process log — those tests
were removed from this file with the in-core adapter itself.
``kind="agent_cli"`` now resolves ONLY via the plugin path, and an
un-migrated config (no plugin registered) gets a targeted migration
hint instead of the generic unknown-kind message (§5.2) — see
``test_agent_cli_kind_without_plugin_raises_targeted_migration_hint``
below.

The loader-level tests reuse the ``importlib.metadata.entry_points``
monkeypatch pattern from ``tests/test_plugins_loader.py`` so they
exercise the real discovery path rather than hand-building a registry.
"""

from __future__ import annotations

import importlib.metadata as md
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from coderouter.adapters.base import BaseAdapter, ChatRequest, ChatResponse, StreamChunk
from coderouter.adapters.openai_compat import OpenAICompatAdapter
from coderouter.adapters.registry import build_adapter
from coderouter.config.schemas import (
    AgentCliConfig,
    Capabilities,
    CodeRouterConfig,
    FallbackChain,
    PluginsConfig,
    ProviderConfig,
)
from coderouter.ingress.app import create_app
from coderouter.metrics import uninstall_collector
from coderouter.plugins import loader as loader_mod
from coderouter.plugins.loader import discover_and_load
from coderouter.plugins.registry import PluginRegistry
from coderouter.routing.fallback import FallbackEngine

# ---------------------------------------------------------------------
# Fake adapter plugin — module-level so entry-point strings
# ("tests.test_plugin_adapter:_ClassName") resolve via importlib, same
# convention as tests/test_plugins_loader.py's synthetic plugins.
# ---------------------------------------------------------------------


class _FakeAgentAdapter(BaseAdapter):
    """Trivial in-process adapter — no I/O, echoes a fixed reply."""

    async def healthcheck(self) -> bool:
        return True

    async def generate(self, request: ChatRequest, *, overrides: Any = None) -> ChatResponse:
        return ChatResponse(
            id="fake-agent-1",
            created=int(time.time()),
            model=self.config.model,
            coderouter_provider=self.name,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello from fake_agent",
                    },
                    "finish_reason": "stop",
                }
            ],
        )

    async def stream(
        self, request: ChatRequest, *, overrides: Any = None
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError  # pragma: no cover - not exercised here
        yield  # pragma: no cover


class _FakeAgentProvider:
    """``coderouter.adapter`` entry point: serves kind="fake_agent"."""

    name = "fake-agent-plugin"
    kind = "fake_agent"

    def __init__(self, **_config: object) -> None:
        pass

    def build(self, config: ProviderConfig) -> BaseAdapter:
        return _FakeAgentAdapter(config)


class _ShadowOpenAICompatProvider:
    """Malicious/buggy plugin that tries to claim an in-core kind.

    ``build_adapter`` must never reach this factory for
    ``kind="openai_compat"`` — in-core resolution always wins.
    """

    name = "shadow-plugin"
    kind = "openai_compat"

    def __init__(self, **_config: object) -> None:
        pass

    def build(self, config: ProviderConfig) -> BaseAdapter:
        raise AssertionError("plugin factory must not be consulted for an in-core kind")


def _fake_agent_provider_config(**overrides: object) -> ProviderConfig:
    kwargs: dict[str, object] = {
        "name": "agent-fake",
        "kind": "fake_agent",
        "model": "fake-model",
        "capabilities": Capabilities(),
    }
    kwargs.update(overrides)
    return ProviderConfig(**kwargs)  # type: ignore[arg-type]


def _entry_points_factory(
    mapping: dict[str, list[tuple[str, str]]],
):
    """Same helper shape as tests/test_plugins_loader.py's version."""

    def fake_entry_points(*, group: str | None = None, **_: Any) -> tuple[md.EntryPoint, ...]:
        if group is None:
            return tuple()
        eps = mapping.get(group, [])
        return tuple(md.EntryPoint(name=n, value=v, group=group) for n, v in eps)

    return fake_entry_points


# ---------------------------------------------------------------------
# build_adapter — direct unit tests
# ---------------------------------------------------------------------


def test_build_adapter_resolves_plugin_kind() -> None:
    reg = PluginRegistry()
    reg.add("adapter", _FakeAgentProvider())

    adapter = build_adapter(_fake_agent_provider_config(), reg)

    assert isinstance(adapter, _FakeAgentAdapter)
    assert adapter.config.kind == "fake_agent"


def test_build_adapter_without_registry_still_resolves_in_core() -> None:
    """``plugin_registry=None`` (legacy call sites) must keep working."""
    provider = ProviderConfig(
        name="local",
        kind="openai_compat",
        base_url="http://localhost:8080/v1",
        model="qwen-coder",
    )
    assert isinstance(build_adapter(provider), OpenAICompatAdapter)


def test_in_core_kind_is_not_shadowed_by_plugin() -> None:
    """A plugin claiming ``openai_compat`` must never be consulted."""
    reg = PluginRegistry()
    reg.add("adapter", _ShadowOpenAICompatProvider())

    provider = ProviderConfig(
        name="local",
        kind="openai_compat",
        base_url="http://localhost:8080/v1",
        model="qwen-coder",
    )
    adapter = build_adapter(provider, reg)

    assert isinstance(adapter, OpenAICompatAdapter)


def test_agent_cli_kind_resolves_via_plugin_when_registered() -> None:
    """Phase 2c: agent_cli is a plugin kind like any other — a factory
    that claims ``kind="agent_cli"`` in the plugin registry now serves
    it directly (no more in-core branch to shadow it,
    docs/designs/agent-cli-plugin-extraction.md §7 "2c" row).
    """

    class _AgentCliLikeProvider:
        name = "agents"
        kind = "agent_cli"

        def __init__(self, **_config: object) -> None:
            pass

        def build(self, config: ProviderConfig) -> BaseAdapter:
            return _FakeAgentAdapter(config)

    reg = PluginRegistry()
    reg.add("adapter", _AgentCliLikeProvider())

    provider = ProviderConfig(
        name="agent",
        kind="agent_cli",
        model="opus",
        agent_cli=AgentCliConfig(agent="claude"),
    )
    adapter = build_adapter(provider, reg)

    assert isinstance(adapter, _FakeAgentAdapter)


@pytest.mark.parametrize("registry", [None, PluginRegistry.empty()])
def test_agent_cli_kind_without_plugin_raises_targeted_migration_hint(
    registry: PluginRegistry | None,
) -> None:
    """Phase 2c (docs/designs/agent-cli-plugin-extraction.md §5.2, §7.1):

    ``kind="agent_cli"`` with no adapter plugin registered (either no
    registry at all, or an empty one without an ``agents`` factory)
    must raise ``ValueError`` carrying the TARGETED migration hint —
    the install command (git+https URL) and the ``plugins.enabled``
    snippet — not just the generic unknown-kind message.
    """
    provider = ProviderConfig(
        name="agent",
        kind="agent_cli",
        model="opus",
        agent_cli=AgentCliConfig(agent="claude"),
    )

    with pytest.raises(ValueError) as excinfo:
        build_adapter(provider, registry)

    message = str(excinfo.value)
    assert "agent_cli" in message
    assert "coderouter-plugin-agents" in message
    assert "git+https://github.com/zephel01/coderouter-plugin-agents" in message
    assert "plugins.enabled" in message
    assert "agents" in message


def test_unknown_kind_error_lists_in_core_and_plugin_kinds() -> None:
    reg = PluginRegistry()
    reg.add("adapter", _FakeAgentProvider())

    provider = _fake_agent_provider_config(kind="totally_unknown")

    with pytest.raises(ValueError) as excinfo:
        build_adapter(provider, reg)

    message = str(excinfo.value)
    assert "totally_unknown" in message
    assert "openai_compat" in message
    assert "anthropic" in message
    assert "fake_agent" in message
    assert "plugins.enabled" in message


def test_unknown_kind_error_with_no_plugins_loaded() -> None:
    provider = _fake_agent_provider_config(kind="fake_agent")

    with pytest.raises(ValueError) as excinfo:
        build_adapter(provider, PluginRegistry.empty())

    message = str(excinfo.value)
    assert "fake_agent" in message
    assert "plugin-provided kinds: []" in message


# ---------------------------------------------------------------------
# Two-stage gate via the real loader (installed vs. enabled)
# ---------------------------------------------------------------------


def _plugins_config() -> CodeRouterConfig:
    return CodeRouterConfig(
        providers=[
            ProviderConfig(
                name="dummy",
                kind="openai_compat",
                base_url="http://localhost:9999",
                model="x",
            )
        ],
        profiles=[FallbackChain(name="default", providers=["dummy"])],
        plugins=PluginsConfig(enabled=["agents"]),
    )


def test_adapter_group_is_active_not_future(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The ``adapter`` group must load without the
    ``plugin-group-not-yet-active`` warning now that it's wired in
    (docs/designs/agent-cli-plugin-extraction.md §2.4).
    """
    fake_eps = _entry_points_factory(
        {
            "coderouter.adapter": [
                ("agents", "tests.test_plugin_adapter:_FakeAgentProvider"),
            ],
        }
    )
    monkeypatch.setattr(loader_mod.md, "entry_points", fake_eps)

    with caplog.at_level("WARNING"):
        reg = discover_and_load(_plugins_config())

    assert reg.count("adapter") == 1
    assert reg.adapters[0].kind == "fake_agent"
    assert not any(rec.msg == "plugin-group-not-yet-active" for rec in caplog.records)


def test_installed_but_not_enabled_adapter_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-stage gate: entry point discoverable but not in
    ``plugins.enabled`` -> ``PluginRegistry.adapters`` stays empty and
    ``build_adapter`` can't resolve the kind.
    """
    fake_eps = _entry_points_factory(
        {
            "coderouter.adapter": [
                ("agents", "tests.test_plugin_adapter:_FakeAgentProvider"),
            ],
        }
    )
    monkeypatch.setattr(loader_mod.md, "entry_points", fake_eps)

    cfg = CodeRouterConfig(
        providers=[
            ProviderConfig(
                name="dummy",
                kind="openai_compat",
                base_url="http://localhost:9999",
                model="x",
            )
        ],
        profiles=[FallbackChain(name="default", providers=["dummy"])],
        plugins=PluginsConfig(enabled=[]),  # NOT enabled
    )
    reg = discover_and_load(cfg)

    assert reg.count("adapter") == 0
    with pytest.raises(ValueError, match="fake_agent"):
        build_adapter(_fake_agent_provider_config(), reg)


# ---------------------------------------------------------------------
# E2E: plugin-provided kind serves a real /v1/chat/completions request
# ---------------------------------------------------------------------


@pytest.fixture
def _fake_agent_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    config = CodeRouterConfig(
        default_profile="fake-agent",
        providers=[_fake_agent_provider_config()],
        profiles=[FallbackChain(name="fake-agent", providers=["agent-fake"])],
        plugins=PluginsConfig(enabled=["agents"]),
    )
    reg = PluginRegistry()
    reg.add("adapter", _FakeAgentProvider())

    monkeypatch.setattr("coderouter.ingress.app.load_config", lambda path=None: config)
    monkeypatch.setattr("coderouter.ingress.app.discover_and_load", lambda cfg: reg)
    uninstall_collector()
    app = create_app()
    with TestClient(app) as tc:
        yield tc
    uninstall_collector()


def test_e2e_plugin_adapter_serves_chat_completions(
    _fake_agent_client: TestClient,
) -> None:
    resp = _fake_agent_client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-CodeRouter-Profile": "fake-agent"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello from fake_agent"
    assert body["coderouter_provider"] == "agent-fake"


# ---------------------------------------------------------------------
# Runtime path: register_provider is the SECOND build_adapter call site
# (fallback.py:1224, §3.5) — verify it also threads the plugin registry
# so a plugin kind registered after startup resolves via the plugin.
# ---------------------------------------------------------------------


def test_register_provider_resolves_plugin_kind_at_runtime() -> None:
    reg = PluginRegistry()
    reg.add("adapter", _FakeAgentProvider())

    # Engine starts with only an in-core provider; the plugin kind is
    # introduced at runtime via register_provider (the launcher path).
    config = CodeRouterConfig(
        default_profile="default",
        providers=[
            ProviderConfig(
                name="static-local",
                kind="openai_compat",
                base_url="http://localhost:8080/v1",
                model="qwen-coder",
            )
        ],
        profiles=[FallbackChain(name="default", providers=["static-local"])],
    )
    engine = FallbackEngine(config, plugins=reg)

    engine.register_provider(_fake_agent_provider_config(), profile_name="launcher")

    adapter = engine._adapters["agent-fake"]
    assert isinstance(adapter, _FakeAgentAdapter)
    assert adapter.config.kind == "fake_agent"
