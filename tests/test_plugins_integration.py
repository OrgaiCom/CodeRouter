"""Integration tests: FallbackEngine + InputFilter / Observer hooks (v2.3.0).

These exercise the engine-side helpers directly
(``_apply_input_filters`` / ``_fanout_observers`` /
``_safe_observe``). We don't actually invoke ``generate_anthropic`` /
``stream_anthropic`` here because doing so would mean stubbing the
entire HTTP / chain dispatch path; the helpers themselves carry the
plugin contract, and the broader engine paths are covered by
``test_fallback*.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from coderouter.config.schemas import CodeRouterConfig
from coderouter.plugins.registry import PluginRegistry
from coderouter.routing.fallback import FallbackEngine
from coderouter.translation.anthropic import AnthropicMessage, AnthropicRequest

# ---------------------------------------------------------------------
# Synthetic plugin classes
# ---------------------------------------------------------------------


class _SystemAppendingFilter:
    """InputFilter that prepends a marker into request.system."""

    name = "marker-filter"

    def __init__(self, marker: str = "[MEMORY]") -> None:
        self.marker = marker
        self.calls = 0

    async def transform(self, request: AnthropicRequest) -> AnthropicRequest:
        self.calls += 1
        new_system = (
            self.marker
            if request.system is None
            else (request.system + self.marker)
            if isinstance(request.system, str)
            else request.system
        )
        return request.model_copy(update={"system": new_system})


class _RaisingFilter:
    name = "raising-filter"

    def __init__(self) -> None:
        self.calls = 0

    async def transform(self, request: AnthropicRequest) -> AnthropicRequest:
        self.calls += 1
        raise RuntimeError("filter exploded")


class _RecordingObserver:
    name = "recorder"

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_request(text: str = "hi") -> AnthropicRequest:
    return AnthropicRequest(
        model="x",
        max_tokens=64,
        messages=[AnthropicMessage(role="user", content=text)],
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_no_plugins_means_no_hook_calls(basic_config: CodeRouterConfig) -> None:
    """With an empty registry, hook helpers are short-circuit no-ops."""
    engine = FallbackEngine(basic_config)

    # InputFilter chain returns request unchanged.
    out = asyncio.run(engine._apply_input_filters(_make_request()))
    assert out.messages[0].content == "hi"

    # Observer fanout with empty registry must NOT spawn tasks.
    engine._fanout_observers("anything", request=None, response=None)


def test_input_filter_chain_runs_in_order(basic_config: CodeRouterConfig) -> None:
    f1 = _SystemAppendingFilter(marker="[A]")
    f2 = _SystemAppendingFilter(marker="[B]")
    reg = PluginRegistry()
    reg.add("input_filter", f1)
    reg.add("input_filter", f2)

    engine = FallbackEngine(basic_config, plugins=reg)
    out = asyncio.run(engine._apply_input_filters(_make_request()))

    # f1 runs first → "[A]"; f2 runs next → "[A][B]".
    assert out.system == "[A][B]"
    assert f1.calls == 1
    assert f2.calls == 1


def test_failing_filter_does_not_abort_chain(
    basic_config: CodeRouterConfig, caplog: pytest.LogCaptureFixture
) -> None:
    bad = _RaisingFilter()
    good = _SystemAppendingFilter(marker="[OK]")

    reg = PluginRegistry()
    reg.add("input_filter", bad)
    reg.add("input_filter", good)

    engine = FallbackEngine(basic_config, plugins=reg)

    with caplog.at_level("WARNING"):
        out = asyncio.run(engine._apply_input_filters(_make_request()))

    # Bad filter raised → its mutation discarded; good filter still ran
    # against the original request.
    assert out.system == "[OK]"
    assert bad.calls == 1
    assert good.calls == 1
    assert any(rec.msg == "input-filter-failed" for rec in caplog.records)


def test_safe_observe_swallows_exceptions(
    basic_config: CodeRouterConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """A raising observer must NOT propagate out of _safe_observe."""

    class BadObserver:
        name = "bad-obs"

        async def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
            raise RuntimeError("observer exploded")

    engine = FallbackEngine(basic_config)
    obs = BadObserver()

    with caplog.at_level("WARNING"):
        asyncio.run(engine._safe_observe(obs, "request_completed", {}))

    assert any(rec.msg == "observer-failed" for rec in caplog.records)


def test_observer_fanout_creates_tasks_when_observers_present(
    basic_config: CodeRouterConfig,
) -> None:
    """Sanity: _fanout_observers spawns one asyncio.Task per observer.

    We can't easily await fire-and-forget tasks from a sync test, so
    we verify the side effect by using the event loop directly:
    create the tasks, give them one tick to run, then inspect the
    observer's recorded events.
    """
    obs = _RecordingObserver()
    reg = PluginRegistry()
    reg.add("observer", obs)
    engine = FallbackEngine(basic_config, plugins=reg)

    async def run() -> None:
        engine._fanout_observers("custom_event", payload_key="value")
        # Yield control so the spawned task gets a chance to run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert len(obs.events) == 1
    event_type, payload = obs.events[0]
    assert event_type == "custom_event"
    assert payload == {"payload_key": "value"}


def test_default_engine_has_empty_plugin_registry(
    basic_config: CodeRouterConfig,
) -> None:
    """Constructing the engine without plugins keeps the registry empty.

    Backward-compat invariant — every existing test/build that does
    ``FallbackEngine(config)`` MUST keep working unchanged.
    """
    engine = FallbackEngine(basic_config)
    assert engine.plugins.is_empty()
