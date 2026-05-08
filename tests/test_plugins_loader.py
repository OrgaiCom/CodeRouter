"""Unit tests for ``coderouter.plugins.loader`` (v2.3.0).

Strategy
========

We don't install a fake plugin package on disk. Instead we patch
``importlib.metadata.entry_points`` to return synthetic
:class:`importlib.metadata.EntryPoint` records that point at classes
defined in the test module. That keeps the test fast and isolated
while exercising the loader's real path: enabled-allowlist filtering,
``__init__`` dispatch with config dict, and degraded-continue on
construction failure.
"""
from __future__ import annotations

import importlib.metadata as md
from collections.abc import Callable
from typing import Any

import pytest

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    PluginsConfig,
    ProviderConfig,
)
from coderouter.plugins import loader as loader_mod
from coderouter.plugins.loader import discover_and_load


# ---------------------------------------------------------------------
# Synthetic plugin classes — module-level so EntryPoint.load() can
# resolve "tests.test_plugins_loader:..." strings via importlib.
# ---------------------------------------------------------------------


class _GoodFilter:
    name = "good-filter"

    def __init__(self, **kwargs: Any) -> None:
        self.received_kwargs = kwargs

    async def transform(self, request: Any) -> Any:
        return request


class _BadInit:
    name = "bad-init"

    def __init__(self, **_: Any) -> None:
        raise RuntimeError("explode on purpose")

    async def transform(self, request: Any) -> Any:  # pragma: no cover
        return request


class _Observer:
    name = "observer-x"

    def __init__(self, **_: Any) -> None:
        pass

    async def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        return None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_config(plugins_cfg: PluginsConfig | None) -> CodeRouterConfig:
    """Smallest valid CodeRouterConfig the loader needs.

    The loader only reads ``config.plugins``; provider/profile shapes
    don't matter for these tests, but pydantic validation requires at
    least one entry, so we supply trivial ones.
    """
    return CodeRouterConfig(
        providers=[
            ProviderConfig(
                name="dummy",
                kind="openai_compat",
                base_url="http://localhost:9999",
                model="x",
            )
        ],
        profiles=[
            FallbackChain(name="default", providers=["dummy"]),
        ],
        plugins=plugins_cfg,
    )


def _entry_points_factory(
    mapping: dict[str, list[tuple[str, str]]],
) -> Callable[..., Any]:
    """Build a function compatible with ``importlib.metadata.entry_points``.

    ``mapping`` keys are full group names (e.g. ``"coderouter.input_filter"``);
    values are ``(name, target)`` pairs where ``target`` is the
    ``module:attribute`` string EntryPoint.load() will import.
    """

    def fake_entry_points(*, group: str | None = None, **_: Any) -> tuple[md.EntryPoint, ...]:
        if group is None:
            return tuple()
        eps = mapping.get(group, [])
        return tuple(md.EntryPoint(name=n, value=v, group=group) for n, v in eps)

    return fake_entry_points


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_no_plugins_block_returns_empty_registry() -> None:
    cfg = _make_config(plugins_cfg=None)
    reg = discover_and_load(cfg)
    assert reg.is_empty()


def test_empty_enabled_list_returns_empty_registry() -> None:
    cfg = _make_config(plugins_cfg=PluginsConfig(enabled=[]))
    reg = discover_and_load(cfg)
    assert reg.is_empty()


def test_enabled_loads_matching_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_eps = _entry_points_factory(
        {
            "coderouter.input_filter": [
                ("good-filter", "tests.test_plugins_loader:_GoodFilter"),
            ],
        }
    )
    monkeypatch.setattr(loader_mod.md, "entry_points", fake_eps)

    cfg = _make_config(
        plugins_cfg=PluginsConfig(
            enabled=["good-filter"],
            config={"good-filter": {"some_key": "some_value"}},
        )
    )

    reg = discover_and_load(cfg)
    assert reg.count("input_filter") == 1
    instance = reg.input_filters[0]
    assert instance.name == "good-filter"
    # Config dict was splatted into __init__:
    assert instance.received_kwargs == {"some_key": "some_value"}


def test_unlisted_plugin_is_silently_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed entry point not on the enabled list is NOT loaded."""
    fake_eps = _entry_points_factory(
        {
            "coderouter.input_filter": [
                ("good-filter", "tests.test_plugins_loader:_GoodFilter"),
                ("other", "tests.test_plugins_loader:_GoodFilter"),
            ],
        }
    )
    monkeypatch.setattr(loader_mod.md, "entry_points", fake_eps)

    cfg = _make_config(plugins_cfg=PluginsConfig(enabled=["good-filter"]))
    reg = discover_and_load(cfg)
    assert reg.count("input_filter") == 1
    assert reg.input_filters[0].received_kwargs == {}


def test_failing_init_does_not_abort_loader(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_eps = _entry_points_factory(
        {
            "coderouter.input_filter": [
                ("bad-init", "tests.test_plugins_loader:_BadInit"),
                ("good-filter", "tests.test_plugins_loader:_GoodFilter"),
            ],
        }
    )
    monkeypatch.setattr(loader_mod.md, "entry_points", fake_eps)

    cfg = _make_config(
        plugins_cfg=PluginsConfig(enabled=["bad-init", "good-filter"])
    )
    with caplog.at_level("ERROR"):
        reg = discover_and_load(cfg)

    # Bad plugin skipped, good plugin still loaded.
    assert reg.count("input_filter") == 1
    assert reg.input_filters[0].name == "good-filter"
    # An error log was emitted for the bad plugin.
    assert any(rec.msg == "plugin-load-failed" for rec in caplog.records)


def test_unknown_enabled_name_emits_not_found_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_eps = _entry_points_factory(
        {
            "coderouter.input_filter": [
                ("good-filter", "tests.test_plugins_loader:_GoodFilter"),
            ],
        }
    )
    monkeypatch.setattr(loader_mod.md, "entry_points", fake_eps)

    cfg = _make_config(plugins_cfg=PluginsConfig(enabled=["typo-name"]))
    with caplog.at_level("WARNING"):
        reg = discover_and_load(cfg)

    assert reg.is_empty()
    assert any(rec.msg == "plugin-not-found" for rec in caplog.records)


def test_observer_group_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_eps = _entry_points_factory(
        {
            "coderouter.observer": [
                ("observer-x", "tests.test_plugins_loader:_Observer"),
            ],
        }
    )
    monkeypatch.setattr(loader_mod.md, "entry_points", fake_eps)

    cfg = _make_config(plugins_cfg=PluginsConfig(enabled=["observer-x"]))
    reg = discover_and_load(cfg)
    assert reg.count("observer") == 1
    assert reg.observers[0].name == "observer-x"


def test_future_group_logs_warning_but_still_loads(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Plugins targeting future groups load, but get a warning log line."""
    fake_eps = _entry_points_factory(
        {
            "coderouter.frontend": [
                # Reuse _GoodFilter as a stand-in — the loader doesn't
                # type-check Protocol membership, only construct the class.
                ("frontend-x", "tests.test_plugins_loader:_GoodFilter"),
            ],
        }
    )
    monkeypatch.setattr(loader_mod.md, "entry_points", fake_eps)

    cfg = _make_config(plugins_cfg=PluginsConfig(enabled=["frontend-x"]))
    with caplog.at_level("WARNING"):
        reg = discover_and_load(cfg)

    assert reg.count("frontend") == 1
    assert any(
        rec.msg == "plugin-group-not-yet-active" for rec in caplog.records
    )
