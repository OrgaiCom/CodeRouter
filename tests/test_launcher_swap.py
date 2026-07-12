"""Tests for Phase 1 on-demand model swap (docs/designs/launcher-model-swap.md).

Sections mirror the design's §8 test plan:

* A. SwapManager unit tests (U1-U7) — spawn coalescing, touch/lease,
  readiness timeout, non-poison spawn failure, TTL sweep, TTL-vs-new-
  request race, lease lifecycle. Backend spawning is faked
  (``spawn_process`` / ``stop_process`` monkeypatched) — no llama-server.
* B. Config load-time validation (U10).
* C. End-to-end via TestClient (I1-I5), using a real local stub HTTP
  server (mirrors ``tests/test_launcher_readiness_restart.py``'s
  pattern) so the whole spawn -> readiness -> register -> dispatch ->
  TTL-unload path runs for real, without llama-server.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from coderouter import launcher_swap
from coderouter.adapters.base import AdapterError
from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    LauncherBackendConfig,
    LauncherConfig,
    LauncherOptionProfile,
    LauncherSwapConfig,
    ProviderConfig,
    SwapModelSpec,
)
from coderouter.ingress.app import create_app
from coderouter.ingress.launcher_routes import ManagedProcess, _registry_for_app
from coderouter.metrics import uninstall_collector

# pyproject.toml sets asyncio_mode = "auto" (pytest-asyncio).


# ---------------------------------------------------------------------------
# A. SwapManager unit tests — faked spawn/stop, no llama-server
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Records register_provider / deregister_provider calls."""

    def __init__(self) -> None:
        self.registered: list[tuple[str, str]] = []
        self.deregistered: list[tuple[str, str]] = []

    def register_provider(
        self, provider: ProviderConfig, profile_name: str = "launcher"
    ) -> dict[str, Any]:
        self.registered.append((provider.name, profile_name))
        return {"provider": provider.name, "profile": profile_name}

    async def deregister_provider(
        self, provider_name: str, profile_name: str = "launcher"
    ) -> bool:
        self.deregistered.append((provider_name, profile_name))
        return True


class _FakeSpawner:
    """Stands in for ``launcher_routes.spawn_process`` / ``stop_process``.

    ``script`` is a per-call behavior queue popped on every
    ``spawn_process`` call; unset (default) behavior is ``"ok"``
    (instant success). Behaviors:

    - ``"ok"``    — spawns, immediately marks ready + running.
    - ``"raise"`` — raises ValueError (spawn itself failed, U4).
    - ``"error"`` — spawns, but readiness resolves to status="error"
      (mirrors a real readiness-timeout/crash outcome).
    - ``"hang"``  — spawns but never sets ``proc.ready`` (U3, readiness
      timeout — the test uses a short ``readiness_timeout_s``).
    """

    def __init__(self) -> None:
        self.spawn_calls: list[dict[str, Any]] = []
        self.stop_calls: list[str] = []
        self.script: list[str] = []
        self._n = 0

    def _next_behavior(self) -> str:
        return self.script.pop(0) if self.script else "ok"

    async def spawn_process(
        self,
        app: Any,
        launcher_cfg: Any,
        *,
        name: str,
        backend: str,
        model_path: str,
        port: int,
        options: dict[str, Any] | None = None,
        extra_args: str = "",
        draft_model_path: str | None = None,
        mtp_mode: str = "auto",
        swap_managed: bool = False,
        swap_model: str | None = None,
    ) -> ManagedProcess:
        self._n += 1
        self.spawn_calls.append(
            {
                "name": name,
                "port": port,
                "n": self._n,
                "swap_managed": swap_managed,
                "swap_model": swap_model,
            }
        )
        behavior = self._next_behavior()
        if behavior == "raise":
            raise ValueError(f"fake spawn failure #{self._n}")
        proc = ManagedProcess(
            id=f"proc-{self._n}",
            name=name,
            backend=backend,
            model_path=model_path,
            port=port,
            options=options or {},
            extra_args=extra_args,
            status="loading",
            swap_managed=swap_managed,
            swap_model=swap_model,
        )
        _registry_for_app(app).add(proc)
        if behavior == "ok":
            proc.status = "running"
            proc.ready.set()
        elif behavior == "error":
            proc.status = "error"
            proc.ready.set()
        # "hang": leave status="loading", ready unset.
        return proc

    async def stop_process(self, app: Any, proc_id: str) -> ManagedProcess:
        self.stop_calls.append(proc_id)
        proc = _registry_for_app(app).get(proc_id)
        proc.stopping = True
        proc.status = "stopped"
        return proc


def _make_manager(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ttl_seconds: float | None = None,
    readiness_timeout_s: float = 2.0,
    port_retry_attempts: int | None = None,
    models: list[SwapModelSpec] | None = None,
) -> tuple[launcher_swap.SwapManager, _FakeSpawner, _FakeEngine]:
    fake = _FakeSpawner()
    monkeypatch.setattr(launcher_swap, "spawn_process", fake.spawn_process)
    monkeypatch.setattr(launcher_swap, "stop_process", fake.stop_process)
    engine = _FakeEngine()
    app = SimpleNamespace(state=SimpleNamespace(engine=engine))
    swap_cfg = LauncherSwapConfig(
        enabled=True,
        ttl_seconds=ttl_seconds,
        readiness_timeout_s=readiness_timeout_s,
        sweep_interval_s=600.0,
        **(
            {}
            if port_retry_attempts is None
            else {"port_retry_attempts": port_retry_attempts}
        ),
        models=models
        or [
            SwapModelSpec(
                name="m1", backend="llama.cpp", model_path="/tmp/m1.gguf", port=19300
            )
        ],
    )
    launcher_cfg = LauncherConfig(model_dirs=["/tmp"], swap=swap_cfg)
    manager = launcher_swap.SwapManager(app, swap_cfg, launcher_cfg)
    return manager, fake, engine


async def test_u1_concurrent_requests_spawn_once(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, fake, _engine = _make_manager(monkeypatch)
    leases = await asyncio.gather(*(manager.ensure_loaded("m1") for _ in range(8)))
    assert len(fake.spawn_calls) == 1
    # H-1/H-2: SwapManager always spawns with the swap_managed marker so
    # the launcher skips generic registration and auto-restart.
    assert fake.spawn_calls[0]["swap_managed"] is True
    assert all(lease.model == "m1" for lease in leases)
    for lease in leases:
        await manager.release_lease(lease)


async def test_u2_already_loaded_returns_immediately_and_touches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, fake, _engine = _make_manager(monkeypatch)
    lease1 = await manager.ensure_loaded("m1")
    await manager.release_lease(lease1)
    state = manager._states["m1"]
    before = state.last_used
    await asyncio.sleep(0.01)
    lease2 = await manager.ensure_loaded("m1")
    assert len(fake.spawn_calls) == 1  # no second spawn
    assert state.last_used > before  # touched
    await manager.release_lease(lease2)


async def test_u3_readiness_timeout_raises_retryable_adapter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, fake, _engine = _make_manager(monkeypatch, readiness_timeout_s=1.0)
    fake.script = ["hang"]
    with pytest.raises(AdapterError) as excinfo:
        await manager.ensure_loaded("m1")
    assert excinfo.value.retryable is True
    assert manager._states["m1"].status == "idle"


async def test_u4_spawn_failure_is_not_poisoned(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, fake, _engine = _make_manager(monkeypatch, readiness_timeout_s=1.0)
    fake.script = ["raise"]
    with pytest.raises(AdapterError):
        await manager.ensure_loaded("m1")
    assert manager._states["m1"].status == "idle"

    # Second attempt (script exhausted -> defaults to "ok") succeeds.
    lease = await manager.ensure_loaded("m1")
    assert lease.model == "m1"
    assert len(fake.spawn_calls) == 2
    await manager.release_lease(lease)


async def test_u5_ttl_sweep_unloads_and_deregisters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ttl_seconds=0: eligible for unload the instant in_flight drops to 0
    # (real wall-clock time always advances by >0 between calls).
    manager, fake, engine = _make_manager(monkeypatch, ttl_seconds=0.0)
    lease = await manager.ensure_loaded("m1")
    assert engine.registered == [("launcher-swap-m1", "launcher-swap-m1")]
    await manager.release_lease(lease)

    await manager.sweep_once()

    assert manager._states["m1"].status == "idle"
    assert manager._states["m1"].proc_id is None
    assert len(fake.stop_calls) == 1
    assert engine.deregistered == [("launcher-swap-m1", "launcher-swap-m1")]


async def test_u5b_ttl_none_disables_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, fake, _engine = _make_manager(monkeypatch, ttl_seconds=None)
    lease = await manager.ensure_loaded("m1")
    await manager.release_lease(lease)
    await manager.sweep_once()
    assert manager._states["m1"].status == "ready"  # untouched
    assert fake.stop_calls == []


async def test_u6_in_flight_protects_from_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, fake, _engine = _make_manager(monkeypatch, ttl_seconds=0.0)
    lease = await manager.ensure_loaded("m1")  # in_flight=1, held

    await manager.sweep_once()

    assert manager._states["m1"].status == "ready"  # NOT unloaded
    assert fake.stop_calls == []
    await manager.release_lease(lease)


async def test_u6b_after_release_and_ttl_expiry_next_request_respawns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, fake, _engine = _make_manager(monkeypatch, ttl_seconds=0.0)
    lease1 = await manager.ensure_loaded("m1")
    await manager.release_lease(lease1)
    await manager.sweep_once()
    assert manager._states["m1"].status == "idle"

    lease2 = await manager.ensure_loaded("m1")
    assert len(fake.spawn_calls) == 2  # re-spawned
    await manager.release_lease(lease2)


# ---------------------------------------------------------------------------
# [Unreleased]: per-model TTL override (SwapModelSpec.ttl_seconds)
# ---------------------------------------------------------------------------


async def test_ttl_override_spec_wins_over_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog entry's own ``ttl_seconds`` takes priority over the global
    value — here the global TTL is far too long to ever fire in this test,
    but the per-model override of 0 makes the model unload immediately."""
    manager, fake, _engine = _make_manager(
        monkeypatch,
        ttl_seconds=100000.0,
        models=[
            SwapModelSpec(
                name="m1",
                backend="llama.cpp",
                model_path="/tmp/m1.gguf",
                port=19300,
                ttl_seconds=0.0,
            )
        ],
    )
    lease = await manager.ensure_loaded("m1")
    await manager.release_lease(lease)

    await manager.sweep_once()

    assert manager._states["m1"].status == "idle"
    assert len(fake.stop_calls) == 1


async def test_ttl_override_unset_falls_back_to_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog entry that leaves ``ttl_seconds`` unset (``None``, the
    default) is governed by the global value, exactly as before this
    feature existed."""
    manager, fake, _engine = _make_manager(
        monkeypatch,
        ttl_seconds=0.0,
        models=[
            SwapModelSpec(
                name="m1",
                backend="llama.cpp",
                model_path="/tmp/m1.gguf",
                port=19300,
                ttl_seconds=None,
            )
        ],
    )
    lease = await manager.ensure_loaded("m1")
    await manager.release_lease(lease)

    await manager.sweep_once()

    assert manager._states["m1"].status == "idle"
    assert len(fake.stop_calls) == 1


async def test_ttl_override_zero_unloads_even_when_global_ttl_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ttl_seconds: 0`` on a catalog entry means the exact same thing as
    the global field's ``0`` (unload as soon as the last lease releases) —
    scoped to just that model, and effective even when the global TTL is
    disabled (``None``) entirely. Also proves the sweeper actually starts
    in this case, since ``start()`` used to gate solely on the global
    value being non-None."""
    manager, fake, _engine = _make_manager(
        monkeypatch,
        ttl_seconds=None,
        models=[
            SwapModelSpec(
                name="m1",
                backend="llama.cpp",
                model_path="/tmp/m1.gguf",
                port=19300,
                ttl_seconds=0.0,
            )
        ],
    )
    assert manager._any_ttl_configured is True
    await manager.start()
    try:
        assert manager._sweeper_task is not None
    finally:
        await manager.stop()

    lease = await manager.ensure_loaded("m1")
    await manager.release_lease(lease)
    await manager.sweep_once()

    assert manager._states["m1"].status == "idle"
    assert len(fake.stop_calls) == 1


async def test_u7_lease_lifecycle_in_flight_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _fake, _engine = _make_manager(monkeypatch)
    lease = await manager.ensure_loaded("m1")
    assert manager._states["m1"].in_flight == 1
    # Simulate a streaming response holding the lease across multiple
    # "chunks" (nothing releases it until the final one).
    await asyncio.sleep(0.01)
    assert manager._states["m1"].in_flight == 1
    await manager.release_lease(lease)
    assert manager._states["m1"].in_flight == 0
    # Double release is a no-op, not a negative counter.
    await manager.release_lease(lease)
    assert manager._states["m1"].in_flight == 0


async def test_match_falls_back_to_model_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _fake, _engine = _make_manager(
        monkeypatch,
        models=[
            SwapModelSpec(
                name="qwen-writer-14b",
                model_pattern="qwen-writer.*",
                backend="llama.cpp",
                model_path="/tmp/m1.gguf",
                port=19301,
            )
        ],
    )
    assert manager.match("qwen-writer-14b") is not None
    assert manager.match("qwen-writer-xl") is not None
    assert manager.match("something-else") is None


async def test_ensure_loaded_unknown_model_raises_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _fake, _engine = _make_manager(monkeypatch)
    with pytest.raises(KeyError):
        await manager.ensure_loaded("not-in-catalog")


async def test_port_none_retries_default_port_retry_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§10 Q2 / [Unreleased] port_retry_attempts: default is 2 additional
    retries (3 attempts total) when ``port`` is unset, each on a freshly
    picked ephemeral port."""
    manager, fake, _engine = _make_manager(
        monkeypatch,
        readiness_timeout_s=1.0,
        models=[
            SwapModelSpec(
                name="m1", backend="llama.cpp", model_path="/tmp/m1.gguf", port=None
            )
        ],
    )
    assert manager._config.port_retry_attempts == 2  # documented default
    fake.script = ["hang", "hang", "ok"]
    lease = await manager.ensure_loaded("m1")
    assert lease.model == "m1"
    assert len(fake.spawn_calls) == 3
    # Every attempt picked a fresh ephemeral port.
    ports = [c["port"] for c in fake.spawn_calls]
    assert len(set(ports)) == 3
    await manager.release_lease(lease)


async def test_port_retry_attempts_exhausted_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All 1 + port_retry_attempts attempts failing -> retryable AdapterError,
    not an infinite loop or a silent extra attempt."""
    manager, fake, _engine = _make_manager(
        monkeypatch,
        readiness_timeout_s=1.0,
        models=[
            SwapModelSpec(
                name="m1", backend="llama.cpp", model_path="/tmp/m1.gguf", port=None
            )
        ],
    )
    fake.script = ["hang", "hang", "hang"]
    with pytest.raises(AdapterError):
        await manager.ensure_loaded("m1")
    assert len(fake.spawn_calls) == 3  # 1 initial + 2 retries, then give up


async def test_port_retry_attempts_configurable_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """port_retry_attempts=0 -> exactly one attempt, no retry (same shape
    as a fixed port, but still ephemeral-port-per-attempt in principle)."""
    manager, fake, _engine = _make_manager(
        monkeypatch,
        readiness_timeout_s=1.0,
        port_retry_attempts=0,
        models=[
            SwapModelSpec(
                name="m1", backend="llama.cpp", model_path="/tmp/m1.gguf", port=None
            )
        ],
    )
    fake.script = ["hang"]
    with pytest.raises(AdapterError):
        await manager.ensure_loaded("m1")
    assert len(fake.spawn_calls) == 1


async def test_fixed_port_never_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, fake, _engine = _make_manager(monkeypatch, readiness_timeout_s=1.0)
    fake.script = ["hang"]
    with pytest.raises(AdapterError):
        await manager.ensure_loaded("m1")
    assert len(fake.spawn_calls) == 1  # fixed port: no second attempt


# ---------------------------------------------------------------------------
# [Unreleased]: registry-litter fix — swap-managed processes never pile up
# as permanent "stopped" rows in the launcher registry.
# (Adapted from the review repro /tmp/review93/test_registry_litter.py with
# assertions inverted: the bug it demonstrated is now fixed.)
# ---------------------------------------------------------------------------


async def test_failed_load_leaves_no_registry_litter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A swap load whose readiness never passes must not leave any
    "stopped" ManagedProcess in the registry — previously each failed
    attempt left one forever (1 + port_retry_attempts rows per failed
    load, all visible in GET /api/launcher/processes and the UI)."""
    from coderouter.ingress.launcher_routes import _registry_for_app

    for attempts in (0, 2, 5):
        manager, fake, _engine = _make_manager(
            monkeypatch,
            readiness_timeout_s=1.0,
            port_retry_attempts=attempts,
            models=[
                SwapModelSpec(
                    name="m1", backend="llama.cpp", model_path="/tmp/m1.gguf", port=None
                )
            ],
        )
        fake.script = ["hang"] * (1 + attempts)
        with pytest.raises(AdapterError):
            await manager.ensure_loaded("m1")
        assert len(fake.spawn_calls) == 1 + attempts  # all attempts made...
        leftover = _registry_for_app(manager._app).all()
        assert leftover == [], (
            f"port_retry_attempts={attempts}: expected an empty registry, "
            f"got {[(p.id, p.status) for p in leftover]}"
        )


async def test_ttl_unload_removes_registry_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TTL unload removes the swap-managed entry from the registry —
    previously every load/unload cycle left one "stopped" row behind."""
    from coderouter.ingress.launcher_routes import _registry_for_app

    manager, fake, _engine = _make_manager(monkeypatch, ttl_seconds=0.0)
    lease = await manager.ensure_loaded("m1")
    reg = _registry_for_app(manager._app)
    assert len(reg.all()) == 1  # loaded and visible
    await manager.release_lease(lease)

    await manager.sweep_once()

    assert manager._states["m1"].status == "idle"
    assert reg.all() == []  # ...and gone after the TTL unload
    assert len(fake.stop_calls) == 1  # still went through stop_process


async def test_registry_removal_skips_non_swap_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The swap_managed guard: _remove_from_registry never removes a
    manually-started process, whose stopped row is deliberate UI history."""
    manager, _fake, _engine = _make_manager(monkeypatch)
    from coderouter.ingress.launcher_routes import _registry_for_app

    reg = _registry_for_app(manager._app)
    manual = ManagedProcess(
        id="manual-1",
        name="manual",
        backend="llama.cpp",
        model_path="/tmp/m.gguf",
        port=18080,
        options={},
        extra_args="",
        status="stopped",
        swap_managed=False,
    )
    reg.add(manual)
    manager._remove_from_registry("manual-1")
    assert reg.get("manual-1") is manual  # untouched
    manager._remove_from_registry("no-such-id")  # unknown id: no-op, no raise


async def test_sweeper_start_stop_is_idempotent_and_cancellable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _fake, _engine = _make_manager(monkeypatch, ttl_seconds=100.0)
    manager._sweep_interval_s = 0.01
    await manager.start()
    await manager.start()  # idempotent, no second task
    task = manager._sweeper_task
    assert task is not None
    await asyncio.sleep(0.03)
    assert not task.done()
    await manager.stop()
    assert manager._sweeper_task is None
    assert task.cancelled() or task.done()


async def test_sweeper_disabled_when_ttl_none(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _fake, _engine = _make_manager(monkeypatch, ttl_seconds=None)
    await manager.start()
    assert manager._sweeper_task is None  # nothing to sweep, no task spawned
    await manager.stop()  # no-op, must not raise


# ---------------------------------------------------------------------------
# B. Config load-time validation (U10)
# ---------------------------------------------------------------------------


def _spec(**overrides: Any) -> SwapModelSpec:
    base: dict[str, Any] = dict(
        name="m1", backend="llama.cpp", model_path="/tmp/m1.gguf", port=19310,
    )
    base.update(overrides)
    return SwapModelSpec(**base)


def test_u10_duplicate_model_name_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate name"):
        LauncherSwapConfig(
            enabled=True,
            models=[_spec(name="dup", port=19312), _spec(name="dup", port=19313)],
        )


def test_u10_duplicate_port_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate port"):
        LauncherSwapConfig(
            enabled=True,
            models=[_spec(name="a", port=19311), _spec(name="b", port=19311)],
        )


def test_u10_bad_model_pattern_regex_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid model_pattern"):
        _spec(model_pattern="(unclosed")


def test_u10_unknown_option_profile_rejected() -> None:
    with pytest.raises(ValidationError, match="option_profile"):
        LauncherConfig(
            model_dirs=["/tmp"],
            option_profiles={"llama.cpp": [LauncherOptionProfile(name="gpu-fast", args={})]},
            swap=LauncherSwapConfig(
                enabled=True, models=[_spec(option_profile="does-not-exist")]
            ),
        )


def test_u10_known_option_profile_accepted() -> None:
    cfg = LauncherConfig(
        model_dirs=["/tmp"],
        option_profiles={"llama.cpp": [LauncherOptionProfile(name="gpu-fast", args={})]},
        swap=LauncherSwapConfig(enabled=True, models=[_spec(option_profile="gpu-fast")]),
    )
    assert cfg.swap is not None


def test_u10_enabled_empty_models_warns(recwarn: pytest.WarningsRecorder) -> None:
    LauncherSwapConfig(enabled=True, models=[])
    assert any("nothing to serve" in str(w.message) for w in recwarn.list)


def test_swap_model_provider_and_profile_names() -> None:
    spec = _spec(name="qwen-coder-14b")
    assert spec.provider_name == "launcher-swap-qwen-coder-14b"
    assert spec.profile_name == "launcher-swap-qwen-coder-14b"


def _base_config_kwargs(swap_cfg: LauncherSwapConfig) -> dict[str, Any]:
    """Swap-only config kwargs — ``providers``/``profiles`` start empty.

    [Unreleased]: demonstrates the relaxed ``CodeRouterConfig`` schema
    (``_check_providers_and_profiles_nonempty``) instead of the old
    ``min_length=1``-driven dummy ``local`` provider / ``default``
    profile placeholder that earlier revisions of this fixture carried
    purely to satisfy the schema — a genuinely swap-only deployment
    needs neither. ``default_profile`` is ``"auto"`` (the only value
    exempt from the pre-injection default_profile-exists check) paired
    with an explicit, empty-rules ``auto_router`` pointed at the first
    swap model's to-be-injected profile — this sidesteps the bundled
    ruleset merge path (which would otherwise demand unrelated
    multi/coding/writing profiles), exactly like the documented
    swap-only example config.
    """
    from coderouter.config.schemas import AutoRouterConfig

    target_profile = swap_cfg.models[0].profile_name if swap_cfg.models else "default"
    return dict(
        allow_paid=False,
        default_profile="auto",
        providers=[],
        profiles=[],
        auto_router=AutoRouterConfig(rules=[], default_rule_profile=target_profile),
        launcher=LauncherConfig(model_dirs=["/tmp"], swap=swap_cfg),
    )


def test_swap_profile_provider_name_collision_rejected() -> None:
    swap_cfg = LauncherSwapConfig(enabled=True, models=[_spec(name="local2")])
    kwargs = _base_config_kwargs(swap_cfg)
    kwargs["providers"].append(
        ProviderConfig(
            name="launcher-swap-local2",
            base_url="http://localhost:9/v1",
            model="x",
        )
    )
    with pytest.raises(ValidationError, match="collide"):
        CodeRouterConfig(**kwargs)


def test_swap_injects_placeholder_profile_and_auto_router_rule() -> None:
    swap_cfg = LauncherSwapConfig(enabled=True, models=[_spec(name="qwen-coder-14b")])
    cfg = CodeRouterConfig(**_base_config_kwargs(swap_cfg))

    # No static provider/profile was declared at all — swap alone fills
    # both in (the provider is registered at runtime on first spawn).
    assert cfg.providers == []
    profile = cfg.profile_by_name("launcher-swap-qwen-coder-14b")
    assert profile.providers == []

    assert cfg.auto_router is not None
    rule_ids = [r.id for r in cfg.auto_router.rules]
    assert "swap:qwen-coder-14b" in rule_ids
    rule = next(r for r in cfg.auto_router.rules if r.id == "swap:qwen-coder-14b")
    assert rule.profile == "launcher-swap-qwen-coder-14b"
    assert rule.match.model_pattern == "qwen\\-coder\\-14b"


def test_swap_rule_injection_disabled_by_flag() -> None:
    """Needs a real ``default`` profile (unlike ``_base_config_kwargs``):

    with ``inject_auto_router_rules=False`` swap never builds an
    ``auto_router`` block, so ``default_profile: auto`` without one would
    fall onto the *bundled* ruleset's multi/coding/writing requirement —
    an unrelated concern this test isn't exercising. A normal static
    profile sidesteps that, independent of the omission feature under
    test elsewhere in this module.
    """
    swap_cfg = LauncherSwapConfig(
        enabled=True,
        inject_auto_router_rules=False,
        models=[_spec(name="qwen-coder-14b")],
    )
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local", base_url="http://localhost:8080/v1", model="qwen"
            )
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(model_dirs=["/tmp"], swap=swap_cfg),
    )
    # Profile still pre-declared (needed for manual routing)...
    cfg.profile_by_name("launcher-swap-qwen-coder-14b")
    # ...but no rule was generated.
    assert cfg.auto_router is None


def test_swap_rules_appended_after_user_auto_router_rules() -> None:
    """Needs a real ``default`` profile (unlike ``_base_config_kwargs``):

    the test asserts the user-supplied ``default_rule_profile: "default"``
    passes through untouched, which requires ``"default"`` to actually be
    a declared profile (``_check_auto_router_profiles_exist`` would
    otherwise reject it) — independent of the omission feature under
    test elsewhere in this module.
    """
    from coderouter.config.schemas import AutoRouterConfig, AutoRouteRule, RuleMatcher

    swap_cfg = LauncherSwapConfig(enabled=True, models=[_spec(name="qwen-coder-14b")])
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local", base_url="http://localhost:8080/v1", model="qwen"
            )
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        auto_router=AutoRouterConfig(
            rules=[
                AutoRouteRule(
                    id="user:my-rule", profile="default", match=RuleMatcher(has_image=True)
                )
            ],
            default_rule_profile="default",
        ),
        launcher=LauncherConfig(model_dirs=["/tmp"], swap=swap_cfg),
    )
    ids = [r.id for r in cfg.auto_router.rules]
    assert ids == ["user:my-rule", "swap:qwen-coder-14b"]
    assert cfg.auto_router.default_rule_profile == "default"  # left untouched


# ---------------------------------------------------------------------------
# B2. [Unreleased]: swap-only minimal config — no dummy providers/profiles
# ---------------------------------------------------------------------------


def _write_swap_only_yaml(tmp_path: Path, *, include_empty_lists: bool) -> Path:
    """The documented swap-only minimal config (top-level ``providers``/
    ``profiles`` entirely omitted, or present-but-empty — both must load).
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "a.gguf").touch()
    empty_lists = "providers: []\nprofiles: []\n" if include_empty_lists else ""
    content = f"""\
default_profile: auto
{empty_lists}auto_router:
  default_rule_profile: launcher-swap-m1
  rules: []
launcher:
  model_dirs: ["{model_dir}"]
  swap:
    enabled: true
    models:
      - name: m1
        backend: llama.cpp
        model_path: {model_dir / "a.gguf"}
        port: 18081
"""
    cfg_path = tmp_path / "providers.yaml"
    cfg_path.write_text(content)
    return cfg_path


def test_swap_only_config_loads_with_omitted_providers_and_profiles(
    tmp_path: Path,
) -> None:
    """[Unreleased]: providers/profiles can be left out of the YAML entirely
    for a swap-only deployment — no unreachable dummy entry required."""
    from coderouter.config.loader import load_config

    cfg = load_config(_write_swap_only_yaml(tmp_path, include_empty_lists=False))
    assert cfg.providers == []
    assert cfg.profile_by_name("launcher-swap-m1").providers == []
    assert cfg.auto_router is not None
    assert cfg.auto_router.default_rule_profile == "launcher-swap-m1"


def test_swap_only_config_loads_with_explicit_empty_lists(tmp_path: Path) -> None:
    """Same minimal config, but with ``providers: []`` / ``profiles: []``
    spelled out explicitly — identical result to omitting them."""
    from coderouter.config.loader import load_config

    cfg = load_config(_write_swap_only_yaml(tmp_path, include_empty_lists=True))
    assert cfg.providers == []
    assert cfg.profile_by_name("launcher-swap-m1").providers == []
    assert cfg.auto_router is not None
    assert cfg.auto_router.default_rule_profile == "launcher-swap-m1"


def test_empty_providers_without_swap_still_rejected() -> None:
    """Regression: swap disabled + empty providers -> still a load-time
    ValidationError (the min_length=1 fail-fast guarantee is preserved,
    just moved into ``_check_providers_and_profiles_nonempty``)."""
    with pytest.raises(ValidationError, match="providers"):
        CodeRouterConfig(
            allow_paid=False,
            default_profile="auto",
            providers=[],
            profiles=[FallbackChain(name="default", providers=["local"])],
            auto_router=None,
        )


def test_empty_profiles_without_swap_still_rejected() -> None:
    """Regression: swap disabled + empty profiles -> still a load-time
    ValidationError."""
    with pytest.raises(ValidationError, match="profiles"):
        CodeRouterConfig(
            allow_paid=False,
            default_profile="default",
            providers=[
                ProviderConfig(
                    name="local", base_url="http://localhost:8080/v1", model="qwen"
                )
            ],
            profiles=[],
        )


def test_empty_providers_and_profiles_rejected_when_swap_models_empty() -> None:
    """Regression: swap enabled but its catalog is empty -> the relaxation
    does not kick in (matches spec: models must have >=1 entry)."""
    with pytest.raises(ValidationError, match="providers"):
        CodeRouterConfig(
            allow_paid=False,
            default_profile="auto",
            providers=[],
            profiles=[],
            auto_router=None,
            launcher=LauncherConfig(
                model_dirs=["/tmp"], swap=LauncherSwapConfig(enabled=True, models=[])
            ),
        )


# ---------------------------------------------------------------------------
# C. End-to-end via TestClient — real spawn, real local stub HTTP server
# ---------------------------------------------------------------------------

_SWAP_BACKEND_BODY = """
import http.server
import json
import sys
import time

def _port():
    argv = sys.argv[1:]
    return int(argv[argv.index("--port") + 1])

_ready_at = time.monotonic() + 0.15

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            if time.monotonic() < _ready_at:
                self.send_response(503)
            else:
                self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/v1/chat/completions"):
            length = int(self.headers.get("content-length", 0))
            self.rfile.read(length)
            body = json.dumps({
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": "swap-backend",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi from swap"},
                    "finish_reason": "stop",
                }],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass

httpd = http.server.HTTPServer(("127.0.0.1", _port()), Handler)
httpd.serve_forever()
"""


def _write_script(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


def _swap_e2e_config(script: Path, *, port: int, ttl_seconds: float | None) -> CodeRouterConfig:
    swap_cfg = LauncherSwapConfig(
        enabled=True,
        ttl_seconds=ttl_seconds,
        readiness_timeout_s=5.0,
        sweep_interval_s=1.0,
        models=[
            SwapModelSpec(
                name="swap-model",
                backend="llama.cpp",
                model_path="/tmp/does-not-need-to-exist.gguf",
                port=port,
                mtp_mode="off",
            )
        ],
    )
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local", base_url="http://localhost:8080/v1", model="qwen-coder"
            ),
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            model_dirs=["/tmp"],
            backends={"llama.cpp": LauncherBackendConfig(binary=str(script))},
            readiness_timeout_s=5.0,
            readiness_poll_interval_s=0.2,
            swap=swap_cfg,
        ),
    )


@contextlib.contextmanager
def _client_with_config(
    cfg: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr("coderouter.ingress.app.load_config", lambda path=None: cfg)
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


def _poll(fn, timeout: float = 5.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    result = fn()
    while not result and time.monotonic() < deadline:
        time.sleep(interval)
        result = fn()
    return result


def test_i1_on_demand_spawn_reaches_200(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(1) an unstarted model's first request gets a 200 with no manual start."""
    script = _write_script(tmp_path / "swap-backend", _SWAP_BACKEND_BODY)
    cfg = _swap_e2e_config(script, port=19320, ttl_seconds=None)
    with _client_with_config(cfg, monkeypatch) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "swap-model",
                "messages": [{"role": "user", "content": "hi"}],
                "profile": "launcher-swap-swap-model",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "hi from swap"

        procs = tc.get("/api/launcher/processes").json()["processes"]
        assert any(p["name"] == "launcher-swap-swap-model" for p in procs)
        # [Unreleased]: GET /api/launcher/processes surfaces swap_managed /
        # swap_model so the /launcher UI can badge on-demand-spawned
        # processes and label which catalog model they back.
        proc = next(p for p in procs if p["name"] == "launcher-swap-swap-model")
        assert proc["swap_managed"] is True
        assert proc["swap_model"] == "swap-model"


def test_i2_non_swap_routing_bypasses_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(2) a request that resolves to a non-swap profile never spawns.

    Review fix M-2: the hook keys on the resolved profile, so neither
    an unrelated model name on the default profile (this test) nor a
    catalog-matching model name routed elsewhere
    (test_review_m2_* in test_launcher_swap_review.py) touches spawn.
    """
    script = _write_script(tmp_path / "swap-backend", _SWAP_BACKEND_BODY)
    cfg = _swap_e2e_config(script, port=19321, ttl_seconds=None)
    with _client_with_config(cfg, monkeypatch) as tc:
        engine = tc.app.state.engine
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "totally-unrelated-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        # Dispatches through the normal chain (provider "local" isn't a
        # real server, so this 502s) — the point is that no swap process
        # was spawned and no swap provider registered.
        assert resp.status_code in (200, 502)
        procs = engine.config.providers
        assert not any(p.name.startswith("launcher-swap-") for p in procs)
        launched = tc.get("/api/launcher/processes").json()["processes"]
        assert launched == []


def test_i3_streaming_survives_sweep_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(3) TTL never fires mid-response even with ttl_seconds=0 + a fast sweep."""
    script = _write_script(tmp_path / "swap-backend", _SWAP_BACKEND_BODY)
    # ttl=0 + sweep every 20ms is deliberately hostile — the in-flight
    # lease (held for the whole non-streaming call here, since httpx
    # to the stub server is a single blocking POST) must still protect it.
    cfg = _swap_e2e_config(script, port=19322, ttl_seconds=0.0)
    with _client_with_config(cfg, monkeypatch) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "swap-model",
                "messages": [{"role": "user", "content": "hi"}],
                "profile": "launcher-swap-swap-model",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "hi from swap"

        # After the lease released, the (now idle) sweeper eventually
        # reclaims it — proves TTL=0 isn't permanently defeated either.
        # [Unreleased] registry-litter fix: a TTL unload now REMOVES the
        # swap-managed entry from the registry entirely (previously it
        # lingered forever as a "stopped" row).
        def _unloaded() -> bool:
            procs = tc.get("/api/launcher/processes").json()["processes"]
            entry = next(
                (p for p in procs if p["name"] == "launcher-swap-swap-model"), None
            )
            return entry is None

        assert _poll(_unloaded, timeout=5.0)


def test_logs_unknown_proc_id_is_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Unreleased] pin: GET /api/launcher/logs/<unknown-id> answers 404.

    The /launcher UI's 404-loop fix relies on exactly this response to
    know a polled process left the registry (server restart under an
    open tab, delete from another client, or a swap-managed entry
    removed by the registry-litter fix) and stop polling it — the 404
    itself is correct behavior and must stay.
    """
    script = _write_script(tmp_path / "swap-backend", _SWAP_BACKEND_BODY)
    cfg = _swap_e2e_config(script, port=19325, ttl_seconds=None)
    with _client_with_config(cfg, monkeypatch) as tc:
        resp = tc.get("/api/launcher/logs/no-such-proc-id?n=200")
        assert resp.status_code == 404, resp.text


def test_i4_spawn_failure_falls_back_in_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(4) a swap spawn failure raises retryable AdapterError -> chain fallback."""
    cfg = _swap_e2e_config(tmp_path / "does-not-exist-binary", port=19323, ttl_seconds=None)
    with _client_with_config(cfg, monkeypatch) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "swap-model",
                "messages": [{"role": "user", "content": "hi"}],
                "profile": "launcher-swap-swap-model",
            },
        )
        # profile "launcher-swap-swap-model" has no OTHER provider to
        # fall back to (single-model dedicated chain, by design), so the
        # chain is exhausted -> 502, not a 500/crash. The key assertion
        # is that the request fails cleanly via NoProvidersAvailableError
        # rather than an unhandled exception.
        assert resp.status_code == 502, resp.text


def test_i5_shutdown_cleans_up_sweeper_and_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(5) lifespan shutdown cancels the sweeper and stops swap processes."""
    script = _write_script(tmp_path / "swap-backend", _SWAP_BACKEND_BODY)
    cfg = _swap_e2e_config(script, port=19324, ttl_seconds=100.0)
    monkeypatch.setattr("coderouter.ingress.app.load_config", lambda path=None: cfg)
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            resp = tc.post(
                "/v1/chat/completions",
                json={
                    "model": "swap-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "profile": "launcher-swap-swap-model",
                },
            )
            assert resp.status_code == 200, resp.text
            swap_manager = app.state.swap
            assert swap_manager._sweeper_task is not None
        # TestClient __exit__ runs the lifespan shutdown.
        assert swap_manager._sweeper_task is None
        registry = app.state.launcher
        procs = registry.all()
        assert all(p.status in ("stopped", "error") for p in procs)
    finally:
        uninstall_collector()
