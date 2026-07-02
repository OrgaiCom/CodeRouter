"""Regression tests for bug H7: recovery probes on restart.

When CodeRouter restarts, ``SelfHealingOrchestrator.load_state`` restores
the set of *excluded* providers from persisted state.  Excluded providers
are filtered out of the fallback chain, so they are never re-attempted and
therefore never observed failing again -- and a fresh UNHEALTHY observation
is the only thing that previously armed a recovery probe.  The net effect
was a permanent exclusion: a single-provider deployment returned 502 for
every request forever.

The fix re-arms one recovery probe per restored-excluded provider from
``_load_all_state``.  These tests verify that behaviour both when an event
loop is already running (probe spawned immediately) and when it is not
(probe deferred, then flushed by ``rearm_pending_recovery_probes``).
"""

from __future__ import annotations

from pathlib import Path

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.guards.self_healing import SelfHealingOrchestrator
from coderouter.routing import FallbackEngine
from coderouter.state.store import StateStore


def _build_engine() -> FallbackEngine:
    """Single-provider engine mirroring the H7 failure scenario."""
    providers = [
        ProviderConfig(
            name="p1",
            kind="openai_compat",
            base_url="http://localhost:11434/v1",
            model="test",
        ),
    ]
    config = CodeRouterConfig(
        providers=providers,
        profiles=[FallbackChain(name="default", providers=["p1"])],
        default_profile="default",
    )
    return FallbackEngine(config)


def _persist_excluded_state(db_path: Path) -> None:
    """Write a state DB that has ``p1`` excluded via self-healing."""
    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=6)

    store = StateStore(db_path)
    store.put("self_healing", "state", orch.save_state())
    store.close()


def test_excluded_provider_exposed_with_profile() -> None:
    """The new accessor maps each excluded provider to its profile."""
    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=6)
    orch.on_unhealthy("p2", profile="coding", consecutive_failures=4)

    profiles = orch.excluded_profiles()
    assert profiles == {"p1": "default", "p2": "coding"}


async def test_attach_state_store_rearms_probe_with_loop(tmp_path: Path) -> None:
    """attach_state_store under a running loop arms a recovery probe.

    This is the production path: the FastAPI lifespan startup calls
    ``attach_state_store`` from inside the event loop.  The excluded
    provider must get a live recovery probe task so it can eventually
    be restored.
    """
    db_path = tmp_path / "coderouter.db"
    _persist_excluded_state(db_path)

    engine = _build_engine()
    # Sanity: exclusion is restored on load.
    store = StateStore(db_path)
    engine.attach_state_store(store)

    assert engine.self_healing.is_excluded("p1")
    # A recovery probe task was armed immediately (loop is running).
    assert "p1" in engine._recovery_tasks
    task = engine._recovery_tasks["p1"]
    assert not task.done()
    # No deferral needed when a loop is available.
    assert not engine._pending_recovery_rearm

    await engine.shutdown_recovery_probes()
    store.close()


def test_load_defers_rearm_without_loop(tmp_path: Path) -> None:
    """Sync attach (no running loop) defers the probe re-arm."""
    db_path = tmp_path / "coderouter.db"
    _persist_excluded_state(db_path)

    engine = _build_engine()
    store = StateStore(db_path)
    # No event loop is running in this plain sync test.
    engine.attach_state_store(store)

    assert engine.self_healing.is_excluded("p1")
    # Probe could not be spawned yet, so it is queued for later.
    assert engine._pending_recovery_rearm == {"p1": "default"}
    assert "p1" not in engine._recovery_tasks

    store.close()


async def test_flush_pending_rearm_spawns_probe(tmp_path: Path) -> None:
    """rearm_pending_recovery_probes drains deferred providers.

    Simulates: state loaded sync (deferred), then the loop comes up and
    the lifespan flushes the queue.
    """
    db_path = tmp_path / "coderouter.db"
    _persist_excluded_state(db_path)

    engine = _build_engine()
    store = StateStore(db_path)

    # Deferred queue populated as if loaded before the loop existed.
    engine._pending_recovery_rearm = {"p1": "default"}

    await engine.rearm_pending_recovery_probes()

    assert "p1" in engine._recovery_tasks
    assert not engine._recovery_tasks["p1"].done()
    # Queue drained.
    assert not engine._pending_recovery_rearm

    await engine.shutdown_recovery_probes()
    store.close()


async def test_rearm_noop_when_nothing_excluded(tmp_path: Path) -> None:
    """No excluded providers -> no probes, no pending queue."""
    engine = _build_engine()
    store = StateStore(tmp_path / "coderouter.db")
    engine.attach_state_store(store)

    assert not engine._recovery_tasks
    assert not engine._pending_recovery_rearm
    await engine.rearm_pending_recovery_probes()  # idempotent no-op
    assert not engine._recovery_tasks

    store.close()
