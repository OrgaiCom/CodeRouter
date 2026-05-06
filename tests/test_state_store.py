"""v2.0-K: StateStore + persistence hook tests.

Test groups:

- **StateStore CRUD**: put/get/delete/get_all/clear round-trip.
- **BudgetTracker persistence**: save_state/load_state round-trip,
  stale month rejection.
- **BackendHealthMonitor persistence**: save_state/load_state round-trip.
- **SelfHealingOrchestrator persistence**: save_state/load_state round-trip.
- **MetricsCollector persistence**: save_state/load_state round-trip.
- **Engine integration**: attach_state_store / save_all_state cycle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coderouter.state.store import StateStore

# ----------------------------------------------------------------------
# Group 1: StateStore CRUD
# ----------------------------------------------------------------------


def test_put_and_get(tmp_path: Path) -> None:
    """put + get round-trip on a simple value."""
    store = StateStore(tmp_path / "test.db")
    store.put("ns1", "key1", {"hello": "world"})
    result = store.get("ns1", "key1")
    assert result == {"hello": "world"}
    store.close()


def test_get_missing_returns_none(tmp_path: Path) -> None:
    """get on a missing key returns None."""
    store = StateStore(tmp_path / "test.db")
    assert store.get("ns1", "no_such_key") is None
    store.close()


def test_put_overwrites(tmp_path: Path) -> None:
    """Second put on the same key overwrites the value."""
    store = StateStore(tmp_path / "test.db")
    store.put("ns1", "key1", 1)
    store.put("ns1", "key1", 2)
    assert store.get("ns1", "key1") == 2
    store.close()


def test_namespaces_are_independent(tmp_path: Path) -> None:
    """Same key in different namespaces are independent."""
    store = StateStore(tmp_path / "test.db")
    store.put("ns1", "key1", "a")
    store.put("ns2", "key1", "b")
    assert store.get("ns1", "key1") == "a"
    assert store.get("ns2", "key1") == "b"
    store.close()


def test_delete(tmp_path: Path) -> None:
    """delete removes a key."""
    store = StateStore(tmp_path / "test.db")
    store.put("ns1", "key1", "value")
    store.delete("ns1", "key1")
    assert store.get("ns1", "key1") is None
    store.close()


def test_get_all(tmp_path: Path) -> None:
    """get_all returns all keys in a namespace."""
    store = StateStore(tmp_path / "test.db")
    store.put("ns1", "a", 1)
    store.put("ns1", "b", 2)
    store.put("ns2", "c", 3)
    result = store.get_all("ns1")
    assert result == {"a": 1, "b": 2}
    store.close()


def test_clear(tmp_path: Path) -> None:
    """clear removes all keys in a namespace."""
    store = StateStore(tmp_path / "test.db")
    store.put("ns1", "a", 1)
    store.put("ns1", "b", 2)
    store.clear("ns1")
    assert store.get_all("ns1") == {}
    store.close()


def test_persistence_across_reopen(tmp_path: Path) -> None:
    """Data survives close + reopen."""
    db_path = tmp_path / "test.db"
    store1 = StateStore(db_path)
    store1.put("ns1", "key1", {"saved": True})
    store1.close()

    store2 = StateStore(db_path)
    result = store2.get("ns1", "key1")
    assert result == {"saved": True}
    store2.close()


def test_complex_values(tmp_path: Path) -> None:
    """Store and retrieve nested/complex JSON values."""
    store = StateStore(tmp_path / "test.db")
    complex_value = {
        "list": [1, 2, 3],
        "nested": {"a": True, "b": None},
        "number": 3.14,
    }
    store.put("ns1", "complex", complex_value)
    result = store.get("ns1", "complex")
    assert result == complex_value
    store.close()


def test_graceful_degradation_bad_path(tmp_path: Path) -> None:
    """Store with an un-writable path degrades gracefully."""
    # /dev/null/impossible can't be a directory
    store = StateStore("/dev/null/impossible/test.db")
    # Should not raise
    store.put("ns1", "key1", "value")
    assert store.get("ns1", "key1") is None
    store.close()


# ----------------------------------------------------------------------
# Group 2: BudgetTracker persistence
# ----------------------------------------------------------------------


def test_budget_save_load_roundtrip() -> None:
    """BudgetTracker save/load round-trip preserves totals."""
    from coderouter.routing.budget import BudgetTracker

    bt1 = BudgetTracker()
    bt1.record("provider_a", 1.50)
    bt1.record("provider_b", 2.75)
    state = bt1.save_state()

    bt2 = BudgetTracker()
    bt2.load_state(state)
    assert bt2.total_for_provider("provider_a") == pytest.approx(1.50)
    assert bt2.total_for_provider("provider_b") == pytest.approx(2.75)


def test_budget_load_stale_month_ignored() -> None:
    """Loading state from a different month is silently ignored."""
    from coderouter.routing.budget import BudgetTracker

    bt = BudgetTracker()
    state = {"month": "1999-01", "totals": {"p1": 100.0}}
    bt.load_state(state)
    assert bt.total_for_provider("p1") == 0.0  # not loaded


def test_budget_load_invalid_data() -> None:
    """Loading invalid state doesn't crash."""
    from coderouter.routing.budget import BudgetTracker

    bt = BudgetTracker()
    bt.load_state("not a dict")  # type: ignore[arg-type]
    bt.load_state(None)  # type: ignore[arg-type]
    bt.load_state({})  # empty dict — no crash


# ----------------------------------------------------------------------
# Group 3: BackendHealthMonitor persistence
# ----------------------------------------------------------------------


def test_health_save_load_roundtrip() -> None:
    """BackendHealthMonitor save/load preserves health states."""
    from coderouter.guards.backend_health import BackendHealthMonitor

    mon1 = BackendHealthMonitor()
    # Drive p1 to DEGRADED (2 consecutive failures with threshold=2).
    mon1.record_attempt("p1", success=False, threshold=2)
    mon1.record_attempt("p1", success=False, threshold=2)
    assert mon1.state_for("p1") == "DEGRADED"

    state = mon1.save_state()

    mon2 = BackendHealthMonitor()
    mon2.load_state(state)
    assert mon2.state_for("p1") == "DEGRADED"


def test_health_load_invalid_state() -> None:
    """Loading bogus health state data doesn't crash."""
    from coderouter.guards.backend_health import BackendHealthMonitor

    mon = BackendHealthMonitor()
    mon.load_state({"p1": {"state": "INVALID", "consecutive_failures": -1}})
    # Invalid state falls back to HEALTHY, negative failures to 0.
    assert mon.state_for("p1") == "HEALTHY"


# ----------------------------------------------------------------------
# Group 4: SelfHealingOrchestrator persistence
# ----------------------------------------------------------------------


def test_self_healing_save_load_roundtrip() -> None:
    """SelfHealingOrchestrator save/load preserves excluded set."""
    from coderouter.guards.self_healing import SelfHealingOrchestrator

    orch1 = SelfHealingOrchestrator()
    orch1.on_unhealthy("p1", profile="default", consecutive_failures=6)
    orch1.on_unhealthy("p2", profile="coding", consecutive_failures=4)

    state = orch1.save_state()

    orch2 = SelfHealingOrchestrator()
    orch2.load_state(state)
    assert orch2.is_excluded("p1")
    assert orch2.is_excluded("p2")
    assert not orch2.is_excluded("p3")


def test_self_healing_load_does_not_override_existing() -> None:
    """load_state doesn't replace providers already excluded."""
    from coderouter.guards.self_healing import SelfHealingOrchestrator

    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=10)

    # Load with different metadata — should not override.
    state = {"p1": {"profile": "other", "consecutive_failures": 1}}
    orch.load_state(state)
    assert orch.is_excluded("p1")


# ----------------------------------------------------------------------
# Group 5: MetricsCollector persistence
# ----------------------------------------------------------------------


def test_metrics_save_load_roundtrip() -> None:
    """MetricsCollector save/load preserves key counters."""
    from coderouter.metrics.collector import MetricsCollector

    mc1 = MetricsCollector(ring_size=16)
    # Manually inject some state.
    mc1._requests_total = 42
    mc1._chain_paid_gate_blocked_total = 5
    mc1._cost_total_usd["provider_a"] = 1.23

    state = mc1.save_state()

    mc2 = MetricsCollector(ring_size=16)
    mc2.load_state(state)
    assert mc2._requests_total == 42
    assert mc2._chain_paid_gate_blocked_total == 5
    assert mc2._cost_total_usd.get("provider_a") == pytest.approx(1.23)


# ----------------------------------------------------------------------
# Group 6: Engine integration
# ----------------------------------------------------------------------


def test_engine_attach_and_save(tmp_path: Path) -> None:
    """Engine attach_state_store + save_all_state round-trip."""
    from coderouter.config.schemas import CodeRouterConfig, FallbackChain, ProviderConfig
    from coderouter.routing import FallbackEngine

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
    engine = FallbackEngine(config)

    # Record some budget spend.
    engine._budget.record("p1", 2.50)

    # Attach store and save.
    store = StateStore(tmp_path / "test.db")
    engine.attach_state_store(store)
    engine.save_all_state()

    # Verify state was persisted.
    budget_state = store.get("budget", "state")
    assert budget_state is not None
    assert budget_state["totals"]["p1"] == pytest.approx(2.50)  # type: ignore[index]

    store.close()
