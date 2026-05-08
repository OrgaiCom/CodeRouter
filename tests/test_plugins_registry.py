"""Unit tests for ``coderouter.plugins.registry`` (v2.3.0)."""
from __future__ import annotations

from coderouter.plugins.registry import PluginRegistry


class _DummyPlugin:
    """Minimal stand-in — Protocol membership is structural."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_empty_factory_is_truly_empty() -> None:
    reg = PluginRegistry.empty()
    assert reg.is_empty()
    assert reg.input_filters == []
    assert reg.observers == []
    assert reg.groups() == []
    assert reg.count("input_filter") == 0
    assert reg.count("observer") == 0


def test_add_into_two_groups() -> None:
    reg = PluginRegistry()
    reg.add("input_filter", _DummyPlugin("a"))
    reg.add("observer", _DummyPlugin("b"))
    reg.add("input_filter", _DummyPlugin("c"))

    assert reg.count("input_filter") == 2
    assert reg.count("observer") == 1
    assert not reg.is_empty()
    assert set(reg.groups()) == {"input_filter", "observer"}


def test_input_filter_order_is_insertion_order() -> None:
    """Filter chain order matters — first inserted runs first."""
    reg = PluginRegistry()
    a = _DummyPlugin("a")
    b = _DummyPlugin("b")
    c = _DummyPlugin("c")
    reg.add("input_filter", a)
    reg.add("input_filter", b)
    reg.add("input_filter", c)

    assert [p.name for p in reg.input_filters] == ["a", "b", "c"]


def test_observers_returned_as_list() -> None:
    reg = PluginRegistry()
    obs1 = _DummyPlugin("o1")
    obs2 = _DummyPlugin("o2")
    reg.add("observer", obs1)
    reg.add("observer", obs2)

    assert reg.observers == [obs1, obs2]


def test_input_filters_returns_independent_copy() -> None:
    """Mutations to the returned list MUST NOT affect the registry."""
    reg = PluginRegistry()
    reg.add("input_filter", _DummyPlugin("a"))

    snapshot = reg.input_filters
    snapshot.append(_DummyPlugin("intruder"))

    # Registry is unchanged — caller mutated their own copy only.
    assert reg.count("input_filter") == 1
    assert reg.input_filters[0].name == "a"


def test_unknown_group_returns_empty_list() -> None:
    """Future hook groups (not yet wired) return [] not KeyError."""
    reg = PluginRegistry()
    # Use a hook group name from PLUGIN_GROUPS_FUTURE.
    assert reg.count("frontend") == 0
    # Direct dict access via the registry's typed surface stays empty.
    # is_empty must be True even when only future groups are referenced.
    assert reg.is_empty()


def test_groups_excludes_empty_buckets() -> None:
    """An add()-then-implicit-empty (defaultdict-style) group is omitted."""
    reg = PluginRegistry()
    # Reading an unknown group via count() must NOT pollute groups().
    _ = reg.count("frontend")
    # Now actually populate one group.
    reg.add("observer", _DummyPlugin("only-obs"))

    assert reg.groups() == ["observer"]
