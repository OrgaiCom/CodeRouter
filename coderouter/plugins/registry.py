"""PluginRegistry — group-keyed container for loaded plugin instances.

The registry is constructed once during config load (by
:func:`coderouter.plugins.loader.discover_and_load`) and held by the
:class:`coderouter.routing.fallback.FallbackEngine` for the process
lifetime. Lookups are O(1) per group; insertion order is preserved so
that ``plugins.enabled`` order in providers.yaml controls the order
filters run.

Group keys are lowercase, the same names used in pyproject.toml
``[project.entry-points."coderouter.<group>"]`` sections (e.g.
``input_filter``, ``observer``).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


class PluginRegistry:
    """In-memory registry of loaded plugin instances grouped by hook kind.

    Use the typed ``input_filters`` / ``observers`` properties on the
    hot path; plugin code should not iterate ``_by_group`` directly.
    """

    def __init__(self) -> None:
        # ``defaultdict(list)`` keeps insertion order, which matters for
        # InputFilter chaining (filters apply in registration order).
        self._by_group: dict[str, list[Any]] = defaultdict(list)

    @classmethod
    def empty(cls) -> PluginRegistry:
        """Convenience constructor for the no-plugin baseline.

        ``FallbackEngine`` defaults to this when ``providers.yaml``
        omits the ``plugins`` block, so all existing call sites
        (``__init__``, tests that build engines via ``__new__``) keep
        their zero-cost behavior.
        """
        return cls()

    def add(self, group: str, instance: Any) -> None:
        """Register an instance under a hook group.

        The instance is appended to the group's list — duplicates are
        not deduplicated here because the loader applies the
        ``enabled`` allowlist upstream (no instance reaches this
        method without an explicit enable).
        """
        self._by_group[group].append(instance)

    @property
    def input_filters(self) -> list[Any]:
        """Plugins registered as ``coderouter.input_filter``.

        Returns a copy so iteration during chain execution can't be
        invalidated by a concurrent registration (registrations only
        happen at startup, but defensive copying is cheap).
        """
        return list(self._by_group.get("input_filter", ()))

    @property
    def observers(self) -> list[Any]:
        """Plugins registered as ``coderouter.observer``."""
        return list(self._by_group.get("observer", ()))

    def is_empty(self) -> bool:
        """True iff no plugin instance has been registered, in any group.

        ``FallbackEngine`` uses this to short-circuit the hook loops
        entirely when no plugins are configured — keeping the no-
        plugin code path bit-identical to v2.2.0 behavior.
        """
        return not any(self._by_group.values())

    def count(self, group: str) -> int:
        """Number of instances registered in ``group``. Used by tests + logs."""
        return len(self._by_group.get(group, ()))

    def groups(self) -> list[str]:
        """All group names that have at least one registered instance."""
        return [g for g, items in self._by_group.items() if items]
