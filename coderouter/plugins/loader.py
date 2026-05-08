"""Plugin discovery + instantiation (v2.3.0).

Reads ``importlib.metadata`` entry points under the ``coderouter.*``
groups, applies the user's explicit ``plugins.enabled`` allowlist,
constructs each plugin with its config dict, and returns a
:class:`PluginRegistry` ready for the engine to consume.

Supply-chain defense
====================

Just having ``coderouter-plugin-X`` installed is **not enough** to
activate a plugin — the user must list the entry-point name in
``providers.yaml`` under ``plugins.enabled``. Without that explicit
opt-in, an installed-but-unlisted plugin is silently skipped (logged
as ``plugin-skipped``). This stops a compromised transitive
dependency from injecting itself into the request flow.

Failure mode: degraded continue
===============================

Plugin loading is best-effort. A failure to import a module, find a
class, or call ``__init__`` is logged at error level and the engine
keeps booting without that plugin. The rationale: a misconfigured
optional plugin shouldn't take down the wire-level router that other
plugins (and the core) depend on. Operators see the failure in logs
and the ``/dashboard`` plugin panel.
"""
from __future__ import annotations

import importlib.metadata as md
from typing import TYPE_CHECKING

from coderouter.logging import get_logger
from coderouter.plugins.registry import PluginRegistry

if TYPE_CHECKING:
    from coderouter.config.schemas import CodeRouterConfig

logger = get_logger(__name__)

# Active hook groups in v2.3.0. The engine wires these into the
# request flow; plugins targeting them will see their methods called
# at runtime.
PLUGIN_GROUPS_V2_3: tuple[str, ...] = ("input_filter", "observer")

# Hook groups whose Protocol contracts are stable but whose engine
# integration is deferred. Listing them here means a plugin author
# can publish today and the loader will silently skip them with a
# clear log line — no surprise crashes when integration lands.
PLUGIN_GROUPS_FUTURE: tuple[str, ...] = (
    "frontend",
    "guard",
    "output_filter",
    "adapter",
)


def discover_and_load(config: CodeRouterConfig) -> PluginRegistry:
    """Load plugins listed in ``config.plugins.enabled``.

    Returns an empty registry when the ``plugins`` block is absent
    from providers.yaml or its ``enabled`` list is empty. The empty
    return is the same one ``PluginRegistry.empty()`` produces, so
    the engine's hook loops short-circuit and incur zero cost in the
    default (no-plugin) configuration.

    Each enabled plugin is instantiated as
    ``cls(**config.plugins.config.get(name, {}))``. The plugin's
    ``__init__`` gets a fresh dict (validation happens inside the
    plugin's own constructor — Core stays out of plugin-specific
    schemas).

    Logs emitted (one per outcome, all structured):

    - ``plugin-loaded`` (info) — module + class loaded and constructed.
    - ``plugin-skipped`` (info) — entry point exists but not enabled.
    - ``plugin-load-failed`` (error) — import or ``__init__`` raised;
      the engine continues without that plugin.
    """
    plugins_cfg = getattr(config, "plugins", None)
    if plugins_cfg is None or not plugins_cfg.enabled:
        return PluginRegistry.empty()

    enabled = set(plugins_cfg.enabled)
    registry = PluginRegistry()

    # Track which enabled names actually matched an entry point so we
    # can warn the operator about typos in ``plugins.enabled`` —
    # otherwise a misspelled name silently does nothing and is hard
    # to diagnose.
    seen_names: set[str] = set()

    for group in PLUGIN_GROUPS_V2_3 + PLUGIN_GROUPS_FUTURE:
        ep_group = f"coderouter.{group}"
        for ep in md.entry_points(group=ep_group):
            if ep.name not in enabled:
                # Installed but not enabled — silently skip.  Logged
                # at info level so operators have a paper trail of
                # plugins they could enable.
                logger.info(
                    "plugin-skipped",
                    extra={
                        "plugin": ep.name,
                        "group": group,
                        "reason": "not in plugins.enabled",
                    },
                )
                continue

            seen_names.add(ep.name)

            # Future-only groups: plugin is installed *and* enabled,
            # but the engine doesn't yet wire this group into the
            # request flow. Load it anyway so the plugin can sanity-
            # check its own construction; just log a clear warning.
            if group in PLUGIN_GROUPS_FUTURE:
                logger.warning(
                    "plugin-group-not-yet-active",
                    extra={
                        "plugin": ep.name,
                        "group": group,
                        "note": (
                            f"engine integration for '{group}' "
                            "is deferred to v2.4+; the plugin will "
                            "be loaded but its hook is never called"
                        ),
                    },
                )

            try:
                cls = ep.load()
                cfg = plugins_cfg.config.get(ep.name, {}) or {}
                instance = cls(**cfg)
                registry.add(group, instance)
                logger.info(
                    "plugin-loaded",
                    extra={
                        "plugin": ep.name,
                        "group": group,
                        "entry_point": ep.value,
                    },
                )
            except Exception as exc:
                # Don't propagate — engine boot must succeed even when
                # an optional plugin is broken.
                logger.error(
                    "plugin-load-failed",
                    extra={
                        "plugin": ep.name,
                        "group": group,
                        "entry_point": ep.value,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )

    # Warn on enabled names that didn't match any installed entry
    # point — most often this means the user typed a name in
    # providers.yaml but forgot to ``pip install`` the corresponding
    # plugin package.
    missing = enabled - seen_names
    if missing:
        for name in sorted(missing):
            logger.warning(
                "plugin-not-found",
                extra={
                    "plugin": name,
                    "hint": (
                        "listed in plugins.enabled but no entry point "
                        "with this name was discovered. Did you forget "
                        "to install the plugin package?"
                    ),
                },
            )

    return registry
