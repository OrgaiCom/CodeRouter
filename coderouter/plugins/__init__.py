"""Plugin SDK — extension points for in-process CodeRouter plugins (v2.3.0).

CodeRouter Core stays at 5 deps. Optional functionality (memory, PII
redaction, observability bridges, alternative ingresses, etc.) ships
as separate ``coderouter-plugin-*`` packages on PyPI. Each plugin
declares one or more *entry points* in its ``pyproject.toml``; this
SDK discovers them at startup, applies the user's explicit ``enabled``
allowlist (supply chain defense, see :func:`loader.discover_and_load`),
and exposes a :class:`registry.PluginRegistry` to the engine.

Six extension points are defined as :mod:`Protocols <typing>` in
:mod:`coderouter.plugins.base`. v2.3.0 implements the engine-side hook
integration for two of them (``input_filter`` and ``observer``); the
remaining four (``frontend``, ``guard``, ``output_filter``,
``adapter``) have a stable Protocol contract so plugins can target
them now, but the engine doesn't yet wire them into the request flow.
That's intentional — adding contracts is cheap, adding hot-path code
isn't, so we wait for a real plugin to drive each integration.

Public API:

- :class:`InputFilter`, :class:`Observer` — implementable today.
- :class:`Frontend`, :class:`Guard`, :class:`OutputFilter`,
  :class:`Adapter` — Protocol-only, integration in v2.4+.
- :func:`discover_and_load` — called once during config load.
- :class:`PluginRegistry` — held by :class:`FallbackEngine`.
"""
from __future__ import annotations

from coderouter.plugins.base import (
    Adapter,
    Frontend,
    Guard,
    InputFilter,
    Observer,
    OutputFilter,
)
from coderouter.plugins.loader import (
    PLUGIN_GROUPS_FUTURE,
    PLUGIN_GROUPS_V2_3,
    discover_and_load,
)
from coderouter.plugins.registry import PluginRegistry

__all__ = [
    # Active hooks
    "InputFilter",
    "Observer",
    # Future hooks (Protocol-only, no engine integration yet)
    "Frontend",
    "Guard",
    "OutputFilter",
    "Adapter",
    # Discovery + container
    "PluginRegistry",
    "discover_and_load",
    "PLUGIN_GROUPS_V2_3",
    "PLUGIN_GROUPS_FUTURE",
]
