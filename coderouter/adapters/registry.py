"""Adapter factory — maps `kind` strings to adapter classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import BaseAdapter
from coderouter.adapters.openai_compat import OpenAICompatAdapter
from coderouter.config.schemas import ProviderConfig

if TYPE_CHECKING:
    from coderouter.plugins.registry import PluginRegistry

# in-core kinds, in resolution order. Kept as a tuple (not derived from
# the if-chain below) so the "Unknown adapter kind" error message and
# the plugin-shadow guard have a single source of truth.
_IN_CORE_KINDS: tuple[str, ...] = ("openai_compat", "anthropic", "agent_cli")


def _plugin_kinds(plugin_registry: PluginRegistry | None) -> list[str]:
    """``kind`` values served by enabled adapter plugins, for error text."""
    if plugin_registry is None:
        return []
    return [factory.kind for factory in plugin_registry.adapters]


def build_adapter(
    provider: ProviderConfig,
    plugin_registry: PluginRegistry | None = None,
) -> BaseAdapter:
    """Construct an adapter from a ProviderConfig.

    Resolution order (docs/designs/agent-cli-plugin-extraction.md §3.2):
    in-core kinds first, then plugin-provided kinds, then a fail-fast
    error. In-core kinds are checked first so a plugin can never shadow
    a kind Core itself guarantees (``openai_compat`` / ``anthropic`` /,
    during the Phase 2b migration window, ``agent_cli``).
    """
    if provider.kind == "openai_compat":
        return OpenAICompatAdapter(provider)
    if provider.kind == "anthropic":
        return AnthropicAdapter(provider)
    if provider.kind == "agent_cli":
        # Imported lazily so the external-agent adapter (and its subprocess /
        # os plumbing) is only pulled in when a config actually uses it.
        from coderouter.adapters.agent_cli import AgentCliAdapter

        return AgentCliAdapter(provider)
    if plugin_registry is not None:
        for factory in plugin_registry.adapters:
            if factory.kind == provider.kind:
                return factory.build(provider)
    raise ValueError(
        f"Unknown adapter kind {provider.kind!r}. "
        f"in-core kinds: {', '.join(_IN_CORE_KINDS)}; "
        f"plugin-provided kinds: {_plugin_kinds(plugin_registry)}. "
        f"If a plugin should provide {provider.kind!r}, ensure it is "
        f"installed AND listed in plugins.enabled."
    )
