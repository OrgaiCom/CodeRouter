"""Adapter factory — maps `kind` strings to adapter classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import BaseAdapter
from coderouter.adapters.openai_compat import OpenAICompatAdapter
from coderouter.config.schemas import ProviderConfig
from coderouter.logging import get_logger

if TYPE_CHECKING:
    from coderouter.plugins.registry import PluginRegistry

logger = get_logger(__name__)

# in-core kinds, in resolution order. Kept as a tuple (not derived from
# the if-chain below) so the "Unknown adapter kind" error message and
# the plugin-shadow guard have a single source of truth.
#
# Phase 2c (docs/designs/agent-cli-plugin-extraction.md §7 row "2c"):
# "agent_cli" is no longer in this tuple. It is served exclusively by
# the coderouter-plugin-agents adapter plugin now that the in-core
# AgentCliAdapter and its build_adapter branch have been removed.
_IN_CORE_KINDS: tuple[str, ...] = ("openai_compat", "anthropic")

# Phase 2c migration hint (docs/designs/agent-cli-plugin-extraction.md
# §5.2): shown ONLY for kind="agent_cli" when no plugin resolves it, so
# operators upgrading from pre-2c configs get a targeted fix instead of
# the generic unknown-kind message.
_AGENT_CLI_MIGRATION_HINT = (
    "kind='agent_cli' is served by the coderouter-plugin-agents plugin "
    "(Core no longer ships an in-core agent_cli adapter as of Phase 2c). "
    'Install it with: pip install "coderouter-plugin-agents @ '
    'git+https://github.com/zephel01/coderouter-plugin-agents" and add '
    "'agents' to plugins.enabled in providers.yaml, e.g.:\n"
    "plugins:\n"
    "  enabled:\n"
    "    - agents"
)


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
    a kind Core itself guarantees (``openai_compat`` / ``anthropic``).

    As of Phase 2c, ``agent_cli`` is no longer an in-core kind — it
    resolves ONLY via the plugin path (``coderouter-plugin-agents``).
    """
    if provider.kind == "openai_compat":
        return OpenAICompatAdapter(provider)
    if provider.kind == "anthropic":
        return AnthropicAdapter(provider)
    if plugin_registry is not None:
        for factory in plugin_registry.adapters:
            if factory.kind == provider.kind:
                return factory.build(provider)
    if provider.kind == "agent_cli":
        # Targeted migration hint (§5.2) instead of the generic message
        # below — this is the single most common post-2c misconfiguration
        # (an un-migrated providers.yaml that still relies on the removed
        # in-core adapter).
        raise ValueError(f"Unknown adapter kind {provider.kind!r}. {_AGENT_CLI_MIGRATION_HINT}")
    raise ValueError(
        f"Unknown adapter kind {provider.kind!r}. "
        f"in-core kinds: {', '.join(_IN_CORE_KINDS)}; "
        f"plugin-provided kinds: {_plugin_kinds(plugin_registry)}. "
        f"If a plugin should provide {provider.kind!r}, ensure it is "
        f"installed AND listed in plugins.enabled."
    )
