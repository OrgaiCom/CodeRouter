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
_IN_CORE_KINDS: tuple[str, ...] = ("openai_compat", "anthropic", "agent_cli")

# Module-level flag so the Phase 2b in-core agent_cli deprecation warning
# (docs/designs/agent-cli-plugin-extraction.md §5.1) fires at most once per
# process, regardless of how many agent_cli providers are built.
_agent_cli_deprecation_logged = False


def _plugin_kinds(plugin_registry: PluginRegistry | None) -> list[str]:
    """``kind`` values served by enabled adapter plugins, for error text."""
    if plugin_registry is None:
        return []
    return [factory.kind for factory in plugin_registry.adapters]


def _warn_agent_cli_in_core_deprecated_once() -> None:
    """Log ``agent-cli-in-core-deprecated`` once per process (§5.1).

    Only called when the in-core ``agent_cli`` branch is taken AND an
    adapter plugin has also registered a ``kind="agent_cli"`` factory —
    i.e. the operator already has ``coderouter-plugin-agents`` installed
    and enabled, but Core's in-core copy still wins resolution order
    (§3.2) during the Phase 2b migration window. Nudges them toward the
    plugin path ahead of its Phase 2c removal from Core.
    """
    global _agent_cli_deprecation_logged
    if _agent_cli_deprecation_logged:
        return
    _agent_cli_deprecation_logged = True
    logger.warning(
        "agent-cli-in-core-deprecated",
        extra={
            "hint": (
                "an 'agent_cli' adapter plugin is installed and enabled, "
                "but Core's in-core agent_cli implementation still served "
                "this request (in-core wins resolution during the Phase 2b "
                "migration window). The in-core copy will be removed in "
                "Phase 2c — no action is required yet, but new setups "
                "should already be relying on the plugin path "
                "(coderouter-plugin-agents)."
            ),
        },
    )


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

        if plugin_registry is not None and any(
            factory.kind == "agent_cli" for factory in plugin_registry.adapters
        ):
            _warn_agent_cli_in_core_deprecated_once()
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
