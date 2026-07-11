"""Plugin SDK Protocol contracts (v2.3.0, Adapter wired in v2.8.0).

Six extension points are defined here. Three are wired into the engine
(:class:`InputFilter`, :class:`Observer` since v2.3.0; :class:`Adapter`
since v2.8.0 — see ``docs/designs/agent-cli-plugin-extraction.md`` §2);
three remain Protocol-only (:class:`Frontend`, :class:`Guard`,
:class:`OutputFilter`) and will get engine integration when a real
plugin drives the requirement — see
``docs/inside/plugin-architecture-draft.md`` §3 for the full design
rationale.

Why declare contracts ahead of integration? It lets a plugin author
build against the SDK *now* (and ship a working ``coderouter.frontend``
plugin once integration lands) without us having to do a
backward-incompatible Protocol revision later. ``runtime_checkable``
is used so :func:`isinstance` checks work in the loader for clearer
error messages.

InputFilter / Observer are async and run repeatedly on the hot path;
failures must NEVER block the engine response, so the engine wraps
every hook call in try/except and degrades gracefully (see
``coderouter/routing/fallback.py`` integration site). Adapter is
different in shape — see its docstring below.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Avoid circular imports at runtime — Protocol typing only needs
    # these for documentation and static analysis.
    from coderouter.adapters.base import BaseAdapter
    from coderouter.config.schemas import CodeRouterConfig, ProviderConfig
    from coderouter.translation.anthropic import (
        AnthropicRequest,
        AnthropicResponse,
    )


# ====================================================================
# Active hooks (engine integration in v2.3.0)
# ====================================================================


@runtime_checkable
class InputFilter(Protocol):
    """Mutates an inbound :class:`AnthropicRequest` before chain dispatch.

    Plugins MUST treat the input as immutable: build the modified
    request with ``request.model_copy(update={...})`` and return the
    new instance. The engine assumes the returned value is a
    *replacement*, not the same object.

    Engine semantics:

    - Filters run sequentially in the order they appear in
      ``plugins.enabled``. The first filter sees the raw inbound
      request; each subsequent filter sees the previous filter's
      output.
    - If :meth:`transform` raises, the engine logs ``input-filter-failed``
      and continues with the *pre-mutation* request for that filter.
      Other filters still run.
    - The transform runs *after* the v1.9-E tool-loop guard but
      *before* chain resolution and the v2.0-F context budget guard.
      That order matters: filters can grow ``request.system`` (e.g.
      memory injection) without bypassing the budget cap, because the
      budget guard reruns over the post-filter payload.
    """

    name: str

    async def transform(self, request: AnthropicRequest) -> AnthropicRequest: ...


@runtime_checkable
class Observer(Protocol):
    """Passive event consumer. Cannot mutate anything.

    The engine calls observers via :func:`asyncio.create_task` (fire
    and forget). An observer that takes 30s won't slow a 200ms
    response down — it just runs in the background. If it raises,
    the engine logs ``observer-failed`` and discards the exception.

    Event vocabulary (v2.3.0):

    - ``request_completed`` — payload: ``{request, response,
      latency_ms, provider}``. Fires after a successful Anthropic or
      OpenAI-compat response. Streaming requests fire this event once,
      after the SSE stream has terminated (not on each chunk).

    Plugins MUST tolerate unknown event types — the vocabulary will
    grow over time, and an old plugin should silently ignore events
    it doesn't recognize rather than crash.
    """

    name: str

    async def on_event(self, event_type: str, payload: dict[str, Any]) -> None: ...


# ====================================================================
# Adapter hook (engine integration in v2.8.0)
# ====================================================================


@runtime_checkable
class Adapter(Protocol):
    """New ``kind`` value in providers.yaml, backed by a plugin.

    The plugin declares the ``kind`` string it serves and a factory
    that turns a matching :class:`~coderouter.config.schemas.ProviderConfig`
    into a :class:`~coderouter.adapters.base.BaseAdapter` instance —
    the same surface :func:`coderouter.adapters.registry.build_adapter`
    uses for in-core kinds, so the engine treats plugin adapters
    indistinguishably from built-ins once registered. See
    ``docs/designs/agent-cli-plugin-extraction.md`` §2 for the design
    rationale.

    Shape-wise this hook differs from :class:`InputFilter` /
    :class:`Observer`: those are instances the engine calls repeatedly
    on the hot path, so they're async. An adapter provider is instead
    a **factory** consulted once per provider at engine startup (and
    again if the provider is re-registered at runtime) — construction
    is I/O-free, so :meth:`build` is synchronous, matching
    ``build_adapter``.
    """

    name: str
    # The ``kind`` value this plugin serves in providers.yaml. Distinct
    # from ``name`` (the plugin's own identifier, used in
    # ``plugins.enabled`` / logs) — kept 1:1 for now; a future
    # ``kinds: tuple[str, ...]`` can generalize this if a real plugin
    # needs to serve more than one kind.
    kind: str

    def build(self, config: ProviderConfig) -> BaseAdapter: ...


# ====================================================================
# Future hooks (Protocol-only, engine integration in v2.4+)
# ====================================================================


@runtime_checkable
class Frontend(Protocol):
    """Alternative ingress (Discord, Telegram, MCP, Voice, ...).

    Frontends *run* the engine rather than being called by it.
    Integration plan: each frontend gets its own
    :func:`asyncio.create_task` started by ``coderouter serve``;
    SIGTERM cancels all of them and waits for cleanup.

    Not yet integrated — Protocol contract only.
    """

    name: str

    async def serve(
        self, engine: Any, config: CodeRouterConfig
    ) -> None: ...


@runtime_checkable
class Guard(Protocol):
    """Reliability guard, parallel to the built-in tool-loop guard.

    Runs synchronously on the hot path, so implementations must be
    cheap. Heavy work (HTTP calls, model invocations, etc.) belongs
    in an :class:`Observer`.

    Not yet integrated — Protocol contract only.
    """

    name: str

    async def check(
        self, request: AnthropicRequest, config: CodeRouterConfig
    ) -> None: ...


@runtime_checkable
class OutputFilter(Protocol):
    """Mutates a response before it returns to the client.

    For streaming, the engine plans to call :meth:`transform` once
    per ``AnthropicStreamEvent``; for non-streaming, once per
    response.

    Not yet integrated — Protocol contract only.
    """

    name: str

    async def transform(
        self, response: AnthropicResponse
    ) -> AnthropicResponse: ...
