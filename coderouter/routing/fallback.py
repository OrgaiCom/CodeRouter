"""Sequential fallback engine.

Behavior (plan.md §7):
    1. Iterate the provider list of the chosen profile in order.
    2. Skip paid providers when ALLOW_PAID is false.
    3. Try generate() / stream() on each. If AdapterError(retryable=True) → next.
    4. If all providers fail, raise NoProvidersAvailableError.

Dual entry points (v0.3.x-1):
    The engine exposes both OpenAI-shaped (generate / stream) and
    Anthropic-shaped (generate_anthropic / stream_anthropic) methods. The
    Anthropic-shaped methods dispatch per-provider on `ProviderConfig.kind`:
        - kind="anthropic":    passthrough — no translation on either leg.
        - kind="openai_compat": translate AnthropicRequest → ChatRequest,
                               call the adapter, translate ChatResponse /
                               stream chunks back. Tool-call repair runs on
                               non-streaming responses; streaming tool-turns
                               are downgraded to non-stream internally
                               (v0.3-D strategy).

    Mixed chains are supported: a profile can list a native Anthropic
    provider first and fall through to an openai_compat provider second.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from coderouter.config.schemas import FallbackChain
    from coderouter.guards.drift_detection import DriftVerdict
    from coderouter.guards.self_healing import SelfHealingOrchestrator
    from coderouter.state.store import StateStore

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import (
    AdapterError,
    BaseAdapter,
    ChatRequest,
    ChatResponse,
    ProviderCallOverrides,
    StreamChunk,
)
from coderouter.adapters.registry import build_adapter
from coderouter.config.schemas import CodeRouterConfig, ProviderConfig
from coderouter.cost import compute_cost_for_attempt
from coderouter.errors import CodeRouterError
from coderouter.guards.backend_health import BackendHealthMonitor
from coderouter.guards.memory_pressure import (
    MemoryPressureGuard,
    is_memory_pressure_error,
)
from coderouter.guards.tool_loop import (
    DEFAULT_LOOP_INJECT_HINT,
    ToolCountExceededError,
    ToolLoopBreakError,
    check_total_tool_count,
    detect_tool_loop,
    inject_loop_break_hint,
)
from coderouter.logging import (
    classify_cache_outcome,
    get_logger,
    log_backend_health_changed,
    log_cache_observed,
    log_chain_budget_exceeded,
    log_chain_memory_pressure_blocked,
    log_chain_paid_gate_blocked,
    log_demote_unhealthy_provider,
    log_memory_pressure_detected,
    log_skip_budget_exceeded,
    log_skip_memory_pressure,
    log_tool_loop_detected,
)
from coderouter.plugins.registry import PluginRegistry
from coderouter.routing.adaptive import AdaptiveAdjuster
from coderouter.routing.budget import BudgetTracker
from coderouter.routing.capability import (
    anthropic_request_has_cache_control,
    anthropic_request_requires_thinking,
    log_capability_degraded,
    provider_supports_cache_control,
    provider_supports_thinking,
    strip_thinking,
)
from coderouter.translation import (
    AnthropicRequest,
    AnthropicResponse,
    AnthropicStreamEvent,
    stream_chat_to_anthropic_events,
    synthesize_anthropic_stream_from_response,
    to_anthropic_response,
    to_chat_request,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# v1.9-A: cache observation helper
#
# Single chokepoint that turns a successful AnthropicResponse into a
# ``cache-observed`` log line. Lives at module scope (not on the engine
# class) so unit tests can feed a synthetic response without spinning up
# a fallback engine.
#
# We pass `request_had_cache_control` in from the caller (already
# computed in v0.5-B for the capability-degraded gate) so we don't
# re-walk the AnthropicRequest tree twice per call.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v1.9-E (L3): tool-loop guard helpers
#
# The detection runs at the entry of generate_anthropic / stream_anthropic
# before chain dispatch. Three policy actions are dispatched off the
# resolved profile's ``tool_loop_action`` field. ``warn`` always logs;
# ``inject`` logs + returns a mutated request; ``break`` logs + raises
# ToolLoopBreakError which the ingress converts to a 400 response.
#
# Engine integration is intentionally minimal — the guard is a pure
# function and the action dispatch is a small switch. We do NOT pass
# the detection through to the chain itself (no per-adapter awareness)
# because the loop signal is request-shape-only and the chain is
# already free to fall back on a different provider for diagnosis.
# ---------------------------------------------------------------------------


def _apply_tool_loop_guard(
    request: AnthropicRequest, *, config: CodeRouterConfig
) -> AnthropicRequest:
    """Run the L3 tool-loop guard and apply the configured action.

    Returns the (possibly mutated) request. Raises
    :class:`ToolLoopBreakError` when the configured action is ``break``
    and a loop was detected. Also raises :class:`ToolCountExceededError`
    when the total tool-call count exceeds ``max_tool_calls`` (v2.2).

    Profile resolution: uses ``request.profile`` (the X-CodeRouter-Mode
    header / explicit body field) and falls back to
    ``config.default_profile``. The profile's
    ``tool_loop_window`` / ``tool_loop_threshold`` /
    ``tool_loop_action`` fields parameterize the guard. When the profile
    is missing (e.g. test harness with a stripped config), the guard
    is a no-op — there's no safe default for "no profile".
    """
    chosen = request.profile or config.default_profile
    try:
        profile = config.profile_by_name(chosen)
    except (KeyError, ValueError):
        # Profile lookup failure is handled elsewhere; the guard
        # silently no-ops so we don't double-error before the chain
        # resolution path produces its own diagnostic.
        return request

    # v2.2: total tool-call count hard cap — runs before streak
    # detection because it's a cheaper O(n) scan that catches a
    # broader class of runaway behavior.
    if profile.max_tool_calls > 0:
        exceeded = check_total_tool_count(
            request,
            max_calls=profile.max_tool_calls,
        )
        if exceeded is not None:
            logger.warning(
                "tool-count-exceeded",
                extra={
                    "profile": profile.name,
                    "total_count": exceeded.total_count,
                    "max_allowed": exceeded.max_allowed,
                    "action": profile.tool_loop_action,
                },
            )
            if profile.tool_loop_action == "break":
                raise ToolCountExceededError(exceeded, profile.name)
            # For "warn" and "inject" actions, log only and continue.
            # The inject action's hint is not meaningful for count
            # exceeded (not a same-tool loop), so we just warn.

    detection = detect_tool_loop(
        request,
        window=profile.tool_loop_window,
        threshold=profile.tool_loop_threshold,
    )
    if detection is None:
        return request

    log_tool_loop_detected(
        logger,
        profile=profile.name,
        tool_name=detection.tool_name,
        repeat_count=detection.repeat_count,
        threshold=profile.tool_loop_threshold,
        window=profile.tool_loop_window,
        action=profile.tool_loop_action,
    )

    if profile.tool_loop_action == "warn":
        return request
    if profile.tool_loop_action == "inject":
        return inject_loop_break_hint(request, hint=DEFAULT_LOOP_INJECT_HINT)
    if profile.tool_loop_action == "break":
        raise ToolLoopBreakError(
            detection,
            profile.name,
            threshold=profile.tool_loop_threshold,
            window=profile.tool_loop_window,
        )
    # Defensive — schema validates the literal so we never reach here.
    return request


# ---------------------------------------------------------------------------
# v2.0-F (L1): context budget guard
#
# Detects when the inbound request's estimated token count approaches
# the target provider's context window. Runs after tool-loop detection,
# before chain dispatch. Two action paths:
#   * ``warn`` — emit a structured log. Does NOT mutate the request.
#   * ``trim`` — ``warn`` + remove old messages to fit within budget.
#     Returns a mutated request (the engine sends the shortened version).
#
# The guard needs the first provider's max_context_tokens to compute
# the budget. It receives the resolved chain (post-_resolve_anthropic_chain)
# and reads the first adapter's config. This is the "most likely to serve"
# provider; if it fails and the chain falls through to a provider with a
# different context window, the budget may be slightly off — acceptable
# because trim is conservative (targets 75%, not 100%).
# ---------------------------------------------------------------------------


def _resolve_max_context_tokens(
    provider_config: ProviderConfig,
) -> int:
    """Resolve the effective max_context_tokens for a provider.

    Precedence:
      1. ProviderConfig.max_context_tokens (explicit declaration)
      2. CapabilityRegistry lookup (model-capabilities.yaml)
      3. DEFAULT_MAX_CONTEXT_TOKENS (128K fallback)
    """
    from coderouter.token_estimation import DEFAULT_MAX_CONTEXT_TOKENS

    # 1. Explicit provider-level declaration
    if provider_config.max_context_tokens is not None:
        return provider_config.max_context_tokens

    # 2. Registry lookup
    from coderouter.routing.capability import get_default_registry

    registry = get_default_registry()
    resolved = registry.lookup(kind=provider_config.kind, model=provider_config.model or "")
    if resolved.max_context_tokens is not None:
        return resolved.max_context_tokens

    # 3. Fallback
    return DEFAULT_MAX_CONTEXT_TOKENS


def _apply_context_budget_guard(
    request: AnthropicRequest,
    *,
    config: CodeRouterConfig,
    first_provider_config: ProviderConfig | None,
) -> tuple[AnthropicRequest, str | None]:
    """Run the L1 context-budget guard and apply the configured action.

    Returns ``(request, status)`` — the (possibly trimmed) request and
    a status string for the response header:
      * ``None``       — guard inactive or below all thresholds.
      * ``"warning"``  — over warn threshold but not trimmed.
      * ``"trimmed"``  — messages were removed to fit the budget.

    Parameters
    ----------
    request
        Inbound Anthropic request (post-tool-loop-guard).
    config
        Full CodeRouter config (for profile resolution).
    first_provider_config
        The ProviderConfig of the first adapter in the resolved chain.
        Used to determine max_context_tokens. None → guard is a no-op.
    """
    from coderouter.guards.context_budget import (
        estimate_context_usage,
        trim_to_budget,
    )
    from coderouter.logging import (
        log_context_budget_trimmed,
        log_context_budget_warning,
    )

    if first_provider_config is None:
        return request, None

    # Resolve profile
    chosen = request.profile or config.default_profile
    try:
        profile = config.profile_by_name(chosen)
    except (KeyError, ValueError):
        return request, None

    # Check if guard is enabled
    if profile.context_budget_action == "off":
        return request, None

    # Resolve context window
    max_ctx = _resolve_max_context_tokens(first_provider_config)

    # Estimate usage
    estimate = estimate_context_usage(
        request,
        max_context_tokens=max_ctx,
        warn_threshold=profile.context_budget_warn_threshold,
        trim_threshold=profile.context_budget_trim_threshold,
    )

    # Not over any threshold → pass through
    if not estimate.over_warn_threshold:
        return request, None

    # Over warn threshold → emit warning
    log_context_budget_warning(
        logger,
        provider=first_provider_config.name,
        profile=profile.name,
        estimated_tokens=estimate.estimated_tokens,
        max_context_tokens=estimate.max_context_tokens,
        usage_ratio=estimate.usage_ratio,
        action=profile.context_budget_action,
    )

    # If action is warn-only, or not over trim threshold → done
    if profile.context_budget_action == "warn" or not estimate.over_trim_threshold:
        return request, "warning"

    # Over trim threshold + action is trim → trim messages
    trimmed_request, trim_result = trim_to_budget(
        request,
        max_context_tokens=max_ctx,
        trim_target=profile.context_budget_trim_target,
        preserve_last_n=profile.context_budget_preserve_last_n,
    )

    if trim_result.messages_removed > 0:
        log_context_budget_trimmed(
            logger,
            provider=first_provider_config.name,
            profile=profile.name,
            messages_removed=trim_result.messages_removed,
            messages_before=trim_result.messages_before,
            messages_after=trim_result.messages_after,
            estimated_tokens_before=trim_result.estimated_tokens_before,
            estimated_tokens_after=trim_result.estimated_tokens_after,
            max_context_tokens=max_ctx,
        )
        return trimmed_request, "trimmed"

    # Trim couldn't remove anything (e.g. only preserve_last_n messages)
    return request, "warning"


def _emit_cache_observed(
    response: AnthropicResponse,
    *,
    provider: str,
    request_had_cache_control: bool,
    streaming: bool,
    provider_config: ProviderConfig | None = None,
    budget: BudgetTracker | None = None,
) -> None:
    """Extract usage / cache fields from an AnthropicResponse and log them.

    The Anthropic ``usage`` block carries cache_read_input_tokens /
    cache_creation_input_tokens via the ``extra="allow"`` config on
    :class:`AnthropicUsage` — the engine never had to care about them
    until v1.9-A. We pull them out of ``model_extra`` rather than typing
    them into the schema because (a) the openai_compat → anthropic
    converter zero-fills usage so ``input_tokens`` / ``output_tokens``
    are always present, but cache fields land only on native
    Anthropic / LM Studio /v1/messages responses, and (b) future
    Anthropic API additions (e.g. ephemeral_5m vs ephemeral_1h
    breakdowns) extend ``model_extra`` without a schema change.

    The ``streaming=True`` arg path is exercised by
    :func:`_emit_cache_observed_streaming` (v1.9-B2). The non-streaming
    helper still accepts ``streaming`` as a parameter so the
    OpenAI-compat downgrade path (which collapses to a single
    AnthropicResponse before re-streaming) can mark its log line as
    streaming without needing the message_delta accumulator.

    v1.9-D: when ``provider_config.cost`` is set, also computes the
    USD cost of this attempt + the counterfactual cache savings and
    folds them into the log payload. ``provider_config=None`` (legacy
    callers) yields zero cost figures and the dashboard reports
    nothing for that attempt.
    """
    usage = response.usage
    extra = usage.model_extra or {}
    raw_read = extra.get("cache_read_input_tokens", 0)
    raw_creation = extra.get("cache_creation_input_tokens", 0)
    cache_read = raw_read if isinstance(raw_read, int) else 0
    cache_creation = raw_creation if isinstance(raw_creation, int) else 0
    # ``usage_present`` is True if either usage was populated by the
    # upstream OR derived in conversion. We treat any non-zero token
    # count as evidence the upstream answered with usage info; an
    # all-zero usage from the openai_compat converter is treated as
    # "unknown" so the no_cache bucket only counts real cache misses.
    usage_present = (
        usage.input_tokens > 0
        or usage.output_tokens > 0
        or cache_read > 0
        or cache_creation > 0
    )
    outcome = classify_cache_outcome(
        usage_present=usage_present,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )

    # v1.9-D: compute per-attempt USD cost using the provider's
    # CostConfig (if any). Local / unconfigured providers yield 0.0
    # so they show up in token counters but contribute nothing to
    # the cost dashboard.
    cost = compute_cost_for_attempt(
        provider_config.cost if provider_config is not None else None,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )

    # v1.10: feed the per-provider monthly running total. The
    # budget tracker is opt-in (None → no recording), so test
    # harnesses that bypass the engine's __init__ skip this branch.
    if budget is not None and cost.total_usd > 0.0:
        budget.record(provider, cost.total_usd)

    log_cache_observed(
        logger,
        provider=provider,
        request_had_cache_control=request_had_cache_control,
        outcome=outcome,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        streaming=streaming,
        cost_usd=cost.total_usd,
        cost_savings_usd=cost.savings_usd,
    )


# ---------------------------------------------------------------------------
# v1.9-B2: streaming usage aggregation
#
# The Anthropic streaming SSE protocol delivers usage in two pieces:
#   - ``message_start`` event carries ``message.usage`` with
#     ``input_tokens`` + (optionally) ``cache_read_input_tokens`` /
#     ``cache_creation_input_tokens``. ``output_tokens`` is typically 0
#     here because the model hasn't generated anything yet.
#   - terminal ``message_delta`` event carries ``usage.output_tokens``
#     (the cumulative final count). On some API minor versions the
#     ``message_delta.usage`` block also restates ``input_tokens`` and
#     the cache fields.
#
# We accumulate with a max-merge per field so a delta that restates a
# previously-seen counter never undercounts. Only ``message_start`` and
# ``message_delta`` events are observed — content_block_* / ping /
# message_stop carry no usage data.
#
# Output shape mirrors :func:`_emit_cache_observed` so the
# ``cache-observed`` log payload is structurally identical between
# streaming and non-streaming, and dashboards / Prometheus aggregators
# don't have to special-case streaming.
# ---------------------------------------------------------------------------


class _StreamUsageAccumulator:
    """Aggregate Anthropic streaming usage across SSE events.

    Per-field semantics:
        - ``input_tokens``: appears in ``message_start.message.usage``;
          may be restated in ``message_delta.usage`` on newer API
          minor versions. We take the max so restatements don't
          undercount.
        - ``output_tokens``: cumulative count, finalized in the terminal
          ``message_delta.usage``. ``message_start`` typically reports 0.
          Max-merge handles both shapes.
        - ``cache_read_input_tokens`` / ``cache_creation_input_tokens``:
          land on ``message_start.message.usage`` for native Anthropic
          and LM Studio /v1/messages. Max-merge across events tolerates
          API additions that may restate them in ``message_delta``.

    ``observed`` flips True the first time any event surfaces a
    non-empty usage block, so the caller can distinguish "stream had
    no usage data at all" (→ ``outcome=unknown``) from "stream had
    usage with all-zero counts" (→ degenerate but we still treat as
    unknown).
    """

    __slots__ = (
        "_current_block_text",
        "_current_block_type",
        "_observed",
        "_text_blocks",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "has_tool_use",
        "input_tokens",
        "output_tokens",
        "stop_reason",
    )

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0
        self._observed = False
        # v2.0-G: tracked for drift detection observation at stream end.
        self.has_tool_use: bool = False
        self.stop_reason: str | None = None
        # v2.0-H: partial content accumulation for mid-stream recovery.
        # Completed text blocks are moved to _text_blocks on content_block_stop.
        # In-progress text is in _current_block_text (list of str fragments).
        self._text_blocks: list[str] = []
        self._current_block_type: str | None = None
        self._current_block_text: list[str] = []

    @property
    def partial_content(self) -> list[dict[str, Any]]:
        """Return accumulated text content as Anthropic content blocks.

        Includes both completed blocks and any in-progress text block
        (useful when the stream is interrupted mid-block). Tool_use blocks
        are excluded because partial JSON is unusable.
        """
        blocks: list[dict[str, Any]] = []
        for text in self._text_blocks:
            if text:
                blocks.append({"type": "text", "text": text})
        # Include in-progress text block if any
        if self._current_block_type == "text" and self._current_block_text:
            blocks.append({"type": "text", "text": "".join(self._current_block_text)})
        return blocks

    def observe(self, event: AnthropicStreamEvent) -> None:
        """Update counters from one stream event (no-op for non-usage events)."""
        if event.type == "message_start":
            message = event.data.get("message") if isinstance(event.data, dict) else None
            usage = (message or {}).get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                self._merge(usage)
        elif event.type == "message_delta":
            usage = event.data.get("usage") if isinstance(event.data, dict) else None
            if isinstance(usage, dict):
                self._merge(usage)
            # v2.0-G: capture stop_reason from the terminal message_delta.
            delta = event.data.get("delta") if isinstance(event.data, dict) else None
            if isinstance(delta, dict) and "stop_reason" in delta:
                self.stop_reason = delta["stop_reason"]
        elif event.type == "content_block_start":
            # v2.0-G: detect tool_use content blocks for drift observation.
            cb = event.data.get("content_block") if isinstance(event.data, dict) else None
            if isinstance(cb, dict):
                block_type = cb.get("type", "")
                if block_type == "tool_use":
                    self.has_tool_use = True
                # v2.0-H: start tracking a new content block.
                self._current_block_type = block_type
                self._current_block_text = []
        elif event.type == "content_block_delta":
            # v2.0-H: accumulate text_delta fragments.
            delta = event.data.get("delta") if isinstance(event.data, dict) else None
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    self._current_block_text.append(text)
        elif event.type == "content_block_stop":
            # v2.0-H: finalize current block.
            if self._current_block_type == "text" and self._current_block_text:
                self._text_blocks.append("".join(self._current_block_text))
            self._current_block_type = None
            self._current_block_text = []

    def _merge(self, usage: dict[str, object]) -> None:
        any_nonzero = False
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            raw = usage.get(field)
            if not isinstance(raw, int):
                continue
            cur: int = getattr(self, field)
            if raw > cur:
                setattr(self, field, raw)
            if raw > 0:
                any_nonzero = True
        # We mark "observed" even on an all-zero usage block — the
        # upstream did emit usage, the figures just aren't interesting.
        # Dashboards see this as "no_cache" rather than "unknown".
        self._observed = self._observed or any_nonzero or bool(usage)

    @property
    def usage_present(self) -> bool:
        """Whether any event surfaced usage data (zero or non-zero)."""
        return self._observed or any(
            (
                self.input_tokens > 0,
                self.output_tokens > 0,
                self.cache_read_input_tokens > 0,
                self.cache_creation_input_tokens > 0,
            )
        )


def _emit_cache_observed_streaming(
    accumulator: _StreamUsageAccumulator,
    *,
    provider: str,
    request_had_cache_control: bool,
    provider_config: ProviderConfig | None = None,
    budget: BudgetTracker | None = None,
) -> None:
    """Streaming counterpart of :func:`_emit_cache_observed` (v1.9-B2).

    Reads the aggregated counters from ``accumulator``, runs them
    through the same outcome classifier and cost calculator the
    non-streaming path uses, and emits a single ``cache-observed`` log
    line tagged ``streaming=True``. The log payload is structurally
    identical to the non-streaming sibling so MetricsCollector /
    Prometheus / `/dashboard` consumers don't have to branch.

    Pre-v1.9-B2 the streaming emission hard-coded ``outcome=unknown``
    with all-zero counters (per the v1.9.0a6 doc-implementation gap
    patch); v1.9-B2 finally fulfills the v1.9-A
    :data:`coderouter.logging.CacheOutcome` docstring promise.
    """
    cache_read = accumulator.cache_read_input_tokens
    cache_creation = accumulator.cache_creation_input_tokens
    input_tokens = accumulator.input_tokens
    output_tokens = accumulator.output_tokens

    outcome = classify_cache_outcome(
        usage_present=accumulator.usage_present,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )

    cost = compute_cost_for_attempt(
        provider_config.cost if provider_config is not None else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )

    # v1.10: same monthly-budget bookkeeping as the non-streaming
    # sibling — record the attempt's USD cost into the per-provider
    # current-month total when a tracker is supplied.
    if budget is not None and cost.total_usd > 0.0:
        budget.record(provider, cost.total_usd)

    log_cache_observed(
        logger,
        provider=provider,
        request_had_cache_control=request_had_cache_control,
        outcome=outcome,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        streaming=True,
        cost_usd=cost.total_usd,
        cost_savings_usd=cost.savings_usd,
    )


class NoProvidersAvailableError(CodeRouterError):
    """Raised when every provider in the chain has failed (or was filtered out)."""

    def __init__(self, profile: str, errors: list[AdapterError]) -> None:
        """Construct with the resolved profile name and per-provider errors.

        ``errors`` may be empty when every provider was filtered out
        before a call was attempted (e.g. the paid-gate blocked the
        whole chain); in that case the rendered message falls back to
        ``"no providers eligible"``.
        """
        self.profile = profile
        self.errors = errors
        detail = " | ".join(str(e) for e in errors) or "no providers eligible"
        super().__init__(f"profile={profile!r}: all providers failed: {detail}")


class MidStreamError(CodeRouterError):
    """Raised when a provider fails AFTER it has already emitted at least
    one chunk to the client. Fallback is not attempted (the client has
    received partial content, so switching providers would corrupt the
    stream). Callers should surface this as a terminal error event.

    v2.0-H: carries ``partial_content`` — the accumulated text blocks
    generated before the failure. The ingress uses this to synthesize
    a graceful stream termination when ``partial_stitch_action: surface``.
    """

    def __init__(
        self,
        provider: str,
        original: AdapterError,
        partial_content: list[dict[str, Any]] | None = None,
    ) -> None:
        """Wrap the underlying :class:`AdapterError` with the provider name.

        The ingress layer catches this and converts it into an in-stream
        ``event: error`` (never a 5xx) because HTTP headers have already
        shipped by the time we know the stream failed.
        """
        self.provider = provider
        self.original = original
        self.partial_content: list[dict[str, Any]] = partial_content or []
        super().__init__(f"provider {provider!r} failed mid-stream: {original}")


# ---------------------------------------------------------------------------
# v0.5.1 A-3: "probable misconfig" warn
#
# Motivation (from v0.5-verify.md §Follow-ons, 2026-04-20 re-verify):
#   The first verify run hit OpenRouter with a mis-read env var and got
#   401 back. The single-provider chain short-circuited as it should, but
#   the surface error was just "all providers failed" — operators had to
#   grep the ``provider-failed`` line to spot the common 401 in the
#   `error` field. A one-line warn at the aggregate level turns that
#   grep-and-diagnose into a directly-readable hint.
#
# Scope:
#   - Fires only when EVERY attempt in the chain returned the SAME
#     non-retryable auth status (401 or 403). A mixed chain (one 401 +
#     one 429, etc.) is ambiguous and stays quiet; so does any chain
#     where at least one error was retryable (transient / rate-limit).
#   - Auth-only by design. 400 "model not found" is also non-retryable
#     but reflects a config-vs-upstream-reality mismatch that a generic
#     "probable misconfig" hint would mis-diagnose. Widening later is
#     cheap if we see the need.
#   - Fires for single-provider chains too (the verify scenario). "Every
#     attempt" is trivially all attempts when there is one.
# ---------------------------------------------------------------------------

_AUTH_STATUS_CODES: Final[frozenset[int]] = frozenset({401, 403})


def _warn_if_uniform_auth_failure(errors: list[AdapterError], *, profile: str) -> None:
    """Emit a ``chain-uniform-auth-failure`` warn when the whole chain 401/403'd.

    Called from each of the four ``raise NoProvidersAvailableError`` sites
    right before the raise. No-op when:
        - ``errors`` is empty (nothing was attempted — e.g. every provider
          was filtered out by paid-blocking).
        - The first error's status is not in ``_AUTH_STATUS_CODES``.
        - Any error has a different status_code, or is retryable.

    The log is intentionally separate from the raised exception (which
    stays unchanged for API stability) — it sits alongside the
    ``provider-failed`` lines and gives operators a single-line diagnosis
    without changing the ingress response shape.
    """
    if not errors:
        return
    status = errors[0].status_code
    if status not in _AUTH_STATUS_CODES:
        return
    for exc in errors:
        if exc.status_code != status or exc.retryable:
            return
    logger.warning(
        "chain-uniform-auth-failure",
        extra={
            "profile": profile,
            "status": status,
            "count": len(errors),
            "providers": [exc.provider for exc in errors],
            "hint": "probable-misconfig",
        },
    )


class FallbackEngine:
    """Sequential fallback router — the core of CodeRouter.

    Holds the resolved :class:`CodeRouterConfig` plus a pre-built adapter
    per provider (adapters are cheap but constructing them per-request
    would repeatedly re-read provider config). Exposes four entry
    points: :meth:`generate` / :meth:`stream` for OpenAI-shaped requests,
    :meth:`generate_anthropic` / :meth:`stream_anthropic` for Anthropic
    Messages API requests. See the module docstring for the per-kind
    translation behavior.
    """

    def __init__(
        self,
        config: CodeRouterConfig,
        plugins: PluginRegistry | None = None,
    ) -> None:
        """Pre-build one adapter per configured provider.

        Adapters are stateless with respect to requests (all state is
        held in the per-call ``ProviderCallOverrides``), so caching by
        provider name across requests is safe and avoids the cost of
        re-parsing YAML / re-resolving env vars on every request.

        v1.9-C: an :class:`AdaptiveAdjuster` is constructed eagerly
        but its observation buffers stay empty until the first profile
        with ``adaptive: true`` actually fires. Adapter calls under
        non-adaptive profiles record nothing — zero observation
        overhead in the default configuration.

        v2.3.0: an optional :class:`PluginRegistry` carries
        InputFilter / Observer instances loaded by
        :func:`coderouter.plugins.discover_and_load`. When omitted (or
        empty), all hook loops in :meth:`generate_anthropic` /
        :meth:`stream_anthropic` short-circuit and the request flow is
        bit-identical to v2.2.0 — that's the zero-cost no-plugin path.
        """
        self.config = config
        # v2.3.0: plugin registry.  Default empty so legacy callers
        # (engine constructed without going through the loader) keep
        # working unchanged.  Stored under ``_plugin_registry`` and
        # surfaced via the ``plugins`` property — same lazy fallback
        # pattern as ``_adaptive`` / ``_budget`` / ``_memory_pressure_guard``
        # so tests that build the engine via ``FallbackEngine.__new__``
        # see an empty registry instead of AttributeError.
        self._plugin_registry: PluginRegistry = plugins or PluginRegistry.empty()
        # v2.3.0: holds strong refs to in-flight Observer fanout tasks
        # so the asyncio event loop's weak-ref bookkeeping doesn't GC
        # them mid-flight (RUF006).  Tasks remove themselves on done
        # via ``add_done_callback(_observer_tasks.discard)`` in
        # :meth:`_fanout_observers`.
        self._observer_tasks: set[asyncio.Task[None]] = set()
        # Cache adapters so we don't re-instantiate per request
        self._adapters: dict[str, BaseAdapter] = {
            p.name: build_adapter(p) for p in config.providers
        }
        # v1.9-C: per-process adaptive routing adjuster (rolling-window
        # latency + error-rate observations, debounced rank changes).
        # Stored under ``_adaptive_adjuster`` and surfaced via the
        # ``_adaptive`` property so legacy tests that bypass __init__
        # via ``__new__`` get a lazily-built default instance instead
        # of an AttributeError.
        self._adaptive_adjuster: AdaptiveAdjuster = AdaptiveAdjuster()
        # v1.10: per-process monthly USD budget tracker. Same lazy
        # property pattern as ``_adaptive`` — legacy tests that
        # construct via ``__new__`` see an auto-built empty tracker.
        # The tracker's running totals reset on UTC calendar-month
        # rollover and on process restart (in-memory only).
        self._budget_tracker: BudgetTracker = BudgetTracker()
        # v1.9-E phase 2 (L2): per-process OOM cooldown tracker.
        # Same lazy-property pattern. Legacy tests via ``__new__``
        # auto-build an empty guard that records nothing — no
        # observation overhead in deployments that leave
        # ``memory_pressure_action`` unset (default ``warn`` is
        # log-only and similarly cheap).
        self._memory_pressure_guard: MemoryPressureGuard = MemoryPressureGuard()
        # v1.9-E phase 2 (L5): per-process backend health state
        # machine. Counts consecutive failures and demotes
        # UNHEALTHY providers to the back of the chain (when the
        # active profile's ``backend_health_action`` is ``demote``).
        # Distinct from v1.9-C ``adaptive`` which handles the
        # gradient case via a rolling window.
        self._backend_health_monitor: BackendHealthMonitor = BackendHealthMonitor()
        # v2.0-J: self-healing orchestrator. Manages provider exclusion,
        # restart, and recovery probing when backend_health_action is
        # "exclude". Composes with the L5 backend health monitor.
        from coderouter.guards.self_healing import SelfHealingOrchestrator

        self._self_healing: SelfHealingOrchestrator = SelfHealingOrchestrator()
        # v2.0-G (L4): per-process drift detection window manager.
        # Stores per-provider rolling observations; the detector is
        # invoked after each provider-ok / provider-failed event and
        # returns a verdict. Action dispatch (promote/reload) reuses
        # the adaptive rank machinery.
        from coderouter.guards.drift_detection import DriftWindow

        self._drift_window: DriftWindow = DriftWindow()
        # Track which providers are currently in drift-demoted state
        # and when their cooldown expires (monotonic timestamp).
        self._drift_demoted: dict[str, float] = {}
        # Last drift verdict (set by _observe_drift_signal for ingress header).
        self._last_drift_verdict: DriftVerdict | None = None
        # v2.0-J: active recovery probe tasks (one per excluded provider).
        self._recovery_tasks: dict[str, asyncio.Task[None]] = {}
        # v2.0-J: shutdown event shared with recovery probe tasks.
        self._recovery_shutdown: asyncio.Event | None = None
        # v2.0-K: persistent state store (None = in-memory only).
        self._state_store: StateStore | None = None

    @property
    def plugins(self) -> PluginRegistry:
        """Return the plugin registry, lazily building an empty one if absent.

        Same legacy-test compatibility pattern as :py:attr:`_adaptive` /
        :py:attr:`_budget`. Tests that construct the engine via
        ``FallbackEngine.__new__`` (bypassing ``__init__``) see an
        empty registry here instead of AttributeError, so the hook
        helpers ``_apply_input_filters`` / ``_fanout_observers``
        short-circuit cleanly without any plugin work happening.
        """
        existing = getattr(self, "_plugin_registry", None)
        if existing is None:
            self._plugin_registry = PluginRegistry.empty()
            existing = self._plugin_registry
        return existing

    @property
    def last_drift_severity(self) -> str | None:
        """Return the severity string of the most recent drift verdict, or None.

        The ingress reads this after generate_anthropic / stream_anthropic to
        set the ``X-CodeRouter-Drift`` response header.  Returns ``"mild"`` or
        ``"severe"`` when drift was detected, ``None`` otherwise.
        """
        v = self._last_drift_verdict
        if v is None or not v.drifted:
            return None
        return v.severity

    @property
    def _adaptive(self) -> AdaptiveAdjuster:
        """Return the adaptive routing adjuster, lazily building one if absent.

        Some legacy tests construct the engine via ``FallbackEngine.__new__``
        and only populate ``config`` + ``_adapters``. The ``_adaptive``
        property covers that case so the engine's recording sites
        always see an adjuster object — at worst, an empty one whose
        observations don't outlive the test.
        """
        existing = getattr(self, "_adaptive_adjuster", None)
        if existing is None:
            self._adaptive_adjuster = AdaptiveAdjuster()
            existing = self._adaptive_adjuster
        return existing

    @property
    def _budget(self) -> BudgetTracker:
        """Return the monthly budget tracker, lazily building one if absent.

        Same legacy-test compatibility pattern as :py:attr:`_adaptive`:
        when the engine is constructed via ``FallbackEngine.__new__``
        (which bypasses ``__init__``), the property hands back a
        freshly built tracker instead of raising ``AttributeError``.
        """
        existing = getattr(self, "_budget_tracker", None)
        if existing is None:
            self._budget_tracker = BudgetTracker()
            existing = self._budget_tracker
        return existing

    @property
    def _memory_pressure(self) -> MemoryPressureGuard:
        """Return the L2 memory-pressure cooldown guard, lazily building one if absent.

        Same legacy-test compatibility pattern as :py:attr:`_adaptive` /
        :py:attr:`_budget`: ``__new__``-constructed engines get a
        fresh empty guard so ``is_pressured`` is always answerable.
        """
        existing = getattr(self, "_memory_pressure_guard", None)
        if existing is None:
            self._memory_pressure_guard = MemoryPressureGuard()
            existing = self._memory_pressure_guard
        return existing

    @property
    def backend_health(self) -> BackendHealthMonitor:
        """Return the L5 backend-health monitor, lazily building one if absent.

        Same legacy-test compatibility pattern as the other guard
        properties — ``__new__``-constructed engines get a fresh
        empty monitor so ``state_for`` is always answerable.

        v2.0-I: promoted from ``_backend_health`` to public ``backend_health``
        so the continuous probe background task can feed results into the
        same state machine. Internal callers continue to work (property
        access is transparent).
        """
        existing = getattr(self, "_backend_health_monitor", None)
        if existing is None:
            self._backend_health_monitor = BackendHealthMonitor()
            existing = self._backend_health_monitor
        return existing

    # Alias for backward compat with internal callers.
    @property
    def _backend_health(self) -> BackendHealthMonitor:
        return self.backend_health

    @property
    def self_healing(self) -> SelfHealingOrchestrator:
        """Return the v2.0-J self-healing orchestrator.

        Lazy init for backward compat with __new__-constructed test engines.
        """
        from coderouter.guards.self_healing import SelfHealingOrchestrator

        existing = getattr(self, "_self_healing", None)
        if existing is None:
            self._self_healing = SelfHealingOrchestrator()
            existing = self._self_healing
        return existing

    def _observe_provider_failure(
        self,
        provider: str,
        exc: AdapterError,
        *,
        profile: str | None,
    ) -> None:
        """Run L2 memory-pressure + L5 backend-health observation on one ``AdapterError``.

        Single chokepoint called from every ``except AdapterError`` site
        in the engine — six call sites total across the four entry
        points (generate / stream / generate_anthropic / stream_anthropic
        x non-stream + mid-stream variants).

        L2 dispatch:
          * action ``off``  → no detection (zero overhead).
          * action ``warn`` → emit ``memory-pressure-detected`` info
                              when the error matches an OOM phrase.
                              Provider is **not** marked pressured, so
                              the chain still tries it on the next
                              request.
          * action ``skip`` → emit ``memory-pressure-detected`` AND
                              ``mark_pressured(provider, cooldown_s)``,
                              so the chain resolver filters the
                              provider out for ``cooldown_s`` seconds.

        L5 dispatch (independent of L2):
          * action ``off``  → no monitoring.
          * action ``warn`` / ``demote`` → record the failure into the
                              :class:`BackendHealthMonitor`. State
                              transitions emit ``backend-health-changed``
                              info. Demotion (``action=demote``) is
                              applied at chain-resolve time, not here.

        The helper is no-op when the profile resolution fails (config
        edge cases, e.g. ``__new__``-constructed engines without
        profiles).
        """
        chosen = profile or self.config.default_profile
        try:
            chain = self.config.profile_by_name(chosen)
        except (KeyError, ValueError):
            return

        # L2: memory-pressure detection.
        if is_memory_pressure_error(exc):
            mp_action = chain.memory_pressure_action
            if mp_action != "off":
                cooldown_s = chain.memory_pressure_cooldown_s
                log_memory_pressure_detected(
                    logger,
                    provider=provider,
                    profile=chosen,
                    action=mp_action,
                    cooldown_s=cooldown_s,
                    error=str(exc),
                )
                if mp_action == "skip":
                    self._memory_pressure.mark_pressured(provider, cooldown_s)

        # L5: backend health.
        bh_action = chain.backend_health_action
        if bh_action != "off":
            transition = self._backend_health.record_attempt(
                provider,
                success=False,
                threshold=chain.backend_health_threshold,
            )
            if transition is not None:
                log_backend_health_changed(
                    logger,
                    provider=transition.provider,
                    profile=chosen,
                    old_state=transition.old_state,
                    new_state=transition.new_state,
                    consecutive_failures=transition.consecutive_failures,
                )
                # v2.0-J: trigger self-healing on UNHEALTHY + exclude.
                if (
                    transition.new_state == "UNHEALTHY"
                    and bh_action == "exclude"
                ):
                    newly_excluded = self.self_healing.on_unhealthy(
                        provider,
                        profile=chosen,
                        consecutive_failures=transition.consecutive_failures,
                    )
                    if newly_excluded:
                        self._spawn_recovery_probe(provider, chain=chain)

    def _observe_provider_success(
        self,
        provider: str,
        *,
        profile: str | None,
    ) -> None:
        """Record one successful attempt into the L5 backend-health monitor.

        A success snaps the provider's state to ``HEALTHY`` and resets
        the consecutive-failure counter to 0. When the transition is
        non-trivial (e.g. UNHEALTHY → HEALTHY recovery), the helper
        emits a ``backend-health-changed`` info line so the recovery
        is visible in the log trail.

        No-op when ``backend_health_action == "off"`` or profile
        resolution fails — same defensive shape as
        :meth:`_observe_provider_failure`.
        """
        chosen = profile or self.config.default_profile
        try:
            chain = self.config.profile_by_name(chosen)
        except (KeyError, ValueError):
            return
        if chain.backend_health_action == "off":
            return
        transition = self._backend_health.record_attempt(
            provider,
            success=True,
            threshold=chain.backend_health_threshold,
        )
        if transition is not None:
            log_backend_health_changed(
                logger,
                provider=transition.provider,
                profile=chosen,
                old_state=transition.old_state,
                new_state=transition.new_state,
                consecutive_failures=transition.consecutive_failures,
            )

    def _spawn_recovery_probe(
        self,
        provider: str,
        *,
        chain: FallbackChain,
    ) -> None:
        """Launch an async recovery probe task for an excluded provider.

        v2.0-J: called by ``_observe_provider_failure`` when a provider
        is newly excluded. The task runs ``recovery_probe_loop`` with
        exponential backoff until the provider recovers or shutdown.

        Safe to call from a sync context — uses ``asyncio.get_event_loop``
        to schedule the task. No-op if no running event loop (e.g. in
        pure-sync tests).
        """
        import asyncio

        from coderouter.guards.self_healing import recovery_probe_loop

        # Find the ProviderConfig for this provider name.
        provider_config = None
        for p in self.config.providers:
            if p.name == provider:
                provider_config = p
                break
        if provider_config is None:
            return

        # Reuse or create a shared shutdown event.
        if self._recovery_shutdown is None:
            self._recovery_shutdown = asyncio.Event()

        # Don't spawn duplicate tasks.
        existing = self._recovery_tasks.get(provider)
        if existing is not None and not existing.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop — skip (sync test context)

        task = loop.create_task(
            recovery_probe_loop(
                provider_config,
                orchestrator=self.self_healing,
                record_fn=self.backend_health.record_attempt,
                health_threshold=chain.backend_health_threshold,
                initial_interval_s=chain.recovery_probe_initial_s,
                max_interval_s=chain.recovery_probe_max_s,
                restart_timeout_s=chain.restart_timeout_s,
                probe_timeout_s=10.0,
                shutdown_event=self._recovery_shutdown,
                profile=chain.name,
            ),
            name=f"recovery-probe-{provider}",
        )
        self._recovery_tasks[provider] = task

    async def shutdown_recovery_probes(self) -> None:
        """Signal all recovery probe tasks to stop and await them.

        Called from the app lifespan shutdown path.
        """
        import contextlib

        if self._recovery_shutdown is not None:
            self._recovery_shutdown.set()
        for task in self._recovery_tasks.values():
            if not task.done():
                with contextlib.suppress(Exception):
                    await task
        self._recovery_tasks.clear()

    # ------------------------------------------------------------------
    # v2.0-K: State persistence
    # ------------------------------------------------------------------

    def attach_state_store(self, store: StateStore) -> None:
        """Attach a :class:`StateStore` and load persisted state.

        Called from the app lifespan startup path when ``state_dir``
        is configured.  Loads budget, health, self-healing, and
        metrics state from the store.
        """
        self._state_store = store
        self._load_all_state()

    def save_all_state(self) -> None:
        """Persist all subsystem state to the attached store.

        Called from the app lifespan shutdown path and optionally
        on a periodic timer.  No-op if no store is attached.
        """
        store = self._state_store
        if store is None:
            return
        import contextlib

        with contextlib.suppress(Exception):
            store.put("budget", "state", self._budget.save_state())
        with contextlib.suppress(Exception):
            store.put("health", "state", self.backend_health.save_state())
        with contextlib.suppress(Exception):
            store.put("self_healing", "state", self.self_healing.save_state())
        # MetricsCollector state is saved separately via the singleton.

    def _load_all_state(self) -> None:
        """Restore subsystem state from the attached store."""
        store = self._state_store
        if store is None:
            return
        import contextlib

        with contextlib.suppress(Exception):
            budget_state = store.get("budget", "state")
            if budget_state is not None:
                self._budget.load_state(budget_state)  # type: ignore[arg-type]
        with contextlib.suppress(Exception):
            health_state = store.get("health", "state")
            if health_state is not None:
                self.backend_health.load_state(health_state)  # type: ignore[arg-type]
        with contextlib.suppress(Exception):
            sh_state = store.get("self_healing", "state")
            if sh_state is not None:
                self.self_healing.load_state(sh_state)  # type: ignore[arg-type]

    def _observe_drift_signal(
        self,
        provider: str,
        *,
        profile: str | None,
        output_tokens: int = 0,
        has_tool_use: bool = False,
        request_had_tools: bool = False,
        stop_reason: str | None = None,
        is_error: bool = False,
        stream: bool = False,
        response_fingerprint: str | None = None,
    ) -> DriftVerdict | None:
        """v2.0-G (L4): record an observation and check for drift.

        Called after every provider-ok / provider-failed event on the
        Anthropic-shaped paths. Returns a :class:`DriftVerdict` when
        drift is detected (drifted=True), None otherwise.

        Side effects on detection:
        - Emits ``drift-detected`` log.
        - If action is ``promote`` or ``reload``, demotes the provider
          via the adaptive rank machinery.

        Parameters
        ----------
        response_fingerprint:
            P1-4: compact content fingerprint from
            :func:`coderouter.guards._fingerprint.fingerprint_response`.
            When set, enables the ``goal_progress_stall`` signal.
            Pass ``None`` (default) to skip that signal.
        """
        from coderouter.guards.drift_detection import (
            SENSITIVITY_PRESETS,
            THRESHOLDS_GOAL,
            ResponseObservation,
            detect_drift,
        )
        from coderouter.logging import log_drift_detected, log_drift_promoted

        chosen = profile or self.config.default_profile
        try:
            chain_cfg = self.config.profile_by_name(chosen)
        except (KeyError, ValueError):
            return None
        if chain_cfg.drift_detection_action == "off":
            return None

        # Update window size if config differs from default
        self._drift_window.max_size = chain_cfg.drift_detection_window_size

        # Record observation
        obs = ResponseObservation(
            provider=provider,
            output_tokens=output_tokens,
            has_tool_use=has_tool_use,
            request_had_tools=request_had_tools,
            stop_reason=stop_reason,
            is_error=is_error,
            stream=stream,
            response_fingerprint=response_fingerprint,
        )
        self._drift_window.record(obs)

        # Check for cooldown recovery
        import time as _time

        demote_expires = self._drift_demoted.get(provider)
        if demote_expires is not None and _time.monotonic() >= demote_expires:
            # Cooldown expired — restore rank and clear drift state
            from coderouter.logging import log_drift_recovered

            elapsed = chain_cfg.drift_detection_cooldown_s
            log_drift_recovered(logger, provider=provider, profile=chosen, after_s=elapsed)
            self._drift_demoted.pop(provider, None)
            self._drift_window.clear(provider)
            return None

        # Don't re-detect while in cooldown
        if provider in self._drift_demoted:
            return None

        # Run detection
        # P1-5: goal_mode overrides the sensitivity preset with the tighter
        # THRESHOLDS_GOAL regardless of drift_detection_sensitivity setting.
        window = self._drift_window.get_window(provider)
        if getattr(chain_cfg, "goal_mode", False):
            thresholds = THRESHOLDS_GOAL
        else:
            thresholds = SENSITIVITY_PRESETS.get(
                chain_cfg.drift_detection_sensitivity, SENSITIVITY_PRESETS["normal"]
            )
        verdict = detect_drift(window, thresholds)

        if not verdict.drifted:
            self._last_drift_verdict = None
            return None

        # Store for ingress response header.
        self._last_drift_verdict = verdict

        # Emit log
        log_drift_detected(
            logger,
            provider=provider,
            profile=chosen,
            severity=verdict.severity,
            reason=verdict.reason,
            action=chain_cfg.drift_detection_action,
            signals=verdict.signals,
        )

        # Action: promote / reload
        if chain_cfg.drift_detection_action in ("promote", "reload"):
            import time as _time_mod

            # Demote via adaptive rank
            self._adaptive.demote(provider, steps=2)
            log_drift_promoted(
                logger,
                provider=provider,
                profile=chosen,
                demoted_to_rank=2,
                cooldown_s=chain_cfg.drift_detection_cooldown_s,
            )
            # Record cooldown expiry
            self._drift_demoted[provider] = (
                _time_mod.monotonic() + chain_cfg.drift_detection_cooldown_s
            )

            # v2.0-G: reload action — attempt Ollama KV cache flush
            # (best-effort, fire-and-forget background task).
            if chain_cfg.drift_detection_action == "reload":
                import asyncio

                from coderouter.guards.drift_actions import attempt_reload

                provider_config = self._adapters[provider].config
                self._reload_task = asyncio.create_task(attempt_reload(provider_config))

        return verdict

    def _resolve_profile_overrides(self, profile_name: str | None) -> ProviderCallOverrides:
        """v0.6-B: build the ProviderCallOverrides for the active profile.

        Invariant across every adapter call on one chain (profiles are
        immutable per request), so callers resolve this once at the top of
        each engine method and pass to every adapter invocation.
        """
        chosen = profile_name or self.config.default_profile
        profile = self.config.profile_by_name(chosen)
        return ProviderCallOverrides(
            timeout_s=profile.timeout_s,
            append_system_prompt=profile.append_system_prompt,
        )

    def _resolve_chain(self, profile_name: str | None) -> list[BaseAdapter]:
        """Return the list of adapters to try, in order, for this profile.

        v0.6-C declarative ALLOW_PAID gate: when the paid gate filters
        the chain to zero adapters, emit ``chain-paid-gate-blocked`` at
        warn level via :func:`log_chain_paid_gate_blocked`. Per-provider
        ``skip-paid-provider`` info lines are still emitted (one per
        blocked provider) so per-provider traceability is intact; the
        warn sits at chain granularity for operator diagnosis.

        v1.10 monthly-budget gate: applied AFTER the paid gate. Each
        provider whose ``cost.monthly_budget_usd`` is set is checked
        against the in-memory ``BudgetTracker`` running total for the
        current UTC calendar month; over-budget providers are filtered
        out with a per-provider ``skip-budget-exceeded`` info line
        (mirror of ``skip-paid-provider``). When the budget gate
        leaves the chain empty *and at least one provider was filtered
        out by the budget gate*, ``chain-budget-exceeded`` warn fires
        — symmetric with ``chain-paid-gate-blocked``.

        v1.9-E phase 2 (L2) memory-pressure gate: applied AFTER the
        budget gate. When the active profile's
        ``memory_pressure_action`` is ``skip``, providers in the
        :class:`MemoryPressureGuard` cooldown window are filtered out
        with ``skip-memory-pressure`` info; when every provider is
        pressured, ``chain-memory-pressure-blocked`` warn fires.
        Action ``warn`` and ``off`` skip the gate entirely (warn
        logging happens at the per-attempt failure site, not at chain
        resolve).
        """
        chosen = profile_name or self.config.default_profile
        chain = self.config.profile_by_name(chosen)

        # Pass 1: paid gate. Same shape as v0.6-C; produces the
        # post-paid candidate list and tracks the names that were
        # filtered out for the aggregate warn at the bottom.
        post_paid: list[tuple[str, BaseAdapter, ProviderConfig]] = []
        blocked_by_paid: list[str] = []
        for prov_name in chain.providers:
            try:
                provider_cfg = self.config.provider_by_name(prov_name)
            except KeyError:
                logger.warning(
                    "skip-unknown-provider",
                    extra={"profile": chosen, "provider": prov_name},
                )
                continue
            if provider_cfg.paid and not self.config.allow_paid:
                logger.info(
                    "skip-paid-provider",
                    extra={"profile": chosen, "provider": prov_name},
                )
                blocked_by_paid.append(prov_name)
                continue
            post_paid.append((prov_name, self._adapters[prov_name], provider_cfg))

        # Pass 2: budget gate. Only applies to providers whose
        # ``cost.monthly_budget_usd`` is set; unset providers are
        # admitted unconditionally. This keeps the gate opt-in —
        # operators with no cost config see zero behavior change.
        post_budget: list[tuple[str, BaseAdapter]] = []
        blocked_by_budget: list[str] = []
        for prov_name, adapter, provider_cfg in post_paid:
            cost_cfg = provider_cfg.cost
            budget_usd = cost_cfg.monthly_budget_usd if cost_cfg is not None else None
            if budget_usd is not None and self._budget.is_over_budget(
                prov_name, budget_usd
            ):
                log_skip_budget_exceeded(
                    logger,
                    provider=prov_name,
                    profile=chosen,
                    monthly_budget_usd=budget_usd,
                    current_total_usd=self._budget.total_for_provider(prov_name),
                    month=self._budget.current_month(),
                )
                blocked_by_budget.append(prov_name)
                continue
            post_budget.append((prov_name, adapter))

        # Pass 3: L2 memory-pressure gate. Only applies when the
        # active profile's ``memory_pressure_action`` is ``skip``;
        # ``warn`` and ``off`` do not filter at chain-resolve time.
        # The detector runs on per-attempt failures (see
        # ``_observe_provider_failure``); this pass consumes the
        # cooldown state the detector populated.
        adapters: list[BaseAdapter] = []
        blocked_by_pressure: list[str] = []
        mp_action = chain.memory_pressure_action
        for prov_name, adapter in post_budget:
            if mp_action == "skip" and self._memory_pressure.is_pressured(prov_name):
                # Round up to >=1 so the ``seconds_until_eligible``
                # field never reports 0 (which would imply already
                # released — but the lazy-expiry sweep would have
                # cleared the entry by the time we read it).
                deadline = self._memory_pressure.pressured_until(prov_name)
                seconds = max(int(deadline - time.monotonic()), 1)
                log_skip_memory_pressure(
                    logger,
                    provider=prov_name,
                    profile=chosen,
                    seconds_until_eligible=seconds,
                )
                blocked_by_pressure.append(prov_name)
                continue
            adapters.append(adapter)

        # Aggregate warns — same precedence rule as the paid gate:
        # fire ONLY when the gate left the chain empty AND at least
        # one provider was filtered out by the gate. A mixed chain
        # with surviving providers stays quiet (normal try-provider
        # narrative covers it).
        # Order of preference for the aggregate warn: pressure → budget
        # → paid. The most recent pass to filter the entire chain
        # gets the warn (later passes see fewer survivors, so a
        # pressure-blocked empty chain should be diagnosed as such
        # even if budget / paid also filtered something earlier).
        if not adapters:
            if blocked_by_pressure:
                log_chain_memory_pressure_blocked(
                    logger,
                    profile=chosen,
                    blocked_providers=blocked_by_pressure,
                )
            elif blocked_by_budget:
                log_chain_budget_exceeded(
                    logger,
                    profile=chosen,
                    blocked_providers=blocked_by_budget,
                    month=self._budget.current_month(),
                )
            elif blocked_by_paid:
                log_chain_paid_gate_blocked(
                    logger,
                    profile=chosen,
                    blocked_providers=blocked_by_paid,
                )
            return adapters

        # Pass 4: L5 backend-health demotion. Only applies when the
        # active profile's ``backend_health_action`` is ``demote`` AND
        # the chain has at least one UNHEALTHY provider AND at least
        # one HEALTHY-or-DEGRADED provider (otherwise demoting a
        # uniformly-UNHEALTHY chain is a no-op). Stable partition:
        # healthy/degraded entries keep their relative order, then
        # all UNHEALTHY entries in their original relative order.
        # NOT a filter — UNHEALTHY providers are still attempted, just
        # last; the engine relies on the existing per-attempt failure
        # path for the actual error reporting.
        if chain.backend_health_action == "demote":
            healthy: list[BaseAdapter] = []
            unhealthy: list[BaseAdapter] = []
            for adapter in adapters:
                if self._backend_health.is_unhealthy(adapter.name):
                    unhealthy.append(adapter)
                else:
                    healthy.append(adapter)
            if unhealthy and healthy:
                # Only emit demote logs when the demotion actually
                # changes the order — uniformly UNHEALTHY chain stays
                # in original order without spamming the log.
                for adapter in unhealthy:
                    log_demote_unhealthy_provider(
                        logger,
                        provider=adapter.name,
                        profile=chosen,
                    )
                adapters = healthy + unhealthy

        # Pass 4b: v2.0-J self-healing exclusion. When the action is
        # "exclude", providers in the orchestrator's excluded set are
        # removed entirely from the chain. Unlike "demote" (which
        # moves to the back), excluded providers are not attempted at
        # all — recovery probes run in the background to detect when
        # they come back. If all providers are excluded, fall through
        # to the existing NoProvidersAvailableError path.
        if chain.backend_health_action == "exclude":
            excluded = self.self_healing.excluded_providers()
            if excluded:
                adapters = [a for a in adapters if a.name not in excluded]

        return adapters

    def _resolve_anthropic_chain(self, request: AnthropicRequest) -> list[tuple[BaseAdapter, bool]]:
        """Resolve a chain, annotating each adapter with a ``will_degrade`` flag.

        v0.5-A capability gate: when ``request`` carries ``thinking: {type:
        enabled}`` and a provider does not support it (per
        ``provider_supports_thinking``), we still include that provider in
        the chain — it becomes a degraded-fallback. The block will be
        stripped before the call and a ``capability-degraded`` log line
        will fire. Capable providers are pulled to the front (stable sort)
        so the user's ordering is preserved within each bucket.

        v1.9-C: when the profile has ``adaptive: true``, the static
        chain is run through :meth:`AdaptiveAdjuster.compute_effective_order`
        BEFORE the thinking-capability split. This way operator-declared
        ordering and adaptive demotions both feed the capability
        bucketing as a single unified order — capable providers stay in
        front, but among them the (possibly-demoted) latency / error-
        rate signal still applies.

        Returns a list of ``(adapter, will_degrade)`` pairs in the order
        they should be tried. When the request has no capability
        requirement, all entries have ``will_degrade=False`` and the order
        matches ``_resolve_chain`` (with adaptive reorder when applicable).
        """
        base = self._resolve_chain(request.profile)

        if self._profile_is_adaptive(request.profile) and base:
            base = self._adaptive.compute_effective_order(base)

        if not anthropic_request_requires_thinking(request):
            return [(a, False) for a in base]

        capable: list[tuple[BaseAdapter, bool]] = []
        degraded: list[tuple[BaseAdapter, bool]] = []
        for adapter in base:
            if provider_supports_thinking(adapter.config):
                capable.append((adapter, False))
            else:
                degraded.append((adapter, True))
        return capable + degraded

    def _profile_is_adaptive(self, profile_name: str | None) -> bool:
        """Return True iff the resolved profile opts into adaptive routing.

        Centralized so both the chain resolver and the recording-side
        path read the same flag from the same source. A missing
        profile (e.g. test harness with stripped config) returns
        False — adaptive defaults off.
        """
        chosen = profile_name or self.config.default_profile
        try:
            profile = self.config.profile_by_name(chosen)
        except (KeyError, ValueError):
            return False
        return profile.adaptive

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming OpenAI-shaped generation with sequential fallback.

        Walks the chain in order, returning the first provider's response.
        On retryable :class:`AdapterError` (transport failure, rate
        limit, upstream 5xx, etc.) the loop advances; on non-retryable
        errors it breaks immediately. When every provider has been tried
        without success, raises :class:`NoProvidersAvailableError` with
        the full per-provider error list so the ingress layer can
        surface a single 502.
        """
        adapters = self._resolve_chain(request.profile)
        overrides = self._resolve_profile_overrides(request.profile)
        errors: list[AdapterError] = []
        for adapter in adapters:
            logger.info(
                "try-provider",
                extra={"provider": adapter.name, "stream": False},
            )
            try:
                response = await adapter.generate(request, overrides=overrides)
                logger.info(
                    "provider-ok",
                    extra={"provider": adapter.name, "stream": False},
                )
                # v1.9-E phase 2 (L5): record success so an UNHEALTHY
                # provider can recover to HEALTHY on a single good
                # response. No-op when ``backend_health_action: off``.
                self._observe_provider_success(
                    adapter.name, profile=request.profile
                )
                return response
            except AdapterError as exc:
                logger.warning(
                    "provider-failed",
                    extra={
                        "provider": adapter.name,
                        "status": exc.status_code,
                        "retryable": exc.retryable,
                        "error": str(exc)[:500],
                    },
                )
                # v1.9-E phase 2 (L2 + L5): observe per-attempt failures
                # for OOM patterns + backend-health state machine.
                self._observe_provider_failure(
                    adapter.name, exc, profile=request.profile
                )
                errors.append(exc)
                if not exc.retryable:
                    break
        profile = request.profile or self.config.default_profile
        _warn_if_uniform_auth_failure(errors, profile=profile)
        raise NoProvidersAvailableError(profile=profile, errors=errors)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream from the first provider that successfully starts streaming.

        Important: once we begin yielding chunks from an adapter, we cannot
        fall back mid-stream (the client has already received partial content).
        We only fall through if the *initial* response is an error.
        """
        adapters: list[BaseAdapter] = self._resolve_chain(request.profile)
        overrides = self._resolve_profile_overrides(request.profile)
        errors: list[AdapterError] = []
        for adapter in adapters:
            logger.info(
                "try-provider",
                extra={"provider": adapter.name, "stream": True},
            )
            stream_iter = adapter.stream(request, overrides=overrides)
            try:
                first = await anext(stream_iter)
            except StopAsyncIteration:
                # Adapter produced zero chunks — treat as failure, try next
                errors.append(AdapterError("empty stream", provider=adapter.name, retryable=True))
                continue
            except AdapterError as exc:
                logger.warning(
                    "provider-failed",
                    extra={
                        "provider": adapter.name,
                        "status": exc.status_code,
                        "retryable": exc.retryable,
                        "error": str(exc)[:500],
                    },
                )
                # v1.9-E phase 2 (L2): observe pre-stream OOM.
                self._observe_provider_failure(
                    adapter.name, exc, profile=request.profile
                )
                errors.append(exc)
                if not exc.retryable:
                    break
                continue

            logger.info(
                "provider-ok",
                extra={"provider": adapter.name, "stream": True},
            )
            # v1.9-E phase 2 (L5): the first chunk landed → treat as
            # success for backend-health bookkeeping. Mid-stream
            # failures don't roll the state back; they're recorded
            # via ``_observe_provider_failure`` separately.
            self._observe_provider_success(
                adapter.name, profile=request.profile
            )
            yield first
            # Mid-stream fallback guard: once the first byte is out the door,
            # any subsequent adapter exception is terminal — we cannot fall
            # back without risking duplicate / interleaved content reaching
            # the client.
            try:
                async for chunk in stream_iter:
                    yield chunk
            except AdapterError as exc:
                logger.warning(
                    "provider-failed-midstream",
                    extra={
                        "provider": adapter.name,
                        "status": exc.status_code,
                        "retryable": exc.retryable,
                        "error": str(exc)[:500],
                    },
                )
                # v1.9-E phase 2 (L2): observe mid-stream OOM too —
                # llama.cpp / Ollama can exhaust VRAM partway through
                # generation when context grows; the next request
                # should respect the cooldown.
                self._observe_provider_failure(
                    adapter.name, exc, profile=request.profile
                )
                raise MidStreamError(adapter.name, exc) from exc
            return

        profile = request.profile or self.config.default_profile
        _warn_if_uniform_auth_failure(errors, profile=profile)
        raise NoProvidersAvailableError(profile=profile, errors=errors)

    # ==================================================================
    # Anthropic-shaped entry points (v0.3.x-1)
    # ==================================================================
    #
    # These exist so the /v1/messages ingress can route to a `kind:
    # "anthropic"` provider without a lossy round-trip through the
    # OpenAI-shaped internal format. Per-provider dispatch:
    #     - AnthropicAdapter: direct passthrough via generate_anthropic /
    #       stream_anthropic — no translation on either leg.
    #     - any other adapter: translate AnthropicRequest → ChatRequest,
    #       call the OpenAI-shaped methods, translate the result back.
    #       Tool-call repair + v0.3-D downgrade happen on this path.

    def apply_context_budget(
        self, request: AnthropicRequest
    ) -> tuple[AnthropicRequest, str | None]:
        """Run the L1 context-budget guard pre-emptively.

        Returns ``(request, header_value)`` where ``header_value`` is:
          * ``None``       — guard inactive or below all thresholds.
          * ``"warning"``  — over warn threshold (logged, not trimmed).
          * ``"trimmed"``  — messages were removed to fit the budget.

        The ingress calls this **before** ``generate_anthropic`` /
        ``stream_anthropic`` so the response header
        ``X-CodeRouter-Context-Budget`` can be set even for streaming
        responses (whose HTTP headers commit before the async generator
        starts iterating).

        The returned request (possibly trimmed) should be passed to the
        engine method; the engine will skip the guard on the second pass
        since the request is already under threshold.
        """
        chain = self._resolve_anthropic_chain(request)
        first_provider_config = chain[0][0].config if chain else None
        return _apply_context_budget_guard(
            request, config=self.config, first_provider_config=first_provider_config,
        )

    # ====================================================================
    # v2.3.0: Plugin SDK hook helpers
    # ====================================================================

    async def _apply_input_filters(
        self, request: AnthropicRequest
    ) -> AnthropicRequest:
        """Run the InputFilter chain over ``request``.

        Filters apply in registration order (= the order the user
        listed them in ``plugins.enabled``). A filter that raises is
        logged at warn level and skipped — its predecessor's output
        flows through unchanged. The chain is short-circuited
        entirely when no plugin is registered (zero overhead).

        Plugin contract: filters MUST return a new
        :class:`AnthropicRequest` (typically via ``model_copy``) and
        not mutate the input in place. The engine treats the return
        value as the new authoritative request and discards the
        previous one.
        """
        if self.plugins.is_empty():
            return request
        filters = self.plugins.input_filters
        if not filters:
            return request
        for filt in filters:
            try:
                request = await filt.transform(request)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "input-filter-failed",
                    extra={
                        "plugin": getattr(filt, "name", filt.__class__.__name__),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    },
                )
        return request

    def _fanout_observers(
        self,
        event_type: str,
        **payload: Any,
    ) -> None:
        """Fan out an Observer event as a fire-and-forget asyncio task.

        The engine never awaits these tasks: a slow observer (e.g. a
        Langfuse uploader) cannot stretch the wire-level latency the
        client measures. Errors are caught inside
        :meth:`_safe_observe` and logged, never propagated.

        ``payload`` is whatever the call site supplied; the receiving
        plugin's :meth:`Observer.on_event` is expected to tolerate
        unknown keys (forward-compat).
        """
        observers = self.plugins.observers
        if not observers:
            return
        # Lazy-init the task set for engines built via ``__new__`` —
        # mirrors the lazy ``plugins`` property pattern so legacy
        # tests that bypass __init__ don't crash here.
        if not hasattr(self, "_observer_tasks"):
            self._observer_tasks = set()
        for obs in observers:
            task = asyncio.create_task(
                self._safe_observe(obs, event_type, payload)
            )
            # Strong-ref keeps the task alive past the loop iteration;
            # ``discard`` cleans up after the task completes (success
            # or exception). Avoids the RUF006 footgun where
            # asyncio.create_task's weakref-only bookkeeping can let
            # the loop GC a fanout-in-progress task.
            self._observer_tasks.add(task)
            task.add_done_callback(self._observer_tasks.discard)

    async def _safe_observe(
        self,
        obs: Any,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Wrap a single Observer.on_event call in try/except.

        Separated from the fanout loop so the test suite can call it
        directly with a synthetic observer + payload.
        """
        try:
            await obs.on_event(event_type, payload)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "observer-failed",
                extra={
                    "plugin": getattr(obs, "name", obs.__class__.__name__),
                    "event": event_type,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            )

    async def generate_anthropic(self, request: AnthropicRequest) -> AnthropicResponse:
        """Non-streaming Anthropic request, per-provider dispatch."""
        # v1.9-E (L3): tool-loop guard runs before chain dispatch so the
        # `inject` action's mutated request flows into the chain
        # naturally. `break` raises ToolLoopBreakError, which the
        # ingress converts to a 400 response.
        request = _apply_tool_loop_guard(request, config=self.config)
        # v2.3.0: input_filter plugin chain runs *after* the built-in
        # tool-loop guard but *before* chain resolution + context
        # budget guard. That order lets a filter (e.g. memory inject)
        # grow ``request.system`` without bypassing the budget cap —
        # the budget guard sees the post-filter payload.
        request = await self._apply_input_filters(request)
        chain = self._resolve_anthropic_chain(request)

        # v2.0-F (L1): context budget guard runs after chain resolution
        # (needs the first provider's max_context_tokens). May trim the
        # request's messages if over the trim threshold.
        # NOTE: When called from the ingress via apply_context_budget()
        # first, the request is already under threshold — this is a
        # cheap no-op re-check (estimate < threshold → returns None).
        first_provider_config = chain[0][0].config if chain else None
        request, _ctx_status = _apply_context_budget_guard(
            request, config=self.config, first_provider_config=first_provider_config,
        )

        overrides = self._resolve_profile_overrides(request.profile)
        errors: list[AdapterError] = []
        tool_names = [t.name for t in request.tools] if request.tools else None
        # v1.9-A: cache observation needs to know whether the request
        # ever asked for caching. Compute once; the v0.5-B gate uses the
        # same value below for the capability-degraded log.
        request_had_cache_control = anthropic_request_has_cache_control(request)

        for adapter, will_degrade in chain:
            is_native = isinstance(adapter, AnthropicAdapter)
            effective_request = request
            if will_degrade:
                # v0.5-A: strip unsupported blocks before handing to this
                # provider and emit a structured log so operators can see
                # the downgrade after the fact. Today only `thinking` is
                # gated; the list is surfaced in the log for forward-compat.
                effective_request = strip_thinking(request)
                log_capability_degraded(
                    logger,
                    provider=adapter.name,
                    dropped=["thinking"],
                    reason="provider-does-not-support",
                )
            # v0.5-B: observability-only gate for cache_control. The
            # field is silently dropped during Anthropic → OpenAI
            # translation for openai_compat providers — no strip is
            # needed here (to_chat_request already handles it) and no
            # chain reorder is done (user ordering preserved). We just
            # emit a log line so operators can see the lossiness.
            if request_had_cache_control and not provider_supports_cache_control(
                adapter.config
            ):
                log_capability_degraded(
                    logger,
                    provider=adapter.name,
                    dropped=["cache_control"],
                    reason="translation-lossy",
                )
            logger.info(
                "try-provider",
                extra={
                    "provider": adapter.name,
                    "stream": False,
                    "native_anthropic": is_native,
                    "degraded": will_degrade,
                },
            )
            # v1.9-C: time the whole adapter call (including any
            # translation hops on the openai_compat path) so the
            # rolling-window median reflects the operator-visible
            # latency, not just the upstream HTTP RTT.
            attempt_started = time.monotonic()
            try:
                # `is_native` is the same test as this `isinstance`; we do
                # it directly here so mypy narrows `adapter` to
                # AnthropicAdapter inside the branch (BaseAdapter itself
                # does not declare the Anthropic-shaped methods).
                if isinstance(adapter, AnthropicAdapter):
                    resp = await adapter.generate_anthropic(effective_request, overrides=overrides)
                else:
                    chat_req = to_chat_request(effective_request)
                    chat_req.stream = False
                    chat_resp = await adapter.generate(chat_req, overrides=overrides)
                    resp = to_anthropic_response(chat_resp, allowed_tool_names=tool_names)
            except AdapterError as exc:
                # v1.9-C: record the failure with its observed latency.
                # Auth-flavored failures (401 / 403) carry no useful
                # latency signal (they short-circuit immediately), so
                # we drop the latency to None and let the error-rate
                # counter alone do the demotion math.
                self._adaptive.record_attempt(
                    adapter.name,
                    latency_ms=(
                        None
                        if exc.status_code in {401, 403}
                        else (time.monotonic() - attempt_started) * 1000.0
                    ),
                    success=False,
                )
                logger.warning(
                    "provider-failed",
                    extra={
                        "provider": adapter.name,
                        "status": exc.status_code,
                        "retryable": exc.retryable,
                        "error": str(exc)[:500],
                    },
                )
                # v1.9-E phase 2 (L2): observe per-attempt OOM
                # signals on the Anthropic-shaped path too.
                self._observe_provider_failure(
                    adapter.name, exc, profile=request.profile
                )
                # v2.0-G (L4): drift detection observation (failure path).
                self._observe_drift_signal(
                    adapter.name,
                    profile=request.profile,
                    is_error=True,
                    request_had_tools=bool(request.tools),
                    stream=False,
                )
                errors.append(exc)
                if not exc.retryable:
                    break
                continue
            else:
                # v1.9-C: record the successful attempt's latency.
                self._adaptive.record_attempt(
                    adapter.name,
                    latency_ms=(time.monotonic() - attempt_started) * 1000.0,
                    success=True,
                )

            logger.info(
                "provider-ok",
                extra={
                    "provider": adapter.name,
                    "stream": False,
                    "native_anthropic": is_native,
                },
            )
            # v1.9-E phase 2 (L5): record success on the Anthropic
            # non-streaming path too — recovery transitions emit
            # backend-health-changed when an UNHEALTHY provider
            # comes back.
            self._observe_provider_success(
                adapter.name, profile=request.profile
            )
            # v2.0-G (L4): drift detection observation (success path).
            # P1-4: compute response fingerprint for goal_progress_stall.
            _fp_text = " ".join(
                getattr(b, "text", "") or (b.get("text", "") if isinstance(b, dict) else "")
                for b in (resp.content or [])
                if (getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)) == "text"
            )
            from coderouter.guards._fingerprint import fingerprint_response as _fp
            self._observe_drift_signal(
                adapter.name,
                profile=request.profile,
                output_tokens=resp.usage.output_tokens if resp.usage else 0,
                has_tool_use=any(
                    getattr(b, "type", None) == "tool_use" for b in (resp.content or [])
                ),
                request_had_tools=bool(request.tools),
                stop_reason=resp.stop_reason,
                stream=False,
                response_fingerprint=_fp(_fp_text) if _fp_text else None,
            )
            # v1.9-A: pair every successful Anthropic response with a
            # cache-observed log line. Native Anthropic / LM Studio
            # /v1/messages report cache_read_input_tokens /
            # cache_creation_input_tokens via usage.model_extra;
            # openai_compat-converted responses fall through to
            # outcome=unknown.
            # v1.9-D: also enrich the log line with per-attempt
            # USD cost + cache savings via the provider's CostConfig.
            _emit_cache_observed(
                resp,
                provider=adapter.name,
                request_had_cache_control=request_had_cache_control,
                streaming=False,
                provider_config=adapter.config,
                budget=self._budget,
            )
            # v2.3.0: observer plugin fanout — fire-and-forget, never
            # blocks the engine response. Latency in ms uses the same
            # per-attempt clock the adaptive recorder used above so a
            # plugin's view of "this request took N ms" matches the
            # /metrics view.
            self._fanout_observers(
                "request_completed",
                request=request,
                response=resp,
                provider=adapter.name,
                latency_ms=(time.monotonic() - attempt_started) * 1000.0,
                stream=False,
            )
            return resp

        profile = request.profile or self.config.default_profile
        _warn_if_uniform_auth_failure(errors, profile=profile)
        raise NoProvidersAvailableError(profile=profile, errors=errors)

    async def stream_anthropic(
        self, request: AnthropicRequest
    ) -> AsyncIterator[AnthropicStreamEvent]:
        """Streaming Anthropic request, per-provider dispatch.

        For non-native providers with tools declared, we use the v0.3-D
        downgrade path (run the request non-streaming internally, repair
        tool calls, then synthesize an Anthropic SSE event sequence) —
        the same logic that used to live in the ingress. Consolidating
        it here keeps the ingress thin and lets native providers bypass
        the downgrade entirely (Anthropic emits structured tool_use
        blocks natively, no repair needed).
        """
        # v1.9-E (L3): tool-loop guard mirrors the non-streaming path.
        request = _apply_tool_loop_guard(request, config=self.config)
        # v2.3.0: input_filter chain — same ordering rules as the
        # non-streaming path. Filters can grow ``request.system``
        # before chain resolution, and the budget guard below sees
        # the post-filter payload.
        request = await self._apply_input_filters(request)
        chain = self._resolve_anthropic_chain(request)

        # v2.0-F (L1): context budget guard mirrors the non-streaming path.
        # See apply_context_budget() note — usually a no-op re-check here.
        first_provider_config = chain[0][0].config if chain else None
        request, _ctx_status = _apply_context_budget_guard(
            request, config=self.config, first_provider_config=first_provider_config,
        )

        overrides = self._resolve_profile_overrides(request.profile)
        errors: list[AdapterError] = []
        tool_names = [t.name for t in request.tools] if request.tools else None
        # v1.9-A: compute once for the v0.5-B capability-degraded gate
        # AND for the cache-observed emission below.
        request_had_cache_control = anthropic_request_has_cache_control(request)

        for adapter, will_degrade in chain:
            is_native = isinstance(adapter, AnthropicAdapter)
            downgrading = (not is_native) and bool(request.tools)
            effective_request = request
            if will_degrade:
                effective_request = strip_thinking(request)
                log_capability_degraded(
                    logger,
                    provider=adapter.name,
                    dropped=["thinking"],
                    reason="provider-does-not-support",
                )
            # v0.5-B: mirror of the non-streaming path — see comment there.
            if request_had_cache_control and not provider_supports_cache_control(
                adapter.config
            ):
                log_capability_degraded(
                    logger,
                    provider=adapter.name,
                    dropped=["cache_control"],
                    reason="translation-lossy",
                )
            logger.info(
                "try-provider",
                extra={
                    "provider": adapter.name,
                    "stream": True,
                    "native_anthropic": is_native,
                    "downgrade": downgrading,
                    "degraded": will_degrade,
                },
            )

            # v1.9-B2: aggregate usage across the whole stream. Each
            # successful event flows through ``acc.observe`` so the
            # cache-observed log at the bottom of the loop carries
            # real input/output/cache-read/cache-creation counts
            # instead of the v1.9.0a6 zero placeholders.
            acc = _StreamUsageAccumulator()

            # Stage 1: acquire an AnthropicStreamEvent iterator. Failures
            # here are candidates for fallback (no bytes have been sent to
            # the client yet).
            event_iter: AsyncIterator[AnthropicStreamEvent]
            first: AnthropicStreamEvent
            try:
                # See the non-streaming branch above: `is_native` and this
                # isinstance test are the same check; we do it inline so
                # mypy narrows for stream_anthropic (not on BaseAdapter).
                if isinstance(adapter, AnthropicAdapter):
                    event_iter = adapter.stream_anthropic(effective_request, overrides=overrides)
                    first = await anext(event_iter)
                elif downgrading:
                    # v0.3-D downgrade: run non-streaming, repair, replay.
                    chat_req = to_chat_request(effective_request)
                    chat_req.stream = False
                    chat_resp = await adapter.generate(chat_req, overrides=overrides)
                    anth_resp = to_anthropic_response(chat_resp, allowed_tool_names=tool_names)
                    event_iter = synthesize_anthropic_stream_from_response(anth_resp)
                    first = await anext(event_iter)
                else:
                    chat_req = to_chat_request(effective_request)
                    chat_req.stream = True
                    event_iter = stream_chat_to_anthropic_events(
                        adapter.stream(chat_req, overrides=overrides)
                    )
                    first = await anext(event_iter)
            except StopAsyncIteration:
                errors.append(AdapterError("empty stream", provider=adapter.name, retryable=True))
                continue
            except AdapterError as exc:
                logger.warning(
                    "provider-failed",
                    extra={
                        "provider": adapter.name,
                        "status": exc.status_code,
                        "retryable": exc.retryable,
                        "error": str(exc)[:500],
                    },
                )
                # v1.9-E phase 2 (L2): observe per-attempt OOM
                # signals on the Anthropic streaming path.
                self._observe_provider_failure(
                    adapter.name, exc, profile=request.profile
                )
                # v2.0-G (L4): drift detection observation (stream failure).
                self._observe_drift_signal(
                    adapter.name,
                    profile=request.profile,
                    is_error=True,
                    request_had_tools=bool(request.tools),
                    stream=True,
                )
                errors.append(exc)
                if not exc.retryable:
                    break
                continue

            logger.info(
                "provider-ok",
                extra={
                    "provider": adapter.name,
                    "stream": True,
                    "native_anthropic": is_native,
                    "downgrade": downgrading,
                },
            )
            # v1.9-E phase 2 (L5): first chunk landed → success for
            # the L5 health monitor on the Anthropic streaming path.
            self._observe_provider_success(
                adapter.name, profile=request.profile
            )
            acc.observe(first)
            yield first
            # Mid-stream guard identical to stream() — any error after the
            # first event is terminal.
            try:
                async for ev in event_iter:
                    acc.observe(ev)
                    yield ev
            except AdapterError as exc:
                logger.warning(
                    "provider-failed-midstream",
                    extra={
                        "provider": adapter.name,
                        "status": exc.status_code,
                        "retryable": exc.retryable,
                        "error": str(exc)[:500],
                    },
                )
                # v1.9-E phase 2 (L2): observe mid-stream OOM on
                # the Anthropic streaming path.
                self._observe_provider_failure(
                    adapter.name, exc, profile=request.profile
                )
                # v2.0-G (L4): drift detection observation (mid-stream failure).
                self._observe_drift_signal(
                    adapter.name,
                    profile=request.profile,
                    is_error=True,
                    request_had_tools=bool(request.tools),
                    stream=True,
                )
                raise MidStreamError(
                    adapter.name, exc, partial_content=acc.partial_content
                ) from exc
            # v2.0-G (L4): drift detection observation (stream success).
            # P1-4: compute response fingerprint for goal_progress_stall.
            _stream_fp_text = " ".join(
                b.get("text", "") for b in acc.partial_content if b.get("type") == "text"
            )
            from coderouter.guards._fingerprint import fingerprint_response as _fp_s
            self._observe_drift_signal(
                adapter.name,
                profile=request.profile,
                output_tokens=acc.output_tokens,
                has_tool_use=acc.has_tool_use,
                request_had_tools=bool(request.tools),
                stop_reason=acc.stop_reason,
                stream=True,
                response_fingerprint=_fp_s(_stream_fp_text) if _stream_fp_text else None,
            )
            # v1.9-B2: pair the successful stream with a cache-observed
            # log line carrying the aggregated usage counters that the
            # ``_StreamUsageAccumulator`` collected from the
            # ``message_start`` + terminal ``message_delta`` events.
            # The non-streaming sibling lives in ``generate_anthropic``;
            # both go through ``classify_cache_outcome`` /
            # ``compute_cost_for_attempt`` for symmetric outcome and
            # cost reporting.
            _emit_cache_observed_streaming(
                acc,
                provider=adapter.name,
                request_had_cache_control=request_had_cache_control,
                provider_config=adapter.config,
                budget=self._budget,
            )
            # v2.3.0: streaming observer fanout fires once, after the
            # SSE terminates successfully. We hand the accumulator's
            # final view (text + usage + tool-use flag) so plugins can
            # treat it like the non-streaming response — they don't
            # need to re-aggregate SSE chunks themselves.
            self._fanout_observers(
                "request_completed",
                request=request,
                response=acc,
                provider=adapter.name,
                latency_ms=None,  # streaming latency is end-of-stream-relative; left to plugin
                stream=True,
            )
            return

        profile = request.profile or self.config.default_profile
        _warn_if_uniform_auth_failure(errors, profile=profile)
        raise NoProvidersAvailableError(profile=profile, errors=errors)
