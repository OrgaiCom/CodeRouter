"""Context budget guard (v2.0-F, L1).

Long-running agent sessions (Claude Code, Cline, OpenClaw, etc.)
accumulate messages that eventually exceed the target model's context
window. Without intervention, the backend returns a 400 error
(Anthropic: ``max_tokens`` violation) or silently truncates the
prompt (Ollama), killing the agent session.

This module provides the engine two pieces:

  1. A **stateless estimator** :func:`estimate_context_usage` that
     computes the approximate context-window fill ratio for a given
     Anthropic request against a declared ``max_context_tokens``.
     Pure function, no I/O.
  2. A **stateless trimmer** :func:`trim_to_budget` that returns a
     new request with old messages removed until the estimated usage
     drops below a target ratio. Pure function, no mutation of the
     input.

Integration with the fallback engine
=====================================

The engine calls these at the ``_apply_context_budget_guard`` site —
**after** tool-loop detection but **before** chain dispatch. The
guard reads the resolved profile's ``context_budget_action`` field:

  * ``off``  — guard is a no-op (default).
  * ``warn`` — compute estimate; if over warn threshold, emit a
               structured log + attach a response header.
  * ``trim`` — ``warn`` behavior + if over trim threshold, call
               :func:`trim_to_budget` and return the shortened
               request to the engine.

Token estimation
================

Uses the shared :func:`~coderouter.token_estimation.estimate_tokens_from_anthropic_request`
(char/4 heuristic, 5-deps invariant). See that module's docstring
for the CJK caveat and recommended threshold compensation.

Trim algorithm
==============

  1. Always preserve the system prompt (not counted toward removal).
  2. Always preserve the last ``preserve_last_n`` messages.
  3. Remove messages from the front (oldest first).
  4. Preserve tool_use / tool_result pairs atomically — if a kept
     message contains a ``tool_result``, also keep the preceding
     ``tool_use`` assistant message (and vice versa).
  5. After removal, re-estimate; if still over ``trim_target``,
     reduce ``preserve_last_n`` by 1 and retry (minimum floor: 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from coderouter.token_estimation import (
    estimate_tokens_from_anthropic_request,
)

if TYPE_CHECKING:
    from coderouter.translation.anthropic import AnthropicRequest


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContextBudgetEstimate:
    """Result of a context-budget estimation check."""

    #: Estimated token count for the full request (system + messages).
    estimated_tokens: int
    #: Declared maximum context window for the target provider.
    max_context_tokens: int
    #: Ratio: estimated_tokens / max_context_tokens (0.0 to ∞).
    usage_ratio: float
    #: True when usage_ratio >= the profile's warn threshold.
    over_warn_threshold: bool
    #: True when usage_ratio >= the profile's trim threshold.
    over_trim_threshold: bool


@dataclass(frozen=True, slots=True)
class TrimResult:
    """Metadata about a trim operation (for logging)."""

    #: Number of messages before trim.
    messages_before: int
    #: Number of messages after trim.
    messages_after: int
    #: Number of messages removed.
    messages_removed: int
    #: Estimated tokens before trim.
    estimated_tokens_before: int
    #: Estimated tokens after trim.
    estimated_tokens_after: int


# ---------------------------------------------------------------------------
# Public API: estimation
# ---------------------------------------------------------------------------


def estimate_context_usage(
    request: AnthropicRequest,
    *,
    max_context_tokens: int,
    warn_threshold: float = 0.80,
    trim_threshold: float = 0.90,
) -> ContextBudgetEstimate:
    """Estimate how full the target provider's context window is.

    Pure function. Does not mutate the request. Returns a
    :class:`ContextBudgetEstimate` with precomputed threshold booleans
    so callers can branch without re-computing ratios.

    Parameters
    ----------
    request
        The inbound Anthropic request to evaluate.
    max_context_tokens
        Declared context window of the target provider (from
        ProviderConfig.max_context_tokens, registry, or fallback 128K).
    warn_threshold
        Ratio at or above which ``over_warn_threshold`` is True.
    trim_threshold
        Ratio at or above which ``over_trim_threshold`` is True.
    """
    estimated = estimate_tokens_from_anthropic_request(
        system=request.system,
        messages=request.messages,
    )
    ratio = estimated / max_context_tokens if max_context_tokens > 0 else 0.0
    return ContextBudgetEstimate(
        estimated_tokens=estimated,
        max_context_tokens=max_context_tokens,
        usage_ratio=ratio,
        over_warn_threshold=ratio >= warn_threshold,
        over_trim_threshold=ratio >= trim_threshold,
    )


# ---------------------------------------------------------------------------
# Public API: trimming
# ---------------------------------------------------------------------------


def trim_to_budget(
    request: AnthropicRequest,
    *,
    max_context_tokens: int,
    trim_target: float = 0.75,
    preserve_last_n: int = 4,
) -> tuple[AnthropicRequest, TrimResult]:
    """Return a new request with old messages removed to fit the budget.

    Pure function — does NOT mutate the input request.

    Algorithm:
      1. Compute target token count = max_context_tokens * trim_target.
      2. Identify messages that MUST be preserved:
         - Last ``preserve_last_n`` messages.
         - Any tool_use / tool_result pairs linked to preserved messages.
      3. Remove messages from the front until estimated tokens ≤ target.
      4. If still over target after removing all removable messages,
         reduce preserve_last_n by 1 and retry (floor: 2 messages).

    Returns
    -------
    tuple[AnthropicRequest, TrimResult]
        The trimmed request (new instance) and metadata about the trim.
    """
    messages = list(request.messages)
    estimated_before = estimate_tokens_from_anthropic_request(
        system=request.system,
        messages=messages,
    )
    target_tokens = int(max_context_tokens * trim_target)
    effective_preserve = min(preserve_last_n, len(messages))

    # Iteratively trim until under target or preserve floor reached
    trimmed_messages = _do_trim(
        messages=messages,
        system=request.system,
        target_tokens=target_tokens,
        preserve_last_n=effective_preserve,
    )

    estimated_after = estimate_tokens_from_anthropic_request(
        system=request.system,
        messages=trimmed_messages,
    )

    result = TrimResult(
        messages_before=len(messages),
        messages_after=len(trimmed_messages),
        messages_removed=len(messages) - len(trimmed_messages),
        estimated_tokens_before=estimated_before,
        estimated_tokens_after=estimated_after,
    )

    # Build new request with trimmed messages.
    # Import here to avoid circular import at module level.
    from coderouter.translation.anthropic import AnthropicMessage

    new_request = request.model_copy(
        update={"messages": [AnthropicMessage(**_msg_to_dict(m)) for m in trimmed_messages]},
    )
    return new_request, result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _msg_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an AnthropicMessage (or dict) to a plain dict for reconstruction."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if isinstance(msg, dict):
        return msg
    return {"role": "user", "content": ""}


def _has_tool_use(msg: Any) -> bool:
    """True if the message contains a tool_use content block."""
    content = msg.content if hasattr(msg, "content") else msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return True
    return False


def _has_tool_result(msg: Any) -> bool:
    """True if the message contains a tool_result content block."""
    content = msg.content if hasattr(msg, "content") else msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
    return False


def _compute_preserve_set(
    messages: list[Any],
    preserve_last_n: int,
) -> set[int]:
    """Compute indices of messages that must be preserved.

    Preserves:
      - The last ``preserve_last_n`` messages.
      - tool_use/tool_result pair integrity: if a preserved message
        has tool_result, also preserve the preceding assistant (tool_use).
        If a preserved message has tool_use, also preserve the following
        user (tool_result).
    """
    n = len(messages)
    if preserve_last_n >= n:
        return set(range(n))

    preserved: set[int] = set(range(n - preserve_last_n, n))

    # Expand to cover tool pairs — iterate until stable
    changed = True
    while changed:
        changed = False
        for idx in list(preserved):
            msg = messages[idx]
            # If this message has tool_result, preserve the preceding
            # assistant message (which should have the matching tool_use)
            if _has_tool_result(msg) and idx > 0 and idx - 1 not in preserved:
                preserved.add(idx - 1)
                changed = True
            # If this message has tool_use, preserve the following
            # user message (which should have the matching tool_result)
            if _has_tool_use(msg) and idx < n - 1 and idx + 1 not in preserved:
                preserved.add(idx + 1)
                changed = True

    return preserved


def _do_trim(
    messages: list[Any],
    system: Any,
    target_tokens: int,
    preserve_last_n: int,
) -> list[Any]:
    """Core trim loop. Reduces preserve_last_n if needed (floor: 2)."""
    current_preserve = preserve_last_n

    while current_preserve >= 2:
        preserved_indices = _compute_preserve_set(messages, current_preserve)
        # Keep only preserved messages (maintain order)
        trimmed = [messages[i] for i in sorted(preserved_indices)]

        estimated = estimate_tokens_from_anthropic_request(
            system=system,
            messages=trimmed,
        )
        if estimated <= target_tokens:
            return trimmed

        # Still over target — reduce preserve count and retry
        current_preserve -= 1

    # Floor reached — return with minimum preservation (last 2)
    preserved_indices = _compute_preserve_set(messages, 2)
    return [messages[i] for i in sorted(preserved_indices)]


__all__ = [
    "ContextBudgetEstimate",
    "TrimResult",
    "estimate_context_usage",
    "trim_to_budget",
]
