"""Heuristic token estimation (v2.0-F, 5-deps invariant).

Provides a shared char/4 token estimator used by:

  - :mod:`coderouter.routing.auto_router` — ``content_token_count_min``
    matcher for longContext auto-switch (v1.10).
  - :mod:`coderouter.guards.context_budget` — L1 context overflow
    detection (v2.0-F).
  - ``POST /v1/messages/count_tokens`` (:mod:`coderouter.ingress.anthropic_routes`).
  - :mod:`coderouter.language_tax` — CJK tax measurement.

No external dependencies. The heuristic is ``total_chars // 4``:

  * English / code: ~4 chars per token (OpenAI's documented rule of
    thumb). Slightly under-estimates for dense code (identifiers are
    often 1 token), slightly over for prose.
  * CJK: ~1-2 chars per token in practice, so char/4 **under-counts**
    (conservative for routing — won't accidentally route a large CJK
    prompt to a model that can't hold it). For context-budget guard
    this is a known gap; operators can lower the warn/trim thresholds
    to compensate (e.g. 0.60 / 0.70 instead of 0.80 / 0.90).

Which content blocks are counted (H-5, v2.12)
=============================================

Up to v2.11.x only ``text`` blocks were counted. Agent clients
(Claude Code, Cline, OpenClaw, …) put almost all of their context in
``tool_result`` and ``tool_use`` blocks, so the estimate was off by
5x at 20 turns and ~29x at 200 turns for a tool-heavy session. The
extractor now dispatches on the block ``type``:

===================== ===========================================
block type            contribution
===================== ===========================================
``text``              ``text`` verbatim
``tool_result``       ``content`` — ``str`` verbatim, or the same
                      per-type rules applied recursively to the
                      block list (so a nested ``image`` is still 0)
``tool_use``          ``name`` + ``json.dumps(input)``
``thinking``          the ``thinking`` string (it is replayed back
                      to the model on the next turn, so it really
                      does occupy the context window)
``image``             **0** — deliberately
``redacted_thinking`` 0 (opaque ciphertext, not model-visible text)
anything else         0 (forward-compatible default)
===================== ===========================================

The ``image`` exclusion is load-bearing, not an oversight: a naive
``json.dumps(block)`` of a request carrying a 400 KB base64 PNG
over-estimates by ~35x (11,789 → 410,781 tokens measured), which
would make the context-budget guard shred the history of any session
that ever pasted a screenshot. Images are billed on their own axis
and must stay at 0.

Known cross-path inconsistency (out of scope for H-5)
-----------------------------------------------------
``POST /v1/messages/count_tokens`` additionally appends the JSON of
the request's ``tools`` array before counting (see
``ingress/anthropic_routes.py``), while
:func:`estimate_tokens_from_anthropic_request` /
:func:`estimate_tokens_from_body` do **not** count tool *definitions*
at all. A request with large tool schemas therefore gets a bigger
number from ``count_tokens`` than the context-budget guard sees. This
is pre-existing behavior and is intentionally left unchanged here —
tracked as a follow-up, not part of the H-5 fix.

Escape hatch
------------
:func:`set_include_tool_content` (wired to the top-level config key
``token_estimation_include_tool_content``) restores the exact v2.11.x
numbers by skipping ``tool_result`` / ``tool_use`` / ``thinking``
again. It exists only to unblock operators surprised by the new
(correct) guard/routing behavior and is scheduled for removal.

Future (v2.0+): detect ``tiktoken`` at import time and offer a
precision mode. The public API is stable either way — callers get
an ``int`` estimate regardless of the backend.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Characters per token assumption. 4 is OpenAI's published rule of
#: thumb for English text. See module docstring for CJK caveat.
CHARS_PER_TOKEN_HEURISTIC: int = 4

#: Conservative fallback when no max_context_tokens is declared for a
#: provider. 128K covers most modern models (Qwen3 32K is lower, but
#: operators who use it should declare explicitly or via registry).
DEFAULT_MAX_CONTEXT_TOKENS: int = 128_000

#: Depth cap for the recursive block walk. ``tool_result.content`` is
#: the only legal nesting level in the Anthropic wire format; the cap
#: just makes a hand-crafted / buggy payload unable to blow the stack.
_MAX_BLOCK_DEPTH: int = 4

#: Block types that never contribute characters, whatever the flags.
#: ``image`` is base64 (or a URL) billed on a separate axis;
#: ``redacted_thinking`` is opaque ciphertext the model never reads as
#: text. Counting either one over-estimates catastrophically.
_ZERO_CHAR_BLOCK_TYPES: frozenset[str] = frozenset(
    {"image", "image_url", "redacted_thinking"}
)

#: Block types gated behind the ``include_tool_content`` switch.
_TOOL_BLOCK_TYPES: frozenset[str] = frozenset({"tool_result", "tool_use", "thinking"})


# ---------------------------------------------------------------------------
# Process-wide opt-out (v2.11.x compatibility escape hatch)
# ---------------------------------------------------------------------------

#: Module-level default for :func:`set_include_tool_content`. Set from
#: ``CodeRouterConfig.token_estimation_include_tool_content`` at config
#: load time. A module global (rather than a parameter threaded through
#: every call site) is required because three of the four consumers —
#: the auto-router, ``count_tokens`` and language-tax — run *outside*
#: any profile context and must all agree on one number per process.
_INCLUDE_TOOL_CONTENT: bool = True


def set_include_tool_content(enabled: bool) -> None:
    """Set the process-wide default for tool-content counting.

    Called from ``CodeRouterConfig``'s validator so a single config key
    controls every consumer of this module consistently. ``False``
    reproduces v2.11.x estimates exactly.
    """
    global _INCLUDE_TOOL_CONTENT
    _INCLUDE_TOOL_CONTENT = bool(enabled)


def get_include_tool_content() -> bool:
    """Return the current process-wide tool-content counting default."""
    return _INCLUDE_TOOL_CONTENT


def _resolve_include(include_tool_content: bool | None) -> bool:
    return _INCLUDE_TOOL_CONTENT if include_tool_content is None else include_tool_content


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _tool_use_text(block: dict[str, Any]) -> str:
    """Serialize a ``tool_use`` block's name + input for char counting.

    ``input`` is arbitrary JSON, so its wire size is the honest proxy
    for how much context it occupies. ``ensure_ascii=False`` keeps CJK
    arguments at their real character count instead of inflating them
    into ``\\uXXXX`` escapes.
    """
    pieces: list[str] = []
    name = block.get("name")
    if isinstance(name, str):
        pieces.append(name)
    raw_input = block.get("input")
    if raw_input is not None:
        try:
            pieces.append(json.dumps(raw_input, ensure_ascii=False))
        except (TypeError, ValueError):
            pieces.append(str(raw_input))
    return "\n".join(pieces)


def _extract_text_from_content(
    content: Any,
    *,
    include_tool_content: bool | None = None,
    _depth: int = 0,
) -> str:
    """Extract concatenated text from a message's ``content`` field.

    Handles:
      - ``str`` (short form) → returned verbatim.
      - ``list[dict]`` (multimodal blocks) → dispatched per block
        ``type``; see the module docstring's table for the exact rules.
        ``image`` blocks always contribute 0 chars; ``tool_result`` /
        ``tool_use`` / ``thinking`` contribute their text unless
        ``include_tool_content`` is False.
      - ``None`` / other → empty string.

    ``include_tool_content=None`` (the default) resolves to the
    process-wide setting from :func:`set_include_tool_content`.
    """
    include = _resolve_include(include_tool_content)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    pieces: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        if btype == "text":
            text = block.get("text")
            if isinstance(text, str):
                pieces.append(text)
            continue
        if btype is None:
            # Fallback for blocks without explicit type but with text
            text = block.get("text")
            if isinstance(text, str):
                pieces.append(text)
            continue
        if btype in _ZERO_CHAR_BLOCK_TYPES:
            # Deliberate 0: see module docstring (35x over-estimate).
            continue
        if btype not in _TOOL_BLOCK_TYPES or not include:
            # Unknown/forward-compat blocks contribute nothing, and the
            # escape hatch drops the tool blocks back to v2.11.x's 0.
            continue

        if btype == "tool_result":
            if _depth >= _MAX_BLOCK_DEPTH:
                continue
            inner = _extract_text_from_content(
                block.get("content"),
                include_tool_content=True,
                _depth=_depth + 1,
            )
            if inner:
                pieces.append(inner)
        elif btype == "tool_use":
            rendered = _tool_use_text(block)
            if rendered:
                pieces.append(rendered)
        elif btype == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str):
                pieces.append(thinking)

    return "\n".join(pieces)


def _count_system_chars(system: Any) -> int:
    """Count characters in the system prompt (str or list-of-blocks form)."""
    if isinstance(system, str):
        return len(system)
    if isinstance(system, list):
        total = 0
        for block in system:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    total += len(text)
        return total
    return 0


def _extract_system_text(system: Any) -> str:
    """Concatenate the system prompt text (str or list-of-blocks form)."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        pieces: list[str] = []
        for block in system:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(pieces)
    return ""


def _message_content(msg: Any) -> Any:
    """Pull ``content`` off a Pydantic message model or a plain dict."""
    if hasattr(msg, "content"):
        return msg.content
    if isinstance(msg, dict):
        return msg.get("content")
    return None


# ---------------------------------------------------------------------------
# Public API: character-level primitives
# ---------------------------------------------------------------------------
#
# The guard's trim loop needs per-message character counts so it can
# subtract as it drops instead of re-walking the whole conversation
# after every removal (O(n^2) → O(n)). Counting *chars* rather than
# per-message tokens keeps the incremental sum bit-identical to a full
# recompute: the heuristic floor-divides the grand total exactly once.


def count_system_chars(system: Any) -> int:
    """Public alias of the system-prompt character count."""
    return _count_system_chars(system)


def count_message_chars(msg: Any, *, include_tool_content: bool | None = None) -> int:
    """Characters a single message contributes to the estimate."""
    return len(
        _extract_text_from_content(
            _message_content(msg),
            include_tool_content=include_tool_content,
        )
    )


def chars_to_tokens(total_chars: int) -> int:
    """Apply the char/4 heuristic to an already-summed character count."""
    return total_chars // CHARS_PER_TOKEN_HEURISTIC


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_tokens_from_body(
    body: dict[str, Any],
    *,
    include_tool_content: bool | None = None,
) -> int:
    """Estimate the prompt's token count from a raw request body dict.

    Works with both Anthropic (``system`` + ``messages``) and OpenAI
    (``messages`` with role=system) shaped bodies. Walks all messages'
    ``content`` fields and the top-level ``system`` (if present),
    sums character counts, divides by :data:`CHARS_PER_TOKEN_HEURISTIC`.

    Image blocks contribute 0 — they're billed differently and don't
    fill the text-token side of the context window. ``tool_result`` /
    ``tool_use`` / ``thinking`` blocks DO count (H-5); see the module
    docstring. Tool *definitions* (``body["tools"]``) are still not
    counted — known inconsistency with ``count_tokens``.

    This is the function formerly known as
    ``coderouter.routing.auto_router._estimate_total_tokens``.
    """
    total_chars = _count_system_chars(body.get("system"))
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                total_chars += len(
                    _extract_text_from_content(
                        msg.get("content"),
                        include_tool_content=include_tool_content,
                    )
                )
    return chars_to_tokens(total_chars)


def estimate_tokens_from_anthropic_request(
    *,
    system: Any,
    messages: list[Any],
    include_tool_content: bool | None = None,
) -> int:
    """Estimate token count for an Anthropic-shaped request.

    Unlike :func:`estimate_tokens_from_body`, this accepts typed
    components directly so callers with a parsed
    :class:`~coderouter.translation.anthropic.AnthropicRequest` don't
    need to round-trip through a raw dict.

    Parameters
    ----------
    system : str | list[dict] | None
        The request's ``system`` field.
    messages : list[AnthropicMessage]
        The request's ``messages`` list. Each item should have a
        ``content`` attribute (str or list of blocks).
    include_tool_content : bool | None
        ``None`` → process-wide default (see
        :func:`set_include_tool_content`). ``False`` reproduces the
        v2.11.x estimate exactly.
    """
    total_chars = _count_system_chars(system)
    for msg in messages:
        # Support both Pydantic model (with .content attribute) and
        # plain dict (test harness convenience).
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content")
        else:
            continue
        total_chars += len(
            _extract_text_from_content(content, include_tool_content=include_tool_content)
        )
    return chars_to_tokens(total_chars)


def extract_text_from_anthropic_request(
    *,
    system: Any,
    messages: list[Any],
    include_tool_content: bool | None = None,
) -> str:
    """Concatenate all text in an Anthropic-shaped request.

    Mirrors :func:`estimate_tokens_from_anthropic_request` but returns
    the raw text (system prompt + every message's countable blocks)
    instead of a char/4 count. Used by :mod:`coderouter.language_tax`
    to feed an accurate tokenizer for language-tax measurement, and by
    ``POST /v1/messages/count_tokens``. Image blocks contribute nothing
    — same rule the char/4 estimator uses.
    """
    pieces: list[str] = []
    sys_text = _extract_system_text(system)
    if sys_text:
        pieces.append(sys_text)
    for msg in messages:
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content")
        else:
            continue
        text = _extract_text_from_content(content, include_tool_content=include_tool_content)
        if text:
            pieces.append(text)
    return "\n".join(pieces)


__all__ = [
    "CHARS_PER_TOKEN_HEURISTIC",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "chars_to_tokens",
    "count_message_chars",
    "count_system_chars",
    "estimate_tokens_from_anthropic_request",
    "estimate_tokens_from_body",
    "extract_text_from_anthropic_request",
    "get_include_tool_content",
    "set_include_tool_content",
]
