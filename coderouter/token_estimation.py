"""Heuristic token estimation (v2.0-F, 5-deps invariant).

Provides a shared char/4 token estimator used by:

  - :mod:`coderouter.routing.auto_router` — ``content_token_count_min``
    matcher for longContext auto-switch (v1.10).
  - :mod:`coderouter.guards.context_budget` — L1 context overflow
    detection (v2.0-F).

No external dependencies. The heuristic is ``total_chars // 4``:

  * English / code: ~4 chars per token (OpenAI's documented rule of
    thumb). Slightly under-estimates for dense code (identifiers are
    often 1 token), slightly over for prose.
  * CJK: ~1-2 chars per token in practice, so char/4 **under-counts**
    (conservative for routing — won't accidentally route a large CJK
    prompt to a model that can't hold it). For context-budget guard
    this is a known gap; operators can lower the warn/trim thresholds
    to compensate (e.g. 0.60 / 0.70 instead of 0.80 / 0.90).

Future (v2.0+): detect ``tiktoken`` at import time and offer a
precision mode. The public API is stable either way — callers get
an ``int`` estimate regardless of the backend.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_text_from_content(content: Any) -> str:
    """Extract concatenated text from a message's ``content`` field.

    Handles:
      - ``str`` (short form) → returned verbatim.
      - ``list[dict]`` (multimodal blocks) → text-type blocks
        concatenated with newlines. Image / tool_use / tool_result
        blocks contribute 0 chars (billed differently or don't
        consume text-token budget).
      - ``None`` / other → empty string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str):
                    pieces.append(text)
            elif btype is None and "text" in block and isinstance(block["text"], str):
                # Fallback for blocks without explicit type but with text
                pieces.append(block["text"])
        return "\n".join(pieces)
    return ""


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_tokens_from_body(body: dict[str, Any]) -> int:
    """Estimate the prompt's token count from a raw request body dict.

    Works with both Anthropic (``system`` + ``messages``) and OpenAI
    (``messages`` with role=system) shaped bodies. Walks all messages'
    ``content`` fields and the top-level ``system`` (if present),
    sums character counts, divides by :data:`CHARS_PER_TOKEN_HEURISTIC`.

    Image blocks contribute 0 — they're billed differently and don't
    fill the text-token side of the context window.

    This is the function formerly known as
    ``coderouter.routing.auto_router._estimate_total_tokens``.
    """
    total_chars = _count_system_chars(body.get("system"))
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                total_chars += len(_extract_text_from_content(msg.get("content")))
    return total_chars // CHARS_PER_TOKEN_HEURISTIC


def estimate_tokens_from_anthropic_request(
    *,
    system: Any,
    messages: list[Any],
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
        total_chars += len(_extract_text_from_content(content))
    return total_chars // CHARS_PER_TOKEN_HEURISTIC


def extract_text_from_anthropic_request(
    *,
    system: Any,
    messages: list[Any],
) -> str:
    """Concatenate all text in an Anthropic-shaped request.

    Mirrors :func:`estimate_tokens_from_anthropic_request` but returns
    the raw text (system prompt + every message's text blocks) instead
    of a char/4 count. Used by :mod:`coderouter.language_tax` to feed an
    accurate tokenizer for language-tax measurement. Non-text blocks
    (images / tool_use / tool_result) contribute nothing — same rule the
    char/4 estimator uses.
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
        text = _extract_text_from_content(content)
        if text:
            pieces.append(text)
    return "\n".join(pieces)


__all__ = [
    "CHARS_PER_TOKEN_HEURISTIC",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "estimate_tokens_from_anthropic_request",
    "estimate_tokens_from_body",
    "extract_text_from_anthropic_request",
]
