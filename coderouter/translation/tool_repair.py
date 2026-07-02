"""Tool-call repair: extract tool invocations that a model wrote as plain text.

Background
----------
Small coding models (e.g. qwen2.5-coder) sometimes respond to a tool-bearing
prompt by *writing* a JSON object describing the tool call in the assistant
message body instead of populating the structured `tool_calls` field. The
downstream Anthropic/OpenAI clients then see regular text and never execute
the tool.

This module scans assistant text for such embedded tool-call JSON and pulls
it back into the OpenAI-shape `tool_calls` list, so the rest of the
translation pipeline (`to_anthropic_response`, stream event emitter) can
produce real `tool_use` content blocks.

Recognised shapes
-----------------
1. Fenced code blocks:
    ```json
    {"name": "Bash", "arguments": {"command": "pwd"}}
    ```
   (the language tag is optional: ``` ...``` also works.)
2. Bare JSON objects embedded in text:
    "Let me check the current directory. {\"name\":\"Bash\",\"arguments\":{}}"
3. Multiple JSON objects in sequence (for multi-call turns).

Each candidate is accepted only if it parses to one of:
    {"name": <str>, "arguments": <dict | str>}          # direct shape
    {"function": {"name": <str>, "arguments": ...}}     # OpenAI shape

If `allowed_tool_names` is provided, the `name` must be in that set;
otherwise any tool-shaped JSON is accepted. Passing the allow-list is
strongly recommended to avoid false positives (a model legitimately
discussing JSON in prose).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

__all__ = ["deduplicate_tool_calls", "repair_tool_calls_in_text"]


# ------------------------------------------------------------------
# Tool-call shape detection + normalisation
# ------------------------------------------------------------------


def _looks_like_tool_call(obj: Any, allowed: set[str] | None) -> tuple[str, Any] | None:
    """Return (name, arguments) if obj looks like a tool call, else None."""
    if not isinstance(obj, dict):
        return None

    # Direct shape: {"name": "...", "arguments": ...}
    name = obj.get("name")
    if isinstance(name, str) and "arguments" in obj and (allowed is None or name in allowed):
        return name, obj["arguments"]

    # OpenAI function shape: {"function": {"name": "...", "arguments": ...}}
    fn = obj.get("function")
    if isinstance(fn, dict):
        inner_name = fn.get("name")
        if (
            isinstance(inner_name, str)
            and "arguments" in fn
            and (allowed is None or inner_name in allowed)
        ):
            return inner_name, fn["arguments"]

    return None


def _normalise_to_openai_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    """Build an OpenAI-shape tool_calls entry."""
    if isinstance(arguments, str):
        args_str = arguments
    elif isinstance(arguments, dict):
        args_str = json.dumps(arguments, ensure_ascii=False)
    else:
        # list / None / anything else — fall back to serialising what we got.
        args_str = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


# ------------------------------------------------------------------
# Scanners: fenced code blocks, then balanced braces in remaining text
# ------------------------------------------------------------------

# Match ```json ... ``` or ``` ... ``` with anything after the fence tag line.
# Group 1 captures the body.
_FENCED_RE = re.compile(
    r"```(?:\w+)?[ \t]*\r?\n(.*?)\r?\n?```",
    re.DOTALL,
)


def _extract_tool_call_fenced_blocks(
    text: str,
    allowed: set[str] | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Pull tool-call-shaped ```...``` blocks out of text.

    Only fenced blocks whose body parses to a recognised tool-call shape are
    removed from the text and returned as normalised OpenAI tool_calls. Any
    other fenced block (prose, real source code, non-tool JSON) is preserved
    verbatim in the returned text — the removal decision and the extraction
    decision use the exact same predicate, so a fenced block is never dropped
    without also being surfaced as a tool call (bug H2: data loss when a
    response mixed a code example with a tool-call block).

    Returns (text_without_tool_fences, tool_calls).
    """
    tool_calls: list[dict[str, Any]] = []

    def _repair(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if not body.startswith("{"):
            return match.group(0)  # keep non-JSON fenced blocks (e.g. code)
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return match.group(0)  # keep unparseable fenced blocks
        hit = _looks_like_tool_call(obj, allowed)
        if hit is None:
            return match.group(0)  # keep non-tool-call JSON blocks
        name, args = hit
        tool_calls.append(_normalise_to_openai_tool_call(name, args))
        return ""  # remove only recognised tool-call blocks

    cleaned = _FENCED_RE.sub(_repair, text)
    return cleaned, tool_calls


def _find_balanced_json_objects(text: str) -> list[tuple[int, int, str]]:
    """Find top-level `{...}` JSON substrings by a brace-counter scan.

    Returns a list of (start, end_exclusive, substring). Handles escape
    sequences and string literals so braces inside JSON strings do not
    confuse the counter. Malformed (unclosed) candidates are skipped.
    """
    out: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        # Scan forward to find a balanced close.
        depth = 0
        j = i
        in_str = False
        escape = False
        while j < n:
            c = text[j]
            if escape:
                escape = False
            elif in_str:
                if c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        out.append((i, j + 1, text[i : j + 1]))
                        i = j + 1
                        break
            j += 1
        else:
            # Ran off the end without closing — skip this `{` and move on.
            i += 1
            continue
    return out


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def repair_tool_calls_in_text(
    text: str,
    allowed_tool_names: list[str] | set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract embedded tool-call JSON from assistant text.

    Returns:
        (cleaned_text, tool_calls)
          cleaned_text  : the input with recognised tool-call JSON removed,
                          stripped of surrounding whitespace.
          tool_calls    : OpenAI-shape tool_calls entries, in the order they
                          appeared in the original text. Each entry has a
                          freshly minted `id` (the source JSON did not carry one).

    If nothing repairable is found, returns (text, []).
    """
    if not isinstance(text, str) or not text:
        return text, []

    allowed: set[str] | None = None if allowed_tool_names is None else set(allowed_tool_names)

    extracted: list[dict[str, Any]] = []

    # 1. Pull tool-call-shaped fenced code blocks out first — they're the most
    #    common shape when a chat-tuned model explains what it's doing. Fenced
    #    blocks that are NOT tool calls (prose, real source code, plain JSON
    #    examples) are left in place so we never drop legitimate content (H2).
    cleaned, fenced_tool_calls = _extract_tool_call_fenced_blocks(text, allowed)
    extracted.extend(fenced_tool_calls)

    # 2. Scan remaining text for bare JSON objects.
    #    We walk from back to front so removals by slicing don't shift
    #    the indices of earlier matches.
    candidates = _find_balanced_json_objects(cleaned)
    # Tentatively evaluate each; keep only the ones that are tool-call-shaped.
    spans_to_remove: list[tuple[int, int]] = []
    repaired_from_bare: list[dict[str, Any]] = []
    for start, end, substr in candidates:
        try:
            obj = json.loads(substr)
        except json.JSONDecodeError:
            continue
        hit = _looks_like_tool_call(obj, allowed)
        if hit is None:
            continue
        name, args = hit
        repaired_from_bare.append(_normalise_to_openai_tool_call(name, args))
        spans_to_remove.append((start, end))

    # Remove the matched spans from the text back-to-front.
    for start, end in reversed(spans_to_remove):
        cleaned = cleaned[:start] + cleaned[end:]

    extracted.extend(repaired_from_bare)

    # Collapse the whitespace left behind by removals.
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # v2.2: deduplicate tool calls within the same response.
    # Small models sometimes output the same tool-call JSON 2-3 times
    # in a single turn. We keep the first occurrence only.
    extracted = deduplicate_tool_calls(extracted)

    return cleaned, extracted


# ------------------------------------------------------------------
# Deduplication (v2.2)
# ------------------------------------------------------------------


def deduplicate_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate tool calls sharing the same (name, arguments).

    Preserves order — the first occurrence wins. Each entry is expected
    to be in OpenAI tool_calls shape (``{"function": {"name": ...,
    "arguments": ...}, ...}``). Entries that lack the expected
    structure are kept unconditionally (conservative fallback).

    This is separate from L3 tool-loop detection (which operates across
    turns in the conversation history). Deduplication operates within a
    single assistant response where the model outputted the same JSON
    tool-call block multiple times.
    """
    if len(tool_calls) <= 1:
        return tool_calls

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function")
        if not isinstance(func, dict):
            # Not in expected shape — keep unconditionally.
            deduped.append(tc)
            continue
        key = (func.get("name", ""), func.get("arguments", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(tc)
    return deduped
