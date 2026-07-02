"""Regression tests for bugs H2 and H6 in the translation layer.

H2 (tool_repair): a response mixing a real code block with a tool-call
    block used to drop *every* fenced block from the cleaned text, so the
    legitimate code example was lost (data loss). Only tool-call-shaped
    fenced blocks may be removed; other fenced blocks must survive.

H6 (convert): when the upstream OpenAI chunk stream yielded zero chunks,
    the translator emitted message_delta / message_stop with no preceding
    message_start — a wire-protocol violation that breaks Claude Code's
    SSE parser. The event stream must always open with message_start.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from coderouter.adapters.base import StreamChunk
from coderouter.translation.convert import stream_chat_to_anthropic_events
from coderouter.translation.tool_repair import repair_tool_calls_in_text

# ----------------------------------------------------------------------
# H2 — fenced code block preservation
# ----------------------------------------------------------------------


def test_h2_code_block_survives_alongside_tool_call() -> None:
    """A normal code block stays in the text; the tool-call block is pulled out."""
    text = (
        "Here is how you print in Python:\n\n"
        "```python\n"
        "print('hello world')\n"
        "```\n\n"
        "Now let me run it:\n\n"
        "```json\n"
        '{"name": "Bash", "arguments": {"command": "python hi.py"}}\n'
        "```\n"
    )
    cleaned, calls = repair_tool_calls_in_text(text, allowed_tool_names=["Bash"])

    # The tool call was extracted.
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "Bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "python hi.py"}

    # The legitimate code block is still present (not dropped as data loss).
    assert "print('hello world')" in cleaned
    assert "```python" in cleaned
    assert "Here is how you print in Python" in cleaned

    # The tool-call block itself was removed from the body.
    assert '"name": "Bash"' not in cleaned
    assert "python hi.py" not in cleaned


def test_h2_non_tool_json_fenced_block_is_preserved() -> None:
    """A fenced JSON example that is not a tool call must not be removed."""
    text = "Config sample:\n\n```json\n" '{"foo": "bar", "count": 3}\n' "```\n"
    cleaned, calls = repair_tool_calls_in_text(text, allowed_tool_names=["Bash"])

    assert calls == []
    assert '"foo": "bar"' in cleaned
    assert "```json" in cleaned


def test_h2_tool_call_only_block_is_removed() -> None:
    """A lone tool-call fenced block is extracted and stripped from the text."""
    text = "```json\n" '{"name": "Bash", "arguments": {"command": "pwd"}}\n' "```"
    cleaned, calls = repair_tool_calls_in_text(text, allowed_tool_names=["Bash"])

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "Bash"
    # Nothing meaningful remains in the body.
    assert cleaned == ""


# ----------------------------------------------------------------------
# H6 — empty stream must still open with message_start
# ----------------------------------------------------------------------


async def _empty_stream() -> AsyncIterator[StreamChunk]:
    return
    yield  # pragma: no cover  (makes this an async generator)


@pytest.mark.asyncio
async def test_h6_empty_stream_starts_with_message_start() -> None:
    events = [ev async for ev in stream_chat_to_anthropic_events(_empty_stream())]
    types = [e.type for e in events]

    # Must begin with message_start (protocol requirement).
    assert types[0] == "message_start"
    # And terminate cleanly.
    assert types[-1] == "message_stop"
    assert "message_delta" in types
    # message_delta / message_stop never precede message_start.
    assert types.index("message_start") < types.index("message_delta")
    assert types.index("message_delta") < types.index("message_stop")


@pytest.mark.asyncio
async def test_h6_empty_stream_message_start_is_well_formed() -> None:
    events = [ev async for ev in stream_chat_to_anthropic_events(_empty_stream())]
    start = events[0]
    assert start.type == "message_start"
    message = start.data["message"]
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["content"] == []
    assert "usage" in message
