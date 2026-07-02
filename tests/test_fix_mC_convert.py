"""Regression tests for the M7/M8/M9 convert-layer bugs.

M7 (stream_chat_to_anthropic_events): when a tool_use block is open and a
    text delta (or a different tool call) arrives, the tool block is closed —
    but subsequent argument fragments for that tool used to be emitted as
    content_block_delta against the now-closed anthropic index. Fragments must
    only target a currently-open block; a reopened tool block must carry the
    original tool id/name (not a placeholder).

M8 (Anthropic<->OpenAI tool_result): the tool_result ``is_error`` flag was
    dropped in both directions. Anthropic->OpenAI now prefixes failing results
    with an "Error: " marker; the reverse leg restores is_error from that
    marker. A round-trip must preserve is_error without doubling the marker.

M9 (stream_anthropic_to_chat_chunks): if the upstream Anthropic event stream
    ends without a message_stop, no finish_reason chunk and no usage chunk were
    emitted, so OpenAI clients hang / see an incomplete response. A truncated
    stream must still be terminated with a finish chunk + usage chunk.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from coderouter.adapters.base import ChatRequest, Message, StreamChunk
from coderouter.translation import (
    AnthropicRequest,
    AnthropicStreamEvent,
    stream_anthropic_to_chat_chunks,
    stream_chat_to_anthropic_events,
    to_anthropic_request,
    to_chat_request,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _chunk(choices: list[dict[str, Any]], **kw: Any) -> StreamChunk:
    return StreamChunk(id="chatcmpl-test", created=0, model="m", choices=choices, **kw)


def _delta_chunk(delta: dict[str, Any]) -> StreamChunk:
    return _chunk([{"index": 0, "delta": delta, "finish_reason": None}])


async def _iter_chunks(chunks: list[StreamChunk]) -> AsyncIterator[StreamChunk]:
    for c in chunks:
        yield c


async def _iter_events(
    events: list[AnthropicStreamEvent],
) -> AsyncIterator[AnthropicStreamEvent]:
    for e in events:
        yield e


def _ev(type_: str, data: dict[str, Any]) -> AnthropicStreamEvent:
    return AnthropicStreamEvent(type=type_, data={"type": type_, **data})


def _finish_reasons(chunks: list[StreamChunk]) -> list[str]:
    """Collect the finish_reason from every chunk that carries one."""
    out: list[str] = []
    for c in chunks:
        for choice in c.choices or []:
            fr = choice.get("finish_reason")
            if fr:
                out.append(str(fr))
    return out


def _usages(chunks: list[StreamChunk]) -> list[dict[str, Any]]:
    """Collect usage dicts from trailing usage-only chunks (no choices)."""
    out: list[dict[str, Any]] = []
    for c in chunks:
        if not c.choices and c.usage is not None:
            out.append(c.usage)
    return out


def _index_events(events: list[AnthropicStreamEvent]) -> None:
    """Assert every content_block_delta targets a currently-open block index.

    Reconstructs the open/close bookkeeping the way the Anthropic SSE parser
    would and fails if a delta references an index that is not currently open
    (the M7 bug). Also checks indices are opened contiguously.
    """
    open_indices: set[int] = set()
    started_indices: set[int] = set()
    next_expected = 0
    for e in events:
        if e.type == "content_block_start":
            idx = e.data["index"]
            assert idx == next_expected, (
                f"non-contiguous content_block_start index {idx}, "
                f"expected {next_expected}"
            )
            next_expected += 1
            assert idx not in open_indices, f"index {idx} started while open"
            open_indices.add(idx)
            started_indices.add(idx)
        elif e.type == "content_block_delta":
            idx = e.data["index"]
            assert idx in open_indices, (
                f"content_block_delta targets index {idx} which is not open "
                f"(open={sorted(open_indices)})"
            )
        elif e.type == "content_block_stop":
            idx = e.data["index"]
            assert idx in open_indices, f"stop for non-open index {idx}"
            open_indices.discard(idx)


# ----------------------------------------------------------------------
# M7 — index integrity across text/tool interleave and parallel tools
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m7_text_after_tool_then_tool_args_no_stale_index() -> None:
    """tool_use opens, text interrupts, then more args arrive for that tool.

    The trailing args must NOT be emitted against the closed tool index.
    """
    chunks = [
        _delta_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "toolu_abc",
                        "function": {"name": "Bash", "arguments": '{"cmd":'},
                    }
                ]
            }
        ),
        # Interleaving text closes the tool_use block.
        _delta_chunk({"content": "thinking..."}),
        # More args for the SAME openai tool index arrive after the close.
        _delta_chunk(
            {"tool_calls": [{"index": 0, "function": {"arguments": ' "pwd"}'}}]}
        ),
    ]
    events = [ev async for ev in stream_chat_to_anthropic_events(_iter_chunks(chunks))]

    _index_events(events)

    # The re-opened tool block must preserve the original id and name.
    reopened_starts = [
        e
        for e in events
        if e.type == "content_block_start"
        and e.data["content_block"].get("type") == "tool_use"
    ]
    for s in reopened_starts:
        assert s.data["content_block"]["id"] == "toolu_abc"
        assert s.data["content_block"]["name"] == "Bash"

    # Reassembling all input_json_delta fragments per anthropic index and
    # concatenating across the tool's blocks yields the full arguments.
    frag = "".join(
        e.data["delta"]["partial_json"]
        for e in events
        if e.type == "content_block_delta"
        and e.data["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(frag) == {"cmd": "pwd"}


@pytest.mark.asyncio
async def test_m7_parallel_tool_calls_interleaved_indices() -> None:
    """Two tools stream with interleaved openai indices (0,1,0,1)."""
    chunks = [
        _delta_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "toolu_0",
                        "function": {"name": "A", "arguments": '{"x":'},
                    }
                ]
            }
        ),
        _delta_chunk(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "toolu_1",
                        "function": {"name": "B", "arguments": '{"y":'},
                    }
                ]
            }
        ),
        # Back to tool 0 — its block was closed when tool 1 opened.
        _delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": " 1}"}}]}),
        # Back to tool 1.
        _delta_chunk({"tool_calls": [{"index": 1, "function": {"arguments": " 2}"}}]}),
    ]
    events = [ev async for ev in stream_chat_to_anthropic_events(_iter_chunks(chunks))]

    _index_events(events)

    # Group input fragments by anthropic block index, then map each block back
    # to its tool id via the preceding content_block_start.
    id_by_index: dict[int, str] = {}
    for e in events:
        if e.type == "content_block_start" and e.data["content_block"].get(
            "type"
        ) == "tool_use":
            id_by_index[e.data["index"]] = e.data["content_block"]["id"]

    frags_by_id: dict[str, str] = {}
    for e in events:
        if (
            e.type == "content_block_delta"
            and e.data["delta"]["type"] == "input_json_delta"
        ):
            tid = id_by_index[e.data["index"]]
            frags_by_id[tid] = frags_by_id.get(tid, "") + e.data["delta"][
                "partial_json"
            ]

    assert json.loads(frags_by_id["toolu_0"]) == {"x": 1}
    assert json.loads(frags_by_id["toolu_1"]) == {"y": 2}


@pytest.mark.asyncio
async def test_m7_single_tool_uninterrupted_still_one_block() -> None:
    """No interleave: a tool whose fragments arrive back-to-back stays one block."""
    chunks = [
        _delta_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "toolu_z",
                        "function": {"name": "A", "arguments": '{"a":'},
                    }
                ]
            }
        ),
        _delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": " 1}"}}]}),
    ]
    events = [ev async for ev in stream_chat_to_anthropic_events(_iter_chunks(chunks))]
    _index_events(events)
    starts = [
        e
        for e in events
        if e.type == "content_block_start"
        and e.data["content_block"].get("type") == "tool_use"
    ]
    # Uninterrupted → exactly one tool_use block (no needless re-open).
    assert len(starts) == 1


# ----------------------------------------------------------------------
# M8 — is_error round-trip preservation
# ----------------------------------------------------------------------


def test_m8_forward_is_error_adds_marker() -> None:
    """Anthropic tool_result is_error -> OpenAI role=tool with 'Error: ' prefix."""
    req = AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                }
            ],
        }
    )
    chat = to_chat_request(req)
    tool_msgs = [m for m in chat.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "Error: boom"


def test_m8_forward_success_no_marker() -> None:
    """A successful tool_result gets no error prefix."""
    req = AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "ok",
                        }
                    ],
                }
            ],
        }
    )
    chat = to_chat_request(req)
    tool_msgs = [m for m in chat.messages if m.role == "tool"]
    assert tool_msgs[0].content == "ok"


def test_m8_reverse_marker_restores_is_error() -> None:
    """OpenAI role=tool 'Error: ...' -> Anthropic tool_result is_error=True."""
    chat = ChatRequest(
        messages=[
            Message(role="tool", tool_call_id="toolu_1", content="Error: boom"),
        ]
    )
    anth = to_anthropic_request(chat)
    # tool result collapses into a user turn with a tool_result block.
    blocks = anth.messages[0].content
    assert isinstance(blocks, list)
    tr = blocks[0]
    assert tr["type"] == "tool_result"
    assert tr["is_error"] is True
    assert tr["content"] == "boom"  # marker stripped


def test_m8_round_trip_preserves_is_error_no_double_marker() -> None:
    """Anthropic -> OpenAI -> Anthropic keeps is_error and does not double 'Error: '."""
    req = AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                }
            ],
        }
    )
    chat = to_chat_request(req)  # forward: adds "Error: "
    anth = to_anthropic_request(chat)  # reverse: strips + restores flag

    blocks = anth.messages[0].content
    assert isinstance(blocks, list)
    tr = blocks[0]
    assert tr["type"] == "tool_result"
    assert tr["is_error"] is True
    assert tr["content"] == "boom"
    # No doubling.
    assert not tr["content"].startswith("Error: Error:")


def test_m8_success_round_trip_has_no_error_flag() -> None:
    """A successful result round-trips without gaining an is_error flag."""
    req = AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "fine",
                        }
                    ],
                }
            ],
        }
    )
    chat = to_chat_request(req)
    anth = to_anthropic_request(chat)
    blocks = anth.messages[0].content
    assert isinstance(blocks, list)
    tr = blocks[0]
    assert "is_error" not in tr
    assert tr["content"] == "fine"


# ----------------------------------------------------------------------
# M9 — truncated stream (no message_stop) still terminates
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m9_no_message_stop_emits_finish_and_usage() -> None:
    """Stream ends after message_delta but before message_stop."""
    events = [
        _ev(
            "message_start",
            {
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-x",
                    "usage": {"input_tokens": 7, "output_tokens": 0},
                }
            },
        ),
        _ev("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        _ev(
            "content_block_delta",
            {"index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        ),
        _ev("content_block_stop", {"index": 0}),
        _ev(
            "message_delta",
            {
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        ),
        # NOTE: no message_stop — upstream truncated here.
    ]
    chunks = [c async for c in stream_anthropic_to_chat_chunks(_iter_events(events))]

    # A finish_reason chunk must be present.
    finish = _finish_reasons(chunks)
    assert finish == ["stop"]

    # A trailing usage chunk (no choices) must be present with accumulated usage.
    usages = _usages(chunks)
    assert len(usages) == 1
    assert usages[0]["prompt_tokens"] == 7
    assert usages[0]["completion_tokens"] == 5
    assert usages[0]["total_tokens"] == 12


@pytest.mark.asyncio
async def test_m9_normal_stop_not_duplicated() -> None:
    """With a proper message_stop, exactly one finish + one usage chunk appear."""
    events = [
        _ev(
            "message_start",
            {
                "message": {
                    "id": "msg_2",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-x",
                    "usage": {"input_tokens": 3, "output_tokens": 0},
                }
            },
        ),
        _ev(
            "message_delta",
            {
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        _ev("message_stop", {}),
    ]
    chunks = [c async for c in stream_anthropic_to_chat_chunks(_iter_events(events))]

    finish = _finish_reasons(chunks)
    usages = _usages(chunks)
    # Exactly one of each — no duplication from the truncation guard.
    assert finish == ["stop"]
    assert len(usages) == 1
    assert usages[0]["completion_tokens"] == 4


@pytest.mark.asyncio
async def test_m9_empty_event_stream_still_terminates() -> None:
    """No events at all: still emit a finish chunk + usage chunk (no hang)."""
    chunks = [c async for c in stream_anthropic_to_chat_chunks(_iter_events([]))]
    assert _finish_reasons(chunks) == ["stop"]
    assert len(_usages(chunks)) == 1
