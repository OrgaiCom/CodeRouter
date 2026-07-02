"""Regression tests for bug fixes H4 and H5.

H4 — continuous probe URL mismatch
    ``probe_one`` for ``kind: anthropic`` built ``base_url + "/v1/messages"``
    while the adapter strips a trailing ``/v1`` first. A ``base_url`` ending
    in ``/v1`` (LM Studio and similar) therefore made the probe hit
    ``/v1/v1/messages`` → constant 404 → healthy backends wrongly demoted.
    The URL builder is now shared via ``anthropic_messages_url``.

H5 — context-budget trim deleted too much
    ``_do_trim`` used to drop every non-preserved message the instant the
    threshold was crossed, collapsing a long history to the last ~4
    messages even on a marginal overflow, and could leave the head as an
    assistant / dangling-tool_result message (Anthropic 400). It now peels
    the oldest messages off one atomic unit at a time, re-estimating each
    step, and normalizes the head to a clean ``user`` message.
"""

from __future__ import annotations

from coderouter.adapters.anthropic_native import anthropic_messages_url

# Import adapters/base first to resolve the circular import between
# coderouter.translation.convert ↔ coderouter.adapters.anthropic_native.
from coderouter.adapters.base import BaseAdapter  # noqa: F401
from coderouter.guards.context_budget import trim_to_budget
from coderouter.token_estimation import estimate_tokens_from_anthropic_request
from coderouter.translation.anthropic import AnthropicMessage, AnthropicRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    messages: list[tuple[str, str | list]],
    system: str | None = None,
) -> AnthropicRequest:
    return AnthropicRequest(
        model="test-model",
        max_tokens=1024,
        system=system,
        messages=[
            AnthropicMessage(role=role, content=content) for role, content in messages
        ],
    )


def _est(request: AnthropicRequest) -> int:
    return estimate_tokens_from_anthropic_request(
        system=request.system, messages=request.messages
    )


def _first_role(request: AnthropicRequest) -> str:
    return request.messages[0].role


def _has_tool_result(msg: AnthropicMessage) -> bool:
    content = msg.content
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
    return False


def _tool_pairs_intact(request: AnthropicRequest) -> bool:
    """Every surviving tool_result must have its tool_use somewhere earlier."""
    use_ids: set[str] = set()
    for msg in request.messages:
        content = msg.content
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tid = block.get("id")
                if isinstance(tid, str):
                    use_ids.add(tid)
            elif block.get("type") == "tool_result":
                tid = block.get("tool_use_id")
                if isinstance(tid, str) and tid not in use_ids:
                    return False  # orphaned tool_result
    return True


# ---------------------------------------------------------------------------
# H4: shared Anthropic /v1/messages URL builder
# ---------------------------------------------------------------------------


class TestH4AnthropicUrl:
    def test_url_matches_with_and_without_trailing_v1(self):
        """Both base_url forms must produce the identical endpoint URL."""
        without = anthropic_messages_url("http://host:1234")
        with_v1 = anthropic_messages_url("http://host:1234/v1")
        assert without == with_v1 == "http://host:1234/v1/messages"

    def test_url_handles_trailing_slash(self):
        assert (
            anthropic_messages_url("http://host:1234/v1/")
            == "http://host:1234/v1/messages"
        )
        assert (
            anthropic_messages_url("http://host:1234/")
            == "http://host:1234/v1/messages"
        )

    def test_url_never_doubles_v1(self):
        """The historic /v1/v1/messages 404 must be impossible."""
        for base in (
            "https://api.anthropic.com",
            "https://api.anthropic.com/v1",
            "https://api.anthropic.com/v1/",
            "http://localhost:1234/v1",
        ):
            url = anthropic_messages_url(base)
            assert "/v1/v1/" not in url
            assert url.endswith("/v1/messages")

    def test_probe_uses_shared_builder(self):
        """probe_one must build the anthropic URL via the shared helper."""
        import inspect

        from coderouter.guards import continuous_probe

        src = inspect.getsource(continuous_probe.probe_one)
        # Must not hand-roll the "+ /v1/messages" concatenation anymore.
        assert "anthropic_messages_url" in src
        assert '/v1/messages"' not in src


# ---------------------------------------------------------------------------
# H5: incremental, minimal-removal trim
# ---------------------------------------------------------------------------


class TestH5MinimalTrim:
    def test_marginal_overflow_removes_minimum(self):
        """A slight overflow trims only a few messages, not the whole history."""
        # 20 messages of ~25 tokens each = ~500 tokens; system ~5 tokens.
        # max_context=700, target=0.75 → 525 tokens. We're ~505/700 = 0.72,
        # so bump message size a touch to land just over target.
        request = _make_request(
            [
                ("user" if i % 2 == 0 else "assistant", "x" * 120)  # ~30 tokens each
                for i in range(20)
            ],
            system="sys",
        )
        before = _est(request)
        assert before > 525  # confirm we start over target

        trimmed, result = trim_to_budget(
            request,
            max_context_tokens=700,
            trim_target=0.75,
            preserve_last_n=4,
        )
        # Only a handful removed — NOT collapsed to the preserve floor.
        assert result.messages_removed > 0
        assert result.messages_after > 4, (
            "marginal overflow collapsed history to the preserve floor"
        )
        # Result is at or below target (525 tokens).
        assert _est(trimmed) <= 525
        # The trim is near-minimal: it removed roughly the deficit, not the
        # whole removable window. The old bug removed all 16 removable
        # messages here (leaving only the 4-message floor); the fix should
        # remove far fewer. Allow +1 for head normalization (dropping a
        # leading assistant so the head is a clean user message).
        before_tokens = _est(request)
        deficit_tokens = before_tokens - 525
        # ~30 tokens/message → minimal messages to remove ≈ deficit / 30.
        minimal_needed = deficit_tokens // 30
        assert result.messages_removed <= minimal_needed + 2, (
            "trim removed far more than the deficit required"
        )
        # Sanity: the head is a clean user message (never assistant).
        assert _first_role(trimmed) == "user"

    def test_no_trim_when_under_target(self):
        """Under target → nothing removed even if guard is invoked."""
        request = _make_request([("user", "hi"), ("assistant", "hello")])
        trimmed, result = trim_to_budget(
            request, max_context_tokens=128000, trim_target=0.75, preserve_last_n=4
        )
        assert result.messages_removed == 0
        assert len(trimmed.messages) == 2

    def test_massive_overflow_head_is_clean_user(self):
        """Even under aggressive trim the head is a user msg w/o tool_result."""
        messages: list[tuple[str, str | list]] = [
            ("user", "x" * 400),
            ("assistant", "y" * 400),
            ("user", "x" * 400),
            ("assistant", "y" * 400),
            ("user", "final question " + "z" * 200),
            ("assistant", "final answer " + "w" * 200),
        ]
        request = _make_request(messages, system="sys " + "s" * 100)
        trimmed, _result = trim_to_budget(
            request,
            max_context_tokens=100,  # impossibly small → aggressive trim
            trim_target=0.50,
            preserve_last_n=4,
        )
        assert _first_role(trimmed) == "user"
        assert not _has_tool_result(trimmed.messages[0])
        assert len(trimmed.messages) >= 1

    def test_tool_pair_not_split_by_head_normalization(self):
        """Head normalization must not orphan a tool_result."""
        messages: list[tuple[str, str | list]] = [
            ("user", "do a thing " + "a" * 300),
            (
                "assistant",
                [
                    {"type": "text", "text": "using tool " + "b" * 300},
                    {"type": "tool_use", "id": "tu_1", "name": "read", "input": {}},
                ],
            ),
            (
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "result " + "c" * 300,
                    }
                ],
            ),
            ("assistant", "answer " + "d" * 300),
            ("user", "next " + "e" * 300),
            (
                "assistant",
                [
                    {"type": "text", "text": "second tool " + "f" * 300},
                    {"type": "tool_use", "id": "tu_2", "name": "write", "input": {}},
                ],
            ),
            (
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_2",
                        "content": "done " + "g" * 300,
                    }
                ],
            ),
            ("assistant", "all done " + "h" * 300),
        ]
        request = _make_request(messages, system="sys " + "s" * 100)
        trimmed, _result = trim_to_budget(
            request,
            max_context_tokens=200,  # force heavy trim
            trim_target=0.50,
            preserve_last_n=4,
        )
        # No orphaned tool_result anywhere in the result.
        assert _tool_pairs_intact(trimmed)
        # Head must be a clean user message.
        assert _first_role(trimmed) == "user"
        assert not _has_tool_result(trimmed.messages[0])

    def test_preserve_floor_respected(self):
        """The last preserve_last_n messages are never removed."""
        request = _make_request(
            [
                ("user" if i % 2 == 0 else "assistant", "x" * 400)
                for i in range(12)
            ],
            system="sys",
        )
        trimmed, _result = trim_to_budget(
            request,
            max_context_tokens=50,  # tiny → maximal trim pressure
            trim_target=0.50,
            preserve_last_n=4,
        )
        # The final message of the original must survive (tail is a floor).
        assert trimmed.messages[-1].content == request.messages[-1].content
        # Trimmed messages remain a contiguous suffix of the original.
        n = len(trimmed.messages)
        for i in range(n):
            assert trimmed.messages[i].content == request.messages[-(n - i)].content
