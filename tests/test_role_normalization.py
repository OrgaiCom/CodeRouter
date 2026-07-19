"""Tests for Claude Code CLI >= 2.1.154 role normalization.

Claude Code 2.1.154+ has a regression where it emits ``role: "system"``
(and reportedly ``ctx`` / ``msg``) inside the Anthropic ``messages``
array, which the spec restricts to ``user`` / ``assistant``.
See anthropics/claude-code#63469, vllm-project/vllm#44000.

These tests cover ``normalize_message_roles`` and its wiring as a
``model_validator(mode="before")`` on ``AnthropicRequest``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# Same import convention as test_translation_anthropic.py: adapters.base
# must load before coderouter.translation, otherwise a pre-existing circular
# import (translation → convert → adapters → anthropic_native → convert)
# blows up at collection time.
import coderouter.adapters.base  # noqa: F401  (import-order anchor)
from coderouter.translation import AnthropicMessage, AnthropicRequest
from coderouter.translation.anthropic import normalize_message_roles


def _req(messages, **kwargs):
    payload = {"model": "m", "max_tokens": 100, "messages": messages, **kwargs}
    return AnthropicRequest.model_validate(payload)


# ------------------------------------------------------------
# The exact Claude Code >= 2.1.154 regression shape
# ------------------------------------------------------------


def test_claude_code_system_role_in_messages_is_accepted():
    req = _req(
        [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "injected system reminder"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    # Mid-conversation system text stays where the client put it, so the
    # cacheable prefix ahead of it is unchanged.
    assert [m.role for m in req.messages] == ["user", "user", "assistant"]
    assert req.messages[1].content == "injected system reminder"
    assert req.system is None


def test_system_role_with_block_list_content():
    req = _req(
        [
            {"role": "user", "content": "q"},
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            },
        ]
    )
    assert req.messages[1].content == "part one\npart two"
    assert req.system is None


def test_mid_conversation_system_does_not_touch_existing_system():
    req = _req(
        [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "extra"},
        ],
        system="original",
    )
    # The system prompt — the cache prefix — must not move.
    assert req.system == "original"
    assert req.messages[1].content == "extra"


def test_mid_conversation_system_does_not_touch_block_list_system():
    req = _req(
        [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "extra"},
        ],
        system=[{"type": "text", "text": "original"}],
    )
    assert req.system == [{"type": "text", "text": "original"}]
    assert req.messages[1].content == "extra"


def test_leading_system_is_hoisted_but_mid_conversation_is_not():
    req = _req(
        [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "q"},
            {"role": "system", "content": "second"},
        ]
    )
    # A leading system message is a genuine system prompt (OpenAI-style
    # client) and belongs at position 0. A later one is per-turn text.
    assert req.system == "first"
    assert [m.role for m in req.messages] == ["user", "user"]
    assert req.messages[1].content == "second"


def test_system_prompt_is_stable_as_conversation_grows():
    """The regression this guards: Claude Code appends a system-reminder each
    turn. If those are hoisted, the system prompt — which sits ahead of the
    whole conversation — changes every request and local backends
    (llama.cpp / LM Studio) reprocess the entire prompt.
    """
    turn1 = _req(
        [
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "reminder A"},
        ],
        system="STABLE PROMPT",
    )
    turn2 = _req(
        [
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "reminder A"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "system", "content": "reminder B"},
        ],
        system="STABLE PROMPT",
    )
    assert turn1.system == turn2.system == "STABLE PROMPT"
    # turn2's message list extends turn1's — an append-only prefix.
    assert [(m.role, m.content) for m in turn2.messages][: len(turn1.messages)] == [
        (m.role, m.content) for m in turn1.messages
    ]


# ------------------------------------------------------------
# ctx / msg / unknown roles
# ------------------------------------------------------------


@pytest.mark.parametrize("role", ["ctx", "msg", "tool", "something_new"])
def test_unknown_role_coerced_to_user_preserving_position(role):
    req = _req(
        [
            {"role": "user", "content": "q"},
            {"role": role, "content": "context blob"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert [m.role for m in req.messages] == ["user", "user", "assistant"]
    assert req.messages[1].content == "context blob"
    assert req.system is None


def test_unknown_role_with_empty_content_dropped():
    req = _req(
        [
            {"role": "user", "content": "q"},
            {"role": "ctx", "content": ""},
            {"role": "msg", "content": []},
        ]
    )
    assert [m.role for m in req.messages] == ["user"]


def test_empty_system_role_message_dropped_without_touching_system():
    req = _req(
        [
            {"role": "user", "content": "q"},
            {"role": "system", "content": ""},
        ]
    )
    assert req.system is None
    assert [m.role for m in req.messages] == ["user"]


# ------------------------------------------------------------
# Spec-conformant requests are untouched
# ------------------------------------------------------------


def test_valid_request_passes_through_unchanged():
    payload = {
        "model": "m",
        "max_tokens": 100,
        "system": "sys",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        ],
    }
    req = AnthropicRequest.model_validate(payload)
    assert req.system == "sys"
    assert [m.role for m in req.messages] == ["user", "assistant"]
    # normalize_message_roles returns the same object when nothing to do
    assert normalize_message_roles(payload) is payload


def test_caller_payload_not_mutated():
    payload = {
        "model": "m",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "s"},
        ],
    }
    AnthropicRequest.model_validate(payload)
    assert len(payload["messages"]) == 2  # original untouched
    assert "system" not in payload


def test_internal_construction_with_model_instances_still_works():
    # convert.to_anthropic_request constructs with AnthropicMessage objects;
    # the before-validator must pass them through.
    req = AnthropicRequest(
        model="m",
        max_tokens=100,
        messages=[AnthropicMessage(role="user", content="q")],
        system="sys",
    )
    assert req.messages[0].role == "user"
    assert req.system == "sys"


def test_message_model_itself_still_rejects_system_role():
    # The strict wire model is unchanged — normalization happens at the
    # request boundary, not by widening the role enum.
    with pytest.raises(ValidationError):
        AnthropicMessage.model_validate({"role": "system", "content": "x"})


def test_all_messages_invalid_roles_yields_empty_messages_but_valid_request():
    req = _req([{"role": "system", "content": "only system"}])
    assert req.messages == []
    assert req.system == "only system"
