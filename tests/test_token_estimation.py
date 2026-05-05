"""v2.0-F: token estimation module tests.

Tests the shared char/4 heuristic used by auto_router (longContext)
and the context budget guard (L1).
"""

from __future__ import annotations

import pytest

from coderouter.token_estimation import (
    CHARS_PER_TOKEN_HEURISTIC,
    DEFAULT_MAX_CONTEXT_TOKENS,
    estimate_tokens_from_anthropic_request,
    estimate_tokens_from_body,
)


# ----------------------------------------------------------------------
# estimate_tokens_from_body (raw dict, backward-compat with auto_router)
# ----------------------------------------------------------------------


class TestEstimateTokensFromBody:
    def test_empty_body(self):
        assert estimate_tokens_from_body({}) == 0

    def test_system_string(self):
        body = {"system": "You are a helpful assistant."}
        expected = len("You are a helpful assistant.") // CHARS_PER_TOKEN_HEURISTIC
        assert estimate_tokens_from_body(body) == expected

    def test_system_list_of_blocks(self):
        body = {
            "system": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        }
        expected = len("Hello") + len("World")
        assert estimate_tokens_from_body(body) == expected // CHARS_PER_TOKEN_HEURISTIC

    def test_messages_string_content(self):
        body = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        total_chars = len("What is 2+2?") + len("4")
        assert estimate_tokens_from_body(body) == total_chars // CHARS_PER_TOKEN_HEURISTIC

    def test_messages_list_content(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "..."}},
                    ],
                }
            ]
        }
        # Image blocks contribute 0
        expected = len("Describe this image") // CHARS_PER_TOKEN_HEURISTIC
        assert estimate_tokens_from_body(body) == expected

    def test_system_plus_messages(self):
        body = {
            "system": "Be helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        total_chars = len("Be helpful.") + len("Hi")
        assert estimate_tokens_from_body(body) == total_chars // CHARS_PER_TOKEN_HEURISTIC


# ----------------------------------------------------------------------
# estimate_tokens_from_anthropic_request (typed components)
# ----------------------------------------------------------------------


class TestEstimateTokensFromAnthropicRequest:
    def test_empty(self):
        assert estimate_tokens_from_anthropic_request(system=None, messages=[]) == 0

    def test_system_string(self):
        result = estimate_tokens_from_anthropic_request(
            system="System prompt here", messages=[]
        )
        assert result == len("System prompt here") // CHARS_PER_TOKEN_HEURISTIC

    def test_messages_with_content_attr(self):
        """Test with objects that have .content attribute (like AnthropicMessage)."""

        class FakeMsg:
            def __init__(self, content):
                self.content = content

        msgs = [FakeMsg("Hello world"), FakeMsg("Goodbye")]
        result = estimate_tokens_from_anthropic_request(system=None, messages=msgs)
        total_chars = len("Hello world") + len("Goodbye")
        assert result == total_chars // CHARS_PER_TOKEN_HEURISTIC

    def test_messages_as_dicts(self):
        """Test with plain dicts (test harness convenience)."""
        msgs = [
            {"content": "First message"},
            {"content": "Second message"},
        ]
        result = estimate_tokens_from_anthropic_request(system=None, messages=msgs)
        total_chars = len("First message") + len("Second message")
        assert result == total_chars // CHARS_PER_TOKEN_HEURISTIC

    def test_combined_system_and_messages(self):
        class FakeMsg:
            def __init__(self, content):
                self.content = content

        result = estimate_tokens_from_anthropic_request(
            system="Be concise.",
            messages=[FakeMsg("Tell me about AI")],
        )
        total_chars = len("Be concise.") + len("Tell me about AI")
        assert result == total_chars // CHARS_PER_TOKEN_HEURISTIC


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------


def test_default_max_context_tokens():
    assert DEFAULT_MAX_CONTEXT_TOKENS == 128_000


def test_chars_per_token_heuristic():
    assert CHARS_PER_TOKEN_HEURISTIC == 4
