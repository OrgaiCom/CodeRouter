"""v2.0-F: token estimation module tests.

Tests the shared char/4 heuristic used by auto_router (longContext)
and the context budget guard (L1).
"""

from __future__ import annotations

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


# ----------------------------------------------------------------------
# H-5: tool_result / tool_use / thinking blocks are counted
#
# Up to v2.11.x ``_extract_text_from_content`` only looked at ``text``
# blocks, so a Claude Code style session — where almost all context
# lives in tool_result payloads — was under-estimated by 5x at 20 turns
# and ~29x at 200. Every test below fails against v2.11.2.
# ----------------------------------------------------------------------


import json  # noqa: E402

from coderouter.token_estimation import (  # noqa: E402
    extract_text_from_anthropic_request,
    get_include_tool_content,
    set_include_tool_content,
)

# NOTE: the process-wide opt-out is restored by the autouse
# ``_restore_token_estimation_scope`` fixture in tests/conftest.py — it
# lives there, not here, because a config built with the key set to
# false leaks the global across *every* test module.


def _user(content):
    return {"role": "user", "content": content}


class TestToolContentCounted:
    def test_tool_result_str_content_counted(self):
        """``tool_result.content`` as a plain string is counted (was 0)."""
        payload = "R" * 5000
        msgs = [
            _user(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": payload,
                    }
                ]
            )
        ]
        assert estimate_tokens_from_anthropic_request(system=None, messages=msgs) == 1250

    def test_tool_result_block_list_counted(self):
        """``content`` as a list of text blocks: the inner text counts."""
        msgs = [
            _user(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "A" * 400},
                            {"type": "text", "text": "B" * 400},
                        ],
                    }
                ]
            )
        ]
        # 400 + "\n" + 400 = 801 chars → 200 tokens
        assert estimate_tokens_from_anthropic_request(system=None, messages=msgs) == 200

    def test_tool_result_image_block_not_counted(self):
        """A base64 image nested in tool_result.content stays at 0 chars.

        This is the over-estimation guard: a naive ``json.dumps(block)``
        counts the base64 payload and inflates the estimate ~35x, which
        would make trim shred the history of any session that pasted a
        screenshot.
        """
        b64 = "Z" * 400_000
        text = "T" * 400
        msgs = [
            _user(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": text},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            },
                        ],
                    }
                ]
            )
        ]
        assert estimate_tokens_from_anthropic_request(system=None, messages=msgs) == 100

    def test_tool_use_input_counted(self):
        """``tool_use.input`` contributes its JSON length."""
        tool_input = {"file_path": "/tmp/x.py", "content": "C" * 340}
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Write",
                        "input": tool_input,
                    }
                ],
            }
        ]
        expected_chars = len("Write") + 1 + len(json.dumps(tool_input, ensure_ascii=False))
        assert (
            estimate_tokens_from_anthropic_request(system=None, messages=msgs)
            == expected_chars // CHARS_PER_TOKEN_HEURISTIC
        )

    def test_tool_use_input_cjk_not_escape_inflated(self):
        """``ensure_ascii=False`` — CJK args count as characters, not \\uXXXX."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "n",
                        "input": {"q": "日" * 100},
                    }
                ],
            }
        ]
        # 6x inflation would show up immediately here.
        assert estimate_tokens_from_anthropic_request(system=None, messages=msgs) < 60

    def test_top_level_image_block_still_zero(self):
        """Pinned existing behavior: a top-level image contributes 0."""
        msgs = [
            _user(
                [
                    {"type": "text", "text": "X" * 40},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "Q" * 400_000,
                        },
                    },
                ]
            )
        ]
        assert estimate_tokens_from_anthropic_request(system=None, messages=msgs) == 10

    def test_thinking_block_policy(self):
        """Policy: ``thinking`` counts, ``redacted_thinking`` does not.

        Extended-thinking blocks are replayed verbatim to the model on
        the following turn (they must be, or tool-use signatures break),
        so they occupy the context window exactly like text.
        ``redacted_thinking.data`` is opaque ciphertext the model never
        reads as text — counting it would be an image-style
        over-estimate.
        """
        thinking = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "T" * 4000, "signature": "sig"}
                ],
            }
        ]
        assert estimate_tokens_from_anthropic_request(system=None, messages=thinking) == 1000

        redacted = [
            {
                "role": "assistant",
                "content": [{"type": "redacted_thinking", "data": "D" * 4000}],
            }
        ]
        assert estimate_tokens_from_anthropic_request(system=None, messages=redacted) == 0

    def test_unknown_block_type_still_zero(self):
        """Forward-compat: an unrecognized block type contributes 0."""
        msgs = [_user([{"type": "some_future_block", "payload": "P" * 4000}])]
        assert estimate_tokens_from_anthropic_request(system=None, messages=msgs) == 0

    def test_extract_text_includes_tool_result(self):
        """The text-extraction path (language tax / count_tokens) agrees."""
        text = extract_text_from_anthropic_request(
            system=None,
            messages=[
                _user(
                    [{"type": "tool_result", "tool_use_id": "t", "content": "PAYLOAD"}]
                )
            ],
        )
        assert "PAYLOAD" in text


class TestIncludeToolContentOptOut:
    """``token_estimation_include_tool_content: false`` → v2.11.x parity."""

    def _session(self):
        return [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling a tool"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "/a/b.py"},
                    },
                    {"type": "thinking", "thinking": "H" * 500},
                ],
            },
            _user(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "F" * 5000,
                    }
                ]
            ),
            _user("plain question"),
        ]

    def test_include_tool_content_flag_off_matches_v2_11(self):
        """Off → only ``text`` blocks count, exactly as v2.11.2 did."""
        msgs = self._session()
        system = "S" * 100

        # v2.11.2 reference value: system + the single text block only.
        v2_11_chars = len(system) + len("calling a tool") + len("plain question")
        off = estimate_tokens_from_anthropic_request(
            system=system, messages=msgs, include_tool_content=False
        )
        assert off == v2_11_chars // CHARS_PER_TOKEN_HEURISTIC

        on = estimate_tokens_from_anthropic_request(
            system=system, messages=msgs, include_tool_content=True
        )
        assert on > off

    def test_process_wide_default_is_honored(self):
        """``set_include_tool_content(False)`` flips every call site."""
        msgs = self._session()
        set_include_tool_content(False)
        assert estimate_tokens_from_anthropic_request(
            system=None, messages=msgs
        ) == estimate_tokens_from_anthropic_request(
            system=None, messages=msgs, include_tool_content=False
        )
        set_include_tool_content(True)
        assert estimate_tokens_from_anthropic_request(
            system=None, messages=msgs
        ) == estimate_tokens_from_anthropic_request(
            system=None, messages=msgs, include_tool_content=True
        )

    def test_body_estimator_honors_flag(self):
        body = {"messages": self._session()}
        assert estimate_tokens_from_body(body, include_tool_content=False) < (
            estimate_tokens_from_body(body, include_tool_content=True)
        )

    def test_config_key_publishes_to_estimator(self):
        """Loading a config with the key false flips the module default."""
        from coderouter.config.schemas import (
            CodeRouterConfig,
            FallbackChain,
            ProviderConfig,
        )

        def _cfg(flag: bool) -> CodeRouterConfig:
            return CodeRouterConfig(
                allow_paid=False,
                default_profile="default",
                token_estimation_include_tool_content=flag,
                providers=[
                    ProviderConfig(
                        name="local",
                        base_url="http://localhost:8080/v1",
                        model="m",
                    )
                ],
                profiles=[FallbackChain(name="default", providers=["local"])],
            )

        _cfg(False)
        assert get_include_tool_content() is False
        _cfg(True)
        assert get_include_tool_content() is True

    def test_config_key_defaults_to_true(self):
        """Existing YAML without the key keeps the corrected behavior."""
        from coderouter.config.schemas import (
            CodeRouterConfig,
            FallbackChain,
            ProviderConfig,
        )

        cfg = CodeRouterConfig(
            allow_paid=False,
            default_profile="default",
            providers=[
                ProviderConfig(
                    name="local", base_url="http://localhost:8080/v1", model="m"
                )
            ],
            profiles=[FallbackChain(name="default", providers=["local"])],
        )
        assert cfg.token_estimation_include_tool_content is True
        assert get_include_tool_content() is True
