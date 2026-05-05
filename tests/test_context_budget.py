"""v2.0-F (L1): context budget guard tests.

Three test groups:

- **Estimation (pure)**: ``estimate_context_usage`` computes ratios
  and threshold booleans correctly.
- **Trim (pure)**: ``trim_to_budget`` removes old messages, preserves
  last N, handles tool pairs atomically.
- **Config validation**: schema fields have correct bounds and defaults.
"""

from __future__ import annotations

import pytest

from coderouter.config.schemas import FallbackChain, ProviderConfig

# Import adapters/routing first to resolve the circular import between
# coderouter.translation.convert ↔ coderouter.adapters.anthropic_native.
# conftest loads schemas first, then test_memory_pressure loads these in
# this order and it works; replicating the pattern here.
from coderouter.adapters.base import BaseAdapter  # noqa: F401
from coderouter.guards.context_budget import (
    ContextBudgetEstimate,
    TrimResult,
    estimate_context_usage,
    trim_to_budget,
)
from coderouter.translation.anthropic import AnthropicMessage, AnthropicRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    messages: list[tuple[str, str | list]],
    system: str | None = None,
) -> AnthropicRequest:
    """Build a minimal AnthropicRequest from (role, content) pairs."""
    return AnthropicRequest(
        model="test-model",
        max_tokens=1024,
        system=system,
        messages=[
            AnthropicMessage(role=role, content=content)
            for role, content in messages
        ],
    )


def _make_long_request(n_messages: int, chars_per_msg: int = 400) -> AnthropicRequest:
    """Build a request with many messages for trim testing."""
    messages = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}: " + "x" * chars_per_msg
        messages.append((role, content))
    return _make_request(messages, system="System prompt " + "y" * 100)


# ---------------------------------------------------------------------------
# Group 1: Estimation (pure)
# ---------------------------------------------------------------------------


class TestEstimateContextUsage:
    def test_empty_request(self):
        request = _make_request([("user", "Hi")])
        estimate = estimate_context_usage(
            request, max_context_tokens=128000
        )
        assert estimate.estimated_tokens == len("Hi") // 4
        assert estimate.max_context_tokens == 128000
        assert estimate.usage_ratio < 0.01
        assert not estimate.over_warn_threshold
        assert not estimate.over_trim_threshold

    def test_over_warn_threshold(self):
        # Create a request that uses ~85% of a 100-token context window
        # 85 tokens * 4 chars/token = 340 chars
        request = _make_request([("user", "x" * 340)])
        estimate = estimate_context_usage(
            request, max_context_tokens=100, warn_threshold=0.80
        )
        assert estimate.usage_ratio >= 0.80
        assert estimate.over_warn_threshold
        assert not estimate.over_trim_threshold  # default trim is 0.90

    def test_over_trim_threshold(self):
        # 95 tokens * 4 = 380 chars
        request = _make_request([("user", "x" * 380)])
        estimate = estimate_context_usage(
            request, max_context_tokens=100, warn_threshold=0.80, trim_threshold=0.90
        )
        assert estimate.over_warn_threshold
        assert estimate.over_trim_threshold

    def test_below_all_thresholds(self):
        # 50 tokens * 4 = 200 chars
        request = _make_request([("user", "x" * 200)])
        estimate = estimate_context_usage(
            request, max_context_tokens=100, warn_threshold=0.80
        )
        assert estimate.usage_ratio == pytest.approx(0.50, abs=0.01)
        assert not estimate.over_warn_threshold
        assert not estimate.over_trim_threshold

    def test_system_prompt_counted(self):
        # system: 200 chars = 50 tokens, message: 200 chars = 50 tokens
        # total: 100 tokens / 100 max = 1.0 ratio
        request = _make_request([("user", "x" * 200)], system="y" * 200)
        estimate = estimate_context_usage(
            request, max_context_tokens=100
        )
        assert estimate.usage_ratio == pytest.approx(1.0, abs=0.01)

    def test_zero_max_context_safe(self):
        """max_context_tokens=0 should not crash (defensive)."""
        request = _make_request([("user", "Hello")])
        estimate = estimate_context_usage(request, max_context_tokens=0)
        assert estimate.usage_ratio == 0.0


# ---------------------------------------------------------------------------
# Group 2: Trim (pure)
# ---------------------------------------------------------------------------


class TestTrimToBudget:
    def test_basic_trim(self):
        # 10 messages of ~100 tokens each = 1000 tokens total
        # max_context = 500, target = 0.75 → target_tokens = 375
        # Should trim most messages, preserve last 4
        request = _make_long_request(10, chars_per_msg=396)  # ~100 tokens each
        trimmed, result = trim_to_budget(
            request,
            max_context_tokens=500,
            trim_target=0.75,
            preserve_last_n=4,
        )
        assert result.messages_removed > 0
        assert result.messages_after <= result.messages_before
        assert result.messages_after >= 2  # minimum floor
        assert len(trimmed.messages) == result.messages_after

    def test_preserve_last_n(self):
        # 20 messages of ~25 tokens each = ~500 tokens + system ~30 tokens
        # max_context = 1000, target = 0.75 → target_tokens = 750
        # Should keep last 4 and remove the rest
        request = _make_long_request(20, chars_per_msg=96)  # ~25 tokens each
        trimmed, result = trim_to_budget(
            request,
            max_context_tokens=1000,
            trim_target=0.75,
            preserve_last_n=4,
        )
        original_msgs = request.messages
        trimmed_msgs = trimmed.messages
        # The last N messages in trimmed should be the last N from original
        # (at minimum, the final messages are always preserved)
        n_preserved = len(trimmed_msgs)
        for i in range(n_preserved):
            assert trimmed_msgs[i].content == original_msgs[-(n_preserved - i)].content

    def test_no_trim_needed(self):
        # Small request, large context window → no trim
        request = _make_request([("user", "Hi"), ("assistant", "Hello")])
        trimmed, result = trim_to_budget(
            request,
            max_context_tokens=128000,
            trim_target=0.75,
            preserve_last_n=4,
        )
        assert result.messages_removed == 0
        assert len(trimmed.messages) == 2

    def test_tool_pair_preservation(self):
        """tool_use + tool_result pairs should be preserved atomically."""
        messages = [
            ("user", "Do something"),
            ("assistant", [
                {"type": "text", "text": "I'll use a tool"},
                {"type": "tool_use", "id": "tu_1", "name": "read", "input": {}},
            ]),
            ("user", [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "file contents"},
            ]),
            ("assistant", "Based on the file..."),
            ("user", "Thanks, now do another thing"),
            ("assistant", [
                {"type": "text", "text": "Let me try"},
                {"type": "tool_use", "id": "tu_2", "name": "write", "input": {}},
            ]),
            ("user", [
                {"type": "tool_result", "tool_use_id": "tu_2", "content": "done"},
            ]),
            ("assistant", "All done!"),
        ]
        request = _make_request(messages, system="System " + "z" * 100)
        trimmed, result = trim_to_budget(
            request,
            max_context_tokens=100,  # force aggressive trim
            trim_target=0.50,
            preserve_last_n=4,
        )
        # Verify tool pairs are intact in the trimmed result
        trimmed_msgs = trimmed.messages
        for i, msg in enumerate(trimmed_msgs):
            content = msg.content
            if isinstance(content, list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                )
                if has_tool_result and i > 0:
                    # Previous message should be assistant with tool_use
                    prev = trimmed_msgs[i - 1]
                    prev_content = prev.content
                    if isinstance(prev_content, list):
                        has_tool_use = any(
                            isinstance(b, dict) and b.get("type") == "tool_use"
                            for b in prev_content
                        )
                        assert has_tool_use, (
                            f"tool_result at index {i} has no preceding tool_use"
                        )

    def test_minimum_floor(self):
        """Even with aggressive trim, at least 2 messages are kept."""
        request = _make_long_request(20, chars_per_msg=400)
        trimmed, result = trim_to_budget(
            request,
            max_context_tokens=10,  # impossibly small
            trim_target=0.50,
            preserve_last_n=4,
        )
        assert len(trimmed.messages) >= 2

    def test_system_prompt_preserved(self):
        """System prompt is never removed by trim."""
        request = _make_long_request(10, chars_per_msg=400)
        trimmed, _ = trim_to_budget(
            request,
            max_context_tokens=200,
            trim_target=0.50,
            preserve_last_n=4,
        )
        assert trimmed.system == request.system


# ---------------------------------------------------------------------------
# Group 3: Config schema validation
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_default_values(self):
        chain = FallbackChain(name="test", providers=["p1"])
        assert chain.context_budget_action == "off"
        assert chain.context_budget_warn_threshold == 0.80
        assert chain.context_budget_trim_threshold == 0.90
        assert chain.context_budget_trim_target == 0.75
        assert chain.context_budget_preserve_last_n == 4

    def test_provider_max_context_tokens_default(self):
        provider = ProviderConfig(
            name="test",
            base_url="http://localhost:8080/v1",
            model="qwen3",
        )
        assert provider.max_context_tokens is None

    def test_provider_max_context_tokens_explicit(self):
        provider = ProviderConfig(
            name="test",
            base_url="http://localhost:8080/v1",
            model="qwen3",
            max_context_tokens=32768,
        )
        assert provider.max_context_tokens == 32768

    def test_threshold_bounds(self):
        """Thresholds must be between 0.1 and 1.0."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            FallbackChain(
                name="test",
                providers=["p1"],
                context_budget_warn_threshold=0.0,  # below min
            )

    def test_action_literals(self):
        """Only off/warn/trim are accepted."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            FallbackChain(
                name="test",
                providers=["p1"],
                context_budget_action="compress",  # not valid
            )
