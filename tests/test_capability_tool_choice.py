"""Unit tests for the S2 tool_choice capability gate + request helpers.

Covers:
    * ``provider_supports_tool_choice`` 3-tier resolution
      (explicit capabilities → registry → kind fallback),
    * ``anthropic_request_has_forced_tool_choice`` forcing-mode detection,
    * ``emulate_tool_choice`` request rewriting (both system shapes),
    * ``strip_cache_control`` marker removal.

Pure-function tests — no engine, no HTTP.
"""

from __future__ import annotations

from coderouter.config.capability_registry import (
    CapabilityRegistry,
    CapabilityRule,
    RegistryCapabilities,
)
from coderouter.config.schemas import Capabilities, ProviderConfig
from coderouter.routing.capability import (
    anthropic_request_has_forced_tool_choice,
    emulate_tool_choice,
    provider_supports_tool_choice,
    strip_cache_control,
)
from coderouter.translation.anthropic import AnthropicRequest


def _anthropic(name: str = "anth", **caps: bool) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
        capabilities=Capabilities(**caps),
    )


def _openai(name: str = "oai", **caps: bool) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="qwen-coder",
        capabilities=Capabilities(**caps),
    )


def _registry(*, kind: str, match: str, tool_choice: bool) -> CapabilityRegistry:
    return CapabilityRegistry.from_rule_lists(
        user=[
            CapabilityRule(
                match=match,
                kind=kind,  # type: ignore[arg-type]
                capabilities=RegistryCapabilities(tool_choice=tool_choice),
            )
        ]
    )


# ----------------------------------------------------------------------
# provider_supports_tool_choice — 3-tier resolution
# ----------------------------------------------------------------------


def test_gate_explicit_capability_wins() -> None:
    """capabilities.tool_choice: true opts an openai_compat provider in,
    overriding the kind heuristic that would otherwise say False."""
    prov = _openai(tool_choice=True)
    # Empty registry so tier 2 never fires; explicit flag is tier 1.
    reg = CapabilityRegistry.from_rule_lists()
    assert provider_supports_tool_choice(prov, registry=reg) is True


def test_gate_registry_true_promotes_openai() -> None:
    """A registry rule declaring tool_choice=true promotes an
    openai_compat model the kind heuristic would reject."""
    prov = _openai()
    reg = _registry(kind="openai_compat", match="qwen-coder", tool_choice=True)
    assert provider_supports_tool_choice(prov, registry=reg) is True


def test_gate_registry_false_hard_disables_anthropic() -> None:
    """A registry rule declaring tool_choice=false hard-disables even an
    anthropic-kind provider that the fallback would otherwise pass."""
    prov = _anthropic()
    reg = _registry(kind="anthropic", match="claude-sonnet-4-6", tool_choice=False)
    assert provider_supports_tool_choice(prov, registry=reg) is False


def test_gate_kind_fallback_anthropic_true_openai_false() -> None:
    """With no explicit flag and no registry opinion, the kind heuristic
    decides: anthropic → True, openai_compat → False."""
    empty = CapabilityRegistry.from_rule_lists()
    assert provider_supports_tool_choice(_anthropic(), registry=empty) is True
    assert provider_supports_tool_choice(_openai(), registry=empty) is False


# ----------------------------------------------------------------------
# anthropic_request_has_forced_tool_choice — forcing-mode detection
# ----------------------------------------------------------------------


def _req(tool_choice: dict | None) -> AnthropicRequest:
    body: dict = {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    return AnthropicRequest.model_validate(body)


def test_forced_true_for_any_and_tool() -> None:
    assert anthropic_request_has_forced_tool_choice(_req({"type": "any"})) is True
    assert (
        anthropic_request_has_forced_tool_choice(
            _req({"type": "tool", "name": "get_weather"})
        )
        is True
    )


def test_forced_false_for_auto_none_and_absent() -> None:
    assert anthropic_request_has_forced_tool_choice(_req({"type": "auto"})) is False
    assert anthropic_request_has_forced_tool_choice(_req({"type": "none"})) is False
    assert anthropic_request_has_forced_tool_choice(_req(None)) is False


# ----------------------------------------------------------------------
# emulate_tool_choice — request rewriting
# ----------------------------------------------------------------------


def test_emulate_tool_strips_choice_and_names_tool_str_system() -> None:
    req = AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 16,
            "system": "base prompt",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
    )
    out = emulate_tool_choice(req)
    assert out.tool_choice is None
    assert isinstance(out.system, str)
    assert out.system.startswith("base prompt")
    assert 'the tool named "get_weather"' in out.system
    assert "Do not respond with plain text." in out.system
    # Original untouched.
    assert req.tool_choice == {"type": "tool", "name": "get_weather"}
    assert req.system == "base prompt"


def test_emulate_any_appends_block_to_list_system() -> None:
    req = AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 16,
            "system": [
                {"type": "text", "text": "base", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "any"},
        }
    )
    out = emulate_tool_choice(req)
    assert out.tool_choice is None
    assert isinstance(out.system, list)
    # Original block preserved verbatim (incl. cache_control), directive
    # appended as a new trailing text block.
    assert out.system[0] == {
        "type": "text",
        "text": "base",
        "cache_control": {"type": "ephemeral"},
    }
    assert "one of the provided tools" in out.system[-1]["text"]


def test_emulate_preserves_profile_and_beta() -> None:
    req = _req({"type": "any"})
    req.profile = "coding"
    req.anthropic_beta = "beta-x"
    out = emulate_tool_choice(req)
    assert out.profile == "coding"
    assert out.anthropic_beta == "beta-x"


# ----------------------------------------------------------------------
# strip_cache_control — marker removal + count
# ----------------------------------------------------------------------


def test_strip_cache_control_removes_all_markers() -> None:
    req = AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 16,
            "system": [
                {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
            ],
            "tools": [
                {
                    "name": "t",
                    "input_schema": {},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            "tool_choice": {"type": "auto"},
        }
    )
    out, removed = strip_cache_control(req)
    assert removed == 3
    assert "cache_control" not in out.system[0]
    assert "cache_control" not in (out.tools[0].model_extra or {})
    assert "cache_control" not in out.messages[0].content[0]
    # Other fields preserved.
    assert out.tool_choice == {"type": "auto"}
    assert out.system[0]["text"] == "sys"
    # Original untouched.
    assert req.system[0]["cache_control"] == {"type": "ephemeral"}


def test_strip_cache_control_no_markers_returns_zero() -> None:
    req = AnthropicRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 16,
            "system": "plain",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    out, removed = strip_cache_control(req)
    assert removed == 0
    assert out.system == "plain"
