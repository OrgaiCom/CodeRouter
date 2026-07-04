"""Config fail-fast tests for the S2/S3 shim action fields.

The new ``FallbackChain.tool_choice_action`` and ``cache_control_action``
are constrained ``Literal`` enums; an out-of-set value must raise a
Pydantic ``ValidationError`` at config-load rather than silently no-op'ing
at request time (same fast-fail philosophy as the other ``*_action``
fields). Also asserts the backward-compatible defaults.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coderouter.config.schemas import Capabilities, FallbackChain


def test_defaults_are_off() -> None:
    chain = FallbackChain(name="default", providers=["a"])
    assert chain.tool_choice_action == "off"
    assert chain.cache_control_action == "off"


def test_valid_tool_choice_actions() -> None:
    for action in ("off", "warn", "emulate"):
        chain = FallbackChain(name="d", providers=["a"], tool_choice_action=action)
        assert chain.tool_choice_action == action


def test_valid_cache_control_actions() -> None:
    for action in ("off", "strip"):
        chain = FallbackChain(name="d", providers=["a"], cache_control_action=action)
        assert chain.cache_control_action == action


def test_invalid_tool_choice_action_raises() -> None:
    with pytest.raises(ValidationError):
        FallbackChain(name="d", providers=["a"], tool_choice_action="bogus")


def test_invalid_cache_control_action_raises() -> None:
    with pytest.raises(ValidationError):
        FallbackChain(name="d", providers=["a"], cache_control_action="drop")


def test_capabilities_tool_choice_default_none_and_bool() -> None:
    assert Capabilities().tool_choice is None
    assert Capabilities(tool_choice=True).tool_choice is True
    assert Capabilities(tool_choice=False).tool_choice is False
