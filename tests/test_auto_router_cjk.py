"""Phase 2 tests: the ``cjk_ratio_min`` auto-route matcher (v2.6).

Steers CJK-heavy turns (which carry the cloud language tax) to a local
profile, while ASCII / code turns fall through to the default profile.
"""

from __future__ import annotations

import pytest

from coderouter.config.schemas import (
    AutoRouterConfig,
    AutoRouteRule,
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
    RuleMatcher,
)
from coderouter.routing.auto_router import classify


def _provider(name: str, model: str = "stub") -> ProviderConfig:
    return ProviderConfig(name=name, base_url="http://localhost:8080/v1", model=model)


@pytest.fixture
def cjk_config() -> CodeRouterConfig:
    """`default_profile: auto`; one rule routing CJK-heavy turns to local."""
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="auto",
        providers=[
            _provider("local-qwen", "qwen2.5:7b"),
            _provider("cloud-sonnet", "claude-sonnet"),
        ],
        profiles=[
            FallbackChain(name="local", providers=["local-qwen"]),
            FallbackChain(name="cloud", providers=["cloud-sonnet"]),
        ],
        auto_router=AutoRouterConfig(
            rules=[
                AutoRouteRule(
                    id="test:cjk-local",
                    profile="local",
                    match=RuleMatcher(cjk_ratio_min=0.3),
                ),
            ],
            default_rule_profile="cloud",
        ),
    )


def test_cjk_heavy_routes_to_local(cjk_config: CodeRouterConfig):
    body = {"messages": [{"role": "user", "content": "この関数のバグを直してください"}]}
    assert classify(body, cjk_config) == "local"


def test_ascii_falls_through_to_cloud(cjk_config: CodeRouterConfig):
    body = {"messages": [{"role": "user", "content": "please fix this bug in my code"}]}
    assert classify(body, cjk_config) == "cloud"


def test_mixed_below_threshold_falls_through(cjk_config: CodeRouterConfig):
    # Mostly ASCII code with a short JA tail -> ratio < 0.3 -> cloud.
    body = {
        "messages": [
            {
                "role": "user",
                "content": "def add(a, b): return a + b  # ok",
            }
        ]
    }
    assert classify(body, cjk_config) == "cloud"


def test_mixed_above_threshold_routes_local(cjk_config: CodeRouterConfig):
    # Heavier JA share pushes ratio over 0.3.
    body = {
        "messages": [
            {"role": "user", "content": "この関数を修正して: def f(): pass"}
        ]
    }
    assert classify(body, cjk_config) == "local"


def test_matcher_rejects_multiple_fields():
    # one-of invariant still enforced after adding cjk_ratio_min.
    with pytest.raises(ValueError, match="exactly one matcher field"):
        RuleMatcher(cjk_ratio_min=0.3, content_contains="x")


def test_matcher_accepts_cjk_ratio_alone():
    m = RuleMatcher(cjk_ratio_min=0.5)
    assert m.cjk_ratio_min == 0.5


def test_matcher_rejects_out_of_range():
    with pytest.raises(ValueError):
        RuleMatcher(cjk_ratio_min=1.5)
