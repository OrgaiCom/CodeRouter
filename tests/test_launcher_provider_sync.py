"""Tests for launcher provider auto-sync (FallbackEngine.register_provider).

The embedded launcher starts backends on arbitrary ports, but routing is
config-driven — before this feature the operator had to hand-edit
providers.yaml (the observed failure: launcher on 8085, config pointing at
8080). register_provider closes the gap in-memory: provider entry + adapter
+ a "launcher" profile whose head is always the most recently started
backend. default_profile must never be touched.
"""

from __future__ import annotations

import pytest

from coderouter.config.schemas import CodeRouterConfig, FallbackChain, ProviderConfig
from coderouter.ingress.launcher_routes import _launcher_provider_config
from coderouter.routing.fallback import FallbackEngine


def _base_config() -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="static-local",
                base_url="http://localhost:11434/v1",
                model="qwen2.5-coder:7b",
            ),
        ],
        profiles=[FallbackChain(name="default", providers=["static-local"])],
    )


def _engine() -> FallbackEngine:
    return FallbackEngine(_base_config())


def test_register_new_provider_adds_config_adapter_and_profile() -> None:
    engine = _engine()
    summary = engine.register_provider(
        _launcher_provider_config("llama.cpp", 8085)
    )

    names = [p.name for p in engine.config.providers]
    assert "launcher-llamacpp-8085" in names
    assert "launcher-llamacpp-8085" in engine._adapters
    chain = engine.config.profile_by_name("launcher")
    assert chain.providers == ["launcher-llamacpp-8085"]
    assert summary["replaced"] is False
    assert summary["persisted"] is False


def test_reregister_same_name_replaces_instead_of_duplicating() -> None:
    engine = _engine()
    engine.register_provider(_launcher_provider_config("llama.cpp", 8085))
    summary = engine.register_provider(
        _launcher_provider_config("llama.cpp", 8085)
    )

    names = [p.name for p in engine.config.providers]
    assert names.count("launcher-llamacpp-8085") == 1
    chain = engine.config.profile_by_name("launcher")
    assert chain.providers.count("launcher-llamacpp-8085") == 1
    assert summary["replaced"] is True


def test_most_recent_backend_moves_to_front_of_launcher_profile() -> None:
    engine = _engine()
    engine.register_provider(_launcher_provider_config("llama.cpp", 8085))
    engine.register_provider(_launcher_provider_config("vllm", 8081))

    chain = engine.config.profile_by_name("launcher")
    assert chain.providers[0] == "launcher-vllm-8081"
    assert chain.providers == ["launcher-vllm-8081", "launcher-llamacpp-8085"]


def test_default_profile_is_never_touched() -> None:
    engine = _engine()
    engine.register_provider(_launcher_provider_config("llama.cpp", 8085))

    assert engine.config.default_profile == "default"
    default_chain = engine.config.profile_by_name("default")
    assert default_chain.providers == ["static-local"]


def test_provider_config_shape_enables_models_passthrough() -> None:
    pconf = _launcher_provider_config("llama.cpp", 8085)
    # Empty model == passthrough provider: /v1/models will surface the
    # upstream's loaded model id, and outbound requests omit the name.
    assert pconf.model == ""
    assert pconf.kind == "openai_compat"
    assert str(pconf.base_url) == "http://localhost:8085/v1"


def test_static_providers_and_adapters_survive_sync() -> None:
    engine = _engine()
    engine.register_provider(_launcher_provider_config("llama.cpp", 8085))

    assert "static-local" in engine._adapters
    assert any(p.name == "static-local" for p in engine.config.providers)


def test_custom_profile_name_is_respected() -> None:
    engine = _engine()
    summary = engine.register_provider(
        _launcher_provider_config("mlx", 9000), profile_name="bench"
    )
    assert summary["profile"] == "bench"
    assert engine.config.profile_by_name("bench").providers == [
        "launcher-mlx-9000"
    ]


@pytest.mark.parametrize(
    ("backend", "port", "expected"),
    [
        ("llama.cpp", 8085, "launcher-llamacpp-8085"),
        ("vllm", 8081, "launcher-vllm-8081"),
        ("mlx", 9000, "launcher-mlx-9000"),
    ],
)
def test_provider_naming(backend: str, port: int, expected: str) -> None:
    assert _launcher_provider_config(backend, port).name == expected
