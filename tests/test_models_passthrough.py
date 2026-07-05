"""Tests for /v1/models upstream passthrough (empty-model providers).

A provider whose ``model`` field is empty (the launcher_gui llama-server
setup) is a passthrough provider: the upstream decides which model is
loaded. /v1/models should surface the upstream's real model id(s) for such
providers so external benchmarks can tell loaded GGUFs apart, while keeping
the historic provider-name entries for everything else — and falling back
to the provider name whenever the upstream cannot be reached.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import CodeRouterConfig, FallbackChain, ProviderConfig
from coderouter.ingress import openai_routes
from coderouter.ingress.app import create_app


def _config(*providers: ProviderConfig) -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=list(providers),
        profiles=[FallbackChain(name="default", providers=[providers[0].name])],
    )


def _client(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    app = create_app()
    app.state.config = config
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    openai_routes._upstream_models_cache.clear()


def _passthrough_provider(name: str = "llama-cpp-local") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url="http://localhost:8080/v1",
        model="",  # passthrough — llama-server decides the model
    )


def _named_provider(name: str = "ollama-qwen") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url="http://localhost:11434/v1",
        model="qwen2.5-coder:7b",
    )


def test_empty_model_provider_surfaces_upstream_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(base_url: str) -> list[str]:
        assert base_url == "http://localhost:8080/v1"
        return ["qwen2.5-coder-7b-q4_k_m.gguf"]

    monkeypatch.setattr(openai_routes, "_fetch_upstream_model_ids", fake_fetch)
    client = _client(_config(_passthrough_provider()), monkeypatch)

    body = client.get("/v1/models").json()
    ids = [m["id"] for m in body["data"]]
    assert ids == ["qwen2.5-coder-7b-q4_k_m.gguf"]
    assert body["data"][0]["owned_by"] == "coderouter/llama-cpp-local"


def test_named_model_provider_keeps_historic_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_fetch(base_url: str) -> list[str]:  # pragma: no cover
        raise AssertionError("passthrough must not fire for named models")

    monkeypatch.setattr(openai_routes, "_fetch_upstream_model_ids", fail_fetch)
    client = _client(_config(_named_provider()), monkeypatch)

    body = client.get("/v1/models").json()
    assert [m["id"] for m in body["data"]] == ["ollama-qwen"]
    assert body["data"][0]["owned_by"] == "coderouter"


def test_upstream_failure_falls_back_to_provider_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(base_url: str) -> list[str]:
        raise ConnectionError("upstream down")

    monkeypatch.setattr(openai_routes, "_fetch_upstream_model_ids", fake_fetch)
    client = _client(_config(_passthrough_provider()), monkeypatch)

    body = client.get("/v1/models").json()
    assert [m["id"] for m in body["data"]] == ["llama-cpp-local"]
    assert body["data"][0]["owned_by"] == "coderouter"


def test_mixed_providers_mix_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(base_url: str) -> list[str]:
        return ["loaded-model.gguf"]

    monkeypatch.setattr(openai_routes, "_fetch_upstream_model_ids", fake_fetch)
    client = _client(
        _config(_passthrough_provider(), _named_provider()), monkeypatch
    )

    body = client.get("/v1/models").json()
    assert [m["id"] for m in body["data"]] == ["loaded-model.gguf", "ollama-qwen"]


def test_upstream_ids_are_ttl_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_fetch(base_url: str) -> list[str]:
        calls.append(base_url)
        return ["loaded-model.gguf"]

    monkeypatch.setattr(openai_routes, "_fetch_upstream_model_ids", fake_fetch)
    client = _client(_config(_passthrough_provider()), monkeypatch)

    client.get("/v1/models")
    client.get("/v1/models")
    assert len(calls) == 1, "second probe within the TTL must hit the cache"


def test_multiple_upstream_ids_all_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(base_url: str) -> list[str]:
        return ["model-a.gguf", "model-b.gguf"]

    monkeypatch.setattr(openai_routes, "_fetch_upstream_model_ids", fake_fetch)
    client = _client(_config(_passthrough_provider()), monkeypatch)

    body = client.get("/v1/models").json()
    assert [m["id"] for m in body["data"]] == ["model-a.gguf", "model-b.gguf"]
