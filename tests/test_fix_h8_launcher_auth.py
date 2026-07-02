"""Security tests for H8 — launcher auth + DNS-rebinding hardening.

Covers three defenses added for the unauthenticated process-launch surface:

1. Host-header validation middleware (DNS-rebinding protection): loopback
   and ``testserver`` Hosts pass, unknown Hosts get 403, and
   ``CODEROUTER_ALLOWED_HOSTS`` widens the allow-list.
2. Token auth on the state-changing launcher endpoints (start/stop/delete):
   unset token keeps the historical open behaviour, a set token demands a
   matching ``X-CodeRouter-Token`` header (401 otherwise).
3. ``_build_cmd`` rejects ``-m`` / ``--model`` re-specification via
   ``options`` or ``extra_args`` (ValueError → 400) so ``model_path`` is the
   only way to select a model.

Shares the stubbed-``load_config`` + fresh-collector scaffolding with
``test_dashboard_endpoint.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import CodeRouterConfig, FallbackChain, ProviderConfig
from coderouter.ingress.app import create_app
from coderouter.ingress.launcher_routes import _build_cmd
from coderouter.metrics import uninstall_collector


@pytest.fixture
def config() -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local",
                base_url="http://localhost:8080/v1",
                model="qwen-coder",
                paid=False,
            ),
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
    )


@pytest.fixture
def client(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """App with a stubbed load_config and a fresh metrics collector."""
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


# ---------------------------------------------------------------------------
# 1. Host-header validation middleware
# ---------------------------------------------------------------------------


def test_default_testserver_host_is_allowed(client: TestClient) -> None:
    """TestClient's default Host (``testserver``) must pass the middleware.

    This is what keeps the existing suite green — it is whitelisted so the
    hardening is transparent to local/CI test runs.
    """
    resp = client.get("/healthz")
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "localhost:8080", "127.0.0.1:9090", "[::1]:8080"],
)
def test_loopback_hosts_allowed(client: TestClient, host: str) -> None:
    """Loopback hostnames (with or without a port) are accepted."""
    resp = client.get("/healthz", headers={"host": host})
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize(
    "host",
    ["evil.example.com", "attacker.test", "coderouter.attacker.com:8080"],
)
def test_unknown_host_rejected_with_403(client: TestClient, host: str) -> None:
    """A non-loopback Host is rejected — the DNS-rebinding guard."""
    resp = client.get("/healthz", headers={"host": host})
    assert resp.status_code == 403, resp.text
    assert "not allowed" in resp.json()["detail"]


def test_host_validation_applies_to_all_routes(client: TestClient) -> None:
    """The guard is global — even the launcher API surface is protected."""
    resp = client.get(
        "/api/launcher/processes", headers={"host": "evil.example.com"}
    )
    assert resp.status_code == 403, resp.text


def test_allowed_hosts_env_widens_allowlist(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CODEROUTER_ALLOWED_HOSTS`` adds extra accepted hostnames."""
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    monkeypatch.setenv("CODEROUTER_ALLOWED_HOSTS", "coderouter.internal, box.lan")
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            ok = tc.get("/healthz", headers={"host": "coderouter.internal"})
            assert ok.status_code == 200, ok.text
            ok2 = tc.get("/healthz", headers={"host": "box.lan:8080"})
            assert ok2.status_code == 200, ok2.text
            bad = tc.get("/healthz", headers={"host": "other.lan"})
            assert bad.status_code == 403, bad.text
    finally:
        uninstall_collector()


# ---------------------------------------------------------------------------
# 2. Launcher token auth (start / stop / delete)
# ---------------------------------------------------------------------------


def test_stop_without_token_configured_is_open(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no token env, stop stays open (404 for missing id, not 401).

    The historical local-only behaviour must be preserved so nothing breaks
    for users who never set the env var.
    """
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    resp = client.post("/api/launcher/stop/does-not-exist")
    # Auth is disabled → we fall through to the normal 404 (unknown proc id).
    assert resp.status_code == 404, resp.text


def test_delete_without_token_configured_is_open(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same open behaviour for delete when no token is configured."""
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    resp = client.delete("/api/launcher/processes/does-not-exist")
    assert resp.status_code == 404, resp.text


def test_stop_with_token_set_requires_header(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the token is set, a missing header yields 401."""
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    resp = client.post("/api/launcher/stop/does-not-exist")
    assert resp.status_code == 401, resp.text


def test_stop_with_wrong_token_is_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mismatching token header yields 401."""
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    resp = client.post(
        "/api/launcher/stop/does-not-exist",
        headers={"X-CodeRouter-Token": "wrong"},
    )
    assert resp.status_code == 401, resp.text


def test_stop_with_correct_token_passes_auth(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching token clears auth — we then hit the normal 404 path."""
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    resp = client.post(
        "/api/launcher/stop/does-not-exist",
        headers={"X-CodeRouter-Token": "s3cret"},
    )
    assert resp.status_code == 404, resp.text


def test_delete_with_correct_token_passes_auth(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete accepts the correct token and proceeds to the 404 path."""
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    resp = client.delete(
        "/api/launcher/processes/does-not-exist",
        headers={"X-CodeRouter-Token": "s3cret"},
    )
    assert resp.status_code == 404, resp.text


def test_start_with_token_set_requires_header(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The start endpoint also demands the token when configured."""
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "x",
            "backend": "llama.cpp",
            "model_path": "/tmp/model.gguf",
            "port": 8080,
        },
    )
    assert resp.status_code == 401, resp.text


def test_launcher_page_injects_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTML must carry the configured token for the inline fetch calls."""
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    body = client.get("/launcher").text
    assert 'const LAUNCHER_TOKEN = "s3cret";' in body


def test_launcher_page_empty_token_by_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no token env the placeholder collapses to an empty string."""
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    body = client.get("/launcher").text
    assert 'const LAUNCHER_TOKEN = "";' in body


# ---------------------------------------------------------------------------
# 3. _build_cmd — model re-specification rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-m", "--model"])
def test_build_cmd_rejects_model_flag_in_extra_args(flag: str) -> None:
    """A model flag smuggled through extra_args raises ValueError."""
    with pytest.raises(ValueError, match="not allowed"):
        _build_cmd(
            "llama.cpp",
            "/models/good.gguf",
            8080,
            {},
            f"{flag} /models/evil.gguf",
        )


@pytest.mark.parametrize("flag", ["-m", "--model"])
def test_build_cmd_rejects_model_flag_in_options(flag: str) -> None:
    """A model flag smuggled through options keys raises ValueError."""
    with pytest.raises(ValueError, match="not allowed"):
        _build_cmd(
            "llama.cpp",
            "/models/good.gguf",
            8080,
            {flag: "/models/evil.gguf"},
            "",
        )


def test_build_cmd_rejects_equals_form_model_flag() -> None:
    """The ``--model=path`` form is caught by comparing before the '='."""
    with pytest.raises(ValueError, match="not allowed"):
        _build_cmd(
            "vllm",
            "/models/good.safetensors",
            8000,
            {},
            "--model=/models/evil.safetensors",
        )


def test_build_cmd_allows_benign_extra_args() -> None:
    """Non-model flags pass through untouched and land in the command."""
    cmd = _build_cmd(
        "llama.cpp",
        "/models/good.gguf",
        8080,
        {"--threads": 8},
        "-ngl 99",
    )
    assert cmd[0].endswith("llama-server")
    assert "-m" in cmd and "/models/good.gguf" in cmd
    assert "--threads" in cmd and "8" in cmd
    assert "-ngl" in cmd and "99" in cmd
    # The vetted model path appears exactly once.
    assert cmd.count("/models/good.gguf") == 1


def test_start_endpoint_returns_400_on_model_override(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a model override via the API surfaces as HTTP 400."""
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "x",
            "backend": "llama.cpp",
            "model_path": "/models/good.gguf",
            "port": 8080,
            "extra_args": "-m /models/evil.gguf",
        },
    )
    assert resp.status_code == 400, resp.text
    assert "not allowed" in resp.json()["detail"]
