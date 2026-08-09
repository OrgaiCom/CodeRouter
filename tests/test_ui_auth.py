"""Tests for the opt-in metrics/dashboard token (v2.14.0).

Three things must hold, and the third is the one that is easy to get
wrong:

1. With no token configured, nothing changes — a Prometheus scrape that
   worked before the upgrade still works.
2. With a token configured, all three surfaces refuse an unauthenticated
   caller.
3. The check runs **before** the payload is assembled, and the page never
   receives the token itself. codex-router ships both of those bugs: its
   ``/health`` answers ahead of its own auth check and leaks the live
   session name, and (until its own fix) ``GET /launcher`` substituted the
   shared secret straight into the HTML.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from coderouter.ingress.app import create_app
from coderouter.ingress.ui_auth import METRICS_TOKEN_ENV, METRICS_TOKEN_HEADER

TOKEN = "metrics-token-abcdefghijklmnop"

PROVIDERS_YAML = """
providers:
  - name: local
    kind: openai_compat
    base_url: http://localhost:8080/v1
    model: ""
profiles:
  - name: default
    providers: [local]
default_profile: default
"""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    config = tmp_path / "providers.yaml"
    config.write_text(PROVIDERS_YAML, encoding="utf-8")
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    return TestClient(create_app(str(config)))


@pytest.fixture()
def secured_client(tmp_path, monkeypatch):
    config = tmp_path / "providers.yaml"
    config.write_text(PROVIDERS_YAML, encoding="utf-8")
    monkeypatch.setenv(METRICS_TOKEN_ENV, TOKEN)
    return TestClient(create_app(str(config)))


# ---------------------------------------------------------------------------
# Default: unchanged behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/metrics.json", "/metrics", "/dashboard"])
def test_endpoints_stay_open_when_no_token_is_configured(client, path: str) -> None:
    """Upgrading must not break an existing scrape or bookmark."""
    assert client.get(path).status_code == 200


def test_dashboard_reports_auth_off_when_unconfigured(client) -> None:
    body = client.get("/dashboard").text
    assert "const AUTH_REQUIRED = false;" in body


# ---------------------------------------------------------------------------
# Configured: every surface closes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/metrics.json", "/metrics", "/dashboard"])
def test_endpoints_reject_a_caller_without_the_token(secured_client, path: str) -> None:
    assert secured_client.get(path).status_code == 401


@pytest.mark.parametrize("path", ["/metrics.json", "/metrics", "/dashboard"])
def test_endpoints_accept_the_matching_token(secured_client, path: str) -> None:
    resp = secured_client.get(path, headers={METRICS_TOKEN_HEADER: TOKEN})
    assert resp.status_code == 200


@pytest.mark.parametrize("path", ["/metrics.json", "/metrics", "/dashboard"])
def test_a_wrong_token_is_rejected(secured_client, path: str) -> None:
    resp = secured_client.get(path, headers={METRICS_TOKEN_HEADER: "nope"})
    assert resp.status_code == 401


def test_token_is_not_accepted_as_a_query_parameter(secured_client) -> None:
    """A token in a URL lands in access logs and browser history."""
    assert secured_client.get(f"/metrics.json?token={TOKEN}").status_code == 401


# ---------------------------------------------------------------------------
# The two codex-router mistakes we are explicitly not repeating
# ---------------------------------------------------------------------------


def test_rejected_request_leaks_no_topology(secured_client) -> None:
    """The 401 must come before the payload is assembled, not after."""
    body = secured_client.get("/metrics.json").text
    assert "local" not in body
    assert "localhost:8080" not in body


def test_dashboard_html_never_contains_the_token(secured_client) -> None:
    resp = secured_client.get("/dashboard", headers={METRICS_TOKEN_HEADER: TOKEN})
    assert resp.status_code == 200
    assert TOKEN not in resp.text
    assert "const AUTH_REQUIRED = true;" in resp.text


def test_metrics_json_scrubs_a_credential_pasted_into_base_url(
    tmp_path, monkeypatch
) -> None:
    """This endpoint hands base_url to a browser — scrub it on the way out."""
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    config = tmp_path / "providers.yaml"
    config.write_text(
        """
providers:
  - name: leaky
    kind: openai_compat
    base_url: https://api.example.com/v1?api_key=abcdefghijklmnop
    model: ""
profiles:
  - name: default
    providers: [leaky]
default_profile: default
""",
        encoding="utf-8",
    )
    with TestClient(create_app(str(config))) as c:
        payload = c.get("/metrics.json").json()
    base_url = payload["config"]["providers"][0]["base_url"]
    assert "abcdefghijklmnop" not in base_url
    assert "[redacted:url-param]" in base_url
