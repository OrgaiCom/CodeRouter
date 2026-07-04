"""Ingress tests for POST /v1/messages/count_tokens (S1 shim).

Exercises the HTTP boundary of the local token-count endpoint: response
shape (``{"input_tokens": N}``), the char/4 heuristic value (integrates
with ``len(text) // 4``), tool JSON inclusion, and the loose validation
(``model`` + non-empty ``messages`` required, ``max_tokens`` not needed).

The endpoint is fully local — no engine round-trip — so we reuse the
``_RecordingEngine`` scaffolding from tests/test_ingress_anthropic.py
only to satisfy ``app.state.engine`` (the route never calls it).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import CodeRouterConfig
from coderouter.ingress.app import create_app
from tests.test_ingress_anthropic import _RecordingEngine, two_profile_config  # noqa: F401


@pytest.fixture
def client(
    two_profile_config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> TestClient:
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config",
        lambda path=None: two_profile_config,
    )
    app = create_app()
    app.state.engine = _RecordingEngine()
    app.state.config = two_profile_config
    return TestClient(app)


def test_count_tokens_returns_input_tokens_schema(client: TestClient) -> None:
    """200 with an ``{"input_tokens": int}`` body per the Anthropic spec."""
    resp = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert list(body.keys()) == ["input_tokens"]
    assert isinstance(body["input_tokens"], int)
    assert body["input_tokens"] >= 0


def test_count_tokens_heuristic_matches_char_over_4(client: TestClient) -> None:
    """With no provider tokenizer declared, the count is len(text) // 4.

    The two_profile_config providers declare no ``tokenizer_path``, so the
    endpoint uses the char/4 heuristic. ``system`` + message text are
    joined with a newline (see extract_text_from_anthropic_request).
    """
    system = "S" * 20
    user = "U" * 40
    # Joined text is system + "\n" + user → 20 + 1 + 40 = 61 chars → 15.
    resp = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "m",
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == (20 + 1 + 40) // 4


def test_count_tokens_no_max_tokens_required(client: TestClient) -> None:
    """count_tokens requests omit max_tokens — must not 4xx on its absence."""
    resp = client.post(
        "/v1/messages/count_tokens",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200


def test_count_tokens_includes_tool_json_length(client: TestClient) -> None:
    """Declared tools add their JSON length to the counted text, so the
    same messages yield a strictly higher count when tools are present."""
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    without = client.post("/v1/messages/count_tokens", json=base).json()["input_tokens"]
    with_tools = client.post(
        "/v1/messages/count_tokens",
        json={
            **base,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        },
    ).json()["input_tokens"]
    assert with_tools > without


def test_count_tokens_empty_messages_400(client: TestClient) -> None:
    """An empty messages list is rejected with 400 (loose validation)."""
    resp = client.post(
        "/v1/messages/count_tokens",
        json={"model": "m", "messages": []},
    )
    assert resp.status_code == 400


def test_count_tokens_missing_messages_400(client: TestClient) -> None:
    resp = client.post("/v1/messages/count_tokens", json={"model": "m"})
    assert resp.status_code == 400


def test_count_tokens_missing_model_400(client: TestClient) -> None:
    resp = client.post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400


def test_count_tokens_block_list_content_counted(client: TestClient) -> None:
    """Text blocks inside a list-form content are counted; non-text
    (image) blocks contribute nothing — same rule as the char/4 estimator."""
    text = "X" * 32
    resp = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://x/y.png"},
                        },
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 32 // 4
