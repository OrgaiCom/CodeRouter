"""Regression tests for the v2.13.0 launcher hardening batch.

Covers four fixes layered on the launcher surface:

* **H-2** — the sweep API no longer accepts a raw ``bench_command`` /
  ``results_dir`` from the request. The bench command lives in
  ``launcher.bench.presets`` and the request may only name a preset key.
  Deprecated fields fail closed with 400; an unknown key is 400.
* **H-3** — the launcher token is never embedded in the served HTML; only a
  boolean ``AUTH_REQUIRED`` flag is. Auth is still enforced server-side.
* **M-1** — ``/api/launcher/start`` and ``/sweep/start`` validate
  ``model_path`` (and ``draft_model_path``) against ``launcher.model_dirs``.
  The path is validated only — never rewritten — so argv is unchanged.
* **M-2** — process-row action buttons carry ``data-*`` attributes dispatched
  through one delegated listener instead of inline ``onclick`` with an
  interpolated process name (XSS).
* **M-3** — the body-size guard is a pure-ASGI middleware that counts the bytes
  actually received, so a chunked request with no ``Content-Length`` can no
  longer bypass the cap.

Reuses the stubbed-``load_config`` + fresh-collector scaffolding from
``test_fix_h8_launcher_auth.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    LauncherBenchConfig,
    LauncherBenchPreset,
    LauncherConfig,
    ProviderConfig,
)
from coderouter.ingress import launcher_routes
from coderouter.ingress.app import BodySizeLimitMiddleware, create_app
from coderouter.ingress.launcher_routes import _LAUNCHER_HTML, _resolve_bench
from coderouter.launcher_devices import render_bench_command
from coderouter.metrics import uninstall_collector

_TOKEN_ENV = "CODEROUTER_LAUNCHER_TOKEN"


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------


def _make_config(
    *,
    model_dirs: list[str] | None,
    presets: dict[str, LauncherBenchPreset] | None = None,
    default_preset: str | None = None,
) -> CodeRouterConfig:
    bench = LauncherBenchConfig(
        runs=5,
        readiness_timeout_s=5.0,
        presets=presets or {},
        default_preset=default_preset,
    )
    launcher = LauncherConfig(model_dirs=model_dirs, bench=bench)
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
        launcher=launcher,
    )


def _client_for(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    monkeypatch.delenv(_TOKEN_ENV, raising=False)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


class _SpawnRecorder:
    """Records the exact model_path handed to spawn_process (no real process)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def spawn_process(
        self,
        app: Any,
        launcher_cfg: Any,
        *,
        name: str,
        backend: str,
        model_path: str,
        port: int,
        options: dict[str, Any] | None = None,
        extra_args: str = "",
        draft_model_path: str | None = None,
        mtp_mode: str = "auto",
        swap_managed: bool = False,
        swap_model: str | None = None,
        device_args: list[str] | None = None,
    ) -> launcher_routes.ManagedProcess:
        self.calls.append(
            {"model_path": model_path, "draft_model_path": draft_model_path}
        )
        proc = launcher_routes.ManagedProcess(
            id=f"proc-{len(self.calls)}",
            name=name,
            backend=backend,
            model_path=model_path,
            port=port,
            options=options or {},
            extra_args=extra_args,
            status="running",
        )
        launcher_routes._registry_for_app(app).add(proc)
        return proc


def _start_body(model_file: str, **over: Any) -> dict[str, Any]:
    body = {
        "name": "t",
        "backend": "llama.cpp",
        "model_path": model_file,
        "port": 9191,
    }
    body.update(over)
    return body


def _sweep_body(model_path: str, **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "backend": "llama.cpp",
        "model_path": model_path,
        "port": 18190,
        "configs": [{"label": "x", "device_ids": ["CUDA0"]}],
    }
    body.update(over)
    return body


# ---------------------------------------------------------------------------
# H-2: bench preset (deprecated fields rejected, unknown key rejected)
# ---------------------------------------------------------------------------


def test_sweep_rejects_bench_command_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _make_config(model_dirs=[str(tmp_path)])
    client = next(gen := _client_for(config, monkeypatch))
    try:
        resp = client.post(
            "/api/launcher/sweep/start",
            json=_sweep_body(str(tmp_path / "m.gguf"), bench_command="rm -rf /"),
        )
        assert resp.status_code == 400, resp.text
        assert "no longer accepted" in resp.json()["detail"]
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_sweep_rejects_results_dir_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _make_config(model_dirs=[str(tmp_path)])
    client = next(gen := _client_for(config, monkeypatch))
    try:
        resp = client.post(
            "/api/launcher/sweep/start",
            json=_sweep_body(str(tmp_path / "m.gguf"), results_dir="/etc"),
        )
        assert resp.status_code == 400, resp.text
        assert "no longer accepted" in resp.json()["detail"]
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_sweep_rejects_unknown_preset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _make_config(
        model_dirs=[str(tmp_path)],
        presets={"fast": LauncherBenchPreset(name="fast", command_template="true")},
    )
    client = next(gen := _client_for(config, monkeypatch))
    try:
        resp = client.post(
            "/api/launcher/sweep/start",
            json=_sweep_body(str(tmp_path / "m.gguf"), bench_preset="nope"),
        )
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "Unknown bench preset" in detail and "fast" in detail
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_sweep_uses_only_config_declared_command() -> None:
    """The command a preset key resolves to comes from config, not the request.

    The request body has no field capable of carrying a command template (the
    endpoint tests above prove the deprecated ones 400), so ``_resolve_bench``
    is the whole surface — it returns the config-declared template verbatim.
    """
    config = _make_config(
        model_dirs=["/tmp"],
        presets={
            "cuda": LauncherBenchPreset(
                name="CUDA bench",
                command_template="llmbench run --tag {config} --runs {runs}",
                runs=9,
            )
        },
    )
    resolved = _resolve_bench(config.launcher, "cuda")
    assert resolved.key == "cuda"
    assert resolved.command_template == "llmbench run --tag {config} --runs {runs}"
    assert resolved.runs == 9


def test_sweep_default_preset_matches_v2120_argv() -> None:
    """With no preset and no config, the default reproduces the v2.12.0 argv."""
    resolved = _resolve_bench(None, None)
    assert resolved.key == "default"
    assert resolved.command_template == "llmbench run --model local-openai --runs {runs}"
    assert resolved.runs == 5
    argv = render_bench_command(
        resolved.command_template, port=8090, config_label="c", runs=resolved.runs
    )
    assert argv == ["llmbench", "run", "--model", "local-openai", "--runs", "5"]


def test_bench_presets_endpoint_hides_command_template(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /bench-presets exposes keys/names but never the command_template."""
    config = _make_config(
        model_dirs=["/tmp"],
        presets={
            "secret": LauncherBenchPreset(
                name="secret", command_template="/home/op/private/bench --key XYZ"
            )
        },
        default_preset="secret",
    )
    client = next(gen := _client_for(config, monkeypatch))
    try:
        data = client.get("/api/launcher/bench-presets").json()
        assert data["default"] == "secret"
        keys = {p["key"] for p in data["presets"]}
        assert "secret" in keys and "default" in keys  # implicit default prepended
        # no entry carries the actual command template, and the private path /
        # secret token never leave the server.
        assert all("command_template" not in p for p in data["presets"])
        blob = str(data)
        assert "/home/op/private/bench" not in blob
        assert "XYZ" not in blob
    finally:
        with pytest.raises(StopIteration):
            next(gen)


# ---------------------------------------------------------------------------
# H-3: token never embedded, auth still enforced
# ---------------------------------------------------------------------------


def test_launcher_html_never_contains_the_token(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(model_dirs=["/tmp"])
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    monkeypatch.setenv(_TOKEN_ENV, "top-secret-value")
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as client:
            body = client.get("/launcher").text
    finally:
        uninstall_collector()
    assert "top-secret-value" not in body
    assert "LAUNCHER_TOKEN =" not in body  # old injection point is gone


def test_launcher_html_reports_auth_required_flag_only(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(model_dirs=["/tmp"])
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    # auth on
    monkeypatch.setenv(_TOKEN_ENV, "abc")
    uninstall_collector()
    app = create_app()
    with TestClient(app) as client:
        on = client.get("/launcher").text
    uninstall_collector()
    assert "const AUTH_REQUIRED = true;" in on
    # auth off
    monkeypatch.delenv(_TOKEN_ENV, raising=False)
    uninstall_collector()
    app = create_app()
    with TestClient(app) as client:
        off = client.get("/launcher").text
    uninstall_collector()
    assert "const AUTH_REQUIRED = false;" in off


def test_token_still_enforced_on_state_changing_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _make_config(model_dirs=[str(tmp_path)])
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    monkeypatch.setenv(_TOKEN_ENV, "sekret")
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as client:
            model = tmp_path / "m.gguf"
            model.write_bytes(b"GGUF")
            # no header → 401
            no_hdr = client.post("/api/launcher/start", json=_start_body(str(model)))
            assert no_hdr.status_code == 401, no_hdr.text
            # wrong header → 401
            bad = client.post(
                "/api/launcher/start",
                json=_start_body(str(model)),
                headers={"X-CodeRouter-Token": "nope"},
            )
            assert bad.status_code == 401, bad.text
            # correct header → passes auth (any non-401 proves the gate opened)
            recorder = _SpawnRecorder()
            monkeypatch.setattr(
                launcher_routes, "spawn_process", recorder.spawn_process
            )
            ok = client.post(
                "/api/launcher/start",
                json=_start_body(str(model)),
                headers={"X-CodeRouter-Token": "sekret"},
            )
            assert ok.status_code != 401, ok.text
    finally:
        uninstall_collector()


# ---------------------------------------------------------------------------
# M-1: model_path traversal validation on start / sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    ["/etc/passwd", "../../../../etc/shadow", "/root/.ssh/id_rsa"],
)
def test_start_rejects_model_path_outside_model_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, bad_path: str
) -> None:
    config = _make_config(model_dirs=[str(tmp_path)])
    client = next(gen := _client_for(config, monkeypatch))
    try:
        resp = client.post("/api/launcher/start", json=_start_body(bad_path))
        assert resp.status_code == 400, resp.text
        assert "model_path" in resp.json()["detail"]
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_start_rejects_draft_model_path_outside_model_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _make_config(model_dirs=[str(tmp_path)])
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    client = next(gen := _client_for(config, monkeypatch))
    try:
        resp = client.post(
            "/api/launcher/start",
            json=_start_body(str(model), draft_model_path="/etc/passwd"),
        )
        assert resp.status_code == 400, resp.text
        assert "draft_model_path" in resp.json()["detail"]
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_sweep_rejects_model_path_outside_model_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _make_config(model_dirs=[str(tmp_path)])
    client = next(gen := _client_for(config, monkeypatch))
    try:
        resp = client.post(
            "/api/launcher/sweep/start", json=_sweep_body("/etc/passwd")
        )
        assert resp.status_code == 400, resp.text
        assert "model_path" in resp.json()["detail"]
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_start_allows_path_inside_model_dirs_and_argv_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A contained path passes and reaches spawn UNCHANGED (validated, not rewritten)."""
    config = _make_config(model_dirs=[str(tmp_path)])
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    recorder = _SpawnRecorder()
    monkeypatch.setattr(launcher_routes, "spawn_process", recorder.spawn_process)
    client = next(gen := _client_for(config, monkeypatch))
    try:
        resp = client.post("/api/launcher/start", json=_start_body(str(model)))
        assert resp.status_code == 200, resp.text
        assert len(recorder.calls) == 1
        # The exact string is handed downstream; no resolve()/symlink rewrite.
        assert recorder.calls[0]["model_path"] == str(model)
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_start_open_when_model_dirs_unconfigured(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """With model_dirs empty the path check is a no-op (historical behaviour)."""
    config = _make_config(model_dirs=[])
    recorder = _SpawnRecorder()
    monkeypatch.setattr(launcher_routes, "spawn_process", recorder.spawn_process)
    client = next(gen := _client_for(config, monkeypatch))
    try:
        resp = client.post(
            "/api/launcher/start", json=_start_body("/anywhere/on/disk/m.gguf")
        )
        assert resp.status_code == 200, resp.text
        assert recorder.calls[0]["model_path"] == "/anywhere/on/disk/m.gguf"
    finally:
        with pytest.raises(StopIteration):
            next(gen)


# ---------------------------------------------------------------------------
# M-2: XSS — no inline onclick, data-* + delegation (template-level assertions)
# ---------------------------------------------------------------------------


def test_process_row_has_no_inline_onclick() -> None:
    assert 'onclick="stopProc' not in _LAUNCHER_HTML
    assert 'onclick="deleteProc' not in _LAUNCHER_HTML
    assert 'onclick="openLog' not in _LAUNCHER_HTML


def test_render_processes_template_uses_data_attributes() -> None:
    assert 'data-act="stop"' in _LAUNCHER_HTML
    assert 'data-act="del"' in _LAUNCHER_HTML
    assert 'data-act="log"' in _LAUNCHER_HTML
    # a single delegated listener on the process table dispatches the actions
    assert 'getElementById("proc-table").addEventListener' in _LAUNCHER_HTML


def test_process_name_with_quote_survives_json_roundtrip() -> None:
    """The process name is only ever emitted through esc() into a data-* attr,
    never interpolated into an inline handler string (which a ``'`` could break
    out of). JS execution can't be asserted here, so we pin the template shape.
    """
    assert 'data-name="${esc(p.name)}"' in _LAUNCHER_HTML
    # the old, breakable inline form must be gone
    assert "openLog('${p.id}'" not in _LAUNCHER_HTML


# ---------------------------------------------------------------------------
# M-3: chunked body size guard (pure-ASGI middleware, byte counting)
# ---------------------------------------------------------------------------


def _drive_body_guard(
    max_bytes: int,
    headers: list[tuple[bytes, bytes]],
    chunks: list[bytes],
) -> int:
    """Run BodySizeLimitMiddleware over a synthetic ASGI request; return status.

    The downstream app drains the whole request body (via the wrapped receive)
    and returns 200 — so any 413 comes from the guard, not the app.
    """

    async def downstream(scope: Any, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body"):
                break
            if message["type"] == "http.disconnect":
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = BodySizeLimitMiddleware(downstream, max_bytes=max_bytes)

    messages: list[dict[str, Any]] = []
    if chunks:
        for i, chunk in enumerate(chunks):
            messages.append(
                {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": i < len(chunks) - 1,
                }
            )
    else:
        messages.append({"type": "http.request", "body": b"", "more_body": False})
    idx = 0

    async def receive() -> dict[str, Any]:
        nonlocal idx
        if idx < len(messages):
            msg = messages[idx]
            idx += 1
            return msg
        return {"type": "http.disconnect"}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "POST", "headers": headers}
    asyncio.run(mw(scope, receive, send))

    for message in sent:
        if message["type"] == "http.response.start":
            return int(message["status"])
    raise AssertionError("no response.start emitted")


def test_chunked_body_over_limit_rejected_with_413() -> None:
    # No Content-Length header (chunked) + total bytes over the cap → 413.
    status = _drive_body_guard(
        max_bytes=1024, headers=[], chunks=[b"x" * 600, b"y" * 600]
    )
    assert status == 413


def test_chunked_body_under_limit_passes() -> None:
    status = _drive_body_guard(
        max_bytes=1024, headers=[], chunks=[b"x" * 100, b"y" * 100]
    )
    assert status == 200


def test_content_length_over_limit_still_413() -> None:
    status = _drive_body_guard(
        max_bytes=1024,
        headers=[(b"content-length", b"5000")],
        chunks=[b"z" * 10],  # short body; header alone triggers the reject
    )
    assert status == 413


def test_malformed_content_length_is_not_a_bypass() -> None:
    # A garbage Content-Length used to make ``declared`` unparseable and slip
    # through; the byte counter now still catches the oversized body.
    status = _drive_body_guard(
        max_bytes=1024,
        headers=[(b"content-length", b"not-a-number")],
        chunks=[b"x" * 4096],
    )
    assert status == 413


def test_get_without_body_unaffected() -> None:
    status = _drive_body_guard(max_bytes=1024, headers=[], chunks=[])
    assert status == 200
