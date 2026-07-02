"""Security tests for M14 — ingress hardening (body limit + launcher paths).

Covers the four defenses added under M14:

1. Request body size limit (DoS): an oversized ``Content-Length`` is rejected
   with 413, a normal body passes, and ``CODEROUTER_MAX_BODY_BYTES`` tunes the
   ceiling. Coexists with the H8 Host-validation middleware.
2. ``/api/launcher/suggest`` validates ``model_path`` against the configured
   ``model_dirs`` (path-traversal / info-disclosure): a path outside the
   configured dirs is 400, a path inside is accepted.
3. The stop endpoint suppresses ``ProcessLookupError`` when the child has
   already died, so it no longer surfaces a 500.
4. ``_tail_logs`` survives a newline-less flood that overruns the stream
   buffer instead of dying with ``LimitOverrunError``.

Shares the stubbed-``load_config`` + fresh-collector scaffolding with
``test_fix_h8_launcher_auth.py``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    LauncherConfig,
    ProviderConfig,
)
from coderouter.ingress.app import _parse_max_body_bytes, create_app
from coderouter.ingress.launcher_routes import (
    ManagedProcess,
    _resolve_within_model_dirs,
    _tail_logs,
)
from coderouter.metrics import uninstall_collector


def _make_config(model_dirs: list[str] | None = None) -> CodeRouterConfig:
    launcher = LauncherConfig(model_dirs=model_dirs) if model_dirs is not None else None
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


@pytest.fixture
def config() -> CodeRouterConfig:
    return _make_config()


def _client_for(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
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


@pytest.fixture
def client(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    yield from _client_for(config, monkeypatch)


# ---------------------------------------------------------------------------
# 1. Request body size limit (413)
# ---------------------------------------------------------------------------


def test_parse_max_body_bytes_defaults_and_overrides() -> None:
    """Env parsing falls back to the default for bad/empty values."""
    default = _parse_max_body_bytes(None)
    assert default == 64 * 1024 * 1024
    assert _parse_max_body_bytes("") == default
    assert _parse_max_body_bytes("not-a-number") == default
    assert _parse_max_body_bytes("0") == default
    assert _parse_max_body_bytes("-5") == default
    assert _parse_max_body_bytes("1024") == 1024


def test_oversized_body_rejected_with_413(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Content-Length above the configured cap returns 413."""
    monkeypatch.setenv("CODEROUTER_MAX_BODY_BYTES", "1024")
    gen = _client_for(config, monkeypatch)
    client = next(gen)
    try:
        big = b"x" * 4096
        resp = client.post(
            "/api/launcher/start",
            data=big,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 413, resp.text
        assert "too large" in resp.json()["detail"]
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_normal_body_passes_the_limit(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small body is not blocked by the size guard (reaches validation)."""
    monkeypatch.setenv("CODEROUTER_MAX_BODY_BYTES", "1048576")
    gen = _client_for(config, monkeypatch)
    client = next(gen)
    try:
        # Well-formed but intentionally-invalid backend so we exercise the
        # handler past the middleware without spawning a process. A 4xx that
        # is NOT 413 proves the body limit let it through.
        resp = client.post(
            "/api/launcher/start",
            json={
                "name": "t",
                "backend": "bogus",
                "model_path": "/tmp/x.gguf",
                "port": 9999,
            },
        )
        assert resp.status_code != 413, resp.text
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_body_limit_coexists_with_host_validation(client: TestClient) -> None:
    """Host validation still fires (403) even with the body middleware added."""
    resp = client.get("/healthz", headers={"host": "evil.example.com"})
    assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# 2. /api/launcher/suggest path traversal
# ---------------------------------------------------------------------------


def test_resolve_within_model_dirs_accepts_contained_path(tmp_path) -> None:
    """A file under a configured model dir resolves successfully."""
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    resolved = _resolve_within_model_dirs(str(model), [str(tmp_path)])
    assert resolved == model.resolve()


def test_resolve_within_model_dirs_rejects_outside_path(tmp_path) -> None:
    """A path outside every configured model dir raises ValueError."""
    with pytest.raises(ValueError):
        _resolve_within_model_dirs("/etc/passwd", [str(tmp_path)])


def test_resolve_within_model_dirs_rejects_traversal(tmp_path) -> None:
    """A ``..`` escape out of the model dir is caught after resolve()."""
    inside = tmp_path / "models"
    inside.mkdir()
    escape = str(inside / ".." / ".." / "etc" / "passwd")
    with pytest.raises(ValueError):
        _resolve_within_model_dirs(escape, [str(inside)])


def test_resolve_within_model_dirs_rejects_when_unconfigured() -> None:
    """No configured model_dirs → nothing can be validated → ValueError."""
    with pytest.raises(ValueError):
        _resolve_within_model_dirs("/tmp/x.gguf", [])


def test_suggest_rejects_traversal_over_http(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The suggest endpoint returns 400 for a path outside model_dirs."""
    (tmp_path / "m.gguf").write_bytes(b"gguf")
    cfg = _make_config(model_dirs=[str(tmp_path)])
    gen = _client_for(cfg, monkeypatch)
    client = next(gen)
    try:
        resp = client.get(
            "/api/launcher/suggest",
            params={"model_path": "/etc/passwd", "backend": "llama.cpp"},
        )
        assert resp.status_code == 400, resp.text
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_suggest_allows_path_inside_model_dirs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model file under a configured dir is accepted and sized."""
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 2048)
    cfg = _make_config(model_dirs=[str(tmp_path)])
    gen = _client_for(cfg, monkeypatch)
    client = next(gen)
    try:
        resp = client.get(
            "/api/launcher/suggest",
            params={"model_path": str(model), "backend": "llama.cpp"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["backend"] == "llama.cpp"
    finally:
        with pytest.raises(StopIteration):
            next(gen)


# ---------------------------------------------------------------------------
# 3. stop with an already-dead process must not 500
# ---------------------------------------------------------------------------


class _DeadProc:
    """Fake asyncio subprocess whose signals raise ProcessLookupError."""

    def __init__(self) -> None:
        self.returncode = None
        self.pid = 4242

    def terminate(self) -> None:
        raise ProcessLookupError("no such process")

    def kill(self) -> None:
        raise ProcessLookupError("no such process")

    async def wait(self) -> int:
        self.returncode = -15
        return -15


def test_stop_suppresses_process_lookup_error(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping a process that already exited returns 200, not 500."""
    gen = _client_for(config, monkeypatch)
    client = next(gen)
    try:
        proc = ManagedProcess(
            id="dead1234",
            name="dead",
            backend="llama.cpp",
            model_path="/tmp/m.gguf",
            port=9001,
            options={},
            extra_args="",
            status="running",
        )
        proc._proc = _DeadProc()
        # Register directly on the app's launcher registry.
        from coderouter.ingress.launcher_routes import LauncherRegistry

        reg = LauncherRegistry()
        reg.add(proc)
        client.app.state.launcher = reg

        resp = client.post("/api/launcher/stop/dead1234")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "stopped"
    finally:
        with pytest.raises(StopIteration):
            next(gen)


# ---------------------------------------------------------------------------
# 4. _tail_logs survives a newline-less overrun
# ---------------------------------------------------------------------------


class _OverrunStream:
    """StreamReader stub: readline() overruns once, then read() drains."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._readline_calls = 0

    async def readline(self) -> bytes:
        # First call: simulate a line longer than the buffer limit.
        if self._readline_calls == 0:
            self._readline_calls += 1
            raise asyncio.LimitOverrunError("chunk exceeds limit", 0)
        # After the chunk was drained, signal EOF.
        return b""

    async def read(self, n: int = -1) -> bytes:
        chunk, self._payload = self._payload, b""
        return chunk

    def at_eof(self) -> bool:
        return not self._payload


class _FakeProc:
    def __init__(self, stdout, stderr) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0
        self.pid = 1

    async def wait(self) -> int:
        return 0


def test_tail_logs_recovers_from_limit_overrun() -> None:
    """A newline-less flood is drained instead of killing the log task."""
    proc = ManagedProcess(
        id="ovr1",
        name="ovr",
        backend="llama.cpp",
        model_path="/tmp/m.gguf",
        port=9002,
        options={},
        extra_args="",
        status="running",
    )
    proc.log_tail = deque(maxlen=200)
    flood = b"A" * (300 * 1024)  # bigger than the 256 KB stream limit
    proc._proc = _FakeProc(_OverrunStream(flood), None)

    asyncio.run(_tail_logs(proc))

    # The drained chunk landed in the tail and the task completed cleanly.
    joined = "\n".join(proc.log_tail)
    assert "AAAA" in joined
    assert proc.status == "stopped"
    assert any("exited with code" in line for line in proc.log_tail)
