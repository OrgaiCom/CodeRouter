"""Tests for launcher readiness gating and generic auto-restart.

Covers two launcher holes fixed alongside the existing MTP startup-crash
fallback (``tests/test_launcher_mtp_fallback.py``):

1. **Readiness gating** — a launched backend used to be registered as a
   routable provider the instant the OS process spawned, before llama-server
   / vllm had actually finished loading the model (``_backend_ready`` /
   ``_wait_ready_and_register``).
2. **Generic auto-restart** — besides the one-shot MTP fallback, a crashed
   launcher process had no supervision at all; it sat in status="error"
   forever (``_attempt_restart``, wired into ``_tail_logs``). Opt-in via
   ``LauncherConfig.auto_restart`` (default False — see schemas.py).

Sections:

* A. ``_backend_ready`` — the single-probe primitive (real localhost
  sockets, no mocking).
* B. ``_wait_ready_and_register`` — the polling loop, with ``_backend_ready``
  monkeypatched for determinism.
* C. ``_attempt_restart`` — the backoff/respawn primitive.
* D. End-to-end via ``POST /api/launcher/start`` against real subprocess
  stubs (mirrors the integration style of ``test_launcher_mtp_fallback.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    LauncherBackendConfig,
    LauncherConfig,
    ProviderConfig,
)
from coderouter.ingress.app import create_app
from coderouter.ingress.launcher_routes import (
    ManagedProcess,
    _attempt_restart,
    _backend_ready,
    _wait_ready_and_register,
)
from coderouter.metrics import uninstall_collector

# pyproject.toml sets asyncio_mode = "auto" (pytest-asyncio) — every
# `async def test_...` below runs directly, no decorator needed.


# ---------------------------------------------------------------------------
# A. _backend_ready — single-probe primitive
# ---------------------------------------------------------------------------


async def _serve_once(port: int, status_line: bytes) -> asyncio.AbstractServer:
    """Start a bare asyncio TCP server on 127.0.0.1:port for one response."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.read(4096)
            writer.write(status_line + b"\r\ncontent-length: 0\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", port)


async def test_backend_ready_llamacpp_true_on_health_200() -> None:
    server = await _serve_once(19180, b"HTTP/1.1 200 OK")
    try:
        assert await _backend_ready("llama.cpp", 19180, probe_timeout_s=2.0) is True
    finally:
        server.close()
        await server.wait_closed()


async def test_backend_ready_llamacpp_false_on_health_503() -> None:
    server = await _serve_once(19181, b"HTTP/1.1 503 Service Unavailable")
    try:
        assert await _backend_ready("llama.cpp", 19181, probe_timeout_s=2.0) is False
    finally:
        server.close()
        await server.wait_closed()


async def test_backend_ready_false_when_nothing_listening() -> None:
    # Nothing bound to this port — connection refused.
    assert await _backend_ready("llama.cpp", 19182, probe_timeout_s=2.0) is False


async def test_backend_ready_vllm_uses_health_endpoint_too() -> None:
    server = await _serve_once(19183, b"HTTP/1.1 200 OK")
    try:
        assert await _backend_ready("vllm", 19183, probe_timeout_s=2.0) is True
    finally:
        server.close()
        await server.wait_closed()


async def test_backend_ready_mlx_falls_back_to_tcp_connect() -> None:
    """mlx has no documented /health — a bare TCP connect is the fallback."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 19184)
    try:
        # No HTTP response is ever sent — only a successful connect matters.
        assert await _backend_ready("mlx", 19184, probe_timeout_s=2.0) is True
    finally:
        server.close()
        await server.wait_closed()


async def test_backend_ready_mlx_false_when_nothing_listening() -> None:
    assert await _backend_ready("mlx", 19185, probe_timeout_s=2.0) is False


# ---------------------------------------------------------------------------
# B. _wait_ready_and_register — polling loop
# ---------------------------------------------------------------------------


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def register_provider(self, provider: ProviderConfig, profile_name: str = "launcher") -> dict:
        self.calls.append((provider.name, str(provider.base_url)))
        return {"provider": provider.name, "profile": profile_name, "replaced": False}


def _fake_app(engine: _FakeEngine | None) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(engine=engine))


def _mp(**overrides: object) -> ManagedProcess:
    base: dict[str, object] = dict(
        id="r1",
        name="x",
        backend="llama.cpp",
        model_path="/m.gguf",
        port=19190,
        options={},
        extra_args="",
        status="loading",
        started_at=time.monotonic(),
    )
    base.update(overrides)
    proc = ManagedProcess(**base)  # type: ignore[arg-type]
    proc._proc = object()  # anything not-None: "the OS process is alive"
    return proc


async def test_wait_ready_registers_once_probe_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _always_ready(backend: str, port: int, *, probe_timeout_s: float) -> bool:
        return True

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes._backend_ready", _always_ready
    )
    engine = _FakeEngine()
    proc = _mp(restart_count=2)
    launcher_cfg = SimpleNamespace(readiness_timeout_s=5.0, readiness_poll_interval_s=0.01)

    await _wait_ready_and_register(proc, _fake_app(engine), launcher_cfg)

    assert proc.status == "running"
    assert proc.restart_count == 0
    assert engine.calls == [("launcher-llamacpp-19190", "http://localhost:19190/v1")]
    assert any("readiness check passed" in ln for ln in proc.log_tail)


async def test_wait_ready_times_out_marks_error_without_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _never_ready(backend: str, port: int, *, probe_timeout_s: float) -> bool:
        return False

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes._backend_ready", _never_ready
    )
    engine = _FakeEngine()
    proc = _mp()
    launcher_cfg = SimpleNamespace(readiness_timeout_s=0.05, readiness_poll_interval_s=0.01)

    await _wait_ready_and_register(proc, _fake_app(engine), launcher_cfg)

    assert proc.status == "error"
    assert engine.calls == []
    assert any("timed out" in ln for ln in proc.log_tail)


async def test_wait_ready_bails_when_process_already_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the process crashed/stopped before the first probe, never register."""
    probe_called = False

    async def _spy_ready(backend: str, port: int, *, probe_timeout_s: float) -> bool:
        nonlocal probe_called
        probe_called = True
        return True

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes._backend_ready", _spy_ready
    )
    engine = _FakeEngine()
    # Simulate _tail_logs already having handled a crash before this task runs.
    proc = _mp(status="error")
    launcher_cfg = SimpleNamespace(readiness_timeout_s=5.0, readiness_poll_interval_s=0.01)

    await _wait_ready_and_register(proc, _fake_app(engine), launcher_cfg)

    assert proc.status == "error"  # untouched
    assert engine.calls == []
    assert probe_called is False


async def test_wait_ready_uses_defaults_when_launcher_cfg_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``launcher:`` block in providers.yaml → module defaults apply."""

    async def _always_ready(backend: str, port: int, *, probe_timeout_s: float) -> bool:
        return True

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes._backend_ready", _always_ready
    )
    engine = _FakeEngine()
    proc = _mp()

    await _wait_ready_and_register(proc, _fake_app(engine), None)

    assert proc.status == "running"
    assert engine.calls == [("launcher-llamacpp-19190", "http://localhost:19190/v1")]


# ---------------------------------------------------------------------------
# C. _attempt_restart — backoff/respawn primitive
# ---------------------------------------------------------------------------


class _FakeRestartProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.stdout = None
        self.stderr = None
        self.returncode = None

    async def wait(self) -> int:
        return 0


async def test_attempt_restart_disabled_by_default() -> None:
    proc = _mp(cmd=["fake-server", "--port", "19190"])
    # launcher_cfg=None → getattr(..., "auto_restart", False) is False.
    assert await _attempt_restart(proc, None) is False
    assert proc.restart_count == 0


async def test_attempt_restart_false_when_flag_explicitly_off() -> None:
    proc = _mp(cmd=["fake-server", "--port", "19190"])
    cfg = SimpleNamespace(auto_restart=False)
    assert await _attempt_restart(proc, cfg) is False


async def test_attempt_restart_respects_max_attempts() -> None:
    proc = _mp(cmd=["fake-server", "--port", "19190"], restart_count=2)
    cfg = SimpleNamespace(
        auto_restart=True,
        auto_restart_max_attempts=2,
        auto_restart_backoff_s=0.001,
        auto_restart_backoff_max_s=0.001,
    )
    assert await _attempt_restart(proc, cfg) is False
    assert proc.restart_count == 2  # unchanged — budget already exhausted
    assert any("giving up" in ln for ln in proc.log_tail)


async def test_attempt_restart_respawns_and_increments_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_exec(*_args: object, **_kwargs: object) -> _FakeRestartProc:
        return _FakeRestartProc(pid=9911)

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes.asyncio.create_subprocess_exec",
        _fake_exec,
    )
    proc = _mp(cmd=["fake-server", "--port", "19190"])
    cfg = SimpleNamespace(
        auto_restart=True,
        auto_restart_max_attempts=3,
        auto_restart_backoff_s=0.001,
        auto_restart_backoff_max_s=0.001,
    )

    ok = await _attempt_restart(proc, cfg)

    assert ok is True
    assert proc.restart_count == 1
    assert proc._proc.pid == 9911
    assert proc.pid == 9911
    assert any("auto-restart started PID 9911" in ln for ln in proc.log_tail)


async def test_attempt_restart_no_cmd_returns_false() -> None:
    proc = _mp(cmd=[])
    cfg = SimpleNamespace(auto_restart=True, auto_restart_max_attempts=3)
    assert await _attempt_restart(proc, cfg) is False


async def test_attempt_restart_skips_when_stopped_during_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _mp(cmd=["fake-server", "--port", "19190"])
    cfg = SimpleNamespace(
        auto_restart=True,
        auto_restart_max_attempts=3,
        auto_restart_backoff_s=0.001,
        auto_restart_backoff_max_s=0.001,
    )

    async def _sleep_then_stop(_seconds: float) -> None:
        proc.stopping = True

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes.asyncio.sleep", _sleep_then_stop
    )
    exec_called = False

    async def _fake_exec(*_args: object, **_kwargs: object) -> _FakeRestartProc:
        nonlocal exec_called
        exec_called = True
        return _FakeRestartProc(pid=1)

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes.asyncio.create_subprocess_exec",
        _fake_exec,
    )

    assert await _attempt_restart(proc, cfg) is False
    assert exec_called is False


# ---------------------------------------------------------------------------
# D. End-to-end via POST /api/launcher/start (real subprocess stubs)
# ---------------------------------------------------------------------------


def _write_script(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


_HEALTH_SERVER_BODY = """
import http.server
import sys
import time

def _port():
    argv = sys.argv[1:]
    return int(argv[argv.index("--port") + 1])

_ready_at = time.monotonic() + 0.3

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            if time.monotonic() < _ready_at:
                self.send_response(503)
            else:
                self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass

httpd = http.server.HTTPServer(("127.0.0.1", _port()), Handler)
httpd.serve_forever()
"""

_GRACEFUL_LONG_RUNNING_BODY = """
import signal
import sys
import time

def _handler(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, _handler)
while True:
    time.sleep(0.05)
"""

_CRASH_BODY = """
import sys
sys.stderr.write("boom\\n")
sys.exit(11)
"""


@pytest.fixture
def config(tmp_path: Path) -> CodeRouterConfig:
    # The stub server's --port arg is read at spawn time, so one script
    # works for every port a test in this module picks.
    script = _write_script(tmp_path / "health-llama-server", _HEALTH_SERVER_BODY)
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
        launcher=LauncherConfig(
            backends={"llama.cpp": LauncherBackendConfig(binary=str(script))},
            readiness_timeout_s=5.0,
            readiness_poll_interval_s=0.2,
        ),
    )


@contextlib.contextmanager
def _client_with_config(
    cfg: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr("coderouter.ingress.app.load_config", lambda path=None: cfg)
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


@pytest.fixture
def client(config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with _client_with_config(config, monkeypatch) as tc:
        yield tc


def _poll(fn, timeout: float = 5.0, interval: float = 0.05):
    """Poll ``fn()`` until it returns truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    result = fn()
    while not result and time.monotonic() < deadline:
        time.sleep(interval)
        result = fn()
    return result


def test_start_stays_loading_then_becomes_running_and_registers(
    client: TestClient,
) -> None:
    """End-to-end readiness gating: loading -> running, provider registered."""
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "slow",
            "backend": "llama.cpp",
            "model_path": "/tmp/does-not-need-to-exist.gguf",
            "port": 19191,
            "mtp_mode": "off",
        },
    )
    assert resp.status_code == 200, resp.text
    proc_id = resp.json()["id"]
    # provider_sync in the response is None now — registration is async.
    assert resp.json()["provider_sync"] is None

    def _status() -> str:
        procs = client.get("/api/launcher/processes").json()["processes"]
        return next(p for p in procs if p["id"] == proc_id)["status"]

    # Immediately after start it must not already claim "running" — the
    # bug this closes is registering/declaring ready before the backend
    # can actually serve.
    first_status = _status()
    assert first_status in ("loading", "starting", "running")

    ok = _poll(lambda: _status() == "running", timeout=5.0)
    assert ok, f"never reached running, last status={_status()!r}"

    chain = client.app.state.engine.config.profile_by_name("launcher")
    assert "launcher-llamacpp-19191" in chain.providers

    with contextlib.suppress(Exception):
        client.post(f"/api/launcher/stop/{proc_id}")


def test_stop_while_loading_prevents_registration(
    client: TestClient,
) -> None:
    """Stopping before readiness must not leave a stale provider registered."""
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "slow2",
            "backend": "llama.cpp",
            "model_path": "/tmp/does-not-need-to-exist.gguf",
            "port": 19192,
            "mtp_mode": "off",
        },
    )
    assert resp.status_code == 200, resp.text
    proc_id = resp.json()["id"]

    # Stop immediately, before the 0.3s health-server warm-up completes.
    stop_resp = client.post(f"/api/launcher/stop/{proc_id}")
    assert stop_resp.status_code == 200, stop_resp.text
    assert stop_resp.json()["status"] == "stopped"

    time.sleep(0.5)  # let the would-be readiness deadline pass
    chain = client.app.state.engine.config.profile_by_name("launcher") if _has_launcher_profile(client) else None
    if chain is not None:
        assert "launcher-llamacpp-19192" not in chain.providers
    procs = client.get("/api/launcher/processes").json()["processes"]
    entry = next(p for p in procs if p["id"] == proc_id)
    assert entry["status"] == "stopped"


def _has_launcher_profile(client: TestClient) -> bool:
    try:
        client.app.state.engine.config.profile_by_name("launcher")
        return True
    except KeyError:
        return False


def test_auto_restart_disabled_by_default_leaves_crashed_process_in_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _write_script(tmp_path / "crash-llama-server", _CRASH_BODY)
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local", base_url="http://localhost:8080/v1", model="qwen-coder"
            ),
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            backends={"llama.cpp": LauncherBackendConfig(binary=str(script))},
        ),
    )
    with _client_with_config(cfg, monkeypatch) as tc:
        resp = tc.post(
            "/api/launcher/start",
            json={
                "name": "crashy",
                "backend": "llama.cpp",
                "model_path": "/tmp/m.gguf",
                "port": 19193,
                "mtp_mode": "off",
            },
        )
        assert resp.status_code == 200, resp.text
        proc_id = resp.json()["id"]

        def _logs() -> list[str]:
            return tc.get(f"/api/launcher/logs/{proc_id}?n=200").json()["logs"]

        _poll(lambda: any("exited with code" in ln for ln in _logs()), timeout=5.0)
        time.sleep(0.3)  # give any (wrongly-firing) restart a chance to show up
        logs = _logs()
        assert not any("auto-restart" in ln for ln in logs), logs
        procs = tc.get("/api/launcher/processes").json()["processes"]
        entry = next(p for p in procs if p["id"] == proc_id)
        assert entry["status"] == "error"


def test_auto_restart_enabled_retries_then_gives_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _write_script(tmp_path / "crash-llama-server", _CRASH_BODY)
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local", base_url="http://localhost:8080/v1", model="qwen-coder"
            ),
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            backends={"llama.cpp": LauncherBackendConfig(binary=str(script))},
            auto_restart=True,
            auto_restart_max_attempts=2,
            auto_restart_backoff_s=0.1,
            auto_restart_backoff_max_s=1.0,
        ),
    )
    with _client_with_config(cfg, monkeypatch) as tc:
        resp = tc.post(
            "/api/launcher/start",
            json={
                "name": "crashy2",
                "backend": "llama.cpp",
                "model_path": "/tmp/m.gguf",
                "port": 19194,
                "mtp_mode": "off",
            },
        )
        assert resp.status_code == 200, resp.text
        proc_id = resp.json()["id"]

        def _logs() -> list[str]:
            return tc.get(f"/api/launcher/logs/{proc_id}?n=200").json()["logs"]

        _poll(lambda: any("giving up" in ln for ln in _logs()), timeout=5.0)
        logs = _logs()
        joined = "\n".join(logs)
        assert "auto-restart attempt 1/2" in joined, joined
        assert "auto-restart attempt 2/2" in joined, joined
        assert "giving up" in joined, joined
        procs = tc.get("/api/launcher/processes").json()["processes"]
        entry = next(p for p in procs if p["id"] == proc_id)
        assert entry["status"] == "error"


def test_intentional_stop_does_not_trigger_auto_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement (a): a deliberate Stop must never be treated as a crash."""
    script = _write_script(
        tmp_path / "graceful-llama-server", _GRACEFUL_LONG_RUNNING_BODY
    )
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local", base_url="http://localhost:8080/v1", model="qwen-coder"
            ),
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            backends={"llama.cpp": LauncherBackendConfig(binary=str(script))},
            auto_restart=True,
            auto_restart_max_attempts=3,
            auto_restart_backoff_s=0.1,
            auto_restart_backoff_max_s=1.0,
        ),
    )
    with _client_with_config(cfg, monkeypatch) as tc:
        resp = tc.post(
            "/api/launcher/start",
            json={
                "name": "graceful",
                "backend": "llama.cpp",
                "model_path": "/tmp/m.gguf",
                "port": 19195,
                "mtp_mode": "off",
            },
        )
        assert resp.status_code == 200, resp.text
        proc_id = resp.json()["id"]

        # Let it establish itself as "loading" (it never opens /health, so it
        # will never reach "running" — that's fine, we're testing stop here).
        time.sleep(0.2)

        stop_resp = tc.post(f"/api/launcher/stop/{proc_id}")
        assert stop_resp.status_code == 200, stop_resp.text
        assert stop_resp.json()["status"] == "stopped"

        # Give a wrongly-firing auto-restart a generous window to appear.
        time.sleep(0.5)

        def _logs() -> list[str]:
            return tc.get(f"/api/launcher/logs/{proc_id}?n=200").json()["logs"]

        logs = _logs()
        assert not any("auto-restart" in ln for ln in logs), logs
        procs = tc.get("/api/launcher/processes").json()["processes"]
        entry = next(p for p in procs if p["id"] == proc_id)
        assert entry["status"] == "stopped"
