"""Tests for the one-shot MTP startup-crash auto-fallback in the launcher.

When speculative flags were added by AUTO detection (``mtp_mode="auto"``, no
explicit ``draft_model_path``, detection actually emitted flags) and the
backend process dies during startup, the launcher relaunches it ONCE without
the speculative flags — some architectures' ``draft-mtp`` support in
llama.cpp is immature and crashes the context init at load. This is never
done for explicit draft models / operator-supplied ``--spec-type``, and never
more than once.

Covers:

* the :func:`_should_mtp_fallback` truth table (unit), and
* the end-to-end retry via ``POST /api/launcher/start`` against a crashing
  backend stub (integration), including the negative ``mtp_mode="off"`` case.
"""

from __future__ import annotations

import struct
import time
from collections.abc import Iterator
from pathlib import Path

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
    _MTP_FALLBACK_WINDOW_SECS,
    ManagedProcess,
    _should_mtp_fallback,
)
from coderouter.metrics import uninstall_collector

# ---------------------------------------------------------------------------
# Synthetic GGUF builder (mirrors tests/test_launcher_mtp.py helpers)
# ---------------------------------------------------------------------------

_MAGIC = b"GGUF"
_T_UINT32 = 4
_T_STRING = 8


def _gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", _T_STRING) + _gguf_string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<I", _T_UINT32) + struct.pack("<I", value)


def _gguf_bytes(arch: str, *, nextn: int | None = None) -> bytes:
    kvs = [_kv_string("general.architecture", arch)]
    if nextn is not None:
        kvs.append(_kv_u32(f"{arch}.nextn_predict_layers", nextn))
    header = _MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 0)
    header += struct.pack("<Q", len(kvs))
    return header + b"".join(kvs)


def _make_gguf(
    path: Path, arch: str, *, nextn: int | None = None, size: int = 0
) -> Path:
    data = _gguf_bytes(arch, nextn=nextn)
    if size > len(data):
        data = data + b"\x00" * (size - len(data))
    path.write_bytes(data)
    return path


def _make_crashing_binary(path: Path) -> Path:
    """Write an executable shell stub that prints to stderr and exits 11."""
    path.write_text(
        "#!/bin/sh\n"
        'echo "llama_init_from_model: failed to initialize the context" 1>&2\n'
        "exit 11\n"
    )
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# _should_mtp_fallback — truth table
# ---------------------------------------------------------------------------


def _mp(**overrides: object) -> ManagedProcess:
    """Build a ManagedProcess primed for the fallback-eligible baseline."""
    base: dict[str, object] = dict(
        id="p1",
        name="x",
        backend="llama.cpp",
        model_path="/m.gguf",
        port=8080,
        options={},
        extra_args="",
        spec_auto=True,
        mtp_fallback_done=False,
        fallback_cmd=["llama-server", "-m", "/m.gguf", "--port", "8080"],
        returncode=11,
        started_at=time.monotonic(),
    )
    base.update(overrides)
    return ManagedProcess(**base)  # type: ignore[arg-type]


def test_should_fallback_auto_nonzero_in_window() -> None:
    assert _should_mtp_fallback(_mp()) is True


def test_should_fallback_explicit_draft_is_false() -> None:
    # spec_auto False (explicit draft / operator --spec-type) → never retry.
    assert _should_mtp_fallback(_mp(spec_auto=False)) is False


def test_should_fallback_already_retried_is_false() -> None:
    assert _should_mtp_fallback(_mp(mtp_fallback_done=True)) is False


def test_should_fallback_returncode_zero_is_false() -> None:
    assert _should_mtp_fallback(_mp(returncode=0)) is False


def test_should_fallback_returncode_none_is_false() -> None:
    assert _should_mtp_fallback(_mp(returncode=None)) is False


def test_should_fallback_no_fallback_cmd_is_false() -> None:
    assert _should_mtp_fallback(_mp(fallback_cmd=None)) is False


def test_should_fallback_out_of_window_is_false() -> None:
    old = time.monotonic() - (_MTP_FALLBACK_WINDOW_SECS + 10.0)
    assert _should_mtp_fallback(_mp(started_at=old)) is False


# ---------------------------------------------------------------------------
# Integration — real subprocess retry via api_start
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> CodeRouterConfig:
    binary = _make_crashing_binary(tmp_path / "crash-llama-server")
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
            backends={"llama.cpp": LauncherBackendConfig(binary=str(binary))},
        ),
    )


@pytest.fixture
def client(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


def _poll_logs(
    client: TestClient, proc_id: str, needle: str, timeout: float = 5.0
) -> list[str]:
    """Poll a process's log until ``needle`` appears or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    logs: list[str] = []
    while time.monotonic() < deadline:
        resp = client.get(f"/api/launcher/logs/{proc_id}?n=200")
        assert resp.status_code == 200, resp.text
        logs = resp.json()["logs"]
        if any(needle in line for line in logs):
            return logs
        time.sleep(0.1)
    return logs


def test_auto_mtp_crash_retries_once_without_spec(
    client: TestClient, tmp_path: Path
) -> None:
    # nextn gguf → auto detection emits `--spec-type draft-mtp`.
    main = _make_gguf(tmp_path / "glm.gguf", "glm4moe", nextn=1, size=4000)
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "glm",
            "backend": "llama.cpp",
            "model_path": str(main),
            "port": 8080,
            "mtp_mode": "auto",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["speculative"] == ["--spec-type", "draft-mtp"]
    proc_id = body["id"]

    logs = _poll_logs(client, proc_id, "retrying without speculative decoding")
    joined = "\n".join(logs)
    assert "retrying without speculative decoding" in joined

    # Exactly two cmd lines: the original (with spec) + one fallback (without).
    cmd_lines = [ln for ln in logs if ln.startswith("[launcher] cmd:")]
    assert len(cmd_lines) == 2, joined
    assert "--spec-type" in cmd_lines[0]
    assert "--spec-type" not in cmd_lines[1]

    # Exactly one retry — no third cmd line, ends in error after fallback dies.
    logs = _poll_logs(client, proc_id, "process exited with code")
    assert "retrying without speculative decoding" in "\n".join(logs)
    assert (
        sum(1 for ln in logs if ln.startswith("[launcher] cmd:")) == 2
    ), "\n".join(logs)

    procs = client.get("/api/launcher/processes").json()["processes"]
    entry = next(p for p in procs if p["id"] == proc_id)
    assert entry["status"] == "error"


def test_off_mode_never_retries(client: TestClient, tmp_path: Path) -> None:
    main = _make_gguf(tmp_path / "glm.gguf", "glm4moe", nextn=1, size=4000)
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "glm-off",
            "backend": "llama.cpp",
            "model_path": str(main),
            "port": 8081,
            "mtp_mode": "off",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["speculative"] == []
    proc_id = body["id"]

    # Wait for the (single) process to exit, then assert no retry ever fired.
    logs = _poll_logs(client, proc_id, "process exited with code")
    joined = "\n".join(logs)
    assert "retrying without speculative decoding" not in joined
    assert sum(1 for ln in logs if ln.startswith("[launcher] cmd:")) == 1, joined
