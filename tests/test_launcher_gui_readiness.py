"""Tests for launcher_gui.py's readiness gating and generic auto-restart.

launcher_gui.py is the Tk desktop launcher, ported here from the Web版
equivalent in coderouter/ingress/launcher_routes.py (see that module's
``_backend_ready`` / ``_wait_ready_and_register`` / ``_attempt_restart`` /
``_tail_logs`` — the implementation pattern this file mirrors).

Two holes existed in the Tk launcher that the Web launcher had already
fixed:

1. **Readiness gating** — a launched backend was shown as "running" the
   instant the OS process spawned, before llama-server / vllm had actually
   finished loading the model (``_backend_ready`` / ``poll_until_ready`` /
   ``_readiness_worker``).
2. **Generic auto-restart** — a crashed launcher process (besides the
   existing one-shot MTP startup-crash retry) had no supervision; it sat in
   status="error" forever. Opt-in via ``LauncherConfig.auto_restart``
   (default False — see ``launcher_gui._DEFAULT_AUTO_RESTART`` and the
   matching field in ``coderouter/config/schemas.py``).

Design note (why these tests can import launcher_gui.py at all): the
module does an unconditional ``import tkinter as tk`` at the top — that is
a standard-library import (no X11 / DISPLAY connection needed) and
succeeds anywhere Python's tkinter package is installed, which is already
a hard prerequisite for running the GUI itself (see the module's own
"追加パッケージ: 不要 (tkinter は Python 標準...)" docstring). None of the
tests below ever instantiate ``tk.Tk()`` / ``LauncherApp`` — only the
plain, Tk-independent module-level functions and dataclasses that the
readiness/auto-restart logic was deliberately factored into
(``_backend_ready``, ``poll_until_ready``, ``_readiness_worker``,
``plan_auto_restart``, ``_exit_status``, ``_load_config``) — so no display
server is required, matching the desktop CI constraint.

Sections:

* A. ``_backend_ready`` — single-probe primitive (real localhost sockets,
  no mocking — mirrors tests/test_launcher_readiness_restart.py section A).
* B. ``poll_until_ready`` — the generic poll loop, with an injected
  clock/sleep so it needs no real timing or I/O.
* C. ``_readiness_worker`` — the glue between a ``ManagedProcess``, a
  ``LauncherConfig`` and the log queue, with ``backend_ready`` injected for
  determinism.
* D. ``plan_auto_restart`` — the backoff/budget decision primitive.
* E. ``_exit_status`` — stopping-aware exit-code -> terminal-status mapping.
* F. ``_load_config`` — providers.yaml ``launcher:`` block parsing for the
  six new fields, including the deliberate omission of ``swap:``, and the
  string-safe bool parsing of ``auto_restart`` (``_safe_bool``).
* G. Removal liveness (``_proc_alive`` / ``_kill_for_removal``) — the
  "error-but-ALIVE after a readiness timeout" orphan fix (adversarial
  repro: /tmp/review93/test_error_alive_orphan.py, assertions inverted).
* H. Stale-generation "ready" write re-guard in ``_readiness_worker``
  (adversarial repro: /tmp/review93/test_readiness_race.py, inverted).
"""

from __future__ import annotations

import contextlib
import http.server
import queue
import socket
import threading

import pytest

# launcher_gui imports tkinter at module level (it IS a Tk app). Every test
# in this module exercises Tk-free pure helpers, but the import itself still
# needs the tkinter package to exist. Skip cleanly on pythons built without
# Tk support (e.g. uv-managed CPython on CI runners) instead of erroring at
# collection time.
pytest.importorskip("tkinter", reason="launcher_gui requires the tkinter package (python3-tk)")

import launcher_gui as lg

# ---------------------------------------------------------------------------
# A. _backend_ready — single-probe primitive
# ---------------------------------------------------------------------------


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    status_code = 200

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(self.status_code)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_a: object) -> None:
        pass


def _run_health_server(port: int, status_code: int) -> http.server.HTTPServer:
    handler = type("Handler", (_HealthHandler,), {"status_code": status_code})
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_backend_ready_llamacpp_true_on_health_200() -> None:
    httpd = _run_health_server(20301, 200)
    try:
        assert lg._backend_ready("llama.cpp", 20301, probe_timeout_s=2.0) is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_backend_ready_llamacpp_false_on_health_503() -> None:
    httpd = _run_health_server(20302, 503)
    try:
        assert lg._backend_ready("llama.cpp", 20302, probe_timeout_s=2.0) is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_backend_ready_false_when_nothing_listening() -> None:
    assert lg._backend_ready("llama.cpp", 20303, probe_timeout_s=1.0) is False


def test_backend_ready_vllm_uses_health_endpoint_too() -> None:
    httpd = _run_health_server(20304, 200)
    try:
        assert lg._backend_ready("vllm", 20304, probe_timeout_s=2.0) is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_backend_ready_mlx_falls_back_to_tcp_connect() -> None:
    """mlx has no documented /health — a bare TCP connect is the fallback."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 20305))
    srv.listen(1)

    def _accept_once() -> None:
        with contextlib.suppress(OSError):
            conn, _addr = srv.accept()
            conn.close()

    threading.Thread(target=_accept_once, daemon=True).start()
    try:
        # No HTTP response is ever sent — only a successful connect matters.
        assert lg._backend_ready("mlx", 20305, probe_timeout_s=2.0) is True
    finally:
        srv.close()


def test_backend_ready_mlx_false_when_nothing_listening() -> None:
    assert lg._backend_ready("mlx", 20306, probe_timeout_s=1.0) is False


# ---------------------------------------------------------------------------
# B. poll_until_ready — generic poll loop (no real I/O or timing)
# ---------------------------------------------------------------------------


def test_poll_until_ready_returns_ready_once_check_succeeds() -> None:
    calls: list[int] = []

    def check() -> bool:
        calls.append(1)
        return len(calls) >= 3

    fake_time = [0.0]
    outcome = lg.poll_until_ready(
        check=check,
        should_abort=lambda: False,
        timeout_s=10.0,
        poll_interval_s=1.0,
        sleep=lambda s: fake_time.__setitem__(0, fake_time[0] + s),
        now=lambda: fake_time[0],
    )
    assert outcome == "ready"
    assert len(calls) == 3


def test_poll_until_ready_times_out_when_never_ready() -> None:
    fake_time = [0.0]
    outcome = lg.poll_until_ready(
        check=lambda: False,
        should_abort=lambda: False,
        timeout_s=3.0,
        poll_interval_s=1.0,
        sleep=lambda s: fake_time.__setitem__(0, fake_time[0] + s),
        now=lambda: fake_time[0],
    )
    assert outcome == "timeout"


def test_poll_until_ready_aborts_before_first_probe() -> None:
    probed: list[int] = []
    outcome = lg.poll_until_ready(
        check=lambda: (probed.append(1) or True),
        should_abort=lambda: True,
        timeout_s=5.0,
        poll_interval_s=1.0,
        sleep=lambda s: None,
        now=lambda: 0.0,
    )
    assert outcome == "aborted"
    assert probed == []  # bailed out before ever probing


def test_poll_until_ready_aborts_when_resolved_during_last_probe() -> None:
    """Covers the "crashed while the last probe was in flight" race: even
    though check() succeeded, should_abort() flipped True in the meantime,
    so the caller must not treat this as a genuine "ready"."""
    state = {"resolved": False}

    def check() -> bool:
        state["resolved"] = True  # simulates a concurrent crash/stop
        return True

    outcome = lg.poll_until_ready(
        check=check,
        should_abort=lambda: state["resolved"],
        timeout_s=5.0,
        poll_interval_s=1.0,
        sleep=lambda s: None,
        now=lambda: 0.0,
    )
    assert outcome == "aborted"


# ---------------------------------------------------------------------------
# C. _readiness_worker — ManagedProcess + LauncherConfig + queue glue
# ---------------------------------------------------------------------------


def _mp(**overrides: object) -> lg.ManagedProcess:
    base: dict[str, object] = dict(
        id="p1",
        name="test",
        backend="llama.cpp",
        model_name="m.gguf",
        port=20401,
        cmd=["true"],
        status="loading",
    )
    base.update(overrides)
    mp = lg.ManagedProcess(**base)  # type: ignore[arg-type]
    mp.proc = object()  # anything not-None: "the OS process is alive"
    return mp


def test_readiness_worker_marks_running_and_resets_restart_count() -> None:
    mp = _mp(restart_count=2, spawn_gen=1)
    cfg = lg.LauncherConfig(readiness_timeout_s=5.0, readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()

    lg._readiness_worker(mp, cfg, q, 1, backend_ready=lambda *a, **kw: True)

    assert mp.status == "running"
    assert mp.restart_count == 0
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert any("readiness check passed" in ln for _pid, ln in items)
    assert any(ln.startswith("_READY_:") for _pid, ln in items)


def test_readiness_worker_times_out_marks_error() -> None:
    mp = _mp(spawn_gen=1)
    cfg = lg.LauncherConfig(readiness_timeout_s=0.05, readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()

    lg._readiness_worker(mp, cfg, q, 1, backend_ready=lambda *a, **kw: False)

    assert mp.status == "error"
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert any("timed out" in ln for _pid, ln in items)


def test_readiness_worker_bails_when_already_resolved() -> None:
    """If the process crashed/stopped before the first probe, never probe."""
    mp = _mp(status="error", spawn_gen=1)
    cfg = lg.LauncherConfig(readiness_timeout_s=5.0, readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()
    probed: list[int] = []

    def _spy(*_a: object, **_kw: object) -> bool:
        probed.append(1)
        return True

    lg._readiness_worker(mp, cfg, q, 1, backend_ready=_spy)

    assert mp.status == "error"  # untouched
    assert probed == []
    assert q.empty()


def test_readiness_worker_ignored_when_superseded_by_newer_spawn() -> None:
    """A stale worker left over from a previous spawn attempt (MTP fallback
    / auto-restart already moved on to spawn_gen 2) must never overwrite
    the status a newer spawn is tracking."""
    mp = _mp(spawn_gen=2)
    cfg = lg.LauncherConfig(readiness_timeout_s=5.0, readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()

    # gen=1 below is the *stale* generation this worker was started for.
    lg._readiness_worker(mp, cfg, q, 1, backend_ready=lambda *a, **kw: True)

    assert mp.status == "loading"  # untouched
    assert q.empty()


# ---------------------------------------------------------------------------
# D. plan_auto_restart — backoff/budget decision primitive
# ---------------------------------------------------------------------------


def test_plan_auto_restart_disabled_by_default() -> None:
    plan = lg.plan_auto_restart(
        auto_restart=False, restart_count=0, max_attempts=3,
        backoff_s=2.0, backoff_max_s=30.0, has_cmd=True,
    )
    assert plan.should_restart is False
    assert plan.log_lines == []


def test_plan_auto_restart_respects_max_attempts() -> None:
    plan = lg.plan_auto_restart(
        auto_restart=True, restart_count=2, max_attempts=2,
        backoff_s=0.001, backoff_max_s=0.001, has_cmd=True,
    )
    assert plan.should_restart is False
    assert any("giving up" in ln for ln in plan.log_lines)


def test_plan_auto_restart_computes_exponential_backoff() -> None:
    plan = lg.plan_auto_restart(
        auto_restart=True, restart_count=0, max_attempts=5,
        backoff_s=2.0, backoff_max_s=30.0, has_cmd=True,
    )
    assert plan.should_restart is True
    assert plan.backoff_s == 2.0
    assert any("attempt 1/5" in ln for ln in plan.log_lines)

    plan2 = lg.plan_auto_restart(
        auto_restart=True, restart_count=3, max_attempts=5,
        backoff_s=2.0, backoff_max_s=30.0, has_cmd=True,
    )
    assert plan2.backoff_s == 16.0  # 2.0 * 2**3
    assert any("attempt 4/5" in ln for ln in plan2.log_lines)


def test_plan_auto_restart_caps_backoff_at_max() -> None:
    plan = lg.plan_auto_restart(
        auto_restart=True, restart_count=10, max_attempts=20,
        backoff_s=2.0, backoff_max_s=30.0, has_cmd=True,
    )
    assert plan.backoff_s == 30.0


def test_plan_auto_restart_no_cmd_returns_false() -> None:
    plan = lg.plan_auto_restart(
        auto_restart=True, restart_count=0, max_attempts=3,
        backoff_s=2.0, backoff_max_s=30.0, has_cmd=False,
    )
    assert plan.should_restart is False


# ---------------------------------------------------------------------------
# E. _exit_status — stopping-aware exit-code -> status mapping
# ---------------------------------------------------------------------------


def test_exit_status_intentional_stop_is_always_stopped() -> None:
    """A deliberate Stop/Kill must never be mislabeled "error", regardless
    of the exit code SIGTERM/SIGKILL happened to produce."""
    assert lg._exit_status(returncode=-15, stopping=True) == "stopped"
    assert lg._exit_status(returncode=1, stopping=True) == "stopped"
    assert lg._exit_status(returncode=0, stopping=True) == "stopped"


def test_exit_status_crash_without_stopping_is_error() -> None:
    assert lg._exit_status(returncode=1, stopping=False) == "error"
    assert lg._exit_status(returncode=-11, stopping=False) == "error"


def test_exit_status_clean_exit_without_stopping_is_stopped() -> None:
    assert lg._exit_status(returncode=0, stopping=False) == "stopped"
    assert lg._exit_status(returncode=None, stopping=False) == "stopped"


# ---------------------------------------------------------------------------
# F. _load_config — providers.yaml `launcher:` block parsing
# ---------------------------------------------------------------------------


def test_load_config_readiness_defaults_when_absent(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text("launcher:\n  model_dirs: []\n")

    cfg = lg._load_config(str(p))

    assert cfg.readiness_timeout_s == lg._DEFAULT_READINESS_TIMEOUT_S == 300.0
    assert cfg.readiness_poll_interval_s == 2.0
    assert cfg.auto_restart is False
    assert cfg.auto_restart_max_attempts == 3
    assert cfg.auto_restart_backoff_s == 2.0
    assert cfg.auto_restart_backoff_max_s == 30.0


def test_load_config_no_launcher_block_still_uses_defaults(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text("allow_paid: false\n")

    cfg = lg._load_config(str(p))

    assert cfg.readiness_timeout_s == 300.0
    assert cfg.auto_restart is False


def test_load_config_readiness_overrides(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text(
        "launcher:\n"
        "  readiness_timeout_s: 60\n"
        "  readiness_poll_interval_s: 0.5\n"
        "  auto_restart: true\n"
        "  auto_restart_max_attempts: 5\n"
        "  auto_restart_backoff_s: 1.0\n"
        "  auto_restart_backoff_max_s: 10.0\n"
    )

    cfg = lg._load_config(str(p))

    assert cfg.readiness_timeout_s == 60.0
    assert cfg.readiness_poll_interval_s == 0.5
    assert cfg.auto_restart is True
    assert cfg.auto_restart_max_attempts == 5
    assert cfg.auto_restart_backoff_s == 1.0
    assert cfg.auto_restart_backoff_max_s == 10.0


def test_load_config_clamps_backoff_when_misordered(tmp_path) -> None:
    """schemas.py's pydantic validator *rejects* backoff_s > backoff_max_s;
    the GUI has no such validation layer, so it clamps instead — a
    malformed config must never block the desktop tool from opening."""
    p = tmp_path / "providers.yaml"
    p.write_text(
        "launcher:\n"
        "  auto_restart_backoff_s: 50.0\n"
        "  auto_restart_backoff_max_s: 5.0\n"
    )

    cfg = lg._load_config(str(p))

    assert cfg.auto_restart_backoff_s == 50.0
    assert cfg.auto_restart_backoff_max_s == 50.0  # clamped up to match floor


def test_load_config_swap_block_is_never_read(tmp_path) -> None:
    """Tk版に swap 連携は入れない — the swap: sub-block (Web版 SwapManager
    専用) must simply be ignored: no crash, and LauncherConfig never gains
    a `swap` attribute."""
    p = tmp_path / "providers.yaml"
    p.write_text(
        "launcher:\n"
        "  swap:\n"
        "    enabled: true\n"
        "    models: []\n"
    )

    cfg = lg._load_config(str(p))

    assert not hasattr(cfg, "swap")


def test_load_config_malformed_readiness_timeout_falls_back_to_default(
    tmp_path,
) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text("launcher:\n  readiness_timeout_s: [not, a, number]\n")

    cfg = lg._load_config(str(p))

    assert cfg.readiness_timeout_s == lg._DEFAULT_READINESS_TIMEOUT_S


def test_load_config_auto_restart_quoted_false_string_is_false(tmp_path) -> None:
    """bool("false") is True — a quoted YAML string must not silently enable
    the side-effectful auto-restart opt-in the user explicitly turned off."""
    p = tmp_path / "providers.yaml"
    p.write_text('launcher:\n  auto_restart: "false"\n')

    cfg = lg._load_config(str(p))

    assert cfg.auto_restart is False


def test_load_config_auto_restart_quoted_true_string_is_true(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text('launcher:\n  auto_restart: "TRUE"\n')  # case-insensitive

    cfg = lg._load_config(str(p))

    assert cfg.auto_restart is True


def test_load_config_auto_restart_garbage_string_falls_back(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text('launcher:\n  auto_restart: "banana"\n')

    cfg = lg._load_config(str(p))

    assert cfg.auto_restart is False  # the (safe) default


# ---------------------------------------------------------------------------
# G. Removal liveness — a readiness-timed-out backend is status="error" but
# ALIVE (holding its port + VRAM). The remove/stop decision must key on
# proc.poll() (via _proc_alive), never on the status string: pre-readiness-
# gating "error" always meant "process exited non-zero" (nothing alive to
# orphan), but the readiness timeout path overloads "error" onto a
# still-running process. Adapted (assertions inverted to the CORRECT
# behavior) from /tmp/review93/test_error_alive_orphan.py.
# ---------------------------------------------------------------------------


class _AliveProc:
    """Stand-in for a live subprocess.Popen (poll() is None => running)."""

    def __init__(self) -> None:
        self.killed = False
        self.terminated = False

    def poll(self) -> None:
        return None

    def kill(self) -> None:
        self.killed = True

    def terminate(self) -> None:
        self.terminated = True


class _DeadProc:
    def __init__(self, returncode: int = -9) -> None:
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode


def test_readiness_timeout_error_process_is_detected_alive() -> None:
    """After a readiness timeout the process sits in status='error' while
    still running — _proc_alive must report True so the removal path
    confirms + kills instead of orphaning it."""
    proc = _AliveProc()
    mp = _mp(spawn_gen=1)
    mp.proc = proc
    cfg = lg.LauncherConfig(readiness_timeout_s=0.05, readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()

    lg._readiness_worker(mp, cfg, q, 1, backend_ready=lambda *a, **kw: False)

    assert mp.status == "error"  # the overloaded state readiness introduced
    assert proc.poll() is None  # ...and yet the OS process is alive
    # THE FIX: liveness, not status, decides whether removal must kill.
    assert lg._proc_alive(mp) is True


def test_kill_for_removal_sets_stopping_before_killing() -> None:
    """Removal of a live process must set stopping=True and kill() — the
    stopping flag is what stops the launch worker thread from auto-
    restarting a process the UI no longer tracks."""
    proc = _AliveProc()
    mp = _mp()
    mp.proc = proc

    lg._kill_for_removal(mp)

    assert mp.stopping is True
    assert proc.killed is True


def test_proc_alive_false_for_dead_process() -> None:
    mp = _mp()
    mp.proc = _DeadProc(returncode=11)
    assert lg._proc_alive(mp) is False


def test_proc_alive_false_when_no_proc_handle() -> None:
    mp = _mp()
    mp.proc = None
    assert lg._proc_alive(mp) is False


def test_error_but_alive_process_would_not_be_auto_restarted_after_removal() -> None:
    """Follow-through of the orphan fix: once _kill_for_removal ran, the
    launch worker's crash handling sees stopping=True — _exit_status maps
    the SIGKILL exit to 'stopped' (not 'error') and the auto-restart branch
    is skipped entirely."""
    proc = _AliveProc()
    mp = _mp(status="error")  # readiness-timed-out but alive
    mp.proc = proc

    lg._kill_for_removal(mp)

    # what _run() computes when the killed child is reaped:
    assert lg._exit_status(-9, mp.stopping) == "stopped"


# ---------------------------------------------------------------------------
# H. Stale-generation "ready" write — _readiness_worker runs in a REAL
# thread (unlike the Web版's single-threaded asyncio task, where re-check
# and write are atomic), so it must re-guard IMMEDIATELY before writing
# status="running". Adapted (assertions inverted to the CORRECT behavior)
# from /tmp/review93/test_readiness_race.py.
# ---------------------------------------------------------------------------


def test_stale_ready_worker_does_not_clobber_newer_generation(monkeypatch) -> None:
    """Simulates _run's crash+auto-restart+respawn sequence executing in the
    scheduling gap AFTER poll_until_ready internally decided "ready" but
    BEFORE _readiness_worker writes: the stale gen=1 worker must bail out
    silently, leaving gen=2's state (loading, restart_count=1) intact."""
    mp = _mp(spawn_gen=1)
    cfg = lg.LauncherConfig()
    q: queue.Queue = queue.Queue()

    def fake_poll(**_kw: object) -> str:
        # exactly _run's crash-handling in that window:
        mp.status = "error"      # _exit_status(rc!=0, stopping=False)
        mp.restart_count = 1     # auto-restart consumed one attempt
        mp.spawn_gen = 2         # respawn bumped the generation
        mp.status = "loading"    # respawn set loading for gen=2
        return "ready"

    monkeypatch.setattr(lg, "poll_until_ready", fake_poll)
    lg._readiness_worker(mp, cfg, q, gen=1)  # the STALE gen=1 worker

    assert mp.status == "loading"  # gen=2's state intact — not "running"
    assert mp.restart_count == 1   # not clobbered back to 0
    assert mp.spawn_gen == 2
    assert q.empty()               # and no _READY_ emitted by the stale worker


def test_ready_worker_bails_when_stopping_set_during_final_gap(monkeypatch) -> None:
    """Same window, different interleaving: the user hit Stop between the
    "ready" decision and the write — the worker must not resurrect the
    process to "running"."""
    mp = _mp(spawn_gen=1)
    cfg = lg.LauncherConfig()
    q: queue.Queue = queue.Queue()

    def fake_poll(**_kw: object) -> str:
        mp.stopping = True  # _do_stop sets this before signalling
        return "ready"

    monkeypatch.setattr(lg, "poll_until_ready", fake_poll)
    lg._readiness_worker(mp, cfg, q, gen=1)

    assert mp.status == "loading"  # untouched
    assert q.empty()
