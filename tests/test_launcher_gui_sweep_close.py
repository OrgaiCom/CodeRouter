"""H-14 regression: closing a bench-sweep window (or the whole app) while a
sweep is running must stop the sweep's llama-server / llmbench children
instead of orphaning them.

launcher_gui.py is the Tk desktop launcher; it imports tkinter at module
level, so — exactly like the sibling GUI test files — we importorskip on a
python built without Tk and otherwise never instantiate ``tk.Tk()`` /
``SweepWindow`` / ``LauncherApp``. Every case here drives the H-14 logic
against either a bare ``_SweepWorker`` (its live-child ledger is fully
Tk-independent) or a lightweight stub ``self`` with the relevant unbound
SweepWindow / LauncherApp methods bound onto it, so no display server and no
real subprocess are ever needed.

The bug being pinned: a sweep child (``server = self._popen(...)`` /
``bench = self._popen(...)`` in ``_SweepWorker._run_one``) is never put in
``app.processes``, so the old ``_on_close`` couldn't reach it and the old
``SweepWindow`` had no ``WM_DELETE_WINDOW`` handler at all — closing it left
the daemon worker running until app exit force-killed the thread mid-flight,
stranding a llama-server holding VRAM + the port.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import types

import pytest

# launcher_gui imports tkinter at module level (it IS a Tk app). Skip cleanly
# on pythons built without Tk support instead of erroring at collection time.
pytest.importorskip(
    "tkinter", reason="launcher_gui requires the tkinter package (python3-tk)"
)

import launcher_gui as lg
from coderouter.launcher_devices import (
    DeviceSelection,
    SweepPlan,
    SweepState,
    SweepStep,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProc:
    """Stand-in for subprocess.Popen with observable terminate/kill."""

    def __init__(self, *, alive: bool = True, returncode: int = 0,
                 wait_timeout: bool = False) -> None:
        self.alive = alive
        self.returncode = returncode
        self.wait_timeout = wait_timeout
        self.stdout = None  # → _pump_logs skips draining
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None if self.alive else self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.wait_timeout and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        self.alive = False
        return self.returncode


class _FakeWorker:
    """Stand-in for a running _SweepWorker: only the H-14 surface."""

    def __init__(self, *, alive: bool = True) -> None:
        self._alive = alive
        self.request_stop_called = False
        self.force_kill_called = False
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return self._alive

    def request_stop(self) -> None:
        self.request_stop_called = True

    def force_kill(self) -> None:
        self.force_kill_called = True

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


class _Var:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


class _Btn:
    def __init__(self) -> None:
        self.state: str | None = None

    def configure(self, **kw: object) -> None:
        if "state" in kw:
            self.state = str(kw["state"])


def _make_popen(procs: list[_FakeProc]):
    it = iter(procs)

    def _popen(*_a: object, **_kw: object) -> _FakeProc:
        return next(it)

    return _popen


def _plan(**overrides: object) -> SweepPlan:
    base = dict(
        steps=[SweepStep(label="CUDA0 単体",
                         selection=DeviceSelection(["CUDA0"]))],
        model_path="/m/model.gguf",
        backend="llama.cpp",
        port=28080,
        bench_cmd_template="llmbench run --runs {runs}",
        results_dir=None,
    )
    base.update(overrides)
    return SweepPlan(**base)  # type: ignore[arg-type]


def _worker(*, popen=None, poll_ready=lambda **_kw: "ready",
            abort=None, cfg=None) -> lg._SweepWorker:
    return lg._SweepWorker(
        _plan(),
        cfg or lg.LauncherConfig(bench_readiness_timeout_s=5.0,
                                 readiness_poll_interval_s=0.01),
        queue.Queue(),
        abort or threading.Event(),
        runs=1,
        popen=popen or _make_popen([]),
        poll_ready=poll_ready,
    )


# Bind the H-14 SweepWindow methods onto a stub self so we can exercise them
# without a real Toplevel.
_WIN_METHODS = ("is_running", "_on_close", "_await_worker_exit",
                "_finish_close", "shutdown_for_app_close")


class _StubWin:
    def __init__(self, worker=None) -> None:
        self._worker = worker
        self._closing = False
        self._poll_job = "poll-job"
        self._sweep_status = _Var()
        self._start_btn = _Btn()
        self._abort_btn = _Btn()
        self.app = types.SimpleNamespace(_sweep_windows=set())
        self.after_calls: list[tuple] = []
        self.after_cancel_calls: list[object] = []
        self.destroyed = False

    def after(self, ms: int, func=None, *args: object) -> str:
        self.after_calls.append((ms, func, args))
        return f"job-{len(self.after_calls)}"

    def after_cancel(self, job: object) -> None:
        self.after_cancel_calls.append(job)

    def destroy(self) -> None:
        self.destroyed = True


def _make_win(worker=None) -> _StubWin:
    stub = _StubWin(worker)
    for name in _WIN_METHODS:
        setattr(stub, name,
                types.MethodType(getattr(lg.SweepWindow, name), stub))
    return stub


class _StubApp:
    def __init__(self, *, sweep_windows=None, processes=None) -> None:
        self._sweep_windows = sweep_windows or set()
        self.processes = processes or {}
        self._cr_proc = None
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


def _make_app(**kw) -> _StubApp:
    stub = _StubApp(**kw)
    stub._on_close = types.MethodType(lg.LauncherApp._on_close, stub)
    return stub


class _FakeSweepWin:
    def __init__(self, *, running: bool = False) -> None:
        self._running = running
        self.shutdown_called = False

    def is_running(self) -> bool:
        return self._running

    def shutdown_for_app_close(self) -> None:
        self.shutdown_called = True


class _MP:
    """Minimal ManagedProcess stand-in for the _on_close terminate loop."""

    def __init__(self, proc: _FakeProc) -> None:
        self.proc = proc
        self.stopping = False


# ---------------------------------------------------------------------------
# Worker-side: the Tk-independent live-child ledger
# ---------------------------------------------------------------------------


def test_request_stop_terminates_live_children() -> None:
    w = _worker()
    live_a, live_b, dead = _FakeProc(), _FakeProc(), _FakeProc(alive=False)
    for p in (live_a, live_b, dead):
        w._track(p)

    w.request_stop()

    assert w._abort.is_set()
    assert live_a.terminated and live_b.terminated
    assert not dead.terminated  # already exited → left alone (no double signal)


def test_force_kill_only_kills_live_children() -> None:
    w = _worker()
    live, dead = _FakeProc(), _FakeProc(alive=False)
    w._track(live)
    w._track(dead)

    w.force_kill()

    assert live.killed
    assert not dead.killed


def test_happy_path_leaves_no_tracked_children() -> None:
    server, bench = _FakeProc(), _FakeProc(returncode=0)
    w = _worker(popen=_make_popen([server, bench]))
    step = w.plan.steps[0]

    w._run_one(step)

    assert step.state == SweepState.DONE
    # server untracked by _terminate, bench untracked right after wait().
    assert w.live_procs() == []


def test_request_stop_during_readiness_terminates_server() -> None:
    server = _FakeProc()
    w = _worker(popen=_make_popen([server]))
    # Simulate the Tk thread calling request_stop() mid-readiness: the injected
    # poll loop trips the stop, then reports the abort back to _run_one.
    w._poll_ready = lambda **_kw: (w.request_stop(), "aborted")[1]
    step = w.plan.steps[0]

    w._run_one(step)

    assert w._abort.is_set()
    assert server.terminated          # SIGTERM'd instead of orphaned
    assert w.live_procs() == []       # and untracked by _terminate


def test_spawn_failure_tracks_nothing() -> None:
    def _boom(*_a: object, **_kw: object):
        raise OSError("no such binary")

    w = _worker(popen=_boom)
    step = w.plan.steps[0]

    w._run_one(step)

    assert step.state == SweepState.FAILED
    assert w.live_procs() == []  # nothing was ever tracked


def test_terminate_untracks_even_when_kill_path_taken() -> None:
    w = _worker()
    proc = _FakeProc(wait_timeout=True)  # first wait() raises → kill path
    w._track(proc)

    w._terminate(proc)

    assert proc.terminated and proc.killed
    assert proc not in w.live_procs()  # finally: untrack regardless of path


# ---------------------------------------------------------------------------
# Window-side: WM_DELETE_WINDOW / grace-timer close (stub self)
# ---------------------------------------------------------------------------


def test_close_without_worker_destroys_immediately() -> None:
    win = _make_win(worker=None)
    win.app._sweep_windows.add(win)

    win._on_close()

    assert win.destroyed
    assert win._closing
    assert win.after_cancel_calls == ["poll-job"]  # poll timer cancelled
    assert win not in win.app._sweep_windows
    # no grace-timer scheduling when there is nothing running
    assert win.after_calls == []


def test_close_while_running_signals_but_does_not_destroy() -> None:
    worker = _FakeWorker(alive=True)
    win = _make_win(worker=worker)

    win._on_close()

    assert worker.request_stop_called       # children SIGTERM'd, non-blocking
    assert not win.destroyed                 # main thread not blocked
    assert win._closing
    assert win._start_btn.state == "disabled"
    assert win._abort_btn.state == "disabled"
    assert "停止中" in win._sweep_status.get()
    # scheduled the grace poll _await_worker_exit(0)
    assert len(win.after_calls) == 1
    ms, func, args = win.after_calls[0]
    assert ms == lg._SWEEP_CLOSE_TICK_MS
    assert func == win._await_worker_exit
    assert args == (0,)


def test_await_worker_exit_reschedules_until_grace() -> None:
    worker = _FakeWorker(alive=True)
    win = _make_win(worker=worker)

    win._await_worker_exit(0)

    assert not win.destroyed
    assert not worker.force_kill_called
    ms, func, args = win.after_calls[-1]
    assert ms == lg._SWEEP_CLOSE_TICK_MS
    assert func == win._await_worker_exit
    assert args == (lg._SWEEP_CLOSE_TICK_MS,)  # waited advanced by one tick


def test_grace_expiry_force_kills_then_destroys() -> None:
    # The orphan regression: worker refuses to die within the grace window, so
    # the still-live child must be SIGKILL'd before the window is destroyed.
    worker = _FakeWorker(alive=True)
    win = _make_win(worker=worker)

    win._await_worker_exit(lg._SWEEP_CLOSE_GRACE_MS)

    assert worker.force_kill_called
    assert win.destroyed
    assert win.after_cancel_calls == ["poll-job"]


def test_worker_exit_during_grace_destroys_without_kill() -> None:
    worker = _FakeWorker(alive=False)  # thread already finished cleanly
    win = _make_win(worker=worker)

    win._await_worker_exit(lg._SWEEP_CLOSE_TICK_MS)

    assert win.destroyed
    assert not worker.force_kill_called  # nothing left to kill


def test_shutdown_for_app_close_terminates_joins_kills_destroys() -> None:
    worker = _FakeWorker(alive=True)
    win = _make_win(worker=worker)

    win.shutdown_for_app_close()

    assert worker.request_stop_called
    assert worker.join_calls == [lg._SWEEP_APP_CLOSE_JOIN_S]  # bounded sync join
    assert worker.force_kill_called
    assert win.destroyed


# ---------------------------------------------------------------------------
# App-side: LauncherApp._on_close (stub self)
# ---------------------------------------------------------------------------


def test_app_close_marks_processes_stopping() -> None:
    mp = _MP(_FakeProc(alive=True))
    app = _make_app(processes={"p1": mp})

    app._on_close()

    # stopping MUST be set before terminate, else the launch worker mistakes
    # SIGTERM for a crash and respawns a new llama-server (orphan).
    assert mp.stopping
    assert mp.proc.terminated
    assert app.destroyed


def test_app_close_drains_sweep_windows(monkeypatch) -> None:
    monkeypatch.setattr(lg.messagebox, "askyesno", lambda *a, **k: True)
    swin = _FakeSweepWin(running=True)
    app = _make_app(sweep_windows={swin})

    app._on_close()

    assert swin.shutdown_called  # sweep children explicitly stopped
    assert app.destroyed


def test_app_close_cancelled_when_user_declines(monkeypatch) -> None:
    monkeypatch.setattr(lg.messagebox, "askyesno", lambda *a, **k: False)
    swin = _FakeSweepWin(running=True)
    mp = _MP(_FakeProc(alive=True))
    app = _make_app(sweep_windows={swin}, processes={"p1": mp})

    app._on_close()

    assert not app.destroyed          # user backed out
    assert not swin.shutdown_called   # nothing torn down
    assert not mp.stopping
    assert not mp.proc.terminated
