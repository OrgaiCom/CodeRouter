"""Regression tests for launcher_gui.py's tagged log-queue events (H-13).

The Tk launcher used to push BOTH control events and the raw stdout of the
child backends onto one queue, with the control events encoded as magic
prefixes *inside the line*::

    self._log_queue.put((proc_id, f"_ERR_:{exc}"))       # control
    self._log_queue.put((proc_id, line))                 # raw child stdout

The drain loop then did ``line.startswith("_ERR_:")``, so the child's own
output could forge control events — classic in-band signaling:

* a single ``_ERR_:...`` line from llama-server ran ``del
  self.processes[proc_id]``, orphaning a **still-running** server: it kept
  its VRAM and its port, disappeared from the process table, and was no
  longer in ``_on_close``'s shutdown set;
* a short ``_SPAWNED_:x`` line (one colon too few) made ``line.split(":", 2)``
  raise ``ValueError``, killing the drain loop for that whole tick and
  discarding every queued line behind it;
* ``_READY_:name:port`` faked a readiness transition in the UI.

llama-server echoes GGUF metadata (chat templates and friends) verbatim on
startup, so a downloaded model — or a wrapper script hung off
``extra_args`` — was enough to trigger this without any adversary.

The fix gives every queue item an out-of-band ``kind`` tag
(``(kind, proc_id, payload)`` for the backend queue, ``(kind, payload)`` for
the CodeRouter queue), and raw child output is *always* enqueued as
``LOG_KIND_LOG`` / ``CR_KIND_LOG`` and never pattern-matched. The tests
below pin that: payload content must be inert.

Design note (same as the sibling GUI test modules): nothing here
instantiates ``tk.Tk()`` / ``LauncherApp``. The dispatch methods
(``_apply_log_event`` / ``_apply_cr_event`` / ``_poll_impl``) are borrowed
onto a plain stub object, which is exactly the point of splitting the
kind→state-transition mapping out of the poll loop.
"""

from __future__ import annotations

import queue
from collections import deque

import pytest

# launcher_gui imports tkinter at module level (it IS a Tk app). Every test
# in this module exercises the Tk-free dispatch helpers, but the import
# itself still needs the tkinter package to exist. Skip cleanly on pythons
# built without Tk support instead of erroring at collection time.
pytest.importorskip("tkinter", reason="launcher_gui requires the tkinter package (python3-tk)")

import launcher_gui as lg

# ---------------------------------------------------------------------------
# Test doubles — the narrow slice of LauncherApp the drain loop touches.
# ---------------------------------------------------------------------------


class _Var:
    """Stand-in for a tk.StringVar."""

    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class _Btn:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def configure(self, **kw: object) -> None:
        self.kwargs.update(kw)


class _Text:
    """Stand-in for the log pane (only exercised for the selected process)."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def configure(self, **_kw: object) -> None:
        pass

    def insert(self, _where: str, text: str) -> None:
        self.chunks.append(text)

    def index(self, _spec: str) -> str:
        return f"{sum(c.count(chr(10)) for c in self.chunks) + 1}.0"

    def delete(self, *_a: object) -> None:
        pass

    def see(self, _where: str) -> None:
        pass


class _FakeApp:
    """Minimal ``self`` for the queue-dispatch methods (no Tk, no display)."""

    # Borrow the real implementations under test.
    _apply_log_event = lg.LauncherApp._apply_log_event
    _apply_cr_event = lg.LauncherApp._apply_cr_event
    _reset_launch_button = lg.LauncherApp._reset_launch_button
    _poll_impl = lg.LauncherApp._poll_impl

    def __init__(self) -> None:
        self._log_queue: queue.Queue = queue.Queue()
        self._cr_log_queue: queue.Queue = queue.Queue()
        self.processes: dict[str, lg.ManagedProcess] = {}
        self.selected_proc_id: str | None = None

        self._pending_probe: object = object()
        self.rendered_probes: list[object] = []

        self._status_msgs: list[str] = []
        self._launch_errs: list[str] = []

        self._port_var = _Var("20500")
        self._name_var = _Var("keep-me")
        self._launch_btn = _Btn()
        self._launch_anim_proc_id: str | None = "anim"
        self._log_text = _Text()

        self._cr_status = "running"
        self._cr_proc: object = object()
        self._cr_log: deque[str] = deque(maxlen=100)
        self._cr_err_var = _Var("")
        self.cr_ui_updates = 0

        self.table_refreshes = 0

    # -- Tk-side collaborators, recorded instead of drawn ------------------
    def _render_devices(self, probe: object) -> None:
        self.rendered_probes.append(probe)

    def _set_status(self, msg: str) -> None:
        self._status_msgs.append(msg)

    def _set_launch_err(self, msg: str) -> None:
        self._launch_errs.append(msg)

    def _update_cr_ui(self) -> None:
        self.cr_ui_updates += 1

    def _refresh_process_table(self) -> None:
        self.table_refreshes += 1


def _managed(proc_id: str = "abc12345", *, status: str = "loading") -> lg.ManagedProcess:
    mp = lg.ManagedProcess(
        id=proc_id,
        name="qwen3-30b",
        backend="llama.cpp",
        model_name="qwen3-30b.gguf",
        port=20501,
        cmd=["llama-server"],
        status=status,
    )
    mp.pid = 4242
    mp.proc = None          # keep _poll_impl's exit sweep out of the way
    return mp


def _app_with_process(proc_id: str = "abc12345", *, selected: bool = True) -> _FakeApp:
    app = _FakeApp()
    app.processes[proc_id] = _managed(proc_id)
    if selected:
        app.selected_proc_id = proc_id
    return app


# ---------------------------------------------------------------------------
# A. Child stdout can never forge a control event (the H-13 regression)
# ---------------------------------------------------------------------------


def test_child_stdout_line_starting_with_err_marker_is_treated_as_log() -> None:
    """THE regression: an ``_ERR_:`` line on the child's stdout used to run
    ``del self.processes[proc_id]``, orphaning a live llama-server."""
    app = _app_with_process()
    mp = app.processes["abc12345"]
    # Exactly what the launch worker enqueues for raw child output.
    app._log_queue.put((lg.LOG_KIND_LOG, "abc12345", "_ERR_:boom"))

    app._poll_impl()

    assert "abc12345" in app.processes, "live process was orphaned by its own stdout"
    assert app.processes["abc12345"] is mp
    assert list(mp.log_lines) == ["_ERR_:boom"]
    assert app._launch_errs == []
    assert "起動失敗" not in app._status_msgs
    # The launch button must not have been reset either.
    assert app._launch_btn.kwargs == {}
    assert app._launch_anim_proc_id == "anim"


def test_child_stdout_line_starting_with_ready_marker_is_treated_as_log() -> None:
    app = _app_with_process()
    mp = app.processes["abc12345"]
    app._log_queue.put((lg.LOG_KIND_LOG, "abc12345", "_READY_:fake:9999"))

    app._poll_impl()

    assert mp.status == "loading"          # no state transition
    assert list(mp.log_lines) == ["_READY_:fake:9999"]
    assert app._status_msgs == []          # no "稼働中: fake" in the status bar
    assert app._port_var.get() == "20500"  # untouched
    assert app._name_var.get() == "keep-me"


def test_child_stdout_line_starting_with_spawned_marker_does_not_crash_drain() -> None:
    """``_SPAWNED_:x`` has one colon too few; the old ``split(":", 2)``
    raised ValueError and threw away the rest of the tick's queue."""
    app = _app_with_process()
    mp = app.processes["abc12345"]
    app._log_queue.put((lg.LOG_KIND_LOG, "abc12345", "_SPAWNED_:x"))
    app._log_queue.put((lg.LOG_KIND_LOG, "abc12345", "load time = 1234 ms"))

    app._poll_impl()   # must not raise

    assert list(mp.log_lines) == ["_SPAWNED_:x", "load time = 1234 ms"]
    assert app._log_queue.empty()
    assert app._port_var.get() == "20500"


def test_malformed_control_payload_skips_only_that_event() -> None:
    """Defence in depth: even a genuine control event with a malformed
    payload must skip just itself, not abort the drain loop."""
    app = _app_with_process()
    mp = app.processes["abc12345"]
    app._log_queue.put((lg.LOG_KIND_SPAWNED, "abc12345", "no-port-here"))
    app._log_queue.put((lg.LOG_KIND_READY, "abc12345", ""))
    app._log_queue.put((lg.LOG_KIND_LOG, "abc12345", "still drained"))

    app._poll_impl()   # must not raise

    lines = list(mp.log_lines)
    assert lines[-1] == "still drained"
    assert any("不正な spawned" in ln for ln in lines)
    assert any("不正な ready" in ln for ln in lines)
    assert app._port_var.get() == "20500"   # no bogus port advance


def test_unknown_kind_is_inert_and_never_interpreted() -> None:
    app = _app_with_process()
    mp = app.processes["abc12345"]
    app._log_queue.put(("totally-unknown", "abc12345", "_ERR_:boom"))

    app._poll_impl()

    assert "abc12345" in app.processes
    assert list(mp.log_lines) == ["_ERR_:boom"]


# ---------------------------------------------------------------------------
# B. Real control events still drive the UI
# ---------------------------------------------------------------------------


def test_error_event_removes_process_and_resets_launch_button() -> None:
    app = _app_with_process()
    app._log_queue.put((lg.LOG_KIND_ERROR, "abc12345", "No such file"))

    app._poll_impl()

    assert "abc12345" not in app.processes
    assert app._launch_errs == ["起動エラー: No such file"]
    assert "起動失敗" in app._status_msgs
    assert app._launch_anim_proc_id is None
    assert app._launch_btn.kwargs["state"] == "normal"


def test_spawned_event_advances_port_and_clears_name() -> None:
    app = _app_with_process()
    app._log_queue.put((lg.LOG_KIND_SPAWNED, "abc12345", "qwen3-30b:20501"))

    app._poll_impl()

    assert app._port_var.get() == "20502"
    assert app._name_var.get() == ""
    assert any("読み込み中: qwen3-30b" in m for m in app._status_msgs)
    assert app.processes["abc12345"].status == "loading"


def test_ready_event_updates_status_for_selected_process() -> None:
    app = _app_with_process()
    app._log_queue.put((lg.LOG_KIND_READY, "abc12345", "qwen3-30b:20501"))

    app._poll_impl()

    assert any("稼働中: qwen3-30b" in m for m in app._status_msgs)


def test_log_event_for_unselected_process_is_stored_but_not_displayed() -> None:
    app = _app_with_process(selected=False)
    app._log_queue.put((lg.LOG_KIND_LOG, "abc12345", "hello"))

    app._poll_impl()

    assert list(app.processes["abc12345"].log_lines) == ["hello"]
    assert app._log_text.chunks == []


def test_log_event_for_selected_process_reaches_the_pane() -> None:
    app = _app_with_process()
    app._log_queue.put((lg.LOG_KIND_LOG, "abc12345", "hello"))

    app._poll_impl()

    assert app._log_text.chunks == ["hello\n"]


# ---------------------------------------------------------------------------
# C. CodeRouter queue — same treatment
# ---------------------------------------------------------------------------


def test_cr_log_queue_uses_tagged_events() -> None:
    """CodeRouter's own stdout printing ``_CR_EXIT_:0`` must not tear the
    supervisor state down."""
    app = _FakeApp()
    live_proc = app._cr_proc
    app._cr_log_queue.put((lg.CR_KIND_LOG, "_CR_EXIT_:0"))
    app._cr_log_queue.put((lg.CR_KIND_LOG, "_CR_ERR_:nope"))
    app._cr_log_queue.put((lg.CR_KIND_LOG, "_CR_OK_:1"))

    app._poll_impl()

    assert app._cr_status == "running"
    assert app._cr_proc is live_proc
    assert app._cr_err_var.get() == ""
    assert app.cr_ui_updates == 0
    assert list(app._cr_log) == ["_CR_EXIT_:0", "_CR_ERR_:nope", "_CR_OK_:1"]


def test_cr_control_events_still_drive_state() -> None:
    app = _FakeApp()
    app._cr_log_queue.put((lg.CR_KIND_OK, "999"))
    app._poll_impl()
    assert app._cr_status == "running"
    assert any("PID 999" in ln for ln in app._cr_log)

    app._cr_log_queue.put((lg.CR_KIND_EXIT, "0"))
    app._poll_impl()
    assert app._cr_status == "stopped"
    assert app._cr_proc is None

    app._cr_log_queue.put((lg.CR_KIND_ERROR, "boom"))
    app._poll_impl()
    assert app._cr_status == "error"
    assert app._cr_err_var.get() == "CodeRouter 起動失敗: boom"


def test_cr_nonzero_exit_marks_error() -> None:
    app = _FakeApp()
    app._cr_log_queue.put((lg.CR_KIND_EXIT, "1"))
    app._poll_impl()
    assert app._cr_status == "error"


# ---------------------------------------------------------------------------
# D. Producers emit tagged events
# ---------------------------------------------------------------------------


def _ready_mp() -> lg.ManagedProcess:
    mp = lg.ManagedProcess(
        id="rdy00001",
        name="qwen3-30b",
        backend="llama.cpp",
        model_name="m.gguf",
        port=20601,
        cmd=["true"],
        status="loading",
    )
    mp.proc = object()
    mp.spawn_gen = 1
    return mp


def test_readiness_worker_emits_ready_event_kind() -> None:
    mp = _ready_mp()
    cfg = lg.LauncherConfig(readiness_timeout_s=5.0, readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()

    lg._readiness_worker(mp, cfg, q, 1, backend_ready=lambda *a, **kw: True)

    items = [q.get_nowait() for _ in range(q.qsize())]
    ready = [it for it in items if it[0] == lg.LOG_KIND_READY]
    assert len(ready) == 1
    kind, proc_id, payload = ready[0]
    assert (kind, proc_id) == ("ready", mp.id)
    assert lg.parse_name_port(payload) == ("qwen3-30b", 20601)
    # No marker prefix survives anywhere in the payloads.
    assert not any(p.startswith("_READY_:") for _k, _p, p in items)


def test_readiness_worker_emits_log_kind_for_plain_lines() -> None:
    mp = _ready_mp()
    cfg = lg.LauncherConfig(readiness_timeout_s=5.0, readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()

    lg._readiness_worker(mp, cfg, q, 1, backend_ready=lambda *a, **kw: True)

    items = [q.get_nowait() for _ in range(q.qsize())]
    assert any(
        kind == lg.LOG_KIND_LOG
        and proc_id == mp.id
        and "readiness check passed" in payload
        for kind, proc_id, payload in items
    )


def test_readiness_worker_timeout_emits_log_kind() -> None:
    mp = _ready_mp()
    cfg = lg.LauncherConfig(readiness_timeout_s=0.05, readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()

    lg._readiness_worker(mp, cfg, q, 1, backend_ready=lambda *a, **kw: False)

    items = [q.get_nowait() for _ in range(q.qsize())]
    assert {kind for kind, _p, _pl in items} == {lg.LOG_KIND_LOG}
    assert any("timed out" in payload for _k, _p, payload in items)


# ---------------------------------------------------------------------------
# E. Devices event — kind tag, not a proc_id sentinel
# ---------------------------------------------------------------------------


def test_devices_event_uses_kind_not_proc_id_sentinel() -> None:
    app = _FakeApp()
    probe = object()
    app._pending_probe = probe

    # The real event: no proc_id at all, just a kind.
    app._log_queue.put((lg.LOG_KIND_DEVICES, "", ""))
    app._poll_impl()
    assert app.rendered_probes == [probe]

    # And the old sentinel is dead: a *log* line whose proc_id happens to be
    # "_DEVICES_" must not re-render the device list. (It matches no known
    # process, so it is simply dropped.)
    app._log_queue.put((lg.LOG_KIND_LOG, "_DEVICES_", ""))
    app._poll_impl()
    assert app.rendered_probes == [probe]


def test_devices_event_needs_no_registered_process() -> None:
    """The devices branch must run before the ``proc_id in processes`` gate —
    it belongs to no process."""
    app = _FakeApp()
    assert app.processes == {}
    app._log_queue.put((lg.LOG_KIND_DEVICES, "", ""))

    app._poll_impl()

    assert len(app.rendered_probes) == 1


# ---------------------------------------------------------------------------
# F. parse_name_port — the defensive payload splitter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("qwen3:20501", ("qwen3", 20501)),
        # Colons in the model name are fine — the port is the LAST field.
        ("qwen3:30b:a3b:20501", ("qwen3:30b:a3b", 20501)),
        ("x", None),
        ("", None),
        (":20501", None),
        ("qwen3:", None),
        ("qwen3:abc", None),
        ("qwen3:20501 ", None),
    ],
)
def test_parse_name_port(payload: str, expected: tuple[str, int] | None) -> None:
    assert lg.parse_name_port(payload) == expected


# ---------------------------------------------------------------------------
# G. Sweep queue kind vocabulary (aligned with LOG_KIND_* / CR_KIND_*)
# ---------------------------------------------------------------------------


def test_sweep_kind_devices_has_no_underscore_prefix() -> None:
    assert lg.SWEEP_KIND_DEVICES == "devices"
    assert lg.SWEEP_KIND_STEP == "step"
    assert lg.SWEEP_KIND_LOG == "log"
    assert lg.SWEEP_KIND_DONE == "done"
