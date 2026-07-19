"""Tests for launcher_gui.py's device selection + bench sweep (Phase 1b GUI).

launcher_gui.py is the Tk desktop launcher. The device-selection and
bench-sweep pure logic was deliberately factored into Tk-independent
module-level functions / dataclasses (mirroring the readiness/auto-restart
split covered by tests/test_launcher_gui_readiness.py) so every case below
exercises them without ever instantiating ``tk.Tk()`` / ``LauncherApp`` /
``SweepWindow`` — no display server, matching the desktop CI constraint
(uv-managed CPython has no tkinter, hence the importorskip below).

Subprocess spawning is fully mocked: no real llama-server / llmbench ever
runs. The shared, frozen core logic lives in coderouter.launcher_devices and
is tested separately; here we only cover the GUI-side glue:

* A. ``_build_cmd(device_args=...)`` — insertion position + backward
  compatibility (the absolute requirement: no device selected ⇒ argv is
  byte-for-byte identical to before) + vllm/mlx ignore device_args.
* B. ``_selection_from_inputs`` — checkbox / manual-entry / tensor-split
  parsing (the pure core of ``_current_selection``).
* C. ``_build_sweep_configs`` — auto-generated sweep configuration matrix.
* D. ``_SweepWorker._run_one`` / ``run`` — the sweep state machine with
  injected fakes (popen / poll_ready), covering the happy path, readiness
  failure, spawn failure, and abort.
* E. ``_load_config`` — the new ``launcher.bench:`` block parsing.
"""

from __future__ import annotations

import queue
import threading

import pytest

# launcher_gui imports tkinter at module level (it IS a Tk app). Skip cleanly
# on pythons built without Tk support instead of erroring at collection time.
pytest.importorskip(
    "tkinter", reason="launcher_gui requires the tkinter package (python3-tk)"
)

import launcher_gui as lg
from coderouter.launcher_devices import (
    DeviceProbe,
    DeviceSelection,
    LlamaDevice,
    SweepPlan,
    SweepState,
    SweepStep,
)

# ---------------------------------------------------------------------------
# A. _build_cmd(device_args=...)
# ---------------------------------------------------------------------------


def test_build_cmd_inserts_device_args_after_port_for_llamacpp() -> None:
    cmd = lg._build_cmd(
        "llama.cpp", "/m/model.gguf", 8080, {}, "", "llama-server",
        None, ["--device", "CUDA0,CUDA1", "--tensor-split", "0.57,0.43"],
    )
    assert cmd == [
        "llama-server", "-m", "/m/model.gguf", "--port", "8080",
        "--device", "CUDA0,CUDA1", "--tensor-split", "0.57,0.43",
    ]


def test_build_cmd_device_args_before_spec_tokens() -> None:
    cmd = lg._build_cmd(
        "llama.cpp", "/m/model.gguf", 8080, {}, "", "llama-server",
        ["--draft", "4"], ["--device", "CUDA0"],
    )
    # device args come right after --port, spec tokens after that.
    port_i = cmd.index("--port")
    dev_i = cmd.index("--device")
    draft_i = cmd.index("--draft")
    assert port_i < dev_i < draft_i


def test_build_cmd_none_device_args_is_backward_compatible() -> None:
    """The absolute backward-compat requirement: omitting device_args (or
    passing None) yields the exact same argv as before the feature existed."""
    baseline = lg._build_cmd(
        "llama.cpp", "/m/model.gguf", 8080, {"-ngl": 99}, "--foo bar",
        "llama-server", ["--draft", "4"],
    )
    with_none = lg._build_cmd(
        "llama.cpp", "/m/model.gguf", 8080, {"-ngl": 99}, "--foo bar",
        "llama-server", ["--draft", "4"], None,
    )
    assert baseline == with_none
    assert "--device" not in with_none


def test_build_cmd_empty_device_args_adds_nothing() -> None:
    cmd = lg._build_cmd(
        "llama.cpp", "/m/model.gguf", 8080, {}, "", "llama-server", None, [],
    )
    assert "--device" not in cmd


def test_build_cmd_vllm_ignores_device_args() -> None:
    cmd = lg._build_cmd(
        "vllm", "/m/model", 8000, {}, "", "python", None,
        ["--device", "CUDA0"],
    )
    assert "--device" not in cmd


def test_build_cmd_mlx_ignores_device_args() -> None:
    cmd = lg._build_cmd(
        "mlx", "/m/model", 8000, {}, "", "python", None,
        ["--device", "Metal"],
    )
    assert "--device" not in cmd


# ---------------------------------------------------------------------------
# B. _selection_from_inputs — the pure core of _current_selection
# ---------------------------------------------------------------------------


def test_selection_from_inputs_empty_is_inactive() -> None:
    sel = lg._selection_from_inputs([], "", "")
    assert sel.device_ids == []
    assert sel.active is False
    assert sel.to_cli_args() == []


def test_selection_from_inputs_uses_checked_ids() -> None:
    sel = lg._selection_from_inputs(["CUDA0", "CUDA1"], "", "0.5,0.5")
    assert sel.device_ids == ["CUDA0", "CUDA1"]
    assert sel.tensor_split == [0.5, 0.5]
    assert sel.to_cli_args() == [
        "--device", "CUDA0,CUDA1", "--tensor-split", "0.5,0.5",
    ]


def test_selection_from_inputs_manual_fallback_when_no_checkboxes() -> None:
    sel = lg._selection_from_inputs([], " CUDA0 , CUDA1 ", "")
    assert sel.device_ids == ["CUDA0", "CUDA1"]


def test_selection_from_inputs_checked_ids_win_over_fallback() -> None:
    sel = lg._selection_from_inputs(["CUDA2"], "CUDA0,CUDA1", "")
    assert sel.device_ids == ["CUDA2"]


def test_selection_from_inputs_malformed_tsplit_is_dropped() -> None:
    sel = lg._selection_from_inputs(["CUDA0", "CUDA1"], "", "abc,0.5")
    # ValueError on float("abc") → whole tensor-split dropped (best-effort).
    assert sel.tensor_split == []


# ---------------------------------------------------------------------------
# C. _build_sweep_configs
# ---------------------------------------------------------------------------


def _dev(id_: str, total: int) -> LlamaDevice:
    return LlamaDevice(id=id_, name=f"GPU {id_}", total_mib=total, free_mib=total)


def test_build_sweep_configs_single_device_no_split() -> None:
    configs = lg._build_sweep_configs([_dev("CUDA0", 32149)])
    assert len(configs) == 1
    _label, sel = configs[0]
    assert sel.device_ids == ["CUDA0"]
    assert sel.tensor_split == []


def test_build_sweep_configs_two_devices_adds_split_config() -> None:
    configs = lg._build_sweep_configs([_dev("CUDA0", 32149), _dev("CUDA1", 24123)])
    # CUDA0単体 + CUDA1単体 + 全デバイス+split
    assert len(configs) == 3
    labels = [c[0] for c in configs]
    assert any("単体" in lbl for lbl in labels)
    combined = configs[-1][1]
    assert combined.device_ids == ["CUDA0", "CUDA1"]
    assert combined.tensor_split == [0.57, 0.43]
    assert abs(sum(combined.tensor_split) - 1.0) < 1e-9


def test_build_sweep_configs_empty_devices() -> None:
    assert lg._build_sweep_configs([]) == []


def test_build_sweep_configs_mac_mtl_blas_only_single_config() -> None:
    """macOS: MTL0 (real GPU) + BLAS: Accelerate (0 MiB) — the 0 MiB BLAS
    device must be excluded, leaving a single MTL0-only config, no split."""
    devices = [_dev("MTL0", 49152), _dev("BLAS", 0)]
    configs = lg._build_sweep_configs(devices)
    assert len(configs) == 1
    label, sel = configs[0]
    assert sel.device_ids == ["MTL0"]
    assert sel.tensor_split == []
    assert "MTL0" in label
    # selectable filtering (via lg's imported helper) drops BLAS entirely.
    assert [d.id for d in lg.selectable_devices(devices)] == ["MTL0"]


def test_build_sweep_configs_cuda_vulkan_no_cross_backend_mix() -> None:
    """CUDA+Vulkan build enumerates the same physical GPU under both backends.
    The auto configs must never mix CUDA and Vulkan ids, and each backend with
    2+ cards gets its own split config."""
    devices = [
        _dev("CUDA0", 32149), _dev("CUDA1", 24123),
        _dev("Vulkan0", 32149), _dev("Vulkan2", 114000),
    ]
    configs = lg._build_sweep_configs(devices)
    # 4 single + "CUDA x2" + "Vulkan x2"
    assert len(configs) == 6
    for _label, sel in configs:
        backends = {lg.backend_of(i) for i in sel.device_ids}
        assert len(backends) <= 1, f"cross-backend mix leaked: {sel.device_ids}"
    # the CUDA multi-card config carries a proportional split summing to 1.0.
    cuda_multi = next(
        sel for lbl, sel in configs
        if len(sel.device_ids) > 1 and lg.backend_of(sel.device_ids[0]) == "CUDA"
    )
    assert cuda_multi.device_ids == ["CUDA0", "CUDA1"]
    assert abs(sum(cuda_multi.tensor_split) - 1.0) < 1e-9


def test_selectable_devices_excludes_zero_mib_blas() -> None:
    devices = [_dev("MTL0", 49152), _dev("BLAS", 0), _dev("CUDA0", 32149)]
    selectable = lg.selectable_devices(devices)
    assert [d.id for d in selectable] == ["MTL0", "CUDA0"]


def test_backend_of_strips_trailing_digits() -> None:
    assert lg.backend_of("CUDA0") == "CUDA"
    assert lg.backend_of("Vulkan2") == "Vulkan"
    assert lg.backend_of("MTL0") == "MTL"
    assert lg.backend_of("BLAS") == "BLAS"


# ---------------------------------------------------------------------------
# D. _SweepWorker — the sweep state machine (all subprocesses mocked)
# ---------------------------------------------------------------------------


class _FakeProc:
    """Stand-in for a subprocess.Popen: alive server or an exited bench."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.stdout = None  # → _pump_logs skips draining
        self.terminated = False
        self.killed = False

    def poll(self) -> None:
        return None  # server stays "alive" during readiness

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _make_popen(procs: list[_FakeProc]):
    it = iter(procs)

    def _popen(*_a: object, **_kw: object) -> _FakeProc:
        return next(it)

    return _popen


def _plan(**overrides: object) -> SweepPlan:
    base = dict(
        steps=[SweepStep(label="CUDA0 単体", selection=DeviceSelection(["CUDA0"]))],
        model_path="/m/model.gguf",
        backend="llama.cpp",
        port=28080,
        bench_cmd_template="llmbench run --runs {runs}",
        results_dir=None,
    )
    base.update(overrides)
    return SweepPlan(**base)  # type: ignore[arg-type]


def test_sweep_worker_happy_path_reaches_done() -> None:
    plan = _plan()
    cfg = lg.LauncherConfig(bench_readiness_timeout_s=5.0,
                            readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()
    server, bench = _FakeProc(), _FakeProc(returncode=0)
    worker = lg._SweepWorker(
        plan, cfg, q, threading.Event(), runs=1,
        popen=_make_popen([server, bench]),
        poll_ready=lambda **_kw: "ready",
    )

    step = plan.steps[0]
    worker._run_one(step)

    assert step.state == SweepState.DONE
    assert step.bench_exit_code == 0
    # the worker emits a step update at each transition (STARTING → BENCHING →
    # DONE): at least three "step" events reached the queue.
    step_events = [p for k, p in _drain(q) if k == "step"]
    assert len(step_events) >= 3
    # server was stopped (port freed) before moving on.
    assert server.terminated is True


def test_sweep_worker_nonzero_bench_exit_is_still_done() -> None:
    """A non-zero bench exit is recorded via bench_exit_code, NOT treated as a
    sweep failure — the run completed and the number is for comparison."""
    plan = _plan()
    cfg = lg.LauncherConfig(bench_readiness_timeout_s=5.0,
                            readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()
    worker = lg._SweepWorker(
        plan, cfg, q, threading.Event(), runs=1,
        popen=_make_popen([_FakeProc(), _FakeProc(returncode=2)]),
        poll_ready=lambda **_kw: "ready",
    )
    step = plan.steps[0]
    worker._run_one(step)
    assert step.state == SweepState.DONE
    assert step.bench_exit_code == 2


def test_sweep_worker_readiness_timeout_marks_failed() -> None:
    plan = _plan()
    cfg = lg.LauncherConfig(bench_readiness_timeout_s=0.05,
                            readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()
    server = _FakeProc()
    worker = lg._SweepWorker(
        plan, cfg, q, threading.Event(), runs=1,
        popen=_make_popen([server]),
        poll_ready=lambda **_kw: "timeout",
    )
    step = plan.steps[0]
    worker._run_one(step)
    assert step.state == SweepState.FAILED
    assert "readiness" in (step.error or "")
    assert server.terminated is True  # server killed even on failure


def test_sweep_worker_spawn_failure_marks_failed() -> None:
    plan = _plan()
    cfg = lg.LauncherConfig()
    q: queue.Queue = queue.Queue()

    def _boom(*_a: object, **_kw: object):
        raise OSError("no such binary")

    worker = lg._SweepWorker(
        plan, cfg, q, threading.Event(), runs=1, popen=_boom,
        poll_ready=lambda **_kw: "ready",
    )
    step = plan.steps[0]
    worker._run_one(step)
    assert step.state == SweepState.FAILED
    assert "起動失敗" in (step.error or "")


def test_sweep_worker_abort_marks_all_steps_aborted() -> None:
    steps = [
        SweepStep(label="A", selection=DeviceSelection(["CUDA0"])),
        SweepStep(label="B", selection=DeviceSelection(["CUDA1"])),
    ]
    plan = _plan(steps=steps)
    cfg = lg.LauncherConfig()
    q: queue.Queue = queue.Queue()
    abort = threading.Event()
    abort.set()  # aborted before the run even starts

    called: list[int] = []
    worker = lg._SweepWorker(
        plan, cfg, q, abort, runs=1,
        popen=lambda *a, **k: called.append(1),  # must never be called
        poll_ready=lambda **_kw: "ready",
    )
    worker.run()

    assert all(s.state == SweepState.ABORTED for s in steps)
    assert called == []  # no process ever spawned
    kinds = [k for k, _ in _drain(q)]
    assert kinds[-1] == "done"


def test_sweep_worker_run_emits_done_at_end() -> None:
    plan = _plan()
    cfg = lg.LauncherConfig(bench_readiness_timeout_s=5.0,
                            readiness_poll_interval_s=0.01)
    q: queue.Queue = queue.Queue()
    worker = lg._SweepWorker(
        plan, cfg, q, threading.Event(), runs=1,
        popen=_make_popen([_FakeProc(), _FakeProc()]),
        poll_ready=lambda **_kw: "ready",
    )
    worker.run()
    kinds = [k for k, _ in _drain(q)]
    assert kinds[-1] == "done"


def _drain(q: queue.Queue) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ---------------------------------------------------------------------------
# E. _load_config — launcher.bench: block parsing
# ---------------------------------------------------------------------------


def test_load_config_bench_defaults_when_absent(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text("launcher:\n  model_dirs: []\n")

    cfg = lg._load_config(str(p))

    assert cfg.bench_command_template == lg._DEFAULT_BENCH_COMMAND_TEMPLATE
    assert cfg.bench_runs == lg._DEFAULT_BENCH_RUNS == 5
    assert cfg.bench_results_dir is None
    assert cfg.bench_readiness_timeout_s == lg._DEFAULT_READINESS_TIMEOUT_S


def test_load_config_no_launcher_block_bench_defaults(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text("allow_paid: false\n")

    cfg = lg._load_config(str(p))

    assert cfg.bench_command_template == lg._DEFAULT_BENCH_COMMAND_TEMPLATE
    assert cfg.bench_runs == 5


def test_load_config_bench_overrides(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text(
        "launcher:\n"
        "  bench:\n"
        "    command_template: 'llmbench run --base-url {base_url} --runs {runs}'\n"
        "    runs: 10\n"
        "    results_dir: /tmp/results\n"
        "    readiness_timeout_s: 120\n"
    )

    cfg = lg._load_config(str(p))

    assert cfg.bench_command_template == (
        "llmbench run --base-url {base_url} --runs {runs}"
    )
    assert cfg.bench_runs == 10
    assert cfg.bench_results_dir == "/tmp/results"
    assert cfg.bench_readiness_timeout_s == 120.0


def test_load_config_bench_malformed_runs_falls_back(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text("launcher:\n  bench:\n    runs: [not, a, number]\n")

    cfg = lg._load_config(str(p))

    assert cfg.bench_runs == lg._DEFAULT_BENCH_RUNS


def test_load_config_bench_blank_template_falls_back(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text("launcher:\n  bench:\n    command_template: '   '\n")

    cfg = lg._load_config(str(p))

    assert cfg.bench_command_template == lg._DEFAULT_BENCH_COMMAND_TEMPLATE


def test_load_config_bench_non_string_results_dir_becomes_none(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text("launcher:\n  bench:\n    results_dir: 123\n")

    cfg = lg._load_config(str(p))

    assert cfg.bench_results_dir is None


def test_device_probe_fallback_render_is_pure(tmp_path) -> None:
    """A failed probe carries ok=False + error; _render_devices consumes it via
    the manual-entry fallback (UI path is display-dependent, but the DeviceProbe
    contract the GUI relies on is asserted here)."""
    probe = DeviceProbe([], ok=False, error="not found")
    assert probe.ok is False
    assert probe.error == "not found"
