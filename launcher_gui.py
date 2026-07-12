#!/usr/bin/env python3
"""
CodeRouter Launcher GUI — tkinter 版
llama.cpp / vllm / mlx と CodeRouter をブラウザなしで起動・管理するデスクトップツール

使い方:
  python3 launcher_gui.py
  python3 launcher_gui.py --config ~/.coderouter/providers.yaml
  uv run python launcher_gui.py

追加パッケージ: 不要 (tkinter は Python 標準、yaml は CodeRouter の依存)

起動フロー:
  launcher_gui.py 起動
    → ① llama.cpp / vllm / mlx を選択モデルで起動 (ポート 8080)
    → ② CodeRouter を起動 (ポート 8088)  ← ★ このGUIから直接起動
    → Claude Code: ANTHROPIC_BASE_URL=http://localhost:8088 claude
"""

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import queue
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

# ---------------------------------------------------------------------------
# YAML loading (optional — graceful fallback)
# ---------------------------------------------------------------------------
try:
    import yaml  # PyYAML (CodeRouter already depends on it)
    def _load_yaml(p: Path) -> dict:
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
except ImportError:
    yaml = None  # type: ignore
    def _load_yaml(p: Path) -> dict:  # type: ignore
        raise RuntimeError(
            "PyYAML が見つかりません。CodeRouter の venv から実行してください:\n"
            "  uv run python launcher_gui.py"
        )

# ---------------------------------------------------------------------------
# MTP / speculative-decoding resolution (optional — shared with the web UI)
# ---------------------------------------------------------------------------
# The GUI can run standalone without the coderouter package on sys.path; when
# the import fails we degrade gracefully and simply skip the MTP features.
try:
    from coderouter.launcher_speculative import resolve_speculative
    _HAS_SPECULATIVE = True
except ImportError:
    resolve_speculative = None  # type: ignore
    _HAS_SPECULATIVE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MODEL_EXTS = {".gguf", ".ggml", ".safetensors", ".bin", ".pt", ".pth"}

_BACKEND_DEFAULTS = {
    "llama.cpp": "llama-server",
    "vllm": "python",
    "mlx": "python",          # mlx_lm.server (Apple Silicon 向け)
}

# CodeRouter のデフォルトポート (README / docs に揃えて 8088)
_CODEROUTER_PORT = 8088

# ── ログ蓄積の上限(ビーチボール対策) ──────────────────────────────────────
# 長時間稼働でログが無制限に溜まり、メインスレッドの処理が追いつかなくなって
# UI が固まる(くるくる)のを防ぐための上限値。
_MAX_LOG_LINES      = 5000   # mp.log_lines / _cr_log のメモリ上限(行数)
_MAX_TEXT_LINES     = 2000   # _log_text ウィジェットの表示行上限
_MAX_LINES_PER_TICK = 1500   # _poll 1回で処理する最大行数(残りは次回へ繰越)

# 自動検出 MTP が起動時にクラッシュした場合、この秒数以内の非ゼロ終了だけを
# 「起動時失敗」とみなし speculative フラグ無しで 1 回だけ再起動する。これより
# 長く稼働してから落ちたものは通常のクラッシュ扱いで再起動しない。
_MTP_FALLBACK_WINDOW_SECS = 180.0

# ---------------------------------------------------------------------------
# Readiness gating / auto-restart defaults
#
# Web 版 (coderouter/ingress/launcher_routes.py の _wait_ready_and_register /
# _attempt_restart) と挙動・既定値を揃える。根拠は
# coderouter/config/schemas.py の LauncherConfig 該当フィールドの docstring。
# GUI にはこの launcher: ブロックを pydantic 検証する仕組みが無いため、値は
# providers.yaml から緩く読み取り、型が壊れていれば既定値にフォールバックする
# (_load_config の他フィールドと同じ寛容な流儀)。
# ---------------------------------------------------------------------------
_DEFAULT_READINESS_TIMEOUT_S = 300.0
_DEFAULT_READINESS_POLL_INTERVAL_S = 2.0
# 個々のプローブ用ネットワークタイムアウト。readiness_timeout_s(既定300s)
# より十分短い固定値にして、1回のプローブが詰まっても loading→error の締切
# 判定が大きく遅れないようにする(既定の poll interval 2.0s よりは長い点に
# 注意 — プローブが3秒詰まれば次のポーリングはその分だけ遅れる)。
_READINESS_PROBE_TIMEOUT_S = 3.0
_DEFAULT_AUTO_RESTART = False
_DEFAULT_AUTO_RESTART_MAX_ATTEMPTS = 3
_DEFAULT_AUTO_RESTART_BACKOFF_S = 2.0
_DEFAULT_AUTO_RESTART_BACKOFF_MAX_S = 30.0

_CONFIG_SEARCH = [
    Path.cwd() / "providers.yaml",
    Path.home() / ".coderouter" / "providers.yaml",
]


@dataclass
class BackendConfig:
    binary: str | None = None


@dataclass
class OptionProfile:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LauncherConfig:
    model_dirs: list[str] = field(default_factory=list)
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    option_profiles: dict[str, list[OptionProfile]] = field(default_factory=dict)
    # --- readiness gating / auto-restart — see the block comment above
    #     _DEFAULT_READINESS_TIMEOUT_S for the source of truth (Web版と同一
    #     既定値)。 swap: は Web 版の SwapManager 専用機能であり、GUI 版では
    #     意図的に読み込まない。
    readiness_timeout_s: float = _DEFAULT_READINESS_TIMEOUT_S
    readiness_poll_interval_s: float = _DEFAULT_READINESS_POLL_INTERVAL_S
    auto_restart: bool = _DEFAULT_AUTO_RESTART
    auto_restart_max_attempts: int = _DEFAULT_AUTO_RESTART_MAX_ATTEMPTS
    auto_restart_backoff_s: float = _DEFAULT_AUTO_RESTART_BACKOFF_S
    auto_restart_backoff_max_s: float = _DEFAULT_AUTO_RESTART_BACKOFF_MAX_S


def _safe_number(raw: dict, key: str, default: float, cast: Callable[[Any], Any]) -> Any:
    """Read ``raw[key]`` through ``cast``, falling back to ``default``.

    The GUI has no pydantic validation for the ``launcher:`` block (unlike
    ``coderouter.config.schemas.LauncherConfig``), so a malformed value
    (wrong type, missing key) must never crash the whole GUI at startup —
    mirrors the already-lenient parsing of ``model_dirs`` / ``backends``
    just above.
    """
    if key not in raw:
        return default
    try:
        return cast(raw[key])
    except (TypeError, ValueError):
        return default


def _safe_bool(raw: dict, key: str, default: bool) -> bool:
    """Read a boolean out of ``raw[key]``, falling back to ``default``.

    Deliberately NOT ``bool(...)``: a quoted YAML string like
    ``auto_restart: "false"`` is a non-empty str, so ``bool("false")`` is
    True — silently enabling a side-effectful opt-in the user explicitly
    tried to turn off. Strings "true"/"false" (case-insensitive) are
    parsed; real bools pass through; anything else falls back.
    """
    if key not in raw:
        return default
    v = raw[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
    return default


def _load_config(path: str | None) -> LauncherConfig:
    cfg_path: Path | None = None
    if path:
        cfg_path = Path(path).expanduser()
    else:
        cfg_path = next((p for p in _CONFIG_SEARCH if p.is_file()), None)

    if cfg_path is None:
        return LauncherConfig()  # empty config — user can still set options manually

    try:
        raw = _load_yaml(cfg_path)
    except Exception as exc:
        messagebox.showwarning("設定ファイル読み込みエラー", str(exc))
        return LauncherConfig()

    launcher_raw = raw.get("launcher", {}) or {}

    # model_dirs
    model_dirs = [str(d) for d in launcher_raw.get("model_dirs", [])]

    # backends
    backends: dict[str, BackendConfig] = {}
    for bk, bv in (launcher_raw.get("backends", {}) or {}).items():
        if isinstance(bv, dict):
            backends[bk] = BackendConfig(binary=bv.get("binary"))

    # option_profiles
    option_profiles: dict[str, list[OptionProfile]] = {}
    for bk, profiles_raw in (launcher_raw.get("option_profiles", {}) or {}).items():
        profs: list[OptionProfile] = []
        for p in (profiles_raw or []):
            if isinstance(p, dict) and "name" in p:
                profs.append(OptionProfile(name=p["name"], args=p.get("args", {}) or {}))
        option_profiles[bk] = profs

    # readiness gating / auto-restart — same keys & defaults as the Web 版
    # ``launcher:`` block (coderouter/config/schemas.py LauncherConfig).
    # ``swap:`` is intentionally never read here (see LauncherConfig above).
    readiness_timeout_s = _safe_number(
        launcher_raw, "readiness_timeout_s", _DEFAULT_READINESS_TIMEOUT_S, float)
    readiness_poll_interval_s = _safe_number(
        launcher_raw, "readiness_poll_interval_s",
        _DEFAULT_READINESS_POLL_INTERVAL_S, float)
    auto_restart = _safe_bool(
        launcher_raw, "auto_restart", _DEFAULT_AUTO_RESTART)
    auto_restart_max_attempts = _safe_number(
        launcher_raw, "auto_restart_max_attempts",
        _DEFAULT_AUTO_RESTART_MAX_ATTEMPTS, int)
    auto_restart_backoff_s = _safe_number(
        launcher_raw, "auto_restart_backoff_s",
        _DEFAULT_AUTO_RESTART_BACKOFF_S, float)
    auto_restart_backoff_max_s = _safe_number(
        launcher_raw, "auto_restart_backoff_max_s",
        _DEFAULT_AUTO_RESTART_BACKOFF_MAX_S, float)
    if auto_restart_backoff_s > auto_restart_backoff_max_s:
        # Same fast-fail *intent* as schemas.py's
        # _check_auto_restart_backoff_ordered validator, but the GUI just
        # clamps instead of refusing to start — a malformed config must
        # never block the desktop tool from opening.
        auto_restart_backoff_max_s = auto_restart_backoff_s

    return LauncherConfig(
        model_dirs=model_dirs,
        backends=backends,
        option_profiles=option_profiles,
        readiness_timeout_s=readiness_timeout_s,
        readiness_poll_interval_s=readiness_poll_interval_s,
        auto_restart=auto_restart,
        auto_restart_max_attempts=auto_restart_max_attempts,
        auto_restart_backoff_s=auto_restart_backoff_s,
        auto_restart_backoff_max_s=auto_restart_backoff_max_s,
    )


def _resolve_binary(backend: str, cfg: LauncherConfig) -> str:
    bc = cfg.backends.get(backend)
    raw = (bc.binary if bc else None) or _BACKEND_DEFAULTS.get(backend, backend)
    return str(Path(raw).expanduser())


def _check_binary(binary: str) -> bool:
    expanded = str(Path(binary).expanduser())
    return Path(expanded).is_file() or shutil.which(expanded) is not None


def _scan_models(model_dirs: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in model_dirs:
        base = Path(d).expanduser()
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in _MODEL_EXTS:
                continue
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            size_gb = p.stat().st_size / (1024 ** 3)
            results.append({
                "path": str(p),
                "name": p.name,
                "dir": str(p.parent),
                "size_gb": round(size_gb, 2),
            })
    return results


def _build_cmd(backend: str, model_path: str, port: int,
               profile_args: dict[str, Any], extra_args: str,
               binary: str,
               spec_tokens: list[str] | None = None) -> list[str]:
    """Assemble the backend launch command.

    ``spec_tokens`` are pre-resolved MTP / speculative-decoding flags (from
    :func:`coderouter.launcher_speculative.resolve_speculative`). For
    llama.cpp they are inserted right after the port args, before the profile
    / extra args.
    """
    if backend == "llama.cpp":
        cmd = [binary, "-m", model_path, "--port", str(port)]
        if spec_tokens:
            cmd.extend(spec_tokens)
    elif backend == "vllm":
        cmd = [binary, "-m", "vllm.entrypoints.openai.api_server",
               "--model", model_path, "--port", str(port)]
    elif backend == "mlx":
        cmd = [binary, "-m", "mlx_lm.server",
               "--model", model_path, "--port", str(port)]
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    for flag, val in profile_args.items():
        if isinstance(val, bool):
            if val:
                cmd.append(str(flag))
        else:
            cmd.extend([str(flag), str(val)])

    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    return cmd


# ---------------------------------------------------------------------------
# Hardware detection + model recommendation (luna-go /models 互換の発想)
# ---------------------------------------------------------------------------

def _detect_hardware() -> dict[str, Any]:
    """ハードウェアを best-effort で検出する (stdlib + CLI、追加依存なし)。"""
    cpu = os.cpu_count() or 4
    ram_gb = 0.0
    with contextlib.suppress(ValueError, OSError, AttributeError):
        ram_gb = (os.sysconf("SC_PHYS_PAGES")
                  * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3))
    if ram_gb <= 0:
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=3)
            ram_gb = int(out.stdout.strip()) / (1024 ** 3)
        except (ValueError, OSError, subprocess.SubprocessError):
            pass
    gpu, vram_gb = "cpu", 0.0
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        gpu, vram_gb = "metal", ram_gb            # ユニファイドメモリ
    elif shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            mb = max((int(x) for x in out.stdout.split() if x.strip().isdigit()),
                     default=0)
            if mb > 0:
                gpu, vram_gb = "cuda", mb / 1024
        except (ValueError, OSError, subprocess.SubprocessError):
            pass
    return {"ram_gb": round(ram_gb, 1), "vram_gb": round(vram_gb, 1),
            "gpu": gpu, "cpu_count": cpu}


def _hw_summary(hw: dict[str, Any]) -> str:
    """検出ハードを 1 行で表す (UI 表示用)。"""
    gpu_label = {"metal": "Metal", "cuda": "CUDA", "cpu": "CPU"}.get(
        hw.get("gpu", "cpu"), "CPU")
    parts = [gpu_label, f"RAM {hw.get('ram_gb', 0):g}GB"]
    if hw.get("gpu") == "cuda" and hw.get("vram_gb"):
        parts.append(f"VRAM {hw['vram_gb']:g}GB")
    return " · ".join(parts)


def _usable_memory_gb(hw: dict[str, Any]) -> float:
    """モデルの重み + KV キャッシュに使えるメモリ量。"""
    if hw.get("gpu") == "cuda":
        return float(hw.get("vram_gb", 0.0))
    return float(hw.get("ram_gb", 0.0))          # metal (ユニファイド) / cpu


def _model_recommendation(size_gb: float, hw: dict[str, Any]) -> dict[str, str]:
    """モデル単位のメモリ適合判定 (luna-go /models 相当)。

    level: "ok" (推奨) | "warn" (メモリ厳しい) | "unknown"
    """
    usable = _usable_memory_gb(hw)
    if usable <= 0 or size_gb <= 0:
        return {"level": "unknown", "label": "—"}
    if size_gb * 1.2 + 2.0 <= usable:
        return {"level": "ok", "label": "推奨"}
    return {"level": "warn", "label": "メモリ厳しい"}


def _suggest_launch_flags(backend: str, size_gb: float,
                          hw: dict[str, Any]) -> str:
    """選択モデル + ハード + バックエンドから推奨起動フラグを提案する。

    バックエンドごとにフラグ体系が違うため分岐する:
      - llama.cpp : -ngl / --ctx-size / --threads を算出
      - vllm      : モデル config からの自動導出に任せる (空文字)
      - mlx       : 統合メモリ前提で起動時フラグ不要 (空文字)
    あくまで目安。他プロセスのメモリ使用や量子化方式までは考慮しない。
    """
    if backend == "mlx":
        # MLX は統合メモリ + Metal 前提。llama.cpp の -ngl に相当する
        # レイヤーオフロードの概念がなく、mlx_lm.server は起動時の
        # 性能チューニングフラグを取らない。
        return ""
    if backend == "vllm":
        # vllm の --max-model-len はモデルの実コンテキスト長に依存する。
        # メモリ量だけのヒューリスティックで値を出すと、モデルの上限を
        # 超えたときに vllm が起動を拒否する。空にしてエンジンの
        # 自動導出 (モデル config) に任せるのが安全。
        return ""

    # llama.cpp (デフォルト)
    usable = _usable_memory_gb(hw)
    weights = size_gb * 1.15                       # 重み + オーバーヘッド概算
    threads = max(1, int(hw.get("cpu_count", 4)) - 2)
    if hw.get("gpu") == "cpu":
        ngl = 0
    elif usable >= weights + 1.0:
        ngl = 99                                   # 全レイヤー GPU に載る
    elif usable > 1.5:
        ngl = max(0, min(99, int(99 * (usable - 0.7) / max(weights, 0.1))))
    else:
        ngl = 0
    headroom = usable - weights - 1.0
    if headroom >= 8:
        ctx = 32768
    elif headroom >= 4:
        ctx = 16384
    elif headroom >= 2:
        ctx = 8192
    else:
        ctx = 4096
    return f"-ngl {ngl} --ctx-size {ctx} --threads {threads}"


# ---------------------------------------------------------------------------
# CodeRouter helpers
# ---------------------------------------------------------------------------

def _find_coderouter_cmd() -> list[str]:
    """CodeRouter の起動コマンドプレフィクスを返す。

    優先順位:
      1. PATH の coderouter (pip install / uvx で入れた場合)
      2. uv run coderouter  (プロジェクト venv)
      3. python -m coderouter (フォールバック)
    """
    if shutil.which("coderouter"):
        return ["coderouter"]
    if shutil.which("uv"):
        return ["uv", "run", "coderouter"]
    return [sys.executable, "-m", "coderouter"]


def _ensure_providers_yaml(llama_port: int) -> tuple[bool, str]:
    """~/.coderouter/providers.yaml が存在しない場合だけ自動生成する。

    Returns:
        (created, path) — created=True なら今回新しく作った。
    """
    config_dir = Path.home() / ".coderouter"
    config_path = config_dir / "providers.yaml"

    if config_path.exists():
        return False, str(config_path)

    config_dir.mkdir(parents=True, exist_ok=True)
    content = f"""\
# CodeRouter providers.yaml — launcher_gui.py により自動生成
# 手動で編集して構いません。詳細は examples/providers.yaml を参照。

allow_paid: false
default_profile: default

providers:
  - name: llama-cpp-local
    kind: openai_compat
    base_url: http://localhost:{llama_port}/v1
    model: ""          # llama-server はモデル名を問わないので空でOK
    timeout_s: 120
    capabilities:
      chat: true
      streaming: true
      tools: true

profiles:
  - name: default
    providers: [llama-cpp-local]
"""
    config_path.write_text(content, encoding="utf-8")
    return True, str(config_path)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

@dataclass
class ManagedProcess:
    id: str
    name: str
    backend: str
    model_name: str
    port: int
    cmd: list[str]
    # "starting" (constructing, pre-spawn) | "loading" (Popen succeeded,
    # waiting on the readiness probe — see _readiness_worker) | "running"
    # (readiness confirmed) | "stopping" (Stop/Kill requested, transient
    # display state) | "stopped" | "error"
    status: str = "starting"
    pid: int | None = None
    returncode: int | None = None
    proc: Any = None
    # 無制限肥大化を防ぐため上限付き deque を使用(古い行から自動破棄)
    log_lines: deque[str] = field(
        default_factory=lambda: deque(maxlen=_MAX_LOG_LINES))
    # Set by _do_stop / _do_kill just before signalling the child. Tells the
    # launch worker thread the exit was requested, not a crash, so it never
    # auto-restarts and never mislabels whatever exit code SIGTERM/SIGKILL
    # produced as "error" (mirrors ManagedProcess.stopping in
    # coderouter/ingress/launcher_routes.py).
    stopping: bool = False
    # Consecutive generic auto-restart attempts since the last readiness
    # success. Reset to 0 by _readiness_worker once it confirms "running".
    restart_count: int = 0
    started_at: float = 0.0
    # Bumped on every (re)spawn — initial launch, MTP fallback relaunch, or
    # generic auto-restart. A readiness worker captures its own generation
    # at start and aborts once it no longer matches, so a stale worker left
    # over from a previous spawn attempt can never overwrite the status set
    # by a newer one (mirrors the supersede-safety of
    # ManagedProcess.ready.clear() in launcher_routes.py).
    spawn_gen: int = 0


# ---------------------------------------------------------------------------
# Readiness gating — ported from coderouter/ingress/launcher_routes.py
# (_backend_ready / _wait_ready_and_register). A backend used to be shown as
# "running" the instant the OS process spawned, before llama-server / vllm
# had actually finished loading the model — the GUI would claim success
# while the model was still loading. Now the launch worker thread stays in
# "loading" until a poll confirms the backend is actually serving, or the
# poll deadline is exceeded (status becomes "error"; the process itself is
# left running so the user can inspect logs / stop it manually).
# ---------------------------------------------------------------------------


def _backend_ready(backend: str, port: int, *, probe_timeout_s: float) -> bool:
    """Best-effort single readiness probe. Never raises.

    llama.cpp and vllm both expose ``GET /health`` (200 once the model is
    loaded and the server is accepting requests; llama.cpp returns 503
    while still loading). Other backends (mlx — mlx_lm.server has no
    documented health endpoint) fall back to a bare TCP connect: it can't
    distinguish "loaded" from "listening", but it is a strict improvement
    over showing "running" before the port is even open.
    """
    if backend in ("llama.cpp", "vllm"):
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            with urllib.request.urlopen(req, timeout=probe_timeout_s) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return False

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=probe_timeout_s):
            return True
    except OSError:
        return False


def poll_until_ready(
    *,
    check: Callable[[], bool],
    should_abort: Callable[[], bool],
    timeout_s: float,
    poll_interval_s: float,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> str:
    """Generic poll loop, decoupled from Tk / threading / sockets.

    Returns one of:
      * ``"ready"``   — ``check()`` returned True before the deadline.
      * ``"timeout"`` — the deadline passed with ``check()`` never True.
      * ``"aborted"`` — ``should_abort()`` returned True (the caller is no
        longer in a loading-eligible state: crashed, stopped, or a newer
        spawn superseded this one) — checked both before every probe and
        immediately after a successful one, so a fast state change can
        never race a stale "ready" outcome in.

    Injectable ``sleep`` / ``now`` make this fully unit-testable without
    real timing. Mirrors the loop shape of
    ``launcher_routes._wait_ready_and_register``.
    """
    deadline = now() + timeout_s
    while now() < deadline:
        if should_abort():
            return "aborted"
        if check():
            return "aborted" if should_abort() else "ready"
        sleep(poll_interval_s)
    return "timeout"


def _readiness_worker(
    mp: ManagedProcess,
    cfg: LauncherConfig,
    log_queue: queue.Queue[tuple[str, str]],
    gen: int,
    *,
    backend_ready: Callable[..., bool] = _backend_ready,
) -> None:
    """Poll ``mp`` for readiness, then flip it to "running" — or "error".

    Runs as an independent daemon thread per spawn (initial, MTP fallback,
    or auto-restart), started right after Popen succeeds. Takes plain data
    (no ``self``) so it can be driven directly in tests without a live Tk
    app — mirrors the signature shape of ``_wait_ready_and_register``.
    """

    def _abort() -> bool:
        return (
            mp.spawn_gen != gen
            or mp.proc is None
            or mp.stopping
            or mp.status not in ("starting", "loading")
        )

    outcome = poll_until_ready(
        check=lambda: backend_ready(
            mp.backend, mp.port, probe_timeout_s=_READINESS_PROBE_TIMEOUT_S
        ),
        should_abort=_abort,
        timeout_s=cfg.readiness_timeout_s,
        poll_interval_s=cfg.readiness_poll_interval_s,
    )

    if outcome == "ready":
        # Re-guard IMMEDIATELY before the write. poll_until_ready re-checks
        # should_abort at the instant it decides "ready", but unlike the Web
        # 版 (asyncio, single-threaded — check and write are atomic), this
        # worker runs in a REAL thread: between that decision and this
        # assignment the launch worker (_run) can crash-handle, consume an
        # auto-restart attempt, and bump spawn_gen for a respawn. Without
        # this re-check a stale worker would clobber the newer generation's
        # status/restart_count.
        if _abort():
            return
        mp.status = "running"
        mp.restart_count = 0
        log_queue.put((mp.id, "[launcher] readiness check passed"))
        log_queue.put((mp.id, f"_READY_:{mp.name}:{mp.port}"))
    elif outcome == "timeout":
        if mp.spawn_gen == gen and mp.status in ("starting", "loading"):
            mp.status = "error"
            log_queue.put((
                mp.id,
                f"[launcher] readiness check timed out after "
                f"{cfg.readiness_timeout_s:.0f}s — process left running but "
                "not confirmed ready"
            ))
    # "aborted": bail out silently, exactly like the web version — a fast
    # crash/stop/respawn must never have a stale probe overwrite its status.


# ---------------------------------------------------------------------------
# Generic auto-restart — ported from launcher_routes._attempt_restart.
# Opt-in via LauncherConfig.auto_restart (default False — see the docstring
# on that field in coderouter/config/schemas.py for the rationale: silently
# respawning a genuinely misconfigured backend forever would be worse than
# just leaving it in status="error").
# ---------------------------------------------------------------------------


@dataclass
class RestartPlan:
    should_restart: bool
    backoff_s: float = 0.0
    log_lines: list[str] = field(default_factory=list)


def plan_auto_restart(
    *,
    auto_restart: bool,
    restart_count: int,
    max_attempts: int,
    backoff_s: float,
    backoff_max_s: float,
    has_cmd: bool,
) -> RestartPlan:
    """Decide whether/how to auto-restart a crashed backend.

    Pure decision logic — no subprocess spawn, no sleep — mirroring
    ``launcher_routes._attempt_restart`` minus the actual respawn (which the
    caller performs after honoring ``mp.stopping`` one more time once the
    backoff sleep completes, exactly like the web version does).
    """
    if not auto_restart:
        return RestartPlan(should_restart=False)
    if restart_count >= max_attempts:
        return RestartPlan(
            should_restart=False,
            log_lines=[
                f"[launcher] auto-restart exhausted "
                f"({restart_count}/{max_attempts} attempts); giving up"
            ],
        )
    if not has_cmd:
        return RestartPlan(should_restart=False)  # nothing to relaunch

    backoff = min(backoff_s * (2 ** restart_count), backoff_max_s)
    return RestartPlan(
        should_restart=True,
        backoff_s=backoff,
        log_lines=[
            f"[launcher] auto-restart attempt {restart_count + 1}/"
            f"{max_attempts} in {backoff:.1f}s"
        ],
    )


def _exit_status(returncode: int | None, stopping: bool) -> str:
    """Map a subprocess exit to a terminal ManagedProcess.status.

    An intentional stop (Stop/Kill button) is always "stopped", regardless
    of the exit code SIGTERM/SIGKILL produced (POSIX SIGTERM typically
    yields a negative returncode) — without this, every deliberate Stop
    would have been mislabeled "error". Mirrors the ``proc.stopping`` check
    in ``launcher_routes._tail_logs``.
    """
    if stopping:
        return "stopped"
    return "stopped" if (returncode or 0) == 0 else "error"


def _proc_alive(mp: ManagedProcess) -> bool:
    """True iff ``mp`` still holds a live OS process.

    Liveness is decided by ``proc.poll()``, never by ``mp.status``: since
    readiness gating, a readiness-timed-out backend sits in status="error"
    while the OS process is still very much alive (holding its port and
    VRAM). Any stop/kill/remove decision keyed on the status set alone
    would drop such a process un-killed and orphan it — poll() is the only
    source of truth.
    """
    return mp.proc is not None and mp.proc.poll() is None


def _kill_for_removal(mp: ManagedProcess) -> None:
    """Force-kill a live process on behalf of a UI removal.

    ``stopping`` is set BEFORE the signal so the launch worker thread
    treats the exit as intentional: no auto-restart of a process the UI no
    longer tracks, and no "error" mislabel from the SIGKILL exit code.
    """
    mp.stopping = True
    with contextlib.suppress(Exception):
        mp.proc.kill()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class LauncherApp(tk.Tk):
    """CodeRouter Launcher — tkinter GUI."""

    # ── colours ─────────────────────────────────────────────────────────────
    BG       = "#0f172a"   # slate-900
    BG2      = "#1e293b"   # slate-800
    BG3      = "#334155"   # slate-700
    FG       = "#e2e8f0"   # slate-200
    FG2      = "#94a3b8"   # slate-400
    ACCENT   = "#6366f1"   # indigo-500
    GREEN    = "#22c55e"
    RED      = "#ef4444"
    YELLOW   = "#eab308"

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()
        self.title("CodeRouter Launcher")
        self.geometry("1100x800")
        self.minsize(900, 650)
        self.configure(bg=self.BG)

        # State
        self.cfg = _load_config(config_path)
        self.models: list[dict[str, Any]] = []
        self.processes: dict[str, ManagedProcess] = {}
        self.selected_proc_id: str | None = None
        self._last_auto_name: str = ""   # _on_model_select が自動入力した名前を記録
        self._hw: dict[str, Any] = {}    # 検出済みハードウェア情報
        self._log_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        # ── CodeRouter プロセス管理 ─────────────────────────────────────────
        self._cr_proc: subprocess.Popen | None = None
        self._cr_status: str = "stopped"   # stopped / starting / running / error
        # 上限付き deque。常駐 CodeRouter の出力で無制限に増えるのを防ぐ。
        self._cr_log: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._cr_log_queue: queue.Queue[str] = queue.Queue()
        self._cr_port: int = _CODEROUTER_PORT

        # ttk style
        self._setup_style()

        # Layout
        self._build_ui()

        # ウィンドウ閉時に CodeRouter + 全バックエンドを停止
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Initial scan
        self.after(100, self._do_scan)

        # Periodic refresh
        self._poll()

    # ── Style ────────────────────────────────────────────────────────────────

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            style.theme_use("default")

        style.configure(".", background=self.BG, foreground=self.FG,
                        fieldbackground=self.BG2, troughcolor=self.BG2,
                        bordercolor=self.BG3, lightcolor=self.BG3,
                        darkcolor=self.BG3, relief="flat")

        style.configure("TFrame", background=self.BG)
        style.configure("Card.TFrame", background=self.BG2, relief="flat")

        style.configure("TLabel", background=self.BG, foreground=self.FG)
        style.configure("Dim.TLabel", background=self.BG2, foreground=self.FG2,
                        font=("monospace", 10))
        style.configure("Title.TLabel", background=self.BG, foreground=self.FG2,
                        font=("sans-serif", 9, "bold"))
        style.configure("Status.TLabel", background=self.BG2, foreground=self.FG2,
                        font=("monospace", 10))

        style.configure("TEntry", fieldbackground=self.BG3, foreground=self.FG,
                        insertcolor=self.FG, relief="flat", borderwidth=1)
        style.configure("TCombobox", fieldbackground=self.BG3, foreground=self.FG,
                        selectbackground=self.ACCENT, selectforeground="white",
                        insertcolor=self.FG, relief="flat", arrowcolor=self.FG2)
        style.map("TCombobox",
                  fieldbackground=[("readonly", self.BG3),
                                   ("disabled", self.BG2)],
                  foreground=[("disabled", self.FG2)],
                  selectbackground=[("readonly", self.ACCENT)])

        style.configure("TButton", background=self.BG3, foreground=self.FG,
                        relief="flat", padding=(8, 4))
        style.map("TButton",
                  background=[("active", self.BG2), ("disabled", self.BG2)],
                  foreground=[("disabled", self.FG2)])

        style.configure("Accent.TButton", background=self.ACCENT, foreground="white",
                        relief="flat", padding=(10, 6), font=("sans-serif", 11, "bold"))
        style.map("Accent.TButton",
                  background=[("active", "#4f46e5"), ("disabled", self.BG3)],
                  foreground=[("disabled", self.FG2)])

        _tv_map = [
            ("background", [
                ("selected", "focus",   self.ACCENT),
                ("selected", "!focus",  self.ACCENT),
                ("active",              self.BG3),
            ]),
            ("foreground", [
                ("selected", "focus",   "white"),
                ("selected", "!focus",  "white"),
                ("active",              self.FG),
            ]),
        ]
        for prop, rules in _tv_map:
            style.map("Treeview", **{prop: rules})

        style.configure("Treeview",
                        background=self.BG2, foreground=self.FG,
                        fieldbackground=self.BG2, rowheight=26,
                        borderwidth=0, relief="flat",
                        highlightthickness=0)
        style.configure("Treeview.Heading",
                        background=self.BG3, foreground=self.FG2,
                        relief="flat", font=("sans-serif", 9, "bold"))

        style.configure("Model.Treeview",
                        background=self.BG2, foreground=self.FG,
                        fieldbackground=self.BG2, rowheight=22,
                        font=("monospace", 10), borderwidth=0, relief="flat",
                        highlightthickness=0)
        for prop, rules in _tv_map:
            style.map("Model.Treeview", **{prop: rules})

        style.configure("TScrollbar", background=self.BG3, troughcolor=self.BG2,
                        relief="flat", arrowsize=12)


    # ── UI Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Top bar
        top = ttk.Frame(self, padding=(12, 8))
        top.pack(fill="x")
        ttk.Label(top, text="CodeRouter Launcher",
                  font=("sans-serif", 14, "bold"),
                  foreground=self.FG, background=self.BG).pack(side="left")

        self._status_var = tk.StringVar(value="準備完了")
        ttk.Label(top, textvariable=self._status_var,
                  style="Status.TLabel").pack(side="right", padx=4)

        sep = tk.Frame(self, height=1, bg=self.BG3)
        sep.pack(fill="x")

        # ── CodeRouter パネル ────────────────────────────────────────────────
        self._build_coderouter_panel()

        sep2 = tk.Frame(self, height=1, bg=self.BG3)
        sep2.pack(fill="x")

        # Main area — left / right split
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=2, minsize=260)
        main.columnconfigure(1, weight=3, minsize=340)
        main.rowconfigure(0, weight=1)

        self._build_models_panel(main)
        self._build_right_panel(main)

    # ── CodeRouter パネル ─────────────────────────────────────────────────────

    def _build_coderouter_panel(self) -> None:
        """CodeRouter 起動/停止コントロールバー。"""
        bar = tk.Frame(self, bg="#1e293b", pady=0)
        bar.pack(fill="x")

        inner = tk.Frame(bar, bg="#1e293b")
        inner.pack(fill="x", padx=12, pady=6)

        # ステータスドット
        self._cr_dot = tk.Label(inner, text="●", fg=self.RED,
                                bg="#1e293b", font=("sans-serif", 11))
        self._cr_dot.pack(side="left")

        # ラベル
        self._cr_label_var = tk.StringVar(value=f"  CodeRouter  :{self._cr_port}  停止中")
        tk.Label(inner, textvariable=self._cr_label_var,
                 fg=self.FG2, bg="#1e293b",
                 font=("sans-serif", 10)).pack(side="left", padx=(0, 10))

        # ポート入力欄(停止中のみ編集可。trace は _cr_conn_var 生成後に設定)
        tk.Label(inner, text="ポート", fg=self.FG2, bg="#1e293b",
                 font=("sans-serif", 9)).pack(side="left", padx=(0, 4))
        self._cr_port_var = tk.StringVar(value=str(_CODEROUTER_PORT))
        self._cr_port_entry = ttk.Entry(inner, textvariable=self._cr_port_var,
                                        width=6)
        self._cr_port_entry.pack(side="left", padx=(0, 10))

        # 起動ボタン
        self._cr_start_btn = tk.Button(
            inner, text="▶ CodeRouter 起動",
            fg="white", bg=self.ACCENT,
            activebackground="#4f46e5", activeforeground="white",
            relief="flat", bd=0, padx=10, pady=4,
            font=("sans-serif", 10, "bold"),
            cursor="hand2",
            command=self._start_coderouter,
        )
        self._cr_start_btn.pack(side="left", padx=(0, 4))

        # 停止ボタン
        self._cr_stop_btn = tk.Button(
            inner, text="■ 停止",
            fg=self.FG, bg=self.BG3,
            activebackground=self.BG2, activeforeground=self.FG,
            relief="flat", bd=0, padx=8, pady=4,
            font=("sans-serif", 10),
            cursor="hand2",
            command=self._stop_coderouter,
            state="disabled",
        )
        self._cr_stop_btn.pack(side="left", padx=(0, 8))

        # アニメーション用(Progressbar 非使用)
        self._cr_anim_running: bool = False

        # 接続文字列(Claude Code 用)
        conn_str = f"ANTHROPIC_BASE_URL=http://localhost:{self._cr_port} ANTHROPIC_AUTH_TOKEN=dummy claude"
        self._cr_conn_var = tk.StringVar(value=conn_str)
        # ポート欄の編集に接続文字列・ラベルを追従させる
        self._cr_port_var.trace_add("write", self._on_cr_port_change)
        tk.Label(inner, text="Claude Code:", fg=self.FG2, bg="#1e293b",
                 font=("sans-serif", 9)).pack(side="left")
        conn_label = tk.Label(inner, textvariable=self._cr_conn_var,
                              fg="#4ade80", bg="#1e293b",
                              font=("monospace", 9), cursor="hand2")
        conn_label.pack(side="left", padx=(4, 0))
        conn_label.bind("<Button-1>", lambda _: self._copy_conn_str())

        # コピーボタン
        tk.Button(
            inner, text="コピー",
            fg=self.FG2, bg=self.BG3,
            activebackground=self.BG2, activeforeground=self.FG,
            relief="flat", bd=0, padx=6, pady=2,
            font=("sans-serif", 9),
            cursor="hand2",
            command=self._copy_conn_str,
        ).pack(side="left", padx=(4, 0))

        # エラー表示
        self._cr_err_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._cr_err_var,
                 fg=self.RED, bg="#1e293b",
                 font=("sans-serif", 9)).pack(side="right", padx=(8, 0))

    def _copy_conn_str(self) -> None:
        conn = self._cr_conn_var.get()
        self.clipboard_clear()
        self.clipboard_append(conn)
        self._cr_err_var.set("✓ コピーしました")
        self.after(2000, lambda: self._cr_err_var.set(""))

    def _on_cr_port_change(self, *_: Any) -> None:
        """ポート欄が編集されたら _cr_port・接続文字列・ラベルを追従させる。"""
        raw = self._cr_port_var.get().strip()
        if raw.isdigit():
            self._cr_port = int(raw)
        # 接続文字列を最新ポートで更新(無効入力時は直前の有効値を維持)
        self._cr_conn_var.set(
            f"ANTHROPIC_BASE_URL=http://localhost:{self._cr_port} "
            f"ANTHROPIC_AUTH_TOKEN=dummy claude"
        )
        self._update_cr_ui()

    def _update_cr_ui(self) -> None:
        """CodeRouter のステータスに合わせて UI を更新する。"""
        if self._cr_status == "running":
            self._cr_dot.configure(fg=self.GREEN)
            self._cr_label_var.set(f"  CodeRouter  :{self._cr_port}  稼働中")
            self._cr_start_btn.configure(state="disabled")
            self._cr_stop_btn.configure(state="normal")
            self._cr_anim_running = False
        elif self._cr_status == "starting":
            self._cr_dot.configure(fg=self.YELLOW)
            self._cr_start_btn.configure(state="disabled")
            self._cr_stop_btn.configure(state="disabled")
            if not self._cr_anim_running:
                self._cr_anim_running = True
                self._cr_anim_tick(0)
        elif self._cr_status == "error":
            self._cr_dot.configure(fg=self.RED)
            self._cr_label_var.set(f"  CodeRouter  :{self._cr_port}  エラー")
            self._cr_start_btn.configure(state="normal")
            self._cr_stop_btn.configure(state="disabled")
            self._cr_anim_running = False
        else:  # stopped
            self._cr_dot.configure(fg=self.RED)
            self._cr_label_var.set(f"  CodeRouter  :{self._cr_port}  停止中")
            self._cr_start_btn.configure(state="normal")
            self._cr_stop_btn.configure(state="disabled")
            self._cr_anim_running = False

        # ポート欄は停止中/エラー時のみ編集可(起動中・稼働中はロック)
        editable = self._cr_status in ("stopped", "error")
        self._cr_port_entry.configure(state="normal" if editable else "disabled")

    _ANIM_CHARS = ("|", "/", "-", "\\")

    def _cr_anim_tick(self, idx: int) -> None:
        """CodeRouter 起動中のテキストアニメーション(after() ベース)。"""
        if not self._cr_anim_running:
            return
        ch = self._ANIM_CHARS[idx % len(self._ANIM_CHARS)]
        self._cr_label_var.set(f"  CodeRouter  :{self._cr_port}  起動中… {ch}")
        self.after(150, self._cr_anim_tick, idx + 1)

    def _launch_anim_tick(self, proc_id: str, idx: int) -> None:
        """llama.cpp 起動中のボタンテキストアニメーション(after() ベース)。"""
        if self._launch_anim_proc_id != proc_id:
            return
        if proc_id not in self.processes or self.processes[proc_id].status not in ("starting",):
            # 起動完了 or エラー → ボタンを元に戻す
            self._launch_btn.configure(
                text="▶ llama.cpp / vllm / mlx 起動", state="normal", cursor="hand2"
            )
            self._launch_anim_proc_id = None
            return
        ch = self._ANIM_CHARS[idx % len(self._ANIM_CHARS)]
        self._launch_btn.configure(text=f"起動中… {ch}")
        self.after(150, self._launch_anim_tick, proc_id, idx + 1)

    # ── CodeRouter 起動 / 停止 ────────────────────────────────────────────────

    def _start_coderouter(self) -> None:
        """CodeRouter をポート欄の値で起動する。providers.yaml がなければ自動生成。"""
        # CodeRouter ポートの検証(ポート欄の値を使用)
        cr_port_raw = self._cr_port_var.get().strip()
        if not cr_port_raw.isdigit() or not (1024 <= int(cr_port_raw) <= 65535):
            self._cr_err_var.set("CodeRouter ポートは 1024-65535 の数字で指定してください")
            return
        self._cr_port = int(cr_port_raw)

        # llama.cpp の現在のポートを取得(フォームの値を使用)
        try:
            llama_port = int(self._port_var.get())
        except (ValueError, AttributeError):
            llama_port = 8080

        # providers.yaml を自動生成(存在しない場合のみ)
        created, yaml_path = _ensure_providers_yaml(llama_port)
        if created:
            self._cr_err_var.set(f"providers.yaml を生成しました: {yaml_path}")
            self.after(4000, lambda: self._cr_err_var.set(""))
            print(f"[CodeRouter] providers.yaml 生成: {yaml_path}", flush=True)

        self._cr_status = "starting"
        self._update_cr_ui()

        cr_port = self._cr_port  # スレッドに渡すためローカルに保持

        def _run() -> None:
            # shutil.which() をスレッド内で実行(メインスレッドをブロックしない)
            cr_cmd = _find_coderouter_cmd()
            cmd = [*cr_cmd, "serve", "--port", str(cr_port)]
            print(f"[CodeRouter] 起動: {' '.join(cmd)}", flush=True)
            try:
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                )
            except Exception as exc:
                self._cr_log_queue.put(f"_CR_ERR_:{exc}")
                return

            self._cr_proc = p
            self._cr_log_queue.put(f"_CR_OK_:{p.pid}")

            assert p.stdout
            for raw in iter(lambda: p.stdout.read(4096), b""):
                for line in raw.decode("utf-8", errors="replace").splitlines():
                    self._cr_log_queue.put(line)
            p.wait()
            self._cr_log_queue.put(f"_CR_EXIT_:{p.returncode}")

        threading.Thread(target=_run, daemon=True).start()

    def _stop_coderouter(self) -> None:
        """CodeRouter を停止する。"""
        if self._cr_proc and self._cr_proc.poll() is None:
            with contextlib.suppress(Exception):
                self._cr_proc.terminate()
            self._cr_log.append("[coderouter] SIGTERM 送信")
        self._cr_status = "stopped"
        self._cr_proc = None
        self._update_cr_ui()

    # ── ウィンドウ閉時 ───────────────────────────────────────────────────────

    def _on_close(self) -> None:
        """ウィンドウを閉じる際に CodeRouter と全バックエンドを停止する。"""
        # CodeRouter 停止
        if self._cr_proc and self._cr_proc.poll() is None:
            with contextlib.suppress(Exception):
                self._cr_proc.terminate()

        # llama.cpp / vllm 停止
        for mp in list(self.processes.values()):
            if mp.proc and mp.proc.poll() is None:
                with contextlib.suppress(Exception):
                    mp.proc.terminate()

        self.destroy()

    # ── Models panel (left) ──────────────────────────────────────────────────

    def _card(self, parent: ttk.Frame, **grid_kw) -> ttk.Frame:
        f = tk.Frame(parent, bg=self.BG2, bd=0, highlightthickness=1,
                     highlightbackground=self.BG3)
        f.grid(**grid_kw)
        return f

    def _build_models_panel(self, parent: ttk.Frame) -> None:
        card = self._card(parent, row=0, column=0, sticky="nsew",
                          padx=(0, 6), pady=0)
        card.rowconfigure(2, weight=1)
        card.columnconfigure(0, weight=1)

        hdr = tk.Frame(card, bg=self.BG2)
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        tk.Label(hdr, text="MODELS", fg=self.FG2, bg=self.BG2,
                 font=("sans-serif", 9, "bold")).pack(side="left")
        self._hw_var = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._hw_var, fg=self.FG2, bg=self.BG2,
                 font=("monospace", 8)).pack(side="left", padx=(8, 0))
        btn = tk.Button(hdr, text="↻ スキャン", fg=self.FG, bg=self.BG3,
                        relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
                        command=self._do_scan)
        btn.pack(side="right")

        self._dirs_var = tk.StringVar(value="スキャン中…")
        tk.Label(card, textvariable=self._dirs_var, fg=self.FG2, bg=self.BG2,
                 font=("monospace", 9), anchor="w", wraplength=240).grid(
            row=1, column=0, sticky="ew", padx=10, pady=(0, 4))

        lf = tk.Frame(card, bg=self.BG2)
        lf.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 8))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self._model_tree = ttk.Treeview(
            lf, style="Model.Treeview",
            show="tree", selectmode="browse",
        )
        # メモリ的に厳しいモデルの行を警告色にする
        self._model_tree.tag_configure("rec_warn", foreground=self.YELLOW)
        sb = ttk.Scrollbar(lf, orient="vertical",
                           command=self._model_tree.yview)
        self._model_tree.configure(yscrollcommand=sb.set)
        self._model_tree.column("#0", stretch=True)
        self._model_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self._model_tree.bind("<<TreeviewSelect>>", self._on_model_select)

    def _do_scan(self) -> None:
        self._set_status("スキャン中…")
        dirs = self.cfg.model_dirs
        if dirs:
            self._dirs_var.set("  ".join(
                str(Path(d).expanduser()) for d in dirs
            ))
        else:
            self._dirs_var.set("model_dirs 未設定")

        def run() -> None:
            models = _scan_models(dirs)
            hw = _detect_hardware()
            self.after(0, lambda: self._populate_models(models, hw))

        threading.Thread(target=run, daemon=True).start()

    def _populate_models(self, models: list[dict],
                         hw: dict[str, Any] | None = None) -> None:
        self.models = models
        if hw is not None:
            self._hw = hw
        self._model_tree.delete(*self._model_tree.get_children())
        for i, m in enumerate(models):
            rec = _model_recommendation(m["size_gb"], self._hw)
            badge = {"ok": "   ✓ 推奨",
                     "warn": "   ⚠ メモリ厳しい"}.get(rec["level"], "")
            tags = ("rec_warn",) if rec["level"] == "warn" else ()
            self._model_tree.insert(
                "", "end", iid=str(i),
                text=f"{m['name']}  ({m['size_gb']} GB){badge}",
                tags=tags,
            )
        if self._hw:
            self._hw_var.set(_hw_summary(self._hw))
        self._set_status(f"モデル {len(models)} 件")

    def _on_model_select(self, _event: Any = None) -> None:
        sel = self._model_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        m = self.models[idx]
        self._model_path_var.set(m["path"])
        # Name が空、または前回ここで自動入力した値のまま(= 手で変更していない)
        # なら選択モデル名で更新する。手入力された名前は上書きしない。
        current = self._name_var.get()
        if not current or current == self._last_auto_name:
            stem = Path(m["name"]).stem[:30]
            self._name_var.set(stem)
            self._last_auto_name = stem

    def _suggest_options(self) -> None:
        """選択中モデル + ハードから推奨起動フラグを算出し追加オプション欄に入れる。

        ハード検出は通常スキャン時に済んでおり (_hw にキャッシュ)、未取得の
        場合のみその場で検出する。検出はほぼ即時 (CUDA 環境のみ nvidia-smi を
        一度呼ぶ程度) のためメインスレッドで同期実行する。
        """
        model_path = self._model_path_var.get().strip()
        if not model_path:
            self._set_launch_err("先にモデルを選択してください")
            return
        hw = self._hw or _detect_hardware()
        self._hw = hw
        try:
            size_gb = Path(model_path).expanduser().stat().st_size / (1024 ** 3)
        except OSError:
            size_gb = 0.0
        backend = self._backend_var.get()
        flags = _suggest_launch_flags(backend, size_gb, hw)
        self._extra_var.set(flags)
        self._hw_var.set(_hw_summary(hw))
        self._set_launch_err("")
        if flags:
            self._set_status(f"推奨値を設定(目安): {_hw_summary(hw)} → {flags}")
        elif backend == "mlx":
            self._set_status(
                f"{_hw_summary(hw)} — MLX は起動時の調整フラグ不要です(目安)")
        elif backend == "vllm":
            self._set_status(
                f"{_hw_summary(hw)} — vllm は起動時フラグ不要"
                "(モデル設定から自動導出)")
        else:
            self._set_status(f"{_hw_summary(hw)} — 推奨フラグなし")

    # ── Right panel ──────────────────────────────────────────────────────────

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=0)
        right.columnconfigure(0, weight=1)

        self._build_launch_panel(right)
        self._build_process_panel(right)
        self._build_log_panel(right)

    # ── Launch form ──────────────────────────────────────────────────────────

    def _build_launch_panel(self, parent: ttk.Frame) -> None:
        card = self._card(parent, row=0, column=0, sticky="ew",
                          padx=0, pady=(0, 6))
        card.columnconfigure(1, weight=1)
        card.columnconfigure(3, weight=1)

        def lbl(text: str, r: int, c: int) -> None:
            tk.Label(card, text=text, fg=self.FG2, bg=self.BG2,
                     font=("sans-serif", 9)).grid(
                row=r, column=c, sticky="w", padx=(10, 4), pady=(6, 0))

        tk.Label(card, text="LAUNCH  llama.cpp / vllm / mlx", fg=self.FG2, bg=self.BG2,
                 font=("sans-serif", 9, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w",
            padx=10, pady=(8, 2))

        lbl("名前", 1, 0)
        self._name_var = tk.StringVar()
        ttk.Entry(card, textvariable=self._name_var).grid(
            row=1, column=1, sticky="ew", padx=(0, 6), pady=(6, 0))

        lbl("ポート", 1, 2)
        self._port_var = tk.StringVar(value="8080")
        ttk.Entry(card, textvariable=self._port_var, width=8).grid(
            row=1, column=3, sticky="ew", padx=(0, 10), pady=(6, 0))

        lbl("バックエンド", 2, 0)
        self._backend_var = tk.StringVar(value="llama.cpp")
        cb = ttk.Combobox(card, textvariable=self._backend_var,
                          values=list(_BACKEND_DEFAULTS.keys()),
                          state="readonly")
        cb.grid(row=2, column=1, columnspan=3, sticky="ew",
                padx=(0, 10), pady=(6, 0))
        cb.bind("<<ComboboxSelected>>", self._on_backend_change)

        self._binary_hint_var = tk.StringVar(value="")
        self._binary_hint_lbl = tk.Label(
            card, textvariable=self._binary_hint_var,
            fg=self.FG2, bg=self.BG2,
            font=("monospace", 9), anchor="w")
        self._binary_hint_lbl.grid(row=3, column=1, columnspan=3,
                                   sticky="ew", padx=(0, 10), pady=(2, 0))

        lbl("モデルパス", 4, 0)
        self._model_path_var = tk.StringVar()
        ttk.Entry(card, textvariable=self._model_path_var).grid(
            row=4, column=1, columnspan=3, sticky="ew",
            padx=(0, 10), pady=(6, 0))

        lbl("オプションプロファイル", 5, 0)
        self._profile_var = tk.StringVar(value="-- なし --")
        self._profile_cb = ttk.Combobox(card, textvariable=self._profile_var,
                                        state="readonly")
        self._profile_cb.grid(row=5, column=1, columnspan=3, sticky="ew",
                              padx=(0, 10), pady=(6, 0))
        self._profile_cb.bind("<<ComboboxSelected>>", self._on_profile_change)

        self._profile_args_var = tk.StringVar(value="")
        tk.Label(card, textvariable=self._profile_args_var,
                 fg=self.FG2, bg=self.BG2,
                 font=("monospace", 9), anchor="w", justify="left").grid(
            row=6, column=1, columnspan=3, sticky="ew",
            padx=(0, 10), pady=(0, 2))

        lbl("MTP/draft gguf (空欄で自動検出)", 7, 0)
        self._draft_path_var = tk.StringVar()
        ttk.Entry(card, textvariable=self._draft_path_var).grid(
            row=7, column=1, columnspan=2, sticky="ew",
            padx=(0, 6), pady=(6, 0))
        self._mtp_auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(card, text="MTP自動検出",
                        variable=self._mtp_auto_var).grid(
            row=7, column=3, sticky="w", padx=(0, 10), pady=(6, 0))

        lbl("追加オプション", 8, 0)
        self._extra_var = tk.StringVar(value="-ngl 99")
        ttk.Entry(card, textvariable=self._extra_var).grid(
            row=8, column=1, columnspan=2, sticky="ew",
            padx=(0, 6), pady=(6, 0))
        tk.Button(card, text="⚙ 推奨値", fg=self.FG, bg=self.BG3,
                  activebackground=self.BG2, activeforeground=self.FG,
                  relief="flat", bd=0, padx=6, pady=3, cursor="hand2",
                  font=("sans-serif", 9),
                  command=self._suggest_options).grid(
            row=8, column=3, sticky="ew", padx=(0, 10), pady=(6, 0))

        # 起動ボタン
        _btn_wrap = tk.Frame(card, bg=self.ACCENT, bd=0)
        _btn_wrap.grid(row=9, column=0, columnspan=4, sticky="ew", padx=10, pady=8)
        self._launch_btn = tk.Button(
            _btn_wrap, text="▶ llama.cpp / vllm / mlx 起動",
            fg="white", bg=self.ACCENT,
            activebackground="#4f46e5", activeforeground="white",
            disabledforeground=self.FG2,
            relief="flat", bd=0, padx=10, pady=7,
            highlightthickness=0, highlightbackground=self.ACCENT,
            font=("sans-serif", 11, "bold"),
            cursor="hand2",
            command=self._do_launch,
        )
        self._launch_btn.pack(fill="both", expand=True)

        self._launch_err_var = tk.StringVar(value="")
        tk.Label(card, textvariable=self._launch_err_var,
                 fg=self.RED, bg=self.BG2,
                 font=("sans-serif", 9), anchor="w", justify="left",
                 wraplength=400).grid(row=10, column=0, columnspan=4,
                                      sticky="ew", padx=10, pady=(0, 6))

        # アニメーション用(Progressbar 非使用)
        self._launch_anim_proc_id: str | None = None

        self.after(200, self._update_binary_hint)
        self.after(200, self._populate_profiles)

    def _on_backend_change(self, _: Any = None) -> None:
        self._update_binary_hint()
        self._populate_profiles()

    def _update_binary_hint(self) -> None:
        """shutil.which() はメインスレッドをブロックするのでスレッドで実行する。"""
        backend = self._backend_var.get()
        binary = _resolve_binary(backend, self.cfg)
        bc = self.cfg.backends.get(backend)
        is_custom = bc is not None and bc.binary is not None

        # 暫定表示(スレッド完了前)
        self._binary_hint_var.set(f"{binary}  (確認中…)")
        self._binary_hint_lbl.configure(fg=self.FG2)

        def _check() -> None:
            found = _check_binary(binary)
            self.after(0, lambda: self._apply_binary_hint(
                backend, binary, found, is_custom))

        threading.Thread(target=_check, daemon=True).start()

    def _apply_binary_hint(self, backend: str, binary: str,
                           found: bool, is_custom: bool) -> None:
        label = "カスタム設定" if is_custom else "PATH"
        status = "✓ 利用可" if found else "✗ 見つかりません"
        self._binary_hint_var.set(f"{binary}  ({label} — {status})")
        self._binary_hint_lbl.configure(fg=self.GREEN if found else self.RED)

        self._launch_btn.config(
            state="normal" if found else "disabled",
            cursor="hand2" if found else "arrow",
        )
        if not found:
            self._set_launch_err(
                f"⚠ {binary} が見つかりません。\n"
                f"バックエンド ({backend}) をインストールするか、providers.yaml の\n"
                f"launcher.backends.{backend}.binary にフルパスを設定してください。"
            )
        else:
            self._set_launch_err("")

    def _populate_profiles(self) -> None:
        backend = self._backend_var.get()
        profiles = self.cfg.option_profiles.get(backend, [])
        names = ["-- なし --"] + [p.name for p in profiles]
        self._profile_cb["values"] = names
        self._profile_var.set("-- なし --")
        self._profile_args_var.set("")

    def _on_profile_change(self, _: Any = None) -> None:
        backend = self._backend_var.get()
        profiles = self.cfg.option_profiles.get(backend, [])
        sel = self._profile_var.get()
        matched = next((p for p in profiles if p.name == sel), None)
        if matched and matched.args:
            lines = []
            for k, v in matched.args.items():
                if isinstance(v, bool):
                    lines.append(k if v else f"# {k}")
                else:
                    lines.append(f"{k} {v}")
            self._profile_args_var.set("  ".join(lines))
        else:
            self._profile_args_var.set("")

    def _set_launch_err(self, msg: str) -> None:
        self._launch_err_var.set(msg)

    # ── Launch / Stop ────────────────────────────────────────────────────────

    def _do_launch(self) -> None:
        name = self._name_var.get().strip()
        port_str = self._port_var.get().strip()
        backend = self._backend_var.get()
        model_path = self._model_path_var.get().strip()
        extra = self._extra_var.get().strip()

        if not name:
            self._set_launch_err("名前を入力してください")
            return
        if not model_path:
            self._set_launch_err("モデルパスを入力してください (左のリストから選択か直接入力)")
            return
        if not port_str.isdigit() or not (1024 <= int(port_str) <= 65535):
            self._set_launch_err("ポートは 1024-65535 の数字で指定してください")
            return

        port = int(port_str)
        binary = _resolve_binary(backend, self.cfg)

        profile_args: dict[str, Any] = {}
        sel_profile = self._profile_var.get()
        if sel_profile != "-- なし --":
            profs = self.cfg.option_profiles.get(backend, [])
            matched = next((p for p in profs if p.name == sel_profile), None)
            if matched:
                profile_args = matched.args

        # MTP / speculative-decoding resolution (llama.cpp only; no-op for
        # other backends). Skipped entirely if the coderouter package is not
        # importable (standalone GUI use).
        spec_tokens: list[str] = []
        spec_notes: list[str] = []
        if _HAS_SPECULATIVE and resolve_speculative is not None:
            draft_path = self._draft_path_var.get().strip() or None
            mtp_mode = "auto" if self._mtp_auto_var.get() else "off"
            user_tokens: list[str] = []
            for flag, val in profile_args.items():
                if isinstance(val, bool):
                    if val:
                        user_tokens.append(str(flag))
                else:
                    user_tokens.extend([str(flag), str(val)])
            if extra.strip():
                with contextlib.suppress(ValueError):
                    user_tokens += shlex.split(extra)
            try:
                spec_tokens, spec_notes = resolve_speculative(
                    backend, model_path, draft_path, mtp_mode, user_tokens)
            except ValueError as e:
                self._set_launch_err(str(e))
                return

        try:
            cmd = _build_cmd(backend, model_path, port, profile_args, extra,
                             binary, spec_tokens)
        except ValueError as e:
            self._set_launch_err(str(e))
            return

        # MTP auto-fallback: only auto-detected speculative flags (MTP自動検出
        # ON, no explicit draft entry, detection emitted flags) qualify for the
        # one-shot startup-crash retry. Rebuild the command without the spec
        # tokens (exact — never spliced) so the retry is precise.
        spec_auto = bool(
            self._mtp_auto_var.get()
            and not self._draft_path_var.get().strip()
            and spec_tokens
        )
        fallback_cmd: list[str] | None = None
        if spec_auto:
            with contextlib.suppress(ValueError):
                fallback_cmd = _build_cmd(backend, model_path, port,
                                          profile_args, extra, binary, None)
        if fallback_cmd is None:
            spec_auto = False

        proc_id = uuid.uuid4().hex[:8]
        mp = ManagedProcess(
            id=proc_id,
            name=name,
            backend=backend,
            model_name=Path(model_path).name,
            port=port,
            cmd=cmd,
            status="starting",
        )

        self.processes[proc_id] = mp
        self._refresh_process_table()
        self._select_process(proc_id)
        self._set_launch_err("")
        self._set_status(f"起動中: {name}…")

        # ボタンアニメーション開始(Progressbar 非使用)
        self._launch_anim_proc_id = proc_id
        self._launch_btn.configure(state="disabled", cursor="arrow")
        self._launch_anim_tick(proc_id, 0)

        def _spawn_readiness_worker() -> None:
            """Kick off (or re-kick after a respawn) the readiness poller."""
            threading.Thread(
                target=_readiness_worker,
                args=(mp, self.cfg, self._log_queue, mp.spawn_gen),
                daemon=True,
            ).start()

        def _run() -> None:
            mp.log_lines.append(f"[launcher] cmd: {' '.join(cmd)}")
            for note in spec_notes:
                mp.log_lines.append(f"[launcher] {note}")
            run_cmd = cmd
            fallback_done = False
            first = True
            while True:
                try:
                    p = subprocess.Popen(
                        run_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        bufsize=0,
                    )
                except Exception as exc:
                    mp.status = "error"
                    self._log_queue.put((proc_id, f"_ERR_:{exc}"))
                    return

                started_at = time.monotonic()
                mp.cmd = run_cmd
                mp.proc  = p
                mp.pid   = p.pid
                mp.returncode = None
                mp.started_at = started_at
                mp.spawn_gen += 1
                # H2 (readiness gating, ported from launcher_routes.py): the
                # status is "loading", not "running", until the readiness
                # worker below confirms the backend is actually serving —
                # registering/declaring success before the model finishes
                # loading is exactly the bug this closes.
                mp.status = "loading"
                _spawn_readiness_worker()
                if first:
                    self._log_queue.put((proc_id, f"_SPAWNED_:{name}:{port}"))
                    first = False

                assert p.stdout
                stdout = p.stdout
                for raw in iter(lambda s=stdout: s.read(4096), b""):
                    for line in raw.decode("utf-8", errors="replace").splitlines():
                        self._log_queue.put((proc_id, line))
                p.wait()
                mp.returncode = p.returncode
                mp.pid = None
                mp.status = _exit_status(p.returncode, mp.stopping)
                self._log_queue.put(
                    (proc_id, f"[launcher] exited (code {p.returncode})"))

                if mp.stopping:
                    # Intentional stop (_do_stop / _do_kill). Never a crash
                    # to heal, regardless of the exit code SIGTERM/SIGKILL
                    # produced, and never eligible for MTP fallback either.
                    return

                # MTP startup crash → relaunch ONCE without speculative flags.
                if (
                    not fallback_done
                    and spec_auto
                    and p.returncode not in (0, None)
                    and (time.monotonic() - started_at)
                    <= _MTP_FALLBACK_WINDOW_SECS
                ):
                    fallback_done = True
                    self._log_queue.put((
                        proc_id,
                        f"[launcher] MTP startup failure detected "
                        f"(exit code {p.returncode}); retrying without "
                        "speculative decoding",
                    ))
                    self._log_queue.put(
                        (proc_id, f"[launcher] cmd: {' '.join(fallback_cmd)}"))
                    run_cmd = fallback_cmd
                    continue

                # Generic auto-restart (opt-in — see LauncherConfig.auto_restart).
                if p.returncode not in (0, None):
                    plan = plan_auto_restart(
                        auto_restart=self.cfg.auto_restart,
                        restart_count=mp.restart_count,
                        max_attempts=self.cfg.auto_restart_max_attempts,
                        backoff_s=self.cfg.auto_restart_backoff_s,
                        backoff_max_s=self.cfg.auto_restart_backoff_max_s,
                        has_cmd=bool(run_cmd),
                    )
                    for ln in plan.log_lines:
                        self._log_queue.put((proc_id, ln))
                    if plan.should_restart:
                        mp.restart_count += 1
                        time.sleep(plan.backoff_s)
                        if mp.stopping:
                            # Stopped while waiting out the backoff — respect it.
                            return
                        continue  # run_cmd unchanged — relaunch same argv
                return

        threading.Thread(target=_run, daemon=True).start()

    def _do_stop(self) -> None:
        pid = self.selected_proc_id
        if not pid or pid not in self.processes:
            return
        mp = self.processes[pid]
        if _proc_alive(mp):
            # Set BEFORE signalling — tells the launch worker thread (and any
            # in-flight readiness worker) this exit was requested, so neither
            # auto-restarts nor mislabels it "error" (mirrors
            # launcher_routes.stop_process setting proc.stopping = True
            # first).
            mp.stopping = True
            mp.status = "stopping"
            with contextlib.suppress(Exception):
                mp.proc.terminate()
            mp.log_lines.append("[launcher] SIGTERM sent")
            self._refresh_process_table()

    def _do_kill(self) -> None:
        pid = self.selected_proc_id
        if not pid or pid not in self.processes:
            return
        mp = self.processes[pid]
        if _proc_alive(mp):
            mp.stopping = True  # same rationale as _do_stop
            with contextlib.suppress(Exception):
                mp.proc.kill()
            mp.log_lines.append("[launcher] SIGKILL sent")
            self._refresh_process_table()

    def _do_remove(self) -> None:
        pid = self.selected_proc_id
        if not pid or pid not in self.processes:
            return
        mp = self.processes[pid]
        # 生存判定は status ではなく poll() で行う — readiness タイムアウト後は
        # status="error" のままOSプロセスが生きているため、status ベースの
        # 判定では kill されずにオーファン化する(_proc_alive の docstring 参照)。
        if _proc_alive(mp):
            if not messagebox.askyesno("確認", f"{mp.name} は実行中です。強制終了して削除しますか?"):
                return
            _kill_for_removal(mp)
        del self.processes[pid]
        self.selected_proc_id = None
        self._refresh_process_table()
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        self._log_title_var.set("ログ")

    # ── Process table ─────────────────────────────────────────────────────────

    def _build_process_panel(self, parent: ttk.Frame) -> None:
        card = self._card(parent, row=1, column=0, sticky="nsew",
                          padx=0, pady=(0, 6))
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        hdr = tk.Frame(card, bg=self.BG2)
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        tk.Label(hdr, text="PROCESSES", fg=self.FG2, bg=self.BG2,
                 font=("sans-serif", 9, "bold")).pack(side="left")
        for label, cmd in [("■ 停止", self._do_stop),
                            ("✕ 削除", self._do_remove)]:
            tk.Button(hdr, text=label, fg=self.FG, bg=self.BG3,
                      relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
                      command=cmd).pack(side="right", padx=2)

        cols = ("name", "backend", "model", "port", "pid", "status")
        self._proc_tree = ttk.Treeview(
            card, columns=cols, show="headings",
            selectmode="browse", height=5,
        )
        for col, w, label in [
            ("name",    120, "NAME"),
            ("backend",  80, "BACKEND"),
            ("model",   200, "MODEL"),
            ("port",     60, "PORT"),
            ("pid",      60, "PID"),
            ("status",   80, "STATUS"),
        ]:
            self._proc_tree.heading(col, text=label)
            self._proc_tree.column(col, width=w, minwidth=40, anchor="w")

        vsb = ttk.Scrollbar(card, orient="vertical",
                            command=self._proc_tree.yview)
        self._proc_tree.configure(yscrollcommand=vsb.set)
        self._proc_tree.grid(row=1, column=0, sticky="nsew", padx=(6, 0), pady=(0, 6))
        vsb.grid(row=1, column=1, sticky="ns", pady=(0, 6), padx=(0, 4))

        self._proc_tree.bind("<<TreeviewSelect>>", self._on_proc_select)
        self._proc_tree.tag_configure("running",  foreground=self.GREEN,  background=self.BG2)
        self._proc_tree.tag_configure("starting", foreground=self.YELLOW, background=self.BG2)
        self._proc_tree.tag_configure("loading",  foreground=self.YELLOW, background=self.BG2)
        self._proc_tree.tag_configure("stopping", foreground=self.YELLOW, background=self.BG2)
        self._proc_tree.tag_configure("stopped",  foreground=self.FG2,    background=self.BG2)
        self._proc_tree.tag_configure("error",    foreground=self.RED,    background=self.BG2)

    def _refresh_process_table(self) -> None:
        sel_id = self.selected_proc_id
        self._proc_tree.delete(*self._proc_tree.get_children())
        for mp in self.processes.values():
            pid_str = str(mp.pid) if mp.pid else "—"
            try:
                self._proc_tree.insert(
                    "", "end", iid=mp.id,
                    values=(mp.name, mp.backend, mp.model_name,
                            mp.port, pid_str, mp.status),
                    tags=(mp.status,),
                )
            except Exception as e:
                print(f"[DEBUG] insert ERROR: {e}", flush=True)
        if sel_id and sel_id in self.processes:
            try:
                self._proc_tree.selection_set(sel_id)
                self._proc_tree.see(sel_id)
            except Exception:
                pass

    def _on_proc_select(self, _: Any = None) -> None:
        sel = self._proc_tree.selection()
        if not sel:
            return
        pid = sel[0]
        # ★ 無限ループ防止ガード:
        # _select_process() 内の selection_set() は、選択が変わらなくても
        # <<TreeviewSelect>> を再発火する。ガードが無いと
        #   _on_proc_select → _select_process → selection_set →
        #   <<TreeviewSelect>> → _on_proc_select → … が無限再帰し GUI が固まる。
        # 既に選択中の ID なら何もしないことで再帰を断ち切る。
        if pid == self.selected_proc_id:
            return
        self._select_process(pid)

    def _select_process(self, proc_id: str) -> None:
        self.selected_proc_id = proc_id
        with contextlib.suppress(Exception):
            self._proc_tree.selection_set(proc_id)
        self._refresh_log_view()

    # ── Log viewer ────────────────────────────────────────────────────────────

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        card = self._card(parent, row=2, column=0, sticky="ew",
                          padx=0, pady=0)
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)
        card.configure(height=160)

        hdr = tk.Frame(card, bg=self.BG2)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=10, pady=(8, 4))
        self._log_title_var = tk.StringVar(value="ログ")
        tk.Label(hdr, textvariable=self._log_title_var,
                 fg=self.FG2, bg=self.BG2,
                 font=("sans-serif", 9, "bold")).pack(side="left")
        tk.Button(hdr, text="クリア", fg=self.FG, bg=self.BG3,
                  relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
                  command=self._clear_log).pack(side="right")

        self._log_text = tk.Text(
            card, bg="#020617", fg=self.FG2,
            font=("monospace", 9), relief="flat", bd=0,
            state="disabled", wrap="none", height=8,
            insertbackground=self.FG,
        )
        vsb = ttk.Scrollbar(card, orient="vertical",
                            command=self._log_text.yview)
        hsb = ttk.Scrollbar(card, orient="horizontal",
                            command=self._log_text.xview)
        self._log_text.configure(yscrollcommand=vsb.set,
                                  xscrollcommand=hsb.set)
        self._log_text.grid(row=1, column=0, sticky="nsew",
                            padx=(6, 0), pady=(0, 0))
        vsb.grid(row=1, column=1, sticky="ns", padx=(0, 4))
        hsb.grid(row=2, column=0, sticky="ew", padx=(6, 0), pady=(0, 4))

    def _refresh_log_view(self) -> None:
        pid = self.selected_proc_id
        if not pid or pid not in self.processes:
            return
        mp = self.processes[pid]
        self._log_title_var.set(f"ログ — {mp.name} (PID {mp.pid or '—'})")
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        # deque はスライス不可のため list 化してから末尾 400 行を取得
        for line in list(mp.log_lines)[-400:]:
            self._log_text.insert("end", line + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        pid = self.selected_proc_id
        if pid and pid in self.processes:
            self.processes[pid].log_lines.clear()
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    # ── Polling ──────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        backlog = False
        try:
            backlog = self._poll_impl()
        except Exception as e:
            print(f"[DEBUG] _poll EXCEPTION: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # バックログが残っていれば短間隔で再開し、
            # UI に制御を返しつつ素早く追従する
            self.after(50 if backlog else 1000, self._poll)

    def _poll_impl(self) -> bool:
        """キューを処理して UI を更新する。

        Returns:
            backlog — 1ティック上限に達し、キューに未処理が残った場合 True。
        """
        changed = False
        pending_log_lines: list[str] = []

        # ── CodeRouter ログキュー処理 ─────────────────────────────────────
        cr_processed = 0
        while cr_processed < _MAX_LINES_PER_TICK:
            try:
                line = self._cr_log_queue.get_nowait()
            except queue.Empty:
                break
            cr_processed += 1

            if line.startswith("_CR_OK_:"):
                pid_str = line[len("_CR_OK_:"):]
                self._cr_status = "running"
                self._cr_log.append(f"[coderouter] 起動しました (PID {pid_str})")
                self._update_cr_ui()
                self._set_status(f"CodeRouter 稼働中 (PID {pid_str})")

            elif line.startswith("_CR_ERR_:"):
                err = line[len("_CR_ERR_:"):]
                self._cr_status = "error"
                self._cr_log.append(f"[coderouter] 起動エラー: {err}")
                self._cr_err_var.set(f"CodeRouter 起動失敗: {err}")
                self._update_cr_ui()

            elif line.startswith("_CR_EXIT_:"):
                rc = line[len("_CR_EXIT_:"):]
                self._cr_status = "stopped" if rc == "0" else "error"
                self._cr_proc = None
                self._cr_log.append(f"[coderouter] 終了 (code {rc})")
                self._update_cr_ui()

            else:
                self._cr_log.append(line)

        # ── llama.cpp / vllm ログキュー処理 ──────────────────────────────
        lc_processed = 0
        while lc_processed < _MAX_LINES_PER_TICK:
            try:
                proc_id, line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            lc_processed += 1

            if proc_id not in self.processes:
                changed = True
                continue

            mp = self.processes[proc_id]

            if line.startswith("_SPAWNED_:"):
                # OS プロセスの起動に成功した段階(readiness 未確認)。
                # フォームは次の起動へ空けるが、mp.status は "loading" の
                # ままで、実際に "稼働中" になるのは _READY_ 受信時。
                parts = line.split(":", 2)
                _, pname, pport = parts
                self._set_status(f"読み込み中: {pname} (PID {mp.pid})")
                self._port_var.set(str(int(pport) + 1))
                self._name_var.set("")
                # アニメーション停止(_launch_anim_tick が次回呼ばれたとき自動停止)
                self._launch_anim_proc_id = None
                self._launch_btn.configure(
                    text="▶ llama.cpp / vllm / mlx 起動", state="normal", cursor="hand2"
                )
                changed = True
                continue

            if line.startswith("_READY_:"):
                # readiness probe が通り、mp.status は既に "running"。
                parts = line.split(":", 2)
                _, pname, _pport = parts
                if proc_id == self.selected_proc_id:
                    self._set_status(f"稼働中: {pname} (PID {mp.pid})")
                changed = True
                continue

            if line.startswith("_ERR_:"):
                err = line[6:]
                del self.processes[proc_id]
                self._set_launch_err(f"起動エラー: {err}")
                self._set_status("起動失敗")
                # アニメーション停止
                self._launch_anim_proc_id = None
                self._launch_btn.configure(
                    text="▶ llama.cpp / vllm / mlx 起動", state="normal", cursor="hand2"
                )
                changed = True
                continue

            mp.log_lines.append(line)
            if proc_id == self.selected_proc_id:
                pending_log_lines.append(line)
            changed = True

        # ログをまとめて1回だけ書き込む(行ごとに configure するとUI固まる)
        if pending_log_lines:
            self._log_text.configure(state="normal")
            self._log_text.insert("end", "\n".join(pending_log_lines) + "\n")
            # ウィジェットが無制限に伸びると insert/描画が遅くなり UI が固まる。
            # 末尾 _MAX_TEXT_LINES 行のみ残して古い行を削除する。
            line_count = int(self._log_text.index("end-1c").split(".")[0])
            if line_count > _MAX_TEXT_LINES:
                self._log_text.delete(
                    "1.0", f"{line_count - _MAX_TEXT_LINES}.0")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")

        # プロセス終了チェック
        for mp in list(self.processes.values()):
            if mp.proc and mp.status in ("running", "starting", "loading"):
                rc = mp.proc.poll()
                if rc is not None:
                    mp.returncode = rc
                    mp.status = _exit_status(rc, mp.stopping)
                    changed = True

        if changed:
            self._refresh_process_table()

        # 1ティック上限に達した場合はキューに未処理が残っている可能性が高い。
        # backlog=True を返し、_poll 側で短間隔の再ポーリングへ切り替える。
        backlog = (cr_processed >= _MAX_LINES_PER_TICK
                   or lc_processed >= _MAX_LINES_PER_TICK)
        return backlog

    # ── Misc ─────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CodeRouter Launcher GUI")
    parser.add_argument("--config", default=None,
                        help="Path to providers.yaml (default: auto-detect)")
    args = parser.parse_args()

    app = LauncherApp(config_path=args.config)
    app.mainloop()


if __name__ == "__main__":
    main()
