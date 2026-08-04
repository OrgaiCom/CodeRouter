"""Launcher routes — ``GET /launcher`` + ``/api/launcher/*``.

llama.cpp / vllm / mlx プロセス管理 UI。

設計方針:
- ダッシュボードと同じ "1ファイル完結" スタイル (Tailwind CDN + inline JS)
- プロセスレジストリは app.state.launcher に持たせる (再起動で消えるが意図通り)
- option_profiles は providers.yaml の launcher: セクションで管理 → コード変更不要で拡張可
- 複数プロセスの同時起動に対応 (UUID ベースの ID 管理)
- llama.cpp / vllm / mlx いずれも同じ key-value args スキーマで統一

エンドポイント:
  GET  /launcher                   → HTML UI
  GET  /api/launcher/models        → model_dirs をスキャンしてリスト返却
  GET  /api/launcher/option-profiles → providers.yaml の option_profiles を返却
  GET  /api/launcher/processes     → 起動中・停止済みプロセス一覧
  POST /api/launcher/start         → プロセス起動
  POST /api/launcher/stop/{id}     → プロセス停止 (SIGTERM → SIGKILL)
  DELETE /api/launcher/processes/{id} → レジストリから削除 (停止済みのみ)
  GET  /api/launcher/logs/{id}     → ログ最新 N 行
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import secrets
import shlex
import shutil
import subprocess
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from coderouter.launcher_devices import (
    DeviceSelection,
    SweepPlan,
    SweepState,
    SweepStep,
    base_backend,
    build_auto_sweep_configs,
    build_sweep_steps,
    detect_llama_devices,
    foreign_device_ids,
    group_by_backend,
    is_port_free,
    load_latest_results,
    render_bench_command,
    resolve_option_profiles,
    selectable_devices,
    suggest_tensor_split,
    variant_of,
)
from coderouter.launcher_speculative import resolve_speculative
from coderouter.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# 背景タスクへの強参照を保持する (create_task の戻り値が GC されるのを防ぐ)
_background_tasks: set[asyncio.Task[Any]] = set()

# ---------------------------------------------------------------------------
# H8: token auth for state-changing endpoints (start / stop / delete)
# ---------------------------------------------------------------------------

# Env var holding the shared secret. When unset, auth is disabled and the
# launcher behaves exactly as before (local-only assumption). When set, every
# state-changing request must carry a matching X-CodeRouter-Token header.
_LAUNCHER_TOKEN_ENV = "CODEROUTER_LAUNCHER_TOKEN"
_LAUNCHER_TOKEN_HEADER = "X-CodeRouter-Token"

# Guard so the "no token configured" warning is only logged once per process.
_token_warning_emitted = False


def _require_launcher_token(request: Request) -> None:
    """Enforce the launcher shared-secret on state-changing endpoints.

    If ``CODEROUTER_LAUNCHER_TOKEN`` is unset the launcher stays open (its
    historical local-only behaviour) and a one-time warning is logged. When
    the env var is set, the ``X-CodeRouter-Token`` header must match it using
    a constant-time comparison, otherwise a 401 is raised.
    """
    global _token_warning_emitted
    expected = os.environ.get(_LAUNCHER_TOKEN_ENV, "")
    if not expected:
        if not _token_warning_emitted:
            logger.warning(
                "launcher-auth-disabled",
                extra={
                    "hint": (
                        f"{_LAUNCHER_TOKEN_ENV} is not set; launcher "
                        "start/stop/delete endpoints are unauthenticated. "
                        "Set it when binding to anything other than loopback."
                    ),
                },
            )
            _token_warning_emitted = True
        return
    provided = request.headers.get(_LAUNCHER_TOKEN_HEADER, "")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing launcher token.")

# ---------------------------------------------------------------------------
# Model file extensions to scan
# ---------------------------------------------------------------------------

_MODEL_EXTS = {".gguf", ".ggml", ".safetensors", ".bin", ".pt", ".pth"}


# ---------------------------------------------------------------------------
# ManagedProcess — 起動したプロセスの状態を保持
# ---------------------------------------------------------------------------


@dataclass
class ManagedProcess:
    """Running or stopped backend process entry."""

    id: str
    name: str
    backend: str         # "llama.cpp" | "vllm" | "mlx"
    model_path: str
    port: int
    options: dict[str, Any]
    extra_args: str
    # "starting" (constructing, pre-spawn) | "loading" (spawned, waiting on
    # the readiness probe) | "running" (readiness confirmed, provider
    # registered) | "stopped" | "error"
    status: str = "starting"
    # MTP / speculative-decoding controls (defaults keep existing call sites
    # working). Recorded for introspection; the resolved flags live in the cmd.
    draft_model_path: str | None = None
    mtp_mode: str = "auto"
    pid: int | None = None
    returncode: int | None = None
    log_tail: deque = field(default_factory=lambda: deque(maxlen=200))
    # MTP auto-fallback state. When the speculative flags were added by AUTO
    # detection and the child dies during startup, the process is relaunched
    # ONCE without them (some archs' draft-mtp support in llama.cpp is
    # immature and crashes the context init). These fields make that possible
    # and impossible to loop.
    spec_tokens: list = field(default_factory=list)
    spec_auto: bool = False
    mtp_fallback_done: bool = False
    fallback_cmd: list | None = None
    started_at: float = 0.0
    # The exact argv currently in effect (updated when the MTP fallback
    # relaunches without spec tokens). Used to respawn on a generic
    # auto-restart — see ``_attempt_restart``.
    cmd: list = field(default_factory=list)
    # Consecutive generic auto-restart attempts since the last healthy
    # (readiness-confirmed) run. Reset to 0 on success.
    restart_count: int = 0
    # Set by api_stop / shutdown_launcher just before signalling the child.
    # Tells _tail_logs the exit was requested, not a crash, so it neither
    # auto-restarts nor mislabels a SIGTERM/SIGKILL exit as "error".
    stopping: bool = False
    # launcher-model-swap.md §10 Q5: set by _wait_ready_and_register once it
    # reaches a terminal outcome (registered successfully, or gave up —
    # timeout / superseded). SwapManager awaits this directly instead of
    # polling ``status``. Always set exactly once per readiness attempt
    # (see the try/finally in _wait_ready_and_register), so a waiter never
    # hangs past its own timeout even on the rare "resolved before the
    # first probe" bail-out path.
    ready: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)
    # True when this process was spawned by the SwapManager
    # (coderouter/launcher_swap.py) rather than the manual UI. Two effects:
    #   * H-1: _wait_ready_and_register skips the GENERIC provider
    #     registration ('launcher-<backend>-<port>' into the shared
    #     "launcher" profile). SwapManager does its own registration under
    #     'launcher-swap-<name>' and its TTL unload can only deregister
    #     that one — the generic entry would leak a dead-port provider +
    #     cached adapter on every TTL cycle (unbounded with ephemeral
    #     ports).
    #   * H-2: _attempt_restart never touches a swap-managed process —
    #     crash recovery is SwapManager's job (the next request re-spawns
    #     under its per-model lock); a launcher auto-restart racing that
    #     re-spawn would fight over the same fixed port.
    swap_managed: bool = False
    # [Unreleased]: the swap catalog model name (SwapModelSpec.name) this
    # process backs, set by SwapManager._spawn via spawn_process's
    # swap_model kwarg. None for a manually-started process (swap_managed
    # is False) or if a future swap-managed spawn path omits it. Purely
    # informational — surfaced by GET /api/launcher/processes so the
    # /launcher UI can label which catalog entry a running process is.
    swap_model: str | None = None
    # asyncio subprocess handle — not serialised
    _proc: Any = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# LauncherRegistry — app.state に格納するレジストリ
# ---------------------------------------------------------------------------


class LauncherRegistry:
    """In-process registry for ManagedProcess instances."""

    def __init__(self) -> None:
        self._procs: dict[str, ManagedProcess] = {}

    def get(self, proc_id: str) -> ManagedProcess:
        try:
            return self._procs[proc_id]
        except KeyError:
            raise KeyError(proc_id) from None

    def add(self, proc: ManagedProcess) -> None:
        self._procs[proc.id] = proc

    def remove(self, proc_id: str) -> None:
        del self._procs[proc_id]

    def all(self) -> list[ManagedProcess]:
        return list(self._procs.values())


def _registry_for_app(app: Any) -> LauncherRegistry:
    """Get or create the LauncherRegistry on ``app.state``.

    Split out from :func:`_registry` so non-HTTP callers (SwapManager,
    coderouter/launcher_swap.py) that only hold the FastAPI ``app`` —
    not a ``Request`` — can reach the same registry that manual
    ``/api/launcher/start`` launches use. Both paths must land in one
    registry so swap-managed processes show up in the Launcher UI too.
    """
    if not hasattr(app.state, "launcher"):
        app.state.launcher = LauncherRegistry()
    return app.state.launcher


def _registry(request: Request) -> LauncherRegistry:
    """Get or create the LauncherRegistry on app.state."""
    return _registry_for_app(request.app)


# ---------------------------------------------------------------------------
# Model scanning
# ---------------------------------------------------------------------------


def _scan_models(model_dirs: list[str]) -> list[dict[str, Any]]:
    """Walk model_dirs and return metadata for each discovered model file."""
    found: list[dict[str, Any]] = []
    for raw in model_dirs:
        base = Path(raw).expanduser().resolve()
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in _MODEL_EXTS:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            found.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "dir": str(p.parent),
                    "size_gb": round(size / (1024**3), 2),
                    "ext": p.suffix.lower(),
                }
            )
    return found


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


# 基底バックエンド名 → 既定の実行ファイル名。キーは常に**基底名**であり、
# ``llama.cpp-cuda`` のようなバリアント名は入らない。参照側は必ず
# ``base_backend()`` を通してから引くこと(launcher_devices §2.0)。
_BACKEND_DEFAULTS: dict[str, str] = {
    "llama.cpp": "llama-server",
    "vllm": "python",
    "mlx": "python",          # mlx_lm.server (Apple Silicon 向け)
}

# H8: flags that re-specify the model path. Allowing these through
# ``options`` / ``extra_args`` would let a caller override the vetted
# ``model_path`` and load an arbitrary file (or, for vllm/mlx, swap the
# ``-m`` module target). We reject them per backend so the model is only
# ever set by the ``model_path`` field.
#   - llama.cpp: ``-m`` / ``--model`` select the GGUF file.
#   - vllm / mlx: ``--model`` selects the model; ``-m`` selects the python
#     module that ``_build_cmd`` pins, so it must not be re-specified either.
#   - llama.cpp: the draft/MTP model is likewise set only via the dedicated
#     ``draft_model_path`` field, so its aliases (``-md`` / ``--model-draft``
#     / ``--spec-draft-model``) are rejected in options / extra_args too. The
#     remaining spec knobs (``--spec-type``, ``--spec-draft-n-max`` …) stay
#     free-form.
_MODEL_FLAGS: dict[str, frozenset[str]] = {
    "llama.cpp": frozenset(
        {"-m", "--model", "-md", "--model-draft", "--spec-draft-model"}
    ),
    "vllm": frozenset({"-m", "--model"}),
    "mlx": frozenset({"-m", "--model"}),
}


def _assert_no_model_override(backend: str, tokens: list[str]) -> None:
    """Raise ValueError if ``tokens`` contain a model-selecting flag.

    ``tokens`` is the flat list of argument tokens sourced from ``options``
    keys or ``shlex.split(extra_args)``. Matching is exact on the flag name;
    ``--model=foo`` style is caught by comparing the part before ``=``.

    ``backend`` may be a variant name (``llama.cpp-cuda``), so the lookup goes
    through :func:`base_backend`. A missing entry must NOT silently disable the
    guard — an unknown base falls back to the union of every banned set, i.e.
    the strictest possible check (fail-closed).
    """
    banned = _MODEL_FLAGS.get(base_backend(backend))
    if banned is None:
        banned = frozenset().union(*_MODEL_FLAGS.values())
    for token in tokens:
        name = token.split("=", 1)[0]
        if name in banned:
            raise ValueError(
                f"Flag {name!r} is not allowed: the model is set by "
                "'model_path' and cannot be re-specified via options or "
                "extra_args."
            )


def _resolve_binary(backend: str, configured: str | None) -> str:
    """Return the executable to use, expanding ~ and env vars.

    ``backend`` may be a variant name; the default-executable fallback is
    keyed by :func:`base_backend` so ``llama.cpp-cuda`` without a configured
    ``binary`` would fall back to ``llama-server`` rather than to the literal
    backend name. In practice variants require ``binary`` (validated at config
    load — see ``LauncherConfig``), so that fallback should be unreachable.
    """
    raw = configured or _BACKEND_DEFAULTS.get(base_backend(backend), backend)
    return str(Path(raw).expanduser())


def _backend_names(backends_cfg: dict[str, Any] | None) -> list[str]:
    """Launcher が提示するバックエンド名の一覧 (順序が UI の並び順)。

    基底 3 つを常に先頭に (従来どおりの順)、そのあとに ``launcher.backends``
    に書かれたバリアントを記述順で並べる。設計 §6。

    ``launcher.backends`` にバリアントを書かない利用者には従来と同じ 3 要素が
    返るので、UI の選択肢も API 応答も完全に不変 —— 特化ビルドは「書いた人に
    だけ見える上級者向けオプション」であり、バッジや注記ではなく構造でそれを
    担保する。
    """
    names = list(_BACKEND_DEFAULTS)
    for name in backends_cfg or {}:
        if name not in names:
            names.append(name)
    return names


def _resolve_backends_sync(
    backends_cfg: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Resolve binary paths and check availability for every backend.

    Performs blocking filesystem I/O (``is_file`` / ``shutil.which``),
    so async-route callers must invoke it via ``asyncio.to_thread``.

    Covers the base backends plus any variant declared in
    ``launcher.backends`` (see :func:`_backend_names`). Each entry gains
    ``base`` / ``variant`` so the UI can label a variant without re-parsing
    the name client-side.
    """
    result: dict[str, dict[str, Any]] = {}
    for backend in _backend_names(backends_cfg):
        base = base_backend(backend)
        default_bin = _BACKEND_DEFAULTS.get(base, backend)
        configured: str | None = None
        if backends_cfg:
            bc = backends_cfg.get(backend)
            if bc and bc.binary:
                configured = bc.binary
        resolved = _resolve_binary(backend, configured)
        expanded = str(Path(resolved).expanduser())
        found = (
            Path(expanded).is_file()               # フルパス指定でファイルが存在
            or shutil.which(expanded) is not None   # PATH から解決可能
        )
        result[backend] = {
            "resolved": resolved,
            "configured": configured or "",
            "default": default_bin,
            "is_custom": configured is not None,
            "found": found,
            # 以下 2 キーはバリアント機能で追加 (基底名では variant=None)。
            "base": base,
            "variant": variant_of(backend),
        }
    return result


# ---------------------------------------------------------------------------
# Hardware detection + model recommendation (luna-go /models 互換の発想)
# ---------------------------------------------------------------------------


def _detect_hardware() -> dict[str, Any]:
    """ハードウェアを best-effort で検出する。

    ブロッキング I/O (sysctl / nvidia-smi) を含むため、async ルートからは
    ``asyncio.to_thread`` 経由で呼ぶこと。
    """
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
    # バリアント名 (llama.cpp-cuda 等) でも同じフラグ体系なので基底名で判定。
    backend = base_backend(backend)
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


def _model_size_gb(path: str) -> float:
    """モデルファイルのサイズ (GB)。失敗時は 0.0 (ブロッキング — to_thread 推奨)。"""
    try:
        return Path(path).expanduser().stat().st_size / (1024 ** 3)
    except OSError:
        return 0.0


def _resolve_within_model_dirs(model_path: str, model_dirs: list[str]) -> Path:
    """Resolve ``model_path`` and assert it lives under a configured model dir.

    M14: ``/api/launcher/suggest`` accepts an arbitrary ``model_path`` query
    param that was previously fed straight into ``expanduser().stat()``. That
    let a caller probe the existence and size of any file on disk (a path
    traversal / information-disclosure primitive). We now resolve the path
    (following ``~`` and ``..``) and require it to be contained in one of the
    resolved ``model_dirs``. Anything outside — or a request when no
    ``model_dirs`` are configured — raises ``ValueError`` (mapped to 400).
    """
    if not model_dirs:
        raise ValueError("No model_dirs configured; model_path cannot be validated.")
    candidate = Path(model_path).expanduser().resolve()
    for raw in model_dirs:
        base = Path(raw).expanduser().resolve()
        if candidate == base or base in candidate.parents:
            return candidate
    raise ValueError("model_path is not under any configured model_dirs.")


def _option_tokens(options: dict[str, Any]) -> list[str]:
    """Flatten an ``options`` dict into CLI tokens (``{flag: val}`` semantics).

    Boolean values become a bare flag when true and vanish when false;
    everything else becomes ``[flag, str(val)]``. Shared by ``_build_cmd`` and
    ``api_start`` so the tokens fed to ``resolve_speculative`` match exactly
    what lands on the command line (no double-parsing drift).
    """
    tokens: list[str] = []
    for flag, val in options.items():
        if isinstance(val, bool):
            if val:
                tokens.append(flag)
        else:
            tokens.extend([flag, str(val)])
    return tokens


def _build_cmd(
    backend: str,
    model_path: str,
    port: int,
    options: dict[str, Any],
    extra_args: str,
    binary: str | None = None,
    spec_tokens: list[str] | None = None,
    device_args: list[str] | None = None,
) -> list[str]:
    """Build the CLI command list for the given backend and options.

    ``binary`` overrides the default executable (``llama-server`` /
    ``python``).  When None, the default is used and PATH resolution
    is left to the OS.

    ``spec_tokens`` are pre-resolved speculative-decoding / MTP flags
    (from :func:`coderouter.launcher_speculative.resolve_speculative`).
    For llama.cpp they are appended right after the port args and BEFORE
    the profile / extra args. They are trusted (the launcher, not the
    caller, produced them) so they bypass the model-override guard even
    though they may contain ``--model-draft``.

    ``device_args`` are pre-resolved llama.cpp device-selection flags
    (``--device`` / ``--tensor-split``) produced by
    :meth:`coderouter.launcher_devices.DeviceSelection.to_cli_args`. They
    are inserted right after the port args and BEFORE ``spec_tokens`` for
    llama.cpp only. When ``None``/empty the argv is byte-for-byte identical
    to the pre-feature output (後方互換の核心 — 未選択なら ``--device`` を
    一切足さない). They are launcher-produced (trusted) so they bypass the
    model-override guard, and never apply to vllm/mlx (device 非対応)。
    """
    exe = _resolve_binary(backend, binary)
    # argv の形はバリアントによらず基底バックエンドで決まる
    # (llama.cpp-cuda も llama.cpp と同じフラグ体系)。
    base = base_backend(backend)

    if base == "llama.cpp":
        cmd: list[str] = [exe, "-m", model_path, "--port", str(port)]
        if device_args:
            cmd.extend(device_args)
        if spec_tokens:
            cmd.extend(spec_tokens)
    elif base == "vllm":
        cmd = [
            exe, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
        ]
    elif base == "mlx":
        cmd = [
            exe, "-m", "mlx_lm.server",
            "--model", model_path,
            "--port", str(port),
        ]
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Expected 'llama.cpp', 'vllm' or 'mlx' "
            "(optionally with a '-<variant>' suffix)."
        )

    # H8: reject model-path re-specification via options keys.
    _assert_no_model_override(backend, list(options.keys()))

    cmd.extend(_option_tokens(options))

    if extra_args.strip():
        extra_tokens = shlex.split(extra_args)
        # H8: reject model-path re-specification via free-form extra_args.
        _assert_no_model_override(backend, extra_tokens)
        cmd.extend(extra_tokens)

    return cmd


# ---------------------------------------------------------------------------
# Log reader background task
# ---------------------------------------------------------------------------


# M14: cap the StreamReader buffer for launched processes. A backend that
# emits a huge amount of output without a newline (e.g. a progress bar that
# only writes ``\r``) would otherwise make ``readline()`` buffer without
# bound. asyncio raises ``LimitOverrunError`` once the limit is exceeded; we
# recover by reading the buffered chunk and continuing so log tailing survives.
_LOG_STREAM_LIMIT = 256 * 1024  # 256 KB

# Window (seconds) after spawn in which a non-zero exit is treated as a
# startup crash eligible for the MTP auto-fallback. A backend that ran fine
# for longer and then died is a normal crash, not a load-time failure, so it
# is never retried.
_MTP_FALLBACK_WINDOW_SECS = 180.0


def _should_mtp_fallback(proc: ManagedProcess) -> bool:
    """True when a just-crashed process qualifies for the one-shot MTP retry.

    Only auto-detected speculative launches that die within the startup
    window are eligible, and only once (guarded by ``mtp_fallback_done``).
    """
    p = proc._proc
    rc = p.returncode if p is not None else proc.returncode
    return (
        proc.spec_auto
        and not proc.mtp_fallback_done
        and bool(proc.fallback_cmd)
        and rc not in (0, None)
        and (time.monotonic() - proc.started_at) <= _MTP_FALLBACK_WINDOW_SECS
    )


# ---------------------------------------------------------------------------
# Readiness gating — hole #2: a launcher-started backend used to be
# registered as a routable provider the instant the OS process spawned, well
# before llama-server / vllm had finished loading the model into memory.
# Requests routed there during load failed (connection refused before the
# HTTP listener is up, or a 503 once it is). Backends are now polled for
# readiness and only registered once they answer, or marked "error" (never
# registered) if they don't within ``readiness_timeout_s``.
# ---------------------------------------------------------------------------

_DEFAULT_READINESS_TIMEOUT_S = 300.0
_DEFAULT_READINESS_POLL_INTERVAL_S = 2.0
# Per-probe network timeout — deliberately much shorter than the poll
# interval so a single stuck probe cannot stall the loading→error deadline.
_READINESS_PROBE_TIMEOUT_S = 3.0


async def _backend_ready(backend: str, port: int, *, probe_timeout_s: float) -> bool:
    """Best-effort single readiness probe. Never raises.

    llama.cpp and vllm both expose ``GET /health`` (200 once the model is
    loaded and the server is accepting requests; llama.cpp returns 503
    while still loading). Other backends (mlx — mlx_lm.server has no
    documented health endpoint) fall back to a bare TCP connect: it can't
    distinguish "loaded" from "listening", but it is a strict improvement
    over registering the provider before the port is even open.

    ``backend`` may be a variant name (``llama.cpp-cuda``): the check goes
    through :func:`base_backend` so a variant keeps the ``/health`` probe.
    Falling through to the bare TCP connect here would silently re-introduce
    the bug readiness gating was added to fix (provider registered before the
    model finished loading).

    The probe targets ``127.0.0.1`` literally, never ``localhost``. On a host
    where ``localhost`` resolves to ``::1`` first — the default on GitHub's
    macOS runners, and common on Macs generally — an httpx request to
    ``http://localhost:<port>`` fails with ``Address family not supported``
    against a backend listening on IPv4 only, which is what ``llama-server``
    does by default (``--host 127.0.0.1``). Readiness then never succeeds and
    the spawn times out with ``status='loading'`` even though the backend is
    up and serving. The bare TCP fallback below already used ``127.0.0.1``;
    this branch was the odd one out.
    """
    if base_backend(backend) in ("llama.cpp", "vllm"):
        try:
            async with httpx.AsyncClient(timeout=probe_timeout_s) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/health")
            return resp.status_code == 200
        except Exception:
            return False

    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port), timeout=probe_timeout_s
        )
    except Exception:
        return False
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()
    return True


def _register_provider(proc: ManagedProcess, app: Any) -> dict[str, Any] | None:
    """Register ``proc`` as a routable provider once it is confirmed ready.

    Shared by the initial spawn, the MTP fallback relaunch, and generic
    auto-restart — every path that brings a backend up must go through the
    same readiness→register sequence. Failure never raises (mirrors the
    original inline try/except in ``api_start``).
    """
    engine = getattr(app.state, "engine", None)
    if engine is None or not hasattr(engine, "register_provider"):
        return None
    try:
        summary = engine.register_provider(
            _launcher_provider_config(proc.backend, proc.port)
        )
        proc.log_tail.append(
            "[launcher] provider sync: "
            f"{summary['provider']} -> profile '{summary['profile']}' (in-memory)"
        )
        return summary
    except Exception as exc:
        logger.warning(
            "launcher provider sync failed",
            extra={"backend": proc.backend, "port": proc.port, "error": str(exc)},
        )
        proc.log_tail.append(f"[launcher] provider sync failed: {exc}")
        return None


async def _wait_ready_and_register(
    proc: ManagedProcess, app: Any, launcher_cfg: Any
) -> None:
    """Poll ``proc`` for readiness, then register it — or time out to 'error'.

    Runs as an independent background task per spawn (initial, MTP
    fallback, or auto-restart). Bails out silently — without touching
    ``proc.status`` — the moment the process is no longer in a
    loading-eligible state (crashed, stopped, or already resolved by a
    concurrent readiness task), so a fast crash never races a stale
    registration in after the fact.

    launcher-model-swap.md §10 Q5: ``proc.ready`` is cleared on entry
    (so a respawn — MTP fallback / auto-restart reusing the same
    ``ManagedProcess`` — starts a fresh readiness cycle instead of
    reusing a stale "done" signal) and is *always* set exactly once via
    the ``finally`` below, on every exit path (registered, timed out, or
    bailed out early). Callers that just want to know "is this attempt
    over" (e.g. SwapManager) can therefore plain-``await proc.ready.wait()``
    with their own timeout, no polling of ``proc.status`` needed.
    """
    proc.ready.clear()
    try:
        timeout_s = getattr(launcher_cfg, "readiness_timeout_s", _DEFAULT_READINESS_TIMEOUT_S)
        poll_interval_s = getattr(
            launcher_cfg, "readiness_poll_interval_s", _DEFAULT_READINESS_POLL_INTERVAL_S
        )
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            if proc._proc is None or proc.status not in ("starting", "loading"):
                return
            if await _backend_ready(proc.backend, proc.port, probe_timeout_s=_READINESS_PROBE_TIMEOUT_S):
                if proc.status not in ("starting", "loading"):
                    return  # crashed / stopped while the last probe was in flight
                proc.status = "running"
                proc.restart_count = 0
                proc.log_tail.append("[launcher] readiness check passed")
                if proc.swap_managed:
                    # H-1: a swap-spawned backend must NOT also be
                    # registered under the generic port-based name into
                    # the shared "launcher" profile — SwapManager
                    # registers (and, on TTL unload, deregisters) its own
                    # 'launcher-swap-<name>' provider; a second, generic
                    # registration would outlive every unload as a
                    # dead-port provider. Readiness confirmation and the
                    # ready-Event signal (finally below) are unchanged.
                    proc.log_tail.append(
                        "[launcher] swap-managed: generic provider "
                        "registration skipped (SwapManager registers its own)"
                    )
                else:
                    _register_provider(proc, app)
                return
            await asyncio.sleep(poll_interval_s)

        if proc.status in ("starting", "loading"):
            proc.status = "error"
            proc.log_tail.append(
                f"[launcher] readiness check timed out after {timeout_s:.0f}s "
                "— process left running but NOT registered as a provider"
            )
    finally:
        proc.ready.set()


# ---------------------------------------------------------------------------
# Generic auto-restart — hole #1: besides the one-shot MTP startup-crash
# fallback above, a launcher-started backend that crashed was left in
# status="error" forever with no supervision. Opt-in via
# ``LauncherConfig.auto_restart`` (see schemas.py for the default rationale).
# ---------------------------------------------------------------------------


async def _attempt_restart(proc: ManagedProcess, launcher_cfg: Any) -> bool:
    """Respawn ``proc.cmd`` after a crash. Returns True iff the child started.

    Backed off exponentially and capped at ``auto_restart_max_attempts``;
    the counter is reset to 0 by ``_wait_ready_and_register`` on the next
    readiness success, so a backend that stabilizes gets a fresh budget.
    """
    if proc.swap_managed:
        # H-2: swap-managed processes are supervised by SwapManager alone —
        # a crashed one is re-spawned by the NEXT request for its model,
        # under the manager's per-model lock. A concurrent launcher
        # auto-restart would race that re-spawn for the same (typically
        # fixed) port. §10 Q4's single-supervisor rule, extended to the
        # crash path.
        return False
    if not getattr(launcher_cfg, "auto_restart", False):
        return False
    max_attempts = getattr(launcher_cfg, "auto_restart_max_attempts", 3)
    if proc.restart_count >= max_attempts:
        proc.log_tail.append(
            f"[launcher] auto-restart exhausted ({proc.restart_count}/{max_attempts} "
            "attempts); giving up"
        )
        return False
    if not proc.cmd:
        return False  # nothing to relaunch (should not happen in practice)

    base = getattr(launcher_cfg, "auto_restart_backoff_s", 2.0)
    cap = getattr(launcher_cfg, "auto_restart_backoff_max_s", 30.0)
    backoff = min(base * (2**proc.restart_count), cap)
    proc.restart_count += 1
    proc.log_tail.append(
        f"[launcher] auto-restart attempt {proc.restart_count}/{max_attempts} "
        f"in {backoff:.1f}s"
    )
    await asyncio.sleep(backoff)

    if proc.stopping:
        # Stopped while we were waiting out the backoff — respect it.
        return False

    try:
        p = await asyncio.create_subprocess_exec(
            *proc.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_LOG_STREAM_LIMIT,
        )
    except (FileNotFoundError, OSError) as exc:
        proc.log_tail.append(f"[launcher] auto-restart failed: {exc}")
        proc.status = "error"
        return False

    proc._proc = p
    proc.pid = p.pid
    proc.returncode = None
    proc.started_at = time.monotonic()
    proc.log_tail.append(f"[launcher] auto-restart started PID {p.pid}")
    return True


async def _tail_logs(
    proc: ManagedProcess, *, app: Any = None, launcher_cfg: Any = None
) -> None:
    """Read stdout+stderr into proc.log_tail until the process exits.

    Loops so that a one-shot MTP fallback (relaunch without the auto-detected
    speculative flags) and generic auto-restart can be tailed by the same
    task. The loop exits once the process is intentionally stopped, exits
    cleanly, or no further retry applies.

    ``app`` / ``launcher_cfg`` are optional so unit tests can drive this
    function directly against a fake process without a FastAPI app (the
    readiness/register and auto-restart machinery is then simply inert).
    """

    async def _drain(stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while True:
            try:
                line = await stream.readline()
            except (asyncio.LimitOverrunError, ValueError):
                # M14: a single line exceeded the buffer limit (no newline in
                # sight). Drain the currently buffered bytes so the reader can
                # make progress instead of raising forever, then continue.
                try:
                    chunk = await stream.read(_LOG_STREAM_LIMIT)
                except (asyncio.LimitOverrunError, ValueError):
                    chunk = b""
                if not chunk:
                    if stream.at_eof():
                        break
                    continue
                proc.log_tail.append(chunk.decode(errors="replace").rstrip())
                continue
            if not line:
                break
            proc.log_tail.append(line.decode(errors="replace").rstrip())

    def _spawn_readiness_task() -> None:
        """Kick off (or re-kick after a respawn) the readiness→register task."""
        if app is None:
            return
        task = asyncio.create_task(_wait_ready_and_register(proc, app, launcher_cfg))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    _spawn_readiness_task()  # covers the initial spawn done by api_start

    while True:
        p = proc._proc
        if p is None:
            return

        await asyncio.gather(_drain(p.stdout), _drain(p.stderr))
        await p.wait()
        proc.returncode = p.returncode
        proc.pid = None
        proc.log_tail.append(
            f"[launcher] process exited with code {p.returncode}"
        )

        if proc.stopping:
            # Intentional stop (api_stop / shutdown_launcher). Never a crash
            # to heal, regardless of the exit code SIGTERM/SIGKILL produced.
            proc.status = "stopped"
            return

        proc.status = "stopped" if (p.returncode or 0) == 0 else "error"

        if _should_mtp_fallback(proc):
            # Auto-detected MTP crashed during startup — retry ONCE without
            # the speculative flags. The port is unchanged; readiness is
            # re-armed below so the relaunch is gated exactly like the
            # initial spawn before anything routes to it.
            rc = proc.returncode
            proc.mtp_fallback_done = True
            proc.log_tail.append(
                f"[launcher] MTP startup failure detected (exit code {rc}); "
                "retrying without speculative decoding"
            )
            proc.log_tail.append(
                f"[launcher] cmd: {' '.join(proc.fallback_cmd)}"
            )
            try:
                p = await asyncio.create_subprocess_exec(
                    *proc.fallback_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=_LOG_STREAM_LIMIT,
                )
            except (FileNotFoundError, OSError) as exc:
                proc.log_tail.append(f"[launcher] fallback relaunch failed: {exc}")
                proc.status = "error"
                return
            proc.cmd = proc.fallback_cmd  # future auto-restarts reuse this argv
            proc._proc = p
            proc.pid = p.pid
            proc.status = "loading"
            proc.returncode = None
            proc.started_at = time.monotonic()
            proc.log_tail.append(f"[launcher] started PID {p.pid}")
            _spawn_readiness_task()
            continue  # tail the relaunched process

        if proc.returncode not in (0, None) and await _attempt_restart(proc, launcher_cfg):
            proc.status = "loading"
            _spawn_readiness_task()
            continue  # tail the restarted process

        return


async def shutdown_launcher(app: Any) -> None:
    """Terminate all managed child processes on CodeRouter shutdown.

    Called from the FastAPI lifespan so that llama.cpp / vllm processes
    started via the Launcher are not left as orphans when CodeRouter exits.
    """
    reg = getattr(app.state, "launcher", None)
    if reg is None:
        return
    procs = reg.all()
    for proc in procs:
        # Mark intentional before signalling: _tail_logs must not treat a
        # shutdown-triggered SIGTERM/SIGKILL as a crash to auto-restart.
        proc.stopping = True
        p = proc._proc
        if p is not None and p.returncode is None:
            with contextlib.suppress(Exception):
                p.terminate()
    for proc in procs:
        p = proc._proc
        if p is None or p.returncode is not None:
            continue
        try:
            await asyncio.wait_for(p.wait(), timeout=5.0)
        except TimeoutError:
            with contextlib.suppress(Exception):
                p.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    name: str
    backend: str
    model_path: str
    port: int
    options: dict[str, Any] = {}
    extra_args: str = ""
    # MTP / speculative-decoding controls (llama.cpp only). ``draft_model_path``
    # is an explicit companion draft/MTP gguf; ``mtp_mode`` is "auto" (detect)
    # or "off" (never emit speculative flags).
    draft_model_path: str | None = None
    mtp_mode: str = "auto"
    # デバイス選択(llama.cpp のみ)。既定は空 → ``DeviceSelection.to_cli_args``
    # が ``[]`` を返し、``_build_cmd`` の argv は現行と 1 バイトも変わらない
    # (既存 Web クライアント/テスト完全不変)。設計 §4.1。
    device_ids: list[str] = Field(default_factory=list)
    tensor_split: list[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@router.get("/api/launcher/models")
async def api_models(request: Request) -> dict[str, Any]:
    """Scan model_dirs and return discovered model files."""
    cfg = request.app.state.config
    launcher_cfg = getattr(cfg, "launcher", None)
    model_dirs: list[str] = launcher_cfg.model_dirs if launcher_cfg else []
    # rglob / stat はブロッキング I/O。イベントループ(= プロキシ全体)を
    # 止めないよう別スレッドへ退避する。
    models = await asyncio.to_thread(_scan_models, model_dirs)
    hw = await asyncio.to_thread(_detect_hardware)
    for m in models:
        m["recommendation"] = _model_recommendation(m.get("size_gb", 0.0), hw)
    return {
        "models": models,
        "model_dirs": model_dirs,
        "hardware": hw,
    }


@router.get("/api/launcher/option-profiles")
async def api_option_profiles(request: Request) -> dict[str, Any]:
    """Return option_profiles from providers.yaml launcher config.

    A backend-variant key inherits the base backend's presets: the response
    for ``llama.cpp-cuda`` is the ``llama.cpp`` list followed by the
    variant-specific list, with same-``name`` presets replaced in place
    (:func:`resolve_option_profiles`). Every declared backend gets an entry
    so the UI can key on the selected backend name directly, and a config
    without variants yields exactly the pre-feature payload.
    """
    cfg = request.app.state.config
    launcher_cfg = getattr(cfg, "launcher", None)
    if not launcher_cfg:
        return {"profiles": {}, "_note": "launcher: block not found in providers.yaml"}
    if not launcher_cfg.option_profiles:
        return {"profiles": {}, "_note": "option_profiles is empty — add option_profiles: under launcher: in providers.yaml"}
    names = list(launcher_cfg.option_profiles)
    for name in _backend_names(launcher_cfg.backends):
        if name not in names:
            names.append(name)
    result: dict[str, list[dict]] = {}
    for backend in names:
        profiles = resolve_option_profiles(launcher_cfg.option_profiles, backend)
        if not profiles and backend not in launcher_cfg.option_profiles:
            continue  # 継承もローカル定義も無いバックエンドはキーを作らない
        result[backend] = [{"name": p.name, "args": p.args} for p in profiles]
    return {"profiles": result}


@router.get("/api/launcher/config-debug")
async def api_launcher_config_debug(request: Request) -> dict[str, Any]:
    """Return the effective launcher config for troubleshooting."""
    cfg = request.app.state.config
    launcher_cfg = getattr(cfg, "launcher", None)
    if not launcher_cfg:
        return {"launcher": None, "message": "launcher: block not found in providers.yaml"}
    return {
        "launcher": {
            "model_dirs": launcher_cfg.model_dirs,
            "backends": {k: {"binary": v.binary} for k, v in launcher_cfg.backends.items()},
            "option_profiles": {
                k: [p.name for p in v]
                for k, v in launcher_cfg.option_profiles.items()
            },
        },
    }


@router.get("/api/launcher/processes")
async def api_processes(request: Request) -> dict[str, Any]:
    """List all managed processes.

    [Unreleased]: ``swap_managed`` / ``swap_model`` let the ``/launcher``
    UI (and other API clients) tell an on-demand SwapManager-spawned
    process apart from a manually-started one, and label which swap
    catalog model it backs.
    """
    reg = _registry(request)
    return {
        "processes": [
            {
                "id": p.id,
                "name": p.name,
                "backend": p.backend,
                "model_path": p.model_path,
                "port": p.port,
                "status": p.status,
                "pid": p.pid,
                "returncode": p.returncode,
                "swap_managed": p.swap_managed,
                "swap_model": p.swap_model,
            }
            for p in reg.all()
        ]
    }


@router.get("/api/launcher/backends")
async def api_backends(request: Request) -> dict[str, Any]:
    """Return resolved binary paths for each backend.

    Used by the UI to display which executable will be invoked.
    Shows configured path (from providers.yaml) or the PATH default.
    """
    cfg = request.app.state.config
    launcher_cfg = getattr(cfg, "launcher", None)
    backends_cfg = (
        launcher_cfg.backends
        if (launcher_cfg and launcher_cfg.backends)
        else None
    )
    # is_file / shutil.which はブロッキング I/O。別スレッドへ退避する。
    result = await asyncio.to_thread(_resolve_backends_sync, backends_cfg)
    return {"backends": result}


def _launcher_provider_config(backend: str, port: int) -> Any:
    """Build the provider entry for a launcher-started backend.

    ``model`` is left EMPTY on purpose: llama-server / vllm / mlx decide
    which model is actually loaded, and the empty-model /v1/models
    passthrough then surfaces the upstream's real model id to benchmark
    clients — swap the GGUF, no config edit needed.

    Name embeds backend + port (``launcher-llamacpp-8085``) so restarting
    on the same port REPLACES the entry instead of piling up duplicates.
    """
    from coderouter.config.schemas import ProviderConfig

    safe_backend = backend.replace(".", "")
    return ProviderConfig(
        name=f"launcher-{safe_backend}-{port}",
        base_url=f"http://localhost:{port}/v1",
        model="",
        timeout_s=120.0,
    )


async def spawn_process(
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
) -> ManagedProcess:
    """Build argv, spawn the child process, and arm readiness/log tailing.

    Extracted from (and still used by) ``POST /api/launcher/start`` so
    ``SwapManager`` (coderouter/launcher_swap.py, launcher-model-swap.md
    §4.4) can spawn an on-demand backend through the exact same
    command-building, model-override guard, and readiness machinery —
    no parallel spawn path exists that could bypass
    ``_assert_no_model_override`` / the per-backend argv shape.

    Raises ``ValueError`` on bad input (bad ``mtp_mode``, unbalanced
    ``extra_args`` quoting, a rejected model-override flag, an unknown
    backend) and ``FileNotFoundError`` when the resolved binary doesn't
    exist. Any other exception from ``create_subprocess_exec`` propagates
    as-is. Callers decide how to surface these (HTTP 400/500 for
    ``api_start``, a retryable ``AdapterError`` for SwapManager).

    ``swap_model`` ([Unreleased]) is purely informational — the swap
    catalog model name (``SwapModelSpec.name``), recorded on the
    resulting ``ManagedProcess`` for ``GET /api/launcher/processes`` /
    the ``/launcher`` UI. Only ``SwapManager._spawn`` passes it; the
    manual ``POST /api/launcher/start`` path leaves it ``None``.
    """
    options = options if options is not None else {}
    configured_binary: str | None = None
    if launcher_cfg and getattr(launcher_cfg, "backends", None):
        bc = launcher_cfg.backends.get(backend)
        if bc and bc.binary:
            configured_binary = bc.binary

    if mtp_mode not in ("auto", "off"):
        raise ValueError(f"mtp_mode must be 'auto' or 'off', got {mtp_mode!r}.")

    # Build the user token list once (options flags + extra_args) so
    # resolve_speculative sees exactly what _build_cmd will emit. shlex.split
    # can raise ValueError on unbalanced quotes — let it propagate.
    user_tokens = _option_tokens(options)
    if extra_args.strip():
        user_tokens += shlex.split(extra_args)

    # Resolve MTP / speculative-decoding flags (llama.cpp only; no-op elsewhere).
    spec_tokens, spec_notes = resolve_speculative(
        backend, model_path, draft_model_path, mtp_mode, user_tokens,
    )

    cmd = _build_cmd(
        backend, model_path, port, options, extra_args,
        binary=configured_binary, spec_tokens=spec_tokens,
        device_args=device_args,
    )

    # Speculative flags qualify for the one-shot startup-crash fallback only
    # when they came from AUTO detection (mtp_mode="auto", no explicit draft
    # model, and detection actually emitted flags). Explicit draft paths and
    # operator-supplied --spec-type are never auto-retried. The fallback cmd
    # is rebuilt from scratch with spec_tokens=None (exact — never spliced).
    spec_auto = mtp_mode == "auto" and draft_model_path is None and bool(spec_tokens)
    fallback_cmd: list[str] | None = None
    if spec_auto:
        try:
            fallback_cmd = _build_cmd(
                backend, model_path, port, options, extra_args,
                binary=configured_binary, spec_tokens=None,
                device_args=device_args,
            )
        except ValueError:
            # A failure to rebuild the non-spec command simply disables the
            # fallback; it must never block the primary start.
            fallback_cmd = None

    proc_id = uuid.uuid4().hex[:8]
    proc = ManagedProcess(
        id=proc_id,
        name=name,
        backend=backend,
        model_path=model_path,
        port=port,
        options=options,
        extra_args=extra_args,
        draft_model_path=draft_model_path,
        mtp_mode=mtp_mode,
        status="starting",
        spec_tokens=spec_tokens,
        spec_auto=spec_auto and fallback_cmd is not None,
        fallback_cmd=fallback_cmd,
        swap_managed=swap_managed,
        swap_model=swap_model,
    )
    proc.log_tail.append(f"[launcher] cmd: {' '.join(cmd)}")
    for note in spec_notes:
        proc.log_tail.append(f"[launcher] {note}")

    try:
        p = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # M14: bound the per-stream StreamReader buffer so a newline-less
            # flood from the child cannot grow it without limit.
            limit=_LOG_STREAM_LIMIT,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Executable not found: {cmd[0]!r}. Is {backend} installed?"
        ) from None

    proc._proc = p
    proc.pid = p.pid
    proc.cmd = cmd
    # H2 (readiness gating): status is "loading", not "running", until the
    # background readiness probe (armed by _tail_logs below) confirms the
    # backend is actually serving — see _wait_ready_and_register. Provider
    # registration happens there too, once ready, instead of synchronously
    # here (registering before the model finishes loading is exactly the
    # bug this closes: requests would route to a backend that isn't up yet).
    proc.status = "loading"
    proc.started_at = time.monotonic()
    proc.log_tail.append(f"[launcher] started PID {p.pid}")

    _registry_for_app(app).add(proc)
    _task = asyncio.create_task(_tail_logs(proc, app=app, launcher_cfg=launcher_cfg))
    _background_tasks.add(_task)
    _task.add_done_callback(_background_tasks.discard)

    return proc


async def stop_process(app: Any, proc_id: str) -> ManagedProcess:
    """Terminate a managed process (SIGTERM, then SIGKILL after 5s).

    Extracted from (and still used by) ``POST /api/launcher/stop/{id}``
    so SwapManager's TTL sweeper can use the identical stop sequence —
    including setting ``stopping = True`` first, which is what makes a
    TTL unload an *intentional* stop that launcher auto-restart (when
    enabled) never treats as a crash to heal (§10 Q4).

    Raises ``KeyError`` when ``proc_id`` isn't in the registry. Idempotent
    on an already-stopped process (no-op signal-wise; returns its current
    status).
    """
    proc = _registry_for_app(app).get(proc_id)  # raises KeyError

    # "loading"/"starting": the readiness probe hasn't confirmed the backend
    # yet, but the OS process is alive and stoppable — must be included here
    # or a slow-loading model could never be cancelled from the UI.
    if proc._proc and proc.status in ("running", "loading", "starting"):
        # Intentional stop: tell _tail_logs so it neither auto-restarts nor
        # mislabels whatever exit code SIGTERM/SIGKILL produces as "error".
        proc.stopping = True
        # M14: the child may already be gone (crashed / reaped) between the
        # status check and the signal. terminate()/kill() then raise
        # ProcessLookupError, which previously escaped as a 500. Suppress it
        # (mirroring shutdown_launcher) so stop is idempotent.
        with contextlib.suppress(ProcessLookupError):
            proc._proc.terminate()
        proc.log_tail.append("[launcher] SIGTERM sent")
        try:
            await asyncio.wait_for(proc._proc.wait(), timeout=5.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc._proc.kill()
            proc.log_tail.append("[launcher] SIGKILL sent (timeout)")
        proc.status = "stopped"
        proc.pid = None

    return proc


@router.post("/api/launcher/start")
async def api_start(req: StartRequest, request: Request) -> dict[str, Any]:
    """Start a new backend process."""
    _require_launcher_token(request)
    cfg = request.app.state.config
    launcher_cfg = getattr(cfg, "launcher", None)
    _assert_backend_declared(launcher_cfg, req.backend)
    # デバイス選択 → CLI 断片(llama.cpp のみ・選択された場合のみ)。未指定なら
    # None を渡し、既存の argv と完全一致(後方互換)。
    device_args: list[str] | None = None
    if base_backend(req.backend) == "llama.cpp" and req.device_ids:
        await _assert_device_ids_known(launcher_cfg, req.backend, req.device_ids)
        device_args = DeviceSelection(
            device_ids=list(req.device_ids), tensor_split=list(req.tensor_split)
        ).to_cli_args()
    try:
        proc = await spawn_process(
            request.app, launcher_cfg,
            name=req.name, backend=req.backend, model_path=req.model_path,
            port=req.port, options=req.options, extra_args=req.extra_args,
            draft_model_path=req.draft_model_path, mtp_mode=req.mtp_mode,
            device_args=device_args,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "id": proc.id,
        "pid": proc.pid,
        "command": proc.cmd,
        # Provider auto-sync now happens asynchronously once the readiness
        # probe passes (see _wait_ready_and_register) — poll GET
        # /api/launcher/processes/{id} (status) or /api/launcher/logs/{id}
        # ("provider sync: ...") for the outcome instead of this response.
        "provider_sync": None,
        "speculative": proc.spec_tokens,
    }


@router.post("/api/launcher/stop/{proc_id}")
async def api_stop(proc_id: str, request: Request) -> dict[str, Any]:
    """Terminate a running process (SIGTERM, then SIGKILL after 5s)."""
    _require_launcher_token(request)
    try:
        proc = await stop_process(request.app, proc_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Process {proc_id!r} not found.") from None

    return {"id": proc_id, "status": proc.status}


@router.delete("/api/launcher/processes/{proc_id}")
async def api_delete(proc_id: str, request: Request) -> dict[str, Any]:
    """Remove a stopped process from the registry."""
    _require_launcher_token(request)
    reg = _registry(request)
    try:
        proc = reg.get(proc_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Process {proc_id!r} not found.") from None
    if proc.status in ("running", "loading", "starting"):
        raise HTTPException(status_code=400, detail="Stop the process before deleting.")
    reg.remove(proc_id)
    return {"deleted": proc_id}


@router.get("/api/launcher/logs/{proc_id}")
async def api_logs(proc_id: str, request: Request, n: int = 100) -> dict[str, Any]:
    """Return the last N log lines for a process."""
    try:
        proc = _registry(request).get(proc_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Process {proc_id!r} not found.") from None
    tail = list(proc.log_tail)
    return {"id": proc_id, "logs": tail[-n:], "total": len(tail)}


@router.get("/api/launcher/suggest")
async def api_suggest(request: Request, model_path: str = "",
                      backend: str = "llama.cpp") -> dict[str, Any]:
    """Suggest launch flags for the given model based on detected hardware.

    クライアントの「推奨値」ボタンから呼ばれる。値はあくまで目安。
    バックエンドごとにフラグ体系が違うため backend も受け取る。

    M14: ``model_path`` is validated against the configured ``model_dirs``
    before it is ``stat``-ed, so this endpoint can no longer be used to probe
    the existence/size of arbitrary filesystem paths.
    """
    hw = await asyncio.to_thread(_detect_hardware)
    size_gb = 0.0
    if model_path:
        cfg = request.app.state.config
        launcher_cfg = getattr(cfg, "launcher", None)
        model_dirs: list[str] = launcher_cfg.model_dirs if launcher_cfg else []
        try:
            resolved = _resolve_within_model_dirs(model_path, model_dirs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        size_gb = await asyncio.to_thread(_model_size_gb, str(resolved))
    return {
        "extra_args": _suggest_launch_flags(backend, size_gb, hw),
        "backend": backend,
        "hardware": hw,
        "size_gb": round(size_gb, 2),
    }


# ---------------------------------------------------------------------------
# デバイス検出 + ベンチスイープ(設計 §4.4 / §4.5)
# ---------------------------------------------------------------------------


def _configured_binary_for(launcher_cfg: Any, backend: str) -> str | None:
    """launcher.backends[backend].binary を取り出す(未設定なら None)。

    ``backend`` はバリアント名 (``llama.cpp-cuda``) でもよい —— ``backends`` は
    バリアント名もキーに取るので、そのまま引ける。
    """
    if launcher_cfg and getattr(launcher_cfg, "backends", None):
        bc = launcher_cfg.backends.get(backend)
        if bc and bc.binary:
            return bc.binary
    return None


def _assert_backend_declared(launcher_cfg: Any, backend: str) -> None:
    """バリアント名は ``launcher.backends`` に宣言済みでなければ 400。

    設計 §8。バリアントの実行ファイルパスは ``launcher.backends`` にしか無い
    ので、宣言されていない名前を受け取ったら**フォールバックせずに拒否**する。
    基底名 (``llama.cpp`` 等) は PATH 解決で動くため従来どおり宣言不要。

    これは同時に「API はバックエンド名しか受け取らず、実行ファイルパスは常に
    オペレータの静的設定から来る」という不変則の実装でもある (§11)。
    """
    if not variant_of(backend):
        return
    declared = getattr(launcher_cfg, "backends", None) or {}
    if backend not in declared:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown backend variant {backend!r}: not declared in "
                "launcher.backends. Add it (with a 'binary' path) to "
                "providers.yaml first."
            ),
        )


async def _assert_device_ids_known(
    launcher_cfg: Any, backend: str, device_ids: list[str]
) -> None:
    """選択デバイス ID がそのビルドに実在しなければ 400 (設計 §7.2)。

    デバイス ID の名前空間はビルドごとに違う (``CUDA0`` と ``Vulkan0`` は同じ
    GPU を指さない)。CUDA ビルドで ``CUDA0`` を選んだまま Vulkan ビルドで起動
    すると ``--device CUDA0`` が渡って llama-server が起動失敗するので、spawn
    前に弾く。

    ``--list-devices`` 自体が失敗した環境では検証をスキップする
    (:func:`foreign_device_ids` の best-effort 契約)。
    """
    if not device_ids:
        return
    configured = _configured_binary_for(launcher_cfg, backend)
    binary = _resolve_binary(backend, configured)
    probe = await asyncio.to_thread(detect_llama_devices, binary)
    unknown = foreign_device_ids(list(device_ids), probe)
    if unknown:
        known = [d.id for d in probe.devices]
        raise HTTPException(
            status_code=400,
            detail=(
                f"Device id(s) {unknown} do not exist in backend {backend!r} "
                f"(detected: {known}). Device ids are build-specific — "
                "re-select devices after switching build."
            ),
        )


@router.get("/api/launcher/devices")
async def api_devices(
    request: Request, backend: str = "llama.cpp", refresh: int = 0
) -> dict[str, Any]:
    """``{binary} --list-devices`` を実行してデバイス一覧を返す(設計 §4.4)。

    ``?refresh=1`` で検出キャッシュを無視して再取得する。検出失敗
    (``ok=false``)時は空リスト + error を返し、UI は手入力へフォールバック
    する。

    レスポンス:
      - ``devices``: 検出した全デバイス(表示用。``BLAS: Accelerate`` 等の
        ``total_mib==0`` デバイスも情報として含む)。
      - ``suggested_tensor_split``: **バックエンド別**の VRAM 比提案。
        ``{"CUDA": [0.57, 0.43], "Vulkan": [...]}`` の形。同一バックエンドに
        selectable(``total_mib>0``)が 2 枚以上あるものだけを含む。実機で
        CUDA+Vulkan が同一物理 GPU を重複列挙する/BLAS が 0 MiB で並ぶため、
        跨バックエンド・0 MiB 込みのフラット提案は不正になるのを避ける。
      - ``auto_configs``: スイープ構成候補(``build_auto_sweep_configs``)を
        JSON 化。``[{"label", "device_ids", "tensor_split"}]``。構成生成ロジック
        をサーバ側に一本化し、フロントはこれを使う。
    """
    cfg = request.app.state.config
    launcher_cfg = getattr(cfg, "launcher", None)
    configured = _configured_binary_for(launcher_cfg, backend)
    binary = _resolve_binary(backend, configured)
    # --list-devices はブロッキング subprocess → イベントループを止めない。
    probe = await asyncio.to_thread(
        detect_llama_devices, binary, use_cache=(refresh == 0)
    )
    # selectable(total_mib>0)だけをバックエンドごとに束ね、2 枚以上のときのみ
    # tensor-split を提案。跨バックエンド混成・BLAS(0 MiB)は除外される。
    per_backend_split: dict[str, list[float]] = {}
    if probe.ok:
        for backend_prefix, members in group_by_backend(
            selectable_devices(probe.devices)
        ).items():
            if len(members) >= 2:
                per_backend_split[backend_prefix] = suggest_tensor_split(members)
    auto_configs = [
        {
            "label": label,
            "device_ids": sel.device_ids,
            "tensor_split": sel.tensor_split,
        }
        for label, sel in build_auto_sweep_configs(probe.devices)
    ]
    return {
        **probe.as_dict(),
        "suggested_tensor_split": per_backend_split,
        "auto_configs": auto_configs,
    }


class SweepConfigItem(BaseModel):
    """スイープの 1 デバイス構成(ラベル + 選択デバイス)。"""

    label: str
    device_ids: list[str] = Field(default_factory=list)
    tensor_split: list[float] = Field(default_factory=list)
    # バリアント横断スイープ: このステップだけ別ビルドで起動する。None なら
    # ``SweepRequest.backend`` を使う=従来と完全に同一の argv。
    backend: str | None = None


class SweepRequest(BaseModel):
    """``POST /api/launcher/sweep/start`` のリクエストボディ(設計 §4.4)。"""

    backend: str = "llama.cpp"
    model_path: str
    port: int = Field(ge=1024, le=65535)
    options: dict[str, Any] = Field(default_factory=dict)
    extra_args: str = ""
    configs: list[SweepConfigItem] = Field(default_factory=list)
    bench_command: str | None = None  # None なら launcher.bench 既定
    runs: int | None = None
    results_dir: str | None = None


class _SweepRunner:
    """デバイス構成を順に「起動→readiness→外部ベンチ→停止→次」で回す。

    既存の :func:`spawn_process` / :func:`stop_process` / ``proc.ready``
    (asyncio.Event)を再利用する薄い asyncio ランナー。同時に 1 スイープ
    のみ(``app.state.launcher_sweep``、swap と同じ排他方針)。ポートは
    スイープ専用に 1 本を構成間で使い回す(各 step は ``stop_process`` が
    プロセス終了=ポート解放まで待ってから次へ進む)。
    """

    def __init__(
        self,
        app: Any,
        launcher_cfg: Any,
        plan: SweepPlan,
        *,
        runs: int | None,
        readiness_timeout_s: float,
    ) -> None:
        self.app = app
        self.launcher_cfg = launcher_cfg
        self.plan = plan
        self.runs = runs
        self.readiness_timeout_s = readiness_timeout_s
        self.sweep_id = uuid.uuid4().hex[:8]
        self.running = False
        self.current_index = -1
        self.log_tail: deque[str] = deque(maxlen=1000)
        self._abort = asyncio.Event()
        self._task: asyncio.Task[Any] | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self.running = True
        self._task = asyncio.create_task(self._run())
        _background_tasks.add(self._task)
        self._task.add_done_callback(_background_tasks.discard)

    def abort(self) -> None:
        self._abort.set()

    def status(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "running": self.running,
            "current_index": self.current_index,
            "steps": [s.as_dict() for s in self.plan.steps],
        }

    # -- internals -----------------------------------------------------------

    async def _run(self) -> None:
        try:
            for i, step in enumerate(self.plan.steps):
                self.current_index = i
                if self._abort.is_set():
                    step.state = SweepState.ABORTED
                    self.log_tail.append(f"[sweep] {step.label}: aborted (skipped)")
                    continue
                await self._run_one(step)
        finally:
            self.current_index = -1
            self.running = False
            self.log_tail.append("[sweep] finished")

    async def _await_ready(self, mp: ManagedProcess) -> str:
        """ready / abort / timeout のいずれかまで待つ。

        ``proc.ready`` の待機に abort を絡めることで、readiness 待ち中でも
        中断要求に即応する(単に ``wait_for`` するだけだと timeout まで
        止まらない)。
        """
        ready_wait = asyncio.ensure_future(mp.ready.wait())
        abort_wait = asyncio.ensure_future(self._abort.wait())
        try:
            done, _pending = await asyncio.wait(
                {ready_wait, abort_wait},
                timeout=self.readiness_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (ready_wait, abort_wait):
                if not t.done():
                    t.cancel()
        if abort_wait in done:
            return "aborted"
        if ready_wait in done:
            return "ready"
        return "timeout"

    async def _safe_stop(self, proc_id: str) -> None:
        # stop_process は SIGTERM→5s→SIGKILL でポート解放まで待つ。次構成の
        # 起動前に確実に空ける。失敗しても握りつぶす(スイープを止めない)。
        with contextlib.suppress(Exception):
            await stop_process(self.app, proc_id)

    async def _run_one(self, step: SweepStep) -> None:
        step.state = SweepState.STARTING
        self.log_tail.append(f"[sweep] {step.label}: starting")
        device_args = step.selection.to_cli_args() or None
        # ステップ個別のバックエンド(バリアント横断スイープ)。未指定なら
        # プラン既定 = 従来の挙動。
        step_backend = step.backend or self.plan.backend
        try:
            mp = await spawn_process(
                self.app,
                self.launcher_cfg,
                name=f"sweep-{step.label}",
                backend=step_backend,
                model_path=self.plan.model_path,
                port=self.plan.port,
                options=self.plan.options,
                extra_args=self.plan.extra_args,
                mtp_mode="off",
                device_args=device_args,
            )
        except Exception as exc:  # spawn 自体が失敗 → FAILED(次構成へ継続)
            step.state = SweepState.FAILED
            step.error = f"spawn failed: {exc}"
            self.log_tail.append(f"[sweep] {step.label}: spawn failed: {exc}")
            return

        outcome = await self._await_ready(mp)
        if outcome == "aborted":
            step.state = SweepState.ABORTED
            self.log_tail.append(f"[sweep] {step.label}: aborted during startup")
            await self._safe_stop(mp.id)
            return
        if outcome == "timeout":
            step.state = SweepState.FAILED
            step.error = "readiness timeout"
            self.log_tail.append(f"[sweep] {step.label}: readiness timeout")
            await self._safe_stop(mp.id)
            return
        if mp.status != "running":
            step.state = SweepState.FAILED
            step.error = f"status={mp.status}"
            self.log_tail.append(
                f"[sweep] {step.label}: not running (status={mp.status})"
            )
            await self._safe_stop(mp.id)
            return

        if self._abort.is_set():
            step.state = SweepState.ABORTED
            self.log_tail.append(f"[sweep] {step.label}: aborted before bench")
            await self._safe_stop(mp.id)
            return

        # ── 外部ベンチ実行 ──
        step.state = SweepState.BENCHING
        step.started_at = time.time()
        self.log_tail.append(f"[sweep] {step.label}: benching")
        argv = render_bench_command(
            self.plan.bench_cmd_template,
            port=self.plan.port,
            config_label=step.label,
            results_dir=self.plan.results_dir,
            runs=self.runs,
        )
        env = {
            **os.environ,
            "OPENAI_BASE_URL": f"http://localhost:{self.plan.port}/v1",
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                env=env,
                cwd=self.plan.results_dir or None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=_LOG_STREAM_LIMIT,
            )
        except (FileNotFoundError, OSError) as exc:
            # bench 起動自体が例外 → FAILED(exit code とは別扱い、設計 §2.4)
            step.state = SweepState.FAILED
            step.error = f"bench spawn failed: {exc}"
            step.ended_at = time.time()
            self.log_tail.append(f"[sweep] {step.label}: bench spawn failed: {exc}")
            await self._safe_stop(mp.id)
            return

        if proc.stdout is not None:
            async for raw in proc.stdout:
                self.log_tail.append(raw.decode("utf-8", "replace").rstrip())
        await proc.wait()
        step.bench_exit_code = proc.returncode
        step.ended_at = time.time()
        self.log_tail.append(
            f"[sweep] {step.label}: bench exit code {proc.returncode}"
        )
        # results 解析(results_dir 指定時のみ)。非ゼロ exit でも比較のため
        # DONE 扱い(失敗ではなく exit code で判別、設計 §2.4)。
        if self.plan.results_dir:
            step.results_path, step.summary = await asyncio.to_thread(
                load_latest_results, self.plan.results_dir, since=step.started_at
            )
        step.state = SweepState.DONE
        await self._safe_stop(mp.id)


def _bench_defaults(launcher_cfg: Any) -> tuple[str, int, str | None, float]:
    """launcher.bench(あれば)からベンチ既定を取り出す。無ければハードコード。"""
    default_template = "llmbench run --model local-openai --runs {runs}"
    default_runs = 5
    default_results_dir: str | None = None
    default_readiness = _DEFAULT_READINESS_TIMEOUT_S
    bench_cfg = getattr(launcher_cfg, "bench", None) if launcher_cfg else None
    if bench_cfg is not None:
        default_template = bench_cfg.command_template
        default_runs = bench_cfg.runs
        default_results_dir = bench_cfg.results_dir
        default_readiness = bench_cfg.readiness_timeout_s
    return default_template, default_runs, default_results_dir, default_readiness


@router.post("/api/launcher/sweep/start")
async def api_sweep_start(req: SweepRequest, request: Request) -> dict[str, Any]:
    """ベンチスイープを開始する(設計 §4.4 / §4.5)。書き込み系 → token 必須。"""
    _require_launcher_token(request)
    app = request.app

    existing = getattr(app.state, "launcher_sweep", None)
    if existing is not None and existing.running:
        raise HTTPException(status_code=409, detail="A sweep is already running.")
    if not req.configs:
        raise HTTPException(status_code=400, detail="configs must not be empty.")

    cfg = app.state.config
    launcher_cfg = getattr(cfg, "launcher", None)
    default_template, default_runs, default_results_dir, default_readiness = (
        _bench_defaults(launcher_cfg)
    )
    bench_command = req.bench_command or default_template
    runs = req.runs if req.runs is not None else default_runs
    results_dir = req.results_dir if req.results_dir is not None else default_results_dir

    # ポート競合: registry の使用中ポート照合 + best-effort な空きチェック。
    reg = _registry_for_app(app)
    for p in reg.all():
        if p.port == req.port and p.status in ("running", "loading", "starting"):
            raise HTTPException(
                status_code=400,
                detail=f"Port {req.port} is in use by process {p.name!r}.",
            )
    if not is_port_free(req.port):
        raise HTTPException(
            status_code=400, detail=f"Port {req.port} is not free."
        )

    # バリアント横断スイープ: 各構成が別ビルドを指せる。宣言されていない
    # バリアント名は通常起動と同じく 400 で弾く(フォールバックしない)。
    _assert_backend_declared(launcher_cfg, req.backend)
    for item in req.configs:
        if item.backend:
            _assert_backend_declared(launcher_cfg, item.backend)

    labeled = [
        (
            item.label,
            DeviceSelection(
                device_ids=list(item.device_ids),
                tensor_split=list(item.tensor_split),
            ),
            item.backend,
        )
        for item in req.configs
    ]
    steps = build_sweep_steps(labeled)
    plan = SweepPlan(
        steps=steps,
        model_path=req.model_path,
        backend=req.backend,
        port=req.port,
        bench_cmd_template=bench_command,
        results_dir=results_dir,
        options=req.options,
        extra_args=req.extra_args,
    )
    runner = _SweepRunner(
        app, launcher_cfg, plan, runs=runs, readiness_timeout_s=default_readiness
    )
    app.state.launcher_sweep = runner
    runner.start()
    return {"sweep_id": runner.sweep_id, "steps": [s.as_dict() for s in steps]}


@router.get("/api/launcher/sweep/status")
async def api_sweep_status(request: Request) -> dict[str, Any]:
    """現在(または直近)のスイープ状態を返す。読み取り系 → 認証なし。"""
    runner = getattr(request.app.state, "launcher_sweep", None)
    if runner is None:
        return {
            "sweep_id": None,
            "running": False,
            "current_index": -1,
            "steps": [],
        }
    return runner.status()


@router.post("/api/launcher/sweep/abort")
async def api_sweep_abort(request: Request) -> dict[str, Any]:
    """進行中スイープに中断要求を出す(設計 §4.4)。書き込み系 → token 必須。"""
    _require_launcher_token(request)
    runner = getattr(request.app.state, "launcher_sweep", None)
    if runner is None:
        raise HTTPException(status_code=404, detail="No sweep to abort.")
    runner.abort()
    return {"aborted": True}


@router.get("/api/launcher/sweep/logs")
async def api_sweep_logs(request: Request, n: int = 200) -> dict[str, Any]:
    """スイープの進行ログ末尾 N 行。読み取り系 → 認証なし。"""
    runner = getattr(request.app.state, "launcher_sweep", None)
    if runner is None:
        return {"logs": [], "total": 0}
    tail = list(runner.log_tail)
    return {"logs": tail[-n:], "total": len(tail)}


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

_LAUNCHER_HTML = r"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CodeRouter Launcher</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .dot { width:.5rem;height:.5rem;border-radius:9999px;display:inline-block; }
    .tabnum { font-variant-numeric:tabular-nums; }
    .badge-swap { font-size:10px; text-transform:uppercase; letter-spacing:.02em;
                  background:rgba(99,102,241,.2); color:#a5b4fc; padding:1px 6px;
                  border-radius:9999px; margin-left:6px; vertical-align:middle;
                  white-space:nowrap; }
    .log-box { font-family:monospace;font-size:.75rem;line-height:1.4;
               overflow-y:auto;max-height:14rem;white-space:pre-wrap;word-break:break-all; }
    .model-row:hover { background:rgba(255,255,255,.04);cursor:pointer; }
    .model-row.selected { background:rgba(99,102,241,.15);border-left:2px solid #6366f1; }
    input, select, textarea {
      background:#1e293b;border:1px solid #334155;color:#f1f5f9;
      border-radius:.375rem;padding:.35rem .6rem;width:100%;font-size:.875rem;
      outline:none;
    }
    input:focus, select:focus, textarea:focus { border-color:#6366f1; }
    .btn-primary {
      background:#6366f1;color:#fff;padding:.4rem 1rem;border-radius:.375rem;
      font-size:.875rem;font-weight:600;cursor:pointer;transition:background .15s;
    }
    .btn-primary:hover { background:#4f46e5; }
    .btn-primary:disabled { background:#475569;cursor:not-allowed; }
    .btn-sm {
      padding:.25rem .6rem;border-radius:.25rem;font-size:.75rem;
      cursor:pointer;font-weight:500;transition:background .15s;
    }
    .btn-red { background:#7f1d1d;color:#fca5a5; }
    .btn-red:hover { background:#991b1b; }
    .btn-slate { background:#334155;color:#94a3b8; }
    .btn-slate:hover { background:#475569; }
    .btn-indigo { background:#312e81;color:#a5b4fc; }
    .btn-indigo:hover { background:#3730a3; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans">

<!-- Header -->
<header class="border-b border-slate-800 px-6 py-3">
  <div class="max-w-7xl mx-auto flex items-center gap-x-6 text-sm">
    <span class="text-lg font-semibold tracking-tight">CodeRouter</span>
    <a href="/dashboard" class="text-slate-400 hover:text-slate-200 transition-colors">Dashboard</a>
    <span class="text-slate-100 font-medium border-b border-indigo-400 pb-0.5">Launcher</span>
    <span id="status-msg" class="ml-auto text-xs text-slate-500"></span>
  </div>
</header>

<main class="max-w-7xl mx-auto p-4 md:p-6 space-y-4">

  <!-- Row 1: Models + Launch form -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">

    <!-- Models panel -->
    <section class="bg-slate-900/60 border border-slate-800 rounded-lg p-4 flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-baseline gap-2">
          <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Models</h2>
          <span id="hw-info" class="text-xs text-slate-500"></span>
        </div>
        <button onclick="fetchModels()" class="btn-sm btn-slate">↻ スキャン</button>
      </div>
      <div id="model-dirs" class="text-xs text-slate-500 space-y-0.5"></div>
      <div id="model-list" class="divide-y divide-slate-800 text-sm flex-1 overflow-y-auto max-h-64">
        <div class="py-2 text-slate-500 text-xs">スキャン中…</div>
      </div>
    </section>

    <!-- Launch form -->
    <section class="bg-slate-900/60 border border-slate-800 rounded-lg p-4 flex flex-col gap-3">
      <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Launch</h2>

      <div class="grid grid-cols-2 gap-2">
        <div>
          <label class="block text-xs text-slate-400 mb-1">名前</label>
          <input id="f-name" type="text" placeholder="my-qwen" />
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">ポート</label>
          <input id="f-port" type="number" value="8080" min="1024" max="65535" />
        </div>
      </div>

      <div>
        <label class="block text-xs text-slate-400 mb-1">バックエンド</label>
        <!-- 選択肢は fetchBackends() が /api/launcher/backends の応答から
             動的生成する。providers.yaml に llama.cpp-cuda のようなバリアント
             を書くとここに増える。書かなければ基底 3 つのまま(従来と同一)。
             初期値は JS が届く前でも表示が崩れないための土台。 -->
        <select id="f-backend" onchange="onBackendChange()">
          <option value="llama.cpp">llama.cpp</option>
          <option value="vllm">vllm</option>
          <option value="mlx">mlx</option>
        </select>
        <div id="binary-hint" class="mt-1 text-xs text-slate-500 min-h-[1.2rem]"></div>
      </div>

      <div>
        <label class="block text-xs text-slate-400 mb-1">モデルパス</label>
        <input id="f-model" type="text" placeholder="← モデル一覧から選択 or 直接入力" />
      </div>

      <div>
        <label class="block text-xs text-slate-400 mb-1">オプションプロファイル</label>
        <select id="f-profile" onchange="onProfileChange()">
          <option value="">-- なし --</option>
        </select>
        <div id="profile-args" class="mt-1 text-xs font-mono text-slate-400 bg-slate-800/50 rounded p-2 hidden"></div>
      </div>

      <div class="grid grid-cols-2 gap-2">
        <div>
          <label class="block text-xs text-slate-400 mb-1">MTP/draft gguf (空欄で自動検出)</label>
          <input id="f-draft" type="text" placeholder="companion .gguf (任意)" />
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">MTP</label>
          <select id="f-mtp">
            <option value="auto">auto</option>
            <option value="off">off</option>
          </select>
        </div>
      </div>

      <div>
        <div class="flex items-center justify-between mb-1">
          <label class="block text-xs text-slate-400">追加オプション(自由入力)</label>
          <button onclick="suggestOptions()" class="btn-sm btn-slate">⚙ 推奨値</button>
        </div>
        <input id="f-extra" type="text" placeholder="-ngl 99 --threads 8" />
      </div>

      <!-- デバイス選択 (llama.cpp のみ) -->
      <div id="device-block" class="hidden">
        <div class="flex items-center justify-between mb-1">
          <label class="block text-xs text-slate-400">デバイス (llama.cpp)</label>
          <button onclick="fetchDevices(true)" class="btn-sm btn-slate">🔍 検出</button>
        </div>
        <div id="device-list" class="text-xs text-slate-400 space-y-1 bg-slate-800/40 rounded p-2">
          <span class="text-slate-600">「🔍 検出」で --list-devices を実行</span>
        </div>
        <div id="tsplit-row" class="mt-2 hidden">
          <label class="block text-xs text-slate-400 mb-1">tensor-split (複数選択時・自動提案は上書き可)</label>
          <input id="f-tsplit" type="text" placeholder="0.57,0.43" oninput="markTsplitManual()" />
          <div id="tsplit-note" class="mt-1 text-xs text-yellow-400 hidden"></div>
        </div>
      </div>

      <button id="btn-launch" onclick="launchProcess()" class="btn-primary w-full mt-1">
        ▶ 起動
      </button>
      <div id="launch-err" class="text-xs text-red-400 hidden"></div>
    </section>
  </div>

  <!-- Row 2: Running processes -->
  <section class="bg-slate-900/60 border border-slate-800 rounded-lg p-4">
    <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-3">Processes</h2>
    <div class="overflow-x-auto">
      <table class="w-full text-sm tabnum">
        <thead class="text-slate-500 text-left">
          <tr>
            <th class="pb-2 font-medium">NAME</th>
            <th class="pb-2 font-medium">BACKEND</th>
            <th class="pb-2 font-medium">MODEL</th>
            <th class="pb-2 font-medium text-right">PORT</th>
            <th class="pb-2 font-medium text-right">PID</th>
            <th class="pb-2 font-medium">STATUS</th>
            <th class="pb-2 font-medium text-right">ACTIONS</th>
          </tr>
        </thead>
        <tbody id="proc-table" class="divide-y divide-slate-800">
          <tr><td colspan="7" class="py-3 text-slate-500 text-xs">プロセスなし</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- Row 2.5: Bench sweep -->
  <section class="bg-slate-900/60 border border-slate-800 rounded-lg p-4">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">📊 Bench Sweep</h2>
      <div class="flex gap-2">
        <button id="sweep-start" onclick="startSweep()" class="btn-sm btn-indigo">▶ 開始</button>
        <button id="sweep-abort" onclick="abortSweep()" class="btn-sm btn-red">■ 中断</button>
      </div>
    </div>
    <p class="text-xs text-slate-500 mb-2">
      起動フォームの「モデルパス」を使用。各構成を 起動→readiness→ベンチ→停止 で順に回します。
      構成は「デバイス検出」後に候補が自動生成されます (単一デバイス環境では tensor-split 構成は出ません)。
    </p>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
      <div>
        <div class="flex items-center justify-between mb-1">
          <label class="block text-xs text-slate-400">構成候補</label>
          <!-- 宣言済みの llama.cpp バリアントを全部プローブして横断構成を作る。
               「CUDA ビルドと Vulkan ビルドどちらが速いか」を 1 回で回すための入口。 -->
          <button id="btn-cross-variant" onclick="buildCrossVariantConfigs()"
                  class="btn-sm btn-slate hidden">⚙ ビルド横断</button>
        </div>
        <div id="sweep-configs" class="text-xs text-slate-400 space-y-1 bg-slate-800/40 rounded p-2 max-h-40 overflow-y-auto">
          <span class="text-slate-600">先に「🔍 検出」でデバイスを取得してください</span>
        </div>
      </div>
      <div class="space-y-2">
        <div>
          <label class="block text-xs text-slate-400 mb-1">ポート (スイープ専用・構成間で共用)</label>
          <input id="sweep-port" type="number" value="8090" min="1024" max="65535" />
        </div>
        <div>
          <label class="block text-xs text-slate-400 mb-1">ベンチコマンド ({port} {config} {base_url} {results_dir} {runs} を置換)</label>
          <input id="sweep-cmd" type="text" placeholder="llmbench run --model local-openai --runs {runs}" />
        </div>
        <div class="grid grid-cols-2 gap-2">
          <div>
            <label class="block text-xs text-slate-400 mb-1">runs</label>
            <input id="sweep-runs" type="number" value="5" min="1" max="1000" />
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">results_dir (任意)</label>
            <input id="sweep-results" type="text" placeholder="results/" />
          </div>
        </div>
      </div>
    </div>
    <div id="sweep-err" class="text-xs text-red-400 mt-2 hidden"></div>
    <div class="overflow-x-auto mt-3">
      <table class="w-full text-sm tabnum">
        <thead class="text-slate-500 text-left">
          <tr>
            <th class="pb-2 font-medium">CONFIG</th>
            <th class="pb-2 font-medium">DEVICES</th>
            <th class="pb-2 font-medium">STATE</th>
            <th class="pb-2 font-medium text-right">EXIT</th>
            <th class="pb-2 font-medium text-right">tok/s</th>
            <th class="pb-2 font-medium text-right">ttft(ms)</th>
          </tr>
        </thead>
        <tbody id="sweep-table" class="divide-y divide-slate-800">
          <tr><td colspan="6" class="py-3 text-slate-500 text-xs">スイープ未実行</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- Row 3: Log viewer (hidden until a process is selected) -->
  <section id="log-panel" class="bg-slate-900/60 border border-slate-800 rounded-lg p-4 hidden">
    <div class="flex items-center justify-between mb-2">
      <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">
        Log: <span id="log-title" class="text-slate-200 normal-case">—</span>
      </h2>
      <div class="flex gap-2">
        <button onclick="refreshLogs()" class="btn-sm btn-slate">↻ 更新</button>
        <button onclick="closeLog()" class="btn-sm btn-slate">✕ 閉じる</button>
      </div>
    </div>
    <div id="log-box" class="log-box bg-slate-950 rounded p-3 text-slate-300"></div>
  </section>

</main>

<script>
(() => {
  "use strict";

  const POLL_MS = 3000;
  // H8: launcher shared-secret. The server substitutes __LAUNCHER_TOKEN__
  // with the configured token (or an empty string when auth is disabled).
  const LAUNCHER_TOKEN = "__LAUNCHER_TOKEN__";
  // Build headers for state-changing requests, adding the token only when set.
  const authHeaders = (base) => {
    const h = Object.assign({}, base || {});
    if (LAUNCHER_TOKEN) h["X-CodeRouter-Token"] = LAUNCHER_TOKEN;
    return h;
  };
  let allProfiles = {};      // backend → [{name, args}]
  const _modelCache = {};    // index → {path, name, dir, size_gb}
  let selectedLogId = null;
  let logAutoScroll = true;
  let _lastAutoName = "";    // selectModel が自動入力した名前

  // ── Helpers ──────────────────────────────────────────────────────────────

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])
  );

  const statusMsg = (msg, ok = true) => {
    const el = document.getElementById("status-msg");
    el.textContent = msg;
    el.className = "ml-auto text-xs " + (ok ? "text-slate-500" : "text-red-400");
    if (ok) setTimeout(() => { if (el.textContent === msg) el.textContent = ""; }, 3000);
  };

  const showLaunchErr = (msg) => {
    const el = document.getElementById("launch-err");
    if (msg) { el.textContent = msg; el.classList.remove("hidden"); }
    else { el.textContent = ""; el.classList.add("hidden"); }
  };

  const statusDot = (status) => {
    const map = {running:"bg-green-500", starting:"bg-yellow-500",
                 loading:"bg-yellow-500", stopped:"bg-slate-500",
                 error:"bg-red-500"};
    return `<span class="dot ${map[status] || "bg-slate-500"} mr-1.5"></span>${esc(status)}`;
  };

  // ── Models ───────────────────────────────────────────────────────────────

  window.fetchModels = async () => {
    statusMsg("モデルスキャン中…");
    try {
      const r = await fetch("/api/launcher/models");
      const d = await r.json();
      renderModelDirs(d.model_dirs || []);
      renderHwInfo(d.hardware);
      renderModels(d.models || []);
      statusMsg(`モデル ${d.models.length} 件`);
    } catch (e) {
      statusMsg("モデルスキャン失敗: " + e.message, false);
    }
  };

  const renderHwInfo = (hw) => {
    const el = document.getElementById("hw-info");
    if (!el) return;
    if (!hw) { el.textContent = ""; return; }
    const gpu = {metal: "Metal", cuda: "CUDA", cpu: "CPU"}[hw.gpu] || "CPU";
    let s = `${gpu} · RAM ${hw.ram_gb}GB`;
    if (hw.gpu === "cuda" && hw.vram_gb) s += ` · VRAM ${hw.vram_gb}GB`;
    el.textContent = s;
  };

  const renderModelDirs = (dirs) => {
    const el = document.getElementById("model-dirs");
    el.innerHTML = dirs.length
      ? dirs.map(d => `<div class="truncate">📂 ${esc(d)}</div>`).join("")
      : '<div class="text-slate-600">model_dirs 未設定 (providers.yaml)</div>';
  };

  const recBadge = (rec) => {
    if (!rec || !rec.label) return "";
    if (rec.level === "ok")
      return `<span class="text-xs shrink-0" style="color:#22c55e">✓ ${esc(rec.label)}</span>`;
    if (rec.level === "warn")
      return `<span class="text-xs shrink-0" style="color:#eab308">⚠ ${esc(rec.label)}</span>`;
    return "";
  };

  const renderModels = (models) => {
    const el = document.getElementById("model-list");
    if (!models.length) {
      el.innerHTML = '<div class="py-2 text-slate-500 text-xs">モデルが見つかりません</div>';
      return;
    }
    el.innerHTML = models.map((m, i) => {
      _modelCache[i] = m;
      return `
      <div class="model-row px-1 py-2" onclick="selectModel(${i})">
        <div class="flex justify-between items-baseline gap-2">
          <span class="truncate">${esc(m.name)}</span>
          <span class="flex items-baseline gap-2 shrink-0">
            ${recBadge(m.recommendation)}
            <span class="text-slate-400 tabnum">${m.size_gb} GB</span>
          </span>
        </div>
        <div class="text-slate-500 text-xs truncate">${esc(m.dir)}</div>
      </div>`;
    }).join("");
  };

  window.suggestOptions = async () => {
    const model = document.getElementById("f-model").value.trim();
    if (!model) { showLaunchErr("先にモデルを選択してください"); return; }
    const backend = document.getElementById("f-backend").value;
    try {
      const r = await fetch("/api/launcher/suggest?model_path="
                            + encodeURIComponent(model)
                            + "&backend=" + encodeURIComponent(backend));
      const d = await r.json();
      if (!r.ok) { showLaunchErr(d.detail || "推奨値の取得に失敗"); return; }
      document.getElementById("f-extra").value = d.extra_args;
      showLaunchErr("");
      if (d.extra_args) {
        statusMsg("推奨値を設定(目安): " + d.extra_args);
      } else if (backend === "mlx") {
        statusMsg("MLX は起動時の調整フラグ不要です(統合メモリで自動)");
      } else if (backend === "vllm") {
        statusMsg("vllm は起動時フラグ不要です(モデル設定から自動導出)");
      } else {
        statusMsg("このバックエンドは推奨フラグの自動設定対象外です");
      }
    } catch (e) {
      showLaunchErr(e.message);
    }
  };

  window.selectModel = (idx) => {
    const m = _modelCache[idx];
    if (!m) return;
    document.getElementById("f-model").value = m.path;
    // 名前が空 or 前回自動入力した値のまま → 選択モデル名で更新(手入力は保護)
    const nameEl = document.getElementById("f-name");
    if (!nameEl.value || nameEl.value === _lastAutoName) {
      _lastAutoName = m.name.replace(/\.[^.]+$/, "").slice(0, 30);
      nameEl.value = _lastAutoName;
    }
    document.querySelectorAll(".model-row").forEach((r, i) => {
      r.classList.toggle("selected", i === idx);
    });
  };

  // ── Backends (binary paths) ───────────────────────────────────────────────

  let allBackends = {};

  const fetchBackends = async () => {
    try {
      const r = await fetch("/api/launcher/backends");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      allBackends = d.backends || {};
      populateBackendSelect();
      updateCrossVariantButton();
    } catch (e) {
      console.error("[Launcher] fetchBackends failed:", e);
    }
    renderBinaryHint();  // always call outside try-catch so errors surface
  };

  // バックエンドセレクトをサーバ応答から作り直す。バリアント
  // (llama.cpp-cuda 等) には ⚙ を付けて「対応ランタイムを自分で入れる
  // 上級者向け」であることを示す。バリアントが 0 個なら従来と同じ 3 択。
  const populateBackendSelect = () => {
    const sel = document.getElementById("f-backend");
    const names = Object.keys(allBackends);
    if (!names.length) return;
    const prev = sel.value;
    sel.innerHTML = names.map((n) => {
      const info = allBackends[n] || {};
      const label = info.variant ? `${n} ⚙` : n;
      return `<option value="${esc(n)}">${esc(label)}</option>`;
    }).join("");
    sel.value = names.includes(prev) ? prev : names[0];
  };

  const renderBinaryHint = () => {
    const backend = document.getElementById("f-backend").value;
    const hint = document.getElementById("binary-hint");
    const btn = document.getElementById("btn-launch");
    const info = allBackends[backend];
    if (!info) {
      hint.innerHTML = '<span class="text-slate-600 text-xs">バイナリ確認中…</span>';
      return;
    }
    const dotColor = info.found ? "#22c55e" : "#ef4444";   // green-500 / red-500
    const dot = `<svg style="display:inline;vertical-align:middle;margin-right:5px;flex-shrink:0" width="8" height="8" viewBox="0 0 8 8"><circle cx="4" cy="4" r="4" fill="${dotColor}"/></svg>`;
    const label = info.is_custom ? "カスタム設定" : "PATH";
    const statusText = info.found ? "利用可" : "見つかりません";
    const pathColor = info.found
      ? (info.is_custom ? "#818cf8" : "#4ade80")  // indigo-400 / green-400
      : "#f87171";  // red-400
    // バリアント (特化ビルド) は対応 GPU ランタイムをユーザー自身が入れて
    // いる前提なので、選択中はその旨を一行添える。基底名では出さない
    // (通常利用者の画面を汚さない)。
    const runtimeNote = info.variant
      ? `<div style="color:#fbbf24;margin-top:2px">⚙ ${esc(info.variant)} ビルド`
        + ` — 対応ランタイム(CUDA / Vulkan / ROCm)が導入済みの環境でのみ動作します</div>`
      : "";
    hint.innerHTML = "<div style=\"display:flex;align-items:center;overflow:hidden\">"
      + dot
      + `<span style="font-family:monospace;color:${pathColor};overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(info.resolved)}</span>`
      + `<span style="color:#64748b;margin-left:6px;white-space:nowrap;flex-shrink:0">(${label} — ${statusText})</span>`
      + "</div>"
      + runtimeNote;
    hint.style.cssText = "overflow:hidden";
    // Enable/disable launch button based on binary availability
    if (!info.found) {
      btn.disabled = true;
      showLaunchErr(`⚠ "${esc(info.resolved)}" が見つかりません。選択中のバックエンド (${esc(backend)}) をインストールするか、providers.yaml の launcher.backends.${esc(backend)}.binary にフルパスを設定してください。`);
    } else {
      btn.disabled = false;
      // Clear error only if it was a binary-not-found error
      const errEl = document.getElementById("launch-err");
      if (errEl.textContent.startsWith("⚠")) showLaunchErr("");
    }
  };

  // ── Option profiles ──────────────────────────────────────────────────────

  const fetchProfiles = async () => {
    try {
      const r = await fetch("/api/launcher/option-profiles");
      const d = await r.json();
      allProfiles = d.profiles || {};
      populateProfileSelect();
      // Show hint if profiles are empty (misconfigured YAML)
      if (d._note) console.warn("[Launcher] option-profiles:", d._note);
    } catch (e) {
      console.error("[Launcher] fetchProfiles error:", e);
    }
  };

  const populateProfileSelect = () => {
    const backend = document.getElementById("f-backend").value;
    const sel = document.getElementById("f-profile");
    const profiles = allProfiles[backend] || [];
    const hint = profiles.length === 0
      ? '<option value="" disabled style="color:#64748b">providers.yaml に option_profiles を追加すると選べます</option>'
      : '';
    sel.innerHTML = '<option value="">-- なし --</option>' + hint +
      profiles.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("");
    renderProfileArgs();
  };

  window.onBackendChange = () => {
    // ★ デバイス id の名前空間はビルドごとに違う ("CUDA0" と "Vulkan0" は
    //   同じ GPU ではない)。バックエンド/バリアントを切り替えたら選択を破棄
    //   して検出やり直しにする。残したままだと --device CUDA0 が Vulkan
    //   ビルドに渡って起動失敗する (サーバ側でも 400 で弾くが、UI 側で
    //   先に消しておくのが本筋)。
    _devices = [];
    _deviceSel = {};
    _deviceOk = false;
    _autoConfigs = [];
    const box = document.getElementById("device-list");
    if (box) {
      box.innerHTML =
        '<span class="text-slate-600">「検出」を押してデバイスを取得</span>';
    }
    populateProfileSelect();
    renderBinaryHint();
    updateDeviceBlock();
  };

  // renderProfileArgs は下で const 宣言されるため、宣言前参照(TDZ)を避けて
  // 呼び出し時に解決されるラッパーにする。
  window.onProfileChange = () => renderProfileArgs();

  const renderProfileArgs = () => {
    const backend = document.getElementById("f-backend").value;
    const idx = document.getElementById("f-profile").value;
    const box = document.getElementById("profile-args");
    if (idx === "") { box.classList.add("hidden"); box.textContent = ""; return; }
    const profiles = allProfiles[backend] || [];
    const p = profiles[parseInt(idx)];
    if (!p) { box.classList.add("hidden"); return; }
    const lines = Object.entries(p.args).map(([k, v]) =>
      typeof v === "boolean" ? (v ? k : `# ${k} (disabled)`) : `${k} ${v}`
    );
    box.textContent = lines.join("  ");
    box.classList.remove("hidden");
  };

  const selectedProfileArgs = () => {
    const backend = document.getElementById("f-backend").value;
    const idx = document.getElementById("f-profile").value;
    if (idx === "") return {};
    const profiles = allProfiles[backend] || [];
    const p = profiles[parseInt(idx)];
    return p ? p.args : {};
  };

  // ── Devices (llama.cpp) ────────────────────────────────────────────────────

  let _devices = [];        // [{id,name,total_mib,free_mib,total_gb,free_gb}]
  let _deviceSel = {};      // id -> checked
  let _deviceOk = false;
  let _autoConfigs = [];    // サーバ生成のスイープ構成候補

  // バックエンド名は "llama.cpp" だけでなく "llama.cpp-cuda" のようなバリアント
  // 名も来る (docs/designs/launcher-multi-build.md)。素の === 比較にすると
  // バリアント選択時にデバイス欄が出なくなるので基底名で判定する。
  const baseBackend = (name) => {
    for (const b of ["llama.cpp", "vllm", "mlx"]) {
      if (name === b || name.startsWith(b + "-")) return b;
    }
    return name;
  };
  const isLlama = () =>
    baseBackend(document.getElementById("f-backend").value) === "llama.cpp";

  // デバイス id → バックエンド接頭辞(末尾数字を除く)。launcher_devices の
  // backend_of と同じ規則。"CUDA0"→"CUDA" / "Vulkan2"→"Vulkan" / "MTL0"→"MTL"。
  const backendOf = (id) => id.replace(/\d+$/, "");

  const updateDeviceBlock = () => {
    const blk = document.getElementById("device-block");
    if (isLlama()) blk.classList.remove("hidden");
    else blk.classList.add("hidden");
  };

  // VRAM (total) 比の tensor-split を末尾で辻褄合わせして 1.0 にする。
  const splitByTotal = (devs) => {
    const total = devs.reduce((a, d) => a + d.total_mib, 0);
    if (!(total > 0) || devs.length < 2) return [];
    const s = devs.map(d => Math.round((d.total_mib / total) * 100) / 100);
    const head = s.slice(0, -1).reduce((a, b) => a + b, 0);
    s[s.length - 1] = Math.round((1 - head) * 100) / 100;
    return s;
  };

  window.fetchDevices = async (force) => {
    if (!isLlama()) { statusMsg("デバイス選択は llama.cpp のみ"); return; }
    const backend = document.getElementById("f-backend").value;
    const url = "/api/launcher/devices?backend=" + encodeURIComponent(backend)
              + (force ? "&refresh=1" : "");
    const box = document.getElementById("device-list");
    box.innerHTML = '<span class="text-slate-600">検出中…</span>';
    try {
      const r = await fetch(url);
      const d = await r.json();
      _deviceOk = !!d.ok;
      _devices = d.devices || [];
      _autoConfigs = d.auto_configs || [];
      _deviceSel = {};
      renderDevices(d);
      buildSweepConfigs();
    } catch (e) {
      box.innerHTML = '<span class="text-red-400">検出失敗: ' + esc(e.message) + '</span>';
    }
  };

  const renderDevices = (d) => {
    const box = document.getElementById("device-list");
    if (!d.ok) {
      // 検出失敗 → カンマ区切り手入力へフォールバック
      box.innerHTML =
        '<div class="text-yellow-400 mb-1">検出失敗: ' + esc(d.error || "不明")
        + ' — 手入力してください</div>'
        + '<input id="device-fallback" type="text" placeholder="CUDA0,CUDA1 (カンマ区切り)" />';
      document.getElementById("tsplit-row").classList.add("hidden");
      return;
    }
    if (!_devices.length) {
      box.innerHTML = '<span class="text-slate-600">デバイスなし</span>';
      document.getElementById("tsplit-row").classList.add("hidden");
      return;
    }
    // total_mib==0 (BLAS: Accelerate 等) は GPU オフロード先にできないので
    // チェックボックスを無効化(表示は残す)。
    box.innerHTML = _devices.map(dev => {
      const disabled = !(dev.total_mib > 0);
      return `<label class="flex items-center gap-2 ${disabled ? "opacity-50" : "cursor-pointer"}">
        <input type="checkbox" style="width:auto" data-devid="${esc(dev.id)}" ${disabled ? "disabled" : ""} onchange="onDeviceToggle()" />
        <span class="font-mono text-slate-300">${esc(dev.id)}</span>
        <span class="text-slate-500 truncate">${esc(dev.name)}</span>
        <span class="text-slate-500 tabnum ml-auto shrink-0">${disabled ? "— (0 MiB)" : dev.free_gb + "/" + dev.total_gb + " GB"}</span>
      </label>`;
    }).join("");
    updateTsplitVisibility();
  };

  window.onDeviceToggle = () => {
    _deviceSel = {};
    document.querySelectorAll('#device-list input[data-devid]').forEach(cb => {
      _deviceSel[cb.getAttribute("data-devid")] = cb.checked;
    });
    updateTsplitVisibility();
  };

  const selectedDeviceIds = () => {
    if (_deviceOk) return _devices.map(d => d.id).filter(id => _deviceSel[id]);
    const fb = document.getElementById("device-fallback");
    if (!fb) return [];
    return fb.value.split(",").map(s => s.trim()).filter(Boolean);
  };

  // 単一デバイス環境 (総数<=1) or 選択<=1 では tensor-split 欄を隠す (§8.5)。
  // tensor-split の自動提案はバックエンド単位。跨バックエンド選択時は提案せず
  // その旨を表示する(手入力は可)。
  const updateTsplitVisibility = () => {
    const ids = selectedDeviceIds();
    const row = document.getElementById("tsplit-row");
    const note = document.getElementById("tsplit-note");
    if (_devices.length <= 1 || ids.length <= 1) { row.classList.add("hidden"); return; }
    row.classList.remove("hidden");
    const el = document.getElementById("f-tsplit");
    const backends = new Set(ids.map(backendOf));
    if (backends.size > 1) {
      // 跨バックエンド → 自動提案しない(自動値は消す。手入力は保持)。
      note.textContent = "バックエンド跨ぎのため tensor-split 自動提案なし(手入力可)";
      note.classList.remove("hidden");
      if (el.dataset.auto === "1") { el.value = ""; }
      return;
    }
    note.classList.add("hidden");
    // 同一バックエンド。ユーザーが手入力していれば尊重 (dataset.auto で判別)。
    if (!el.value.trim() || el.dataset.auto === "1") {
      const sel = _devices.filter(d => _deviceSel[d.id]);
      el.value = splitByTotal(sel).join(",");
      el.dataset.auto = "1";
    }
  };

  window.markTsplitManual = () => {
    document.getElementById("f-tsplit").dataset.auto = "0";
  };

  // ── Bench sweep ────────────────────────────────────────────────────────────

  let _sweepConfigs = [];   // [{label, device_ids, tensor_split, checked}]
  let _sweepPollTimer = null;

  // スイープ構成候補はサーバの auto_configs(build_auto_sweep_configs)を使う。
  // 0 MiB 除外・バックエンド単位のグループ化・跨バックエンド混成の除外は
  // すべてサーバ側ロジックに一本化されている。
  const buildSweepConfigs = () => {
    const backend = document.getElementById("f-backend").value;
    _sweepConfigs = (_autoConfigs || []).map(c => ({
      label: c.label,
      device_ids: c.device_ids || [],
      tensor_split: c.tensor_split || [],
      backend: backend,
      checked: true,
    }));
    renderSweepConfigs();
  };

  // 宣言済みの llama.cpp バリアントを全部プローブし、ビルド横断の構成を作る。
  // ラベルにビルド名を前置するので {config} 経由でベンチ結果も見分けられる。
  // ビルド間の混成構成は作らない(1 プロセスは 1 実行ファイルなので原理的に不可)。
  window.buildCrossVariantConfigs = async () => {
    const names = Object.keys(allBackends).filter(n => baseBackend(n) === "llama.cpp");
    if (names.length < 2) {
      showSweepErr("横断できるビルドが 1 つしかありません。providers.yaml の "
                   + "launcher.backends に llama.cpp-cuda などを追加してください");
      return;
    }
    showSweepErr("");
    const box = document.getElementById("sweep-configs");
    box.innerHTML = '<span class="text-slate-600">全ビルドを検出中…</span>';
    const collected = [];
    for (const name of names) {
      try {
        const r = await fetch("/api/launcher/devices?backend=" + encodeURIComponent(name));
        const d = await r.json();
        if (!r.ok || !d.ok) continue;   // 検出できないビルドは黙って飛ばす
        const prefix = (allBackends[name] || {}).variant || name;
        for (const c of (d.auto_configs || [])) {
          collected.push({
            label: prefix + " / " + c.label,
            device_ids: c.device_ids || [],
            tensor_split: c.tensor_split || [],
            backend: name,
            checked: true,
          });
        }
      } catch (_) {}
    }
    if (!collected.length) {
      showSweepErr("どのビルドでもデバイスを検出できませんでした");
      renderSweepConfigs();
      return;
    }
    _sweepConfigs = collected;
    renderSweepConfigs();
    statusMsg(`ビルド横断構成 ${collected.length} 件を生成 (${names.length} ビルド)`);
  };

  const renderSweepConfigs = () => {
    const box = document.getElementById("sweep-configs");
    if (!_sweepConfigs.length) {
      box.innerHTML = '<span class="text-slate-600">先に「🔍 検出」でデバイスを取得してください</span>';
      return;
    }
    box.innerHTML = _sweepConfigs.map((c, i) =>
      `<label class="flex items-center gap-2 cursor-pointer">
        <input type="checkbox" style="width:auto" data-cfg="${i}" ${c.checked ? "checked" : ""} onchange="onSweepCfgToggle()" />
        <span class="text-slate-300">${esc(c.label)}</span>
        <span class="text-slate-500 font-mono ml-auto">${esc(c.device_ids.join(","))}${c.tensor_split.length ? " · " + c.tensor_split.join(",") : ""}</span>
      </label>`
    ).join("");
  };

  // 「⚙ ビルド横断」ボタンは llama.cpp のビルドが 2 つ以上宣言されている
  // ときだけ出す(バリアントを書いていない利用者の画面は従来どおり)。
  const updateCrossVariantButton = () => {
    const btn = document.getElementById("btn-cross-variant");
    if (!btn) return;
    const n = Object.keys(allBackends).filter(x => baseBackend(x) === "llama.cpp").length;
    btn.classList.toggle("hidden", n < 2);
  };

  window.onSweepCfgToggle = () => {
    document.querySelectorAll('#sweep-configs input[data-cfg]').forEach(cb => {
      _sweepConfigs[parseInt(cb.getAttribute("data-cfg"))].checked = cb.checked;
    });
  };

  const showSweepErr = (msg) => {
    const el = document.getElementById("sweep-err");
    if (msg) { el.textContent = msg; el.classList.remove("hidden"); }
    else { el.textContent = ""; el.classList.add("hidden"); }
  };

  window.startSweep = async () => {
    showSweepErr("");
    const model = document.getElementById("f-model").value.trim();
    if (!model) { showSweepErr("起動フォームでモデルパスを選択してください"); return; }
    const configs = _sweepConfigs.filter(c => c.checked)
      .map(c => ({label: c.label, device_ids: c.device_ids,
                  tensor_split: c.tensor_split, backend: c.backend || null}));
    if (!configs.length) { showSweepErr("構成を 1 つ以上選択してください"); return; }
    const port = parseInt(document.getElementById("sweep-port").value);
    const cmd = document.getElementById("sweep-cmd").value.trim();
    const runs = parseInt(document.getElementById("sweep-runs").value);
    const results = document.getElementById("sweep-results").value.trim();
    const body = {
      backend: "llama.cpp", model_path: model, port, configs,
      bench_command: cmd || null,
      runs: isNaN(runs) ? null : runs,
      results_dir: results || null,
    };
    try {
      const r = await fetch("/api/launcher/sweep/start", {
        method: "POST", headers: authHeaders({"Content-Type": "application/json"}),
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) { showSweepErr(d.detail || "スイープ開始失敗"); return; }
      statusMsg("スイープ開始: " + d.sweep_id);
      startSweepPolling();
    } catch (e) { showSweepErr(e.message); }
  };

  window.abortSweep = async () => {
    try {
      const r = await fetch("/api/launcher/sweep/abort", {method: "POST", headers: authHeaders()});
      if (r.ok) statusMsg("スイープ中断要求"); else statusMsg("中断対象なし", false);
    } catch (e) { showSweepErr(e.message); }
  };

  const startSweepPolling = () => {
    if (_sweepPollTimer) return;
    _sweepPollTimer = setInterval(pollSweep, POLL_MS);
    pollSweep();
  };

  const pollSweep = async () => {
    try {
      const r = await fetch("/api/launcher/sweep/status");
      const d = await r.json();
      renderSweepSteps(d.steps || []);
      if (d.running && !_sweepPollTimer) {
        // リロードでスイープ進行中を検知 → ポーリング再開。
        _sweepPollTimer = setInterval(pollSweep, POLL_MS);
      } else if (!d.running && _sweepPollTimer) {
        clearInterval(_sweepPollTimer); _sweepPollTimer = null;
      }
    } catch (_) {}
  };

  const sweepStateColor = (s) => ({
    pending: "text-slate-500", starting: "text-yellow-400",
    benching: "text-indigo-300", done: "text-green-400",
    failed: "text-red-400", aborted: "text-slate-400"
  }[s] || "text-slate-400");

  const renderSweepSteps = (steps) => {
    const tb = document.getElementById("sweep-table");
    if (!steps.length) {
      tb.innerHTML = '<tr><td colspan="6" class="py-3 text-slate-500 text-xs">スイープ未実行</td></tr>';
      return;
    }
    tb.innerHTML = steps.map(s => {
      const sm = s.summary || {};
      const tok = sm.tokens_per_sec != null ? sm.tokens_per_sec : "—";
      const ttft = sm.ttft_ms != null ? sm.ttft_ms : "—";
      const dev = (s.device_ids || []).join(",")
        + ((s.tensor_split || []).length ? " · " + s.tensor_split.join(",") : "")
        + (s.backend ? "  [" + s.backend + "]" : "");
      const exit = s.bench_exit_code != null ? s.bench_exit_code : "—";
      return `<tr>
        <td class="py-2 pr-3">${esc(s.label)}</td>
        <td class="py-2 pr-3 font-mono text-slate-400">${esc(dev)}</td>
        <td class="py-2 pr-3 ${sweepStateColor(s.state)}">${esc(s.state)}</td>
        <td class="py-2 pr-3 text-right">${esc(String(exit))}</td>
        <td class="py-2 pr-3 text-right">${esc(String(tok))}</td>
        <td class="py-2 pr-3 text-right">${esc(String(ttft))}</td>
      </tr>`;
    }).join("");
  };

  // ── Launch ───────────────────────────────────────────────────────────────

  window.launchProcess = async () => {
    showLaunchErr("");
    const name = document.getElementById("f-name").value.trim();
    const port = parseInt(document.getElementById("f-port").value);
    const backend = document.getElementById("f-backend").value;
    const model = document.getElementById("f-model").value.trim();
    const extra = document.getElementById("f-extra").value.trim();
    const draft = document.getElementById("f-draft").value.trim();
    const mtp = document.getElementById("f-mtp").value;

    if (!name) { showLaunchErr("名前を入力してください"); return; }
    if (!model) { showLaunchErr("モデルパスを入力してください"); return; }
    if (!port || port < 1024 || port > 65535) { showLaunchErr("ポートは 1024-65535"); return; }

    // デバイス選択 (llama.cpp のみ)。未選択なら空 → サーバ側で --device 不付与。
    let deviceIds = [], tensorSplit = [];
    if (baseBackend(backend) === "llama.cpp") {
      deviceIds = selectedDeviceIds();
      const ts = document.getElementById("f-tsplit").value.trim();
      if (ts && deviceIds.length > 1) {
        tensorSplit = ts.split(",").map(x => parseFloat(x)).filter(x => !isNaN(x));
      }
    }

    const btn = document.getElementById("btn-launch");
    btn.disabled = true;
    btn.textContent = "起動中…";

    try {
      const res = await fetch("/api/launcher/start", {
        method: "POST",
        headers: authHeaders({"Content-Type": "application/json"}),
        body: JSON.stringify({name, backend, model_path: model, port,
                              options: selectedProfileArgs(), extra_args: extra,
                              draft_model_path: draft || null, mtp_mode: mtp,
                              device_ids: deviceIds, tensor_split: tensorSplit}),
      });
      const d = await res.json();
      if (!res.ok) { showLaunchErr(d.detail || "起動失敗"); return; }
      statusMsg(`起動: ${name} (PID ${d.pid})`);
      // reset form name/port only
      document.getElementById("f-name").value = "";
      document.getElementById("f-port").value = String(port + 1);
    } catch (e) {
      showLaunchErr(e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "▶ 起動";
    }
  };

  // ── Processes ────────────────────────────────────────────────────────────

  const fetchProcesses = async () => {
    try {
      const r = await fetch("/api/launcher/processes");
      const d = await r.json();
      const procs = d.processes || [];
      renderProcesses(procs);
      // [Unreleased] 404-loop fix: if the process whose logs we're
      // polling is no longer in the registry (deleted from another
      // client, swap TTL unload removed it, or the server restarted
      // under a still-open tab), stop the log polling BEFORE it even
      // issues a 404 — otherwise poll() would hit
      // /api/launcher/logs/<stale-id> every POLL_MS forever.
      if (selectedLogId && !procs.some(p => p.id === selectedLogId)) {
        markLogGone();
      }
    } catch (_) {}
  };

  const renderProcesses = (procs) => {
    const tbody = document.getElementById("proc-table");
    if (!procs.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="py-3 text-slate-500 text-xs">プロセスなし</td></tr>';
      return;
    }
    tbody.innerHTML = procs.map(p => {
      const modelName = p.model_path.split("/").pop();
      const isActive = p.status === "running" || p.status === "loading" || p.status === "starting";
      const stopBtn = isActive
        ? `<button onclick="stopProc('${p.id}')" class="btn-sm btn-red">■ 停止</button>`
        : "";
      const delBtn = !isActive
        ? `<button onclick="deleteProc('${p.id}')" class="btn-sm btn-slate ml-1">✕</button>`
        : "";
      const logBtn = `<button onclick="openLog('${p.id}','${esc(p.name)}')" class="btn-sm btn-indigo ml-1">📋 ログ</button>`;
      // [Unreleased]: swap badge — marks processes SwapManager spawned
      // on demand, distinct from manually-started ones. Title shows the
      // catalog model name when the backend supplied it.
      const swapBadge = p.swap_managed
        ? `<span class="badge-swap" title="${esc(p.swap_model ? "swap model: " + p.swap_model : "swap-managed")}">swap</span>`
        : "";
      return `<tr>
        <td class="py-2 pr-3 font-medium">${esc(p.name)}${swapBadge}</td>
        <td class="py-2 pr-3 text-slate-400">${esc(p.backend)}</td>
        <td class="py-2 pr-3 text-slate-400 truncate max-w-[10rem]" title="${esc(p.model_path)}">${esc(modelName)}</td>
        <td class="py-2 pr-3 text-right">${p.port}</td>
        <td class="py-2 pr-3 text-right text-slate-400">${p.pid ?? "—"}</td>
        <td class="py-2 pr-3">${statusDot(p.status)}</td>
        <td class="py-2 text-right whitespace-nowrap">${stopBtn}${logBtn}${delBtn}</td>
      </tr>`;
    }).join("");
  };

  window.stopProc = async (id) => {
    if (!confirm("プロセスを停止しますか?")) return;
    const r = await fetch(`/api/launcher/stop/${id}`, {method:"POST", headers: authHeaders()});
    const d = await r.json();
    statusMsg(`停止: ${d.status}`);
    await fetchProcesses();
    if (selectedLogId === id) await refreshLogs();
  };

  window.deleteProc = async (id) => {
    if (!confirm("レジストリから削除しますか?")) return;
    await fetch(`/api/launcher/processes/${id}`, {method:"DELETE", headers: authHeaders()});
    if (selectedLogId === id) closeLog();
    await fetchProcesses();
  };

  // ── Log viewer ───────────────────────────────────────────────────────────

  window.openLog = async (id, name) => {
    selectedLogId = id;
    document.getElementById("log-title").textContent = name;
    document.getElementById("log-panel").classList.remove("hidden");
    await refreshLogs();
  };

  window.refreshLogs = async () => {
    if (!selectedLogId) return;
    try {
      const r = await fetch(`/api/launcher/logs/${selectedLogId}?n=200`);
      // [Unreleased] 404-loop fix: the id is gone from the registry —
      // stop polling instead of retrying (and 404-spamming the serve
      // log) every POLL_MS forever. The server keeps answering 404 for
      // unknown ids (correct); the client just has to take the hint.
      if (r.status === 404) { markLogGone(); return; }
      const d = await r.json();
      const box = document.getElementById("log-box");
      box.textContent = d.logs.join("\n") || "(ログなし)";
      if (logAutoScroll) box.scrollTop = box.scrollHeight;
    } catch (e) {
      document.getElementById("log-box").textContent = "ログ取得失敗: " + e.message;
    }
  };

  // [Unreleased] 404-loop fix: the process we were tailing no longer
  // exists server-side. Clear selectedLogId (which is what stops both
  // poll()'s refreshLogs call and any manual ↻) but keep the panel open
  // with a terminal message so the user sees WHY the logs stopped.
  const markLogGone = () => {
    selectedLogId = null;
    document.getElementById("log-box").textContent = "(process removed)";
  };

  window.closeLog = () => {
    selectedLogId = null;
    document.getElementById("log-panel").classList.add("hidden");
  };

  // ── Init + polling ───────────────────────────────────────────────────────

  const init = async () => {
    updateDeviceBlock();
    await Promise.all([fetchModels(), fetchProfiles(), fetchBackends(), fetchProcesses()]);
    pollSweep();  // 既存スイープがあれば表示 (running なら以後のポーリングを継続)
  };

  const poll = async () => {
    await fetchProcesses();
    if (selectedLogId) await refreshLogs();
  };

  init();
  setInterval(poll, POLL_MS);
})();
</script>

</body>
</html>
"""


@router.get("/launcher", response_class=HTMLResponse)
async def launcher_page() -> HTMLResponse:
    """Serve the launcher single-page UI.

    H8: inject the configured launcher token so the inline JS can attach it
    to state-changing requests. When the env var is unset the placeholder
    becomes an empty string and the UI keeps working unauthenticated (the
    historical local-only default). The token is escaped for a JS string
    context to avoid breaking out of the double-quoted literal.
    """
    token = os.environ.get(_LAUNCHER_TOKEN_ENV, "")
    safe_token = (
        token.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    html = _LAUNCHER_HTML.replace("__LAUNCHER_TOKEN__", safe_token)
    return HTMLResponse(content=html)
