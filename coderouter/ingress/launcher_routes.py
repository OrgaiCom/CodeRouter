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
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
    status: str = "starting"   # "starting" | "running" | "stopped" | "error"
    # MTP / speculative-decoding controls (defaults keep existing call sites
    # working). Recorded for introspection; the resolved flags live in the cmd.
    draft_model_path: str | None = None
    mtp_mode: str = "auto"
    pid: int | None = None
    returncode: int | None = None
    log_tail: deque = field(default_factory=lambda: deque(maxlen=200))
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


def _registry(request: Request) -> LauncherRegistry:
    """Get or create the LauncherRegistry on app.state."""
    if not hasattr(request.app.state, "launcher"):
        request.app.state.launcher = LauncherRegistry()
    return request.app.state.launcher


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
    """
    banned = _MODEL_FLAGS.get(backend, frozenset())
    for token in tokens:
        name = token.split("=", 1)[0]
        if name in banned:
            raise ValueError(
                f"Flag {name!r} is not allowed: the model is set by "
                "'model_path' and cannot be re-specified via options or "
                "extra_args."
            )


def _resolve_binary(backend: str, configured: str | None) -> str:
    """Return the executable to use, expanding ~ and env vars."""
    raw = configured or _BACKEND_DEFAULTS.get(backend, backend)
    return str(Path(raw).expanduser())


def _resolve_backends_sync(
    backends_cfg: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Resolve binary paths and check availability for every backend.

    Performs blocking filesystem I/O (``is_file`` / ``shutil.which``),
    so async-route callers must invoke it via ``asyncio.to_thread``.
    """
    result: dict[str, dict[str, Any]] = {}
    for backend, default_bin in _BACKEND_DEFAULTS.items():
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
    """
    exe = _resolve_binary(backend, binary)

    if backend == "llama.cpp":
        cmd: list[str] = [exe, "-m", model_path, "--port", str(port)]
        if spec_tokens:
            cmd.extend(spec_tokens)
    elif backend == "vllm":
        cmd = [
            exe, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
        ]
    elif backend == "mlx":
        cmd = [
            exe, "-m", "mlx_lm.server",
            "--model", model_path,
            "--port", str(port),
        ]
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Expected 'llama.cpp', 'vllm' or 'mlx'."
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


async def _tail_logs(proc: ManagedProcess) -> None:
    """Read stdout+stderr into proc.log_tail until the process exits."""
    p = proc._proc
    if p is None:
        return

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

    await asyncio.gather(_drain(p.stdout), _drain(p.stderr))
    await p.wait()
    proc.returncode = p.returncode
    proc.pid = None
    proc.status = "stopped" if (p.returncode or 0) == 0 else "error"
    proc.log_tail.append(
        f"[launcher] process exited with code {p.returncode}"
    )


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
    """Return option_profiles from providers.yaml launcher config."""
    cfg = request.app.state.config
    launcher_cfg = getattr(cfg, "launcher", None)
    if not launcher_cfg:
        return {"profiles": {}, "_note": "launcher: block not found in providers.yaml"}
    if not launcher_cfg.option_profiles:
        return {"profiles": {}, "_note": "option_profiles is empty — add option_profiles: under launcher: in providers.yaml"}
    result: dict[str, list[dict]] = {}
    for backend, profiles in launcher_cfg.option_profiles.items():
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
    """List all managed processes."""
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


@router.post("/api/launcher/start")
async def api_start(req: StartRequest, request: Request) -> dict[str, Any]:
    """Start a new backend process."""
    _require_launcher_token(request)
    # Resolve binary path from providers.yaml launcher.backends
    cfg = request.app.state.config
    launcher_cfg = getattr(cfg, "launcher", None)
    configured_binary: str | None = None
    if launcher_cfg and launcher_cfg.backends:
        bc = launcher_cfg.backends.get(req.backend)
        if bc and bc.binary:
            configured_binary = bc.binary

    # Validate the MTP mode up front (only "auto" / "off" are accepted).
    if req.mtp_mode not in ("auto", "off"):
        raise HTTPException(
            status_code=400,
            detail=f"mtp_mode must be 'auto' or 'off', got {req.mtp_mode!r}.",
        )

    # Build the user token list once (options flags + extra_args) so
    # resolve_speculative sees exactly what _build_cmd will emit. shlex.split
    # can raise on unbalanced quotes → surface as 400 like other bad input.
    try:
        user_tokens = _option_tokens(req.options)
        if req.extra_args.strip():
            user_tokens += shlex.split(req.extra_args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Resolve MTP / speculative-decoding flags (llama.cpp only; no-op elsewhere).
    try:
        spec_tokens, spec_notes = resolve_speculative(
            req.backend,
            req.model_path,
            req.draft_model_path,
            req.mtp_mode,
            user_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        cmd = _build_cmd(
            req.backend, req.model_path, req.port,
            req.options, req.extra_args,
            binary=configured_binary,
            spec_tokens=spec_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    proc_id = uuid.uuid4().hex[:8]
    proc = ManagedProcess(
        id=proc_id,
        name=req.name,
        backend=req.backend,
        model_path=req.model_path,
        port=req.port,
        options=req.options,
        extra_args=req.extra_args,
        draft_model_path=req.draft_model_path,
        mtp_mode=req.mtp_mode,
        status="starting",
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
        raise HTTPException(
            status_code=400,
            detail=f"Executable not found: {cmd[0]!r}. Is {req.backend} installed?",
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    proc._proc = p
    proc.pid = p.pid
    proc.status = "running"
    proc.log_tail.append(f"[launcher] started PID {p.pid}")

    _registry(request).add(proc)
    _task = asyncio.create_task(_tail_logs(proc))
    _background_tasks.add(_task)
    _task.add_done_callback(_background_tasks.discard)

    # Provider auto-sync: register the just-started backend as a routable
    # provider so requests can reach it without hand-editing providers.yaml
    # (in-memory only — see FallbackEngine.register_provider docstring).
    # Sync failure must never fail the start itself.
    provider_sync: dict[str, Any] | None = None
    engine = getattr(request.app.state, "engine", None)
    if engine is not None and hasattr(engine, "register_provider"):
        try:
            provider_sync = engine.register_provider(
                _launcher_provider_config(req.backend, req.port)
            )
            proc.log_tail.append(
                "[launcher] provider sync: "
                f"{provider_sync['provider']} -> profile "
                f"'{provider_sync['profile']}' (in-memory)"
            )
        except Exception as exc:
            logger.warning(
                "launcher provider sync failed",
                extra={"backend": req.backend, "port": req.port, "error": str(exc)},
            )
            proc.log_tail.append(f"[launcher] provider sync failed: {exc}")

    return {
        "id": proc_id,
        "pid": p.pid,
        "command": cmd,
        "provider_sync": provider_sync,
        "speculative": spec_tokens,
    }


@router.post("/api/launcher/stop/{proc_id}")
async def api_stop(proc_id: str, request: Request) -> dict[str, Any]:
    """Terminate a running process (SIGTERM, then SIGKILL after 5s)."""
    _require_launcher_token(request)
    try:
        proc = _registry(request).get(proc_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Process {proc_id!r} not found.") from None

    if proc._proc and proc.status == "running":
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
    if proc.status == "running":
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
                 stopped:"bg-slate-500", error:"bg-red-500"};
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
    } catch (e) {
      console.error("[Launcher] fetchBackends failed:", e);
    }
    renderBinaryHint();  // always call outside try-catch so errors surface
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
    hint.innerHTML = dot
      + `<span style="font-family:monospace;color:${pathColor};overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(info.resolved)}</span>`
      + `<span style="color:#64748b;margin-left:6px;white-space:nowrap;flex-shrink:0">(${label} — ${statusText})</span>`;
    hint.style.cssText = "display:flex;align-items:center;gap:0;overflow:hidden";
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
    populateProfileSelect();
    renderBinaryHint();
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

    const btn = document.getElementById("btn-launch");
    btn.disabled = true;
    btn.textContent = "起動中…";

    try {
      const res = await fetch("/api/launcher/start", {
        method: "POST",
        headers: authHeaders({"Content-Type": "application/json"}),
        body: JSON.stringify({name, backend, model_path: model, port,
                              options: selectedProfileArgs(), extra_args: extra,
                              draft_model_path: draft || null, mtp_mode: mtp}),
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
      renderProcesses(d.processes || []);
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
      const stopBtn = p.status === "running"
        ? `<button onclick="stopProc('${p.id}')" class="btn-sm btn-red">■ 停止</button>`
        : "";
      const delBtn = p.status !== "running"
        ? `<button onclick="deleteProc('${p.id}')" class="btn-sm btn-slate ml-1">✕</button>`
        : "";
      const logBtn = `<button onclick="openLog('${p.id}','${esc(p.name)}')" class="btn-sm btn-indigo ml-1">📋 ログ</button>`;
      return `<tr>
        <td class="py-2 pr-3 font-medium">${esc(p.name)}</td>
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
      const d = await r.json();
      const box = document.getElementById("log-box");
      box.textContent = d.logs.join("\n") || "(ログなし)";
      if (logAutoScroll) box.scrollTop = box.scrollHeight;
    } catch (e) {
      document.getElementById("log-box").textContent = "ログ取得失敗: " + e.message;
    }
  };

  window.closeLog = () => {
    selectedLogId = null;
    document.getElementById("log-panel").classList.add("hidden");
  };

  // ── Init + polling ───────────────────────────────────────────────────────

  const init = async () => {
    await Promise.all([fetchModels(), fetchProfiles(), fetchBackends(), fetchProcesses()]);
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
