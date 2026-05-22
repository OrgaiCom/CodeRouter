#!/usr/bin/env python3
"""
CodeRouter Launcher GUI — tkinter 版
llama.cpp / vllm と CodeRouter をブラウザなしで起動・管理するデスクトップツール

使い方:
  python3 launcher_gui.py
  python3 launcher_gui.py --config ~/.coderouter/providers.yaml
  uv run python launcher_gui.py

追加パッケージ: 不要 (tkinter は Python 標準、yaml は CodeRouter の依存)

起動フロー:
  launcher_gui.py 起動
    → ① llama.cpp / vllm を選択モデルで起動 (ポート 8080)
    → ② CodeRouter を起動 (ポート 8088)  ← ★ このGUIから直接起動
    → Claude Code: ANTHROPIC_BASE_URL=http://localhost:8088 claude
"""

from __future__ import annotations

import argparse
import os
import platform
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk, font as tkfont, messagebox

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
# Config
# ---------------------------------------------------------------------------

_MODEL_EXTS = {".gguf", ".ggml", ".safetensors", ".bin", ".pt", ".pth"}

_BACKEND_DEFAULTS = {
    "llama.cpp": "llama-server",
    "vllm": "python",
}

# CodeRouter のデフォルトポート (README / docs に揃えて 8088)
_CODEROUTER_PORT = 8088

# ── ログ蓄積の上限（ビーチボール対策） ──────────────────────────────────────
# 長時間稼働でログが無制限に溜まり、メインスレッドの処理が追いつかなくなって
# UI が固まる（くるくる）のを防ぐための上限値。
_MAX_LOG_LINES      = 5000   # mp.log_lines / _cr_log のメモリ上限（行数）
_MAX_TEXT_LINES     = 2000   # _log_text ウィジェットの表示行上限
_MAX_LINES_PER_TICK = 1500   # _poll 1回で処理する最大行数（残りは次回へ繰越）

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

    return LauncherConfig(
        model_dirs=model_dirs,
        backends=backends,
        option_profiles=option_profiles,
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
               binary: str) -> list[str]:
    if backend == "llama.cpp":
        cmd = [binary, "-m", model_path, "--port", str(port)]
    elif backend == "vllm":
        cmd = [binary, "-m", "vllm.entrypoints.openai.api_server",
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
    try:
        ram_gb = (os.sysconf("SC_PHYS_PAGES")
                  * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3))
    except (ValueError, OSError, AttributeError):
        pass
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


def _suggest_launch_flags(size_gb: float, hw: dict[str, Any]) -> str:
    """選択モデル + ハードから -ngl / --ctx-size / --threads を提案する。

    あくまで目安。他プロセスのメモリ使用や量子化方式までは考慮しない。
    """
    threads = max(1, int(hw.get("cpu_count", 4)) - 2)
    usable = _usable_memory_gb(hw)
    weights = size_gb * 1.15                       # 重み + オーバーヘッド概算
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
    status: str = "starting"   # starting / running / stopped / error
    pid: int | None = None
    returncode: int | None = None
    proc: Any = None
    # 無制限肥大化を防ぐため上限付き deque を使用（古い行から自動破棄）
    log_lines: deque[str] = field(
        default_factory=lambda: deque(maxlen=_MAX_LOG_LINES))


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

        # ポート入力欄（停止中のみ編集可。trace は _cr_conn_var 生成後に設定）
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

        # アニメーション用（Progressbar 非使用）
        self._cr_anim_running: bool = False

        # 接続文字列（Claude Code 用）
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
        # 接続文字列を最新ポートで更新（無効入力時は直前の有効値を維持）
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

        # ポート欄は停止中／エラー時のみ編集可（起動中・稼働中はロック）
        editable = self._cr_status in ("stopped", "error")
        self._cr_port_entry.configure(state="normal" if editable else "disabled")

    _ANIM_CHARS = ["|", "/", "-", "\\"]

    def _cr_anim_tick(self, idx: int) -> None:
        """CodeRouter 起動中のテキストアニメーション（after() ベース）。"""
        if not self._cr_anim_running:
            return
        ch = self._ANIM_CHARS[idx % len(self._ANIM_CHARS)]
        self._cr_label_var.set(f"  CodeRouter  :{self._cr_port}  起動中… {ch}")
        self.after(150, self._cr_anim_tick, idx + 1)

    def _launch_anim_tick(self, proc_id: str, idx: int) -> None:
        """llama.cpp 起動中のボタンテキストアニメーション（after() ベース）。"""
        if self._launch_anim_proc_id != proc_id:
            return
        if proc_id not in self.processes or self.processes[proc_id].status not in ("starting",):
            # 起動完了 or エラー → ボタンを元に戻す
            self._launch_btn.configure(
                text="▶ llama.cpp / vllm 起動", state="normal", cursor="hand2"
            )
            self._launch_anim_proc_id = None
            return
        ch = self._ANIM_CHARS[idx % len(self._ANIM_CHARS)]
        self._launch_btn.configure(text=f"起動中… {ch}")
        self.after(150, self._launch_anim_tick, proc_id, idx + 1)

    # ── CodeRouter 起動 / 停止 ────────────────────────────────────────────────

    def _start_coderouter(self) -> None:
        """CodeRouter をポート欄の値で起動する。providers.yaml がなければ自動生成。"""
        # CodeRouter ポートの検証（ポート欄の値を使用）
        cr_port_raw = self._cr_port_var.get().strip()
        if not cr_port_raw.isdigit() or not (1024 <= int(cr_port_raw) <= 65535):
            self._cr_err_var.set("CodeRouter ポートは 1024–65535 の数字で指定してください")
            return
        self._cr_port = int(cr_port_raw)

        # llama.cpp の現在のポートを取得（フォームの値を使用）
        try:
            llama_port = int(self._port_var.get())
        except (ValueError, AttributeError):
            llama_port = 8080

        # providers.yaml を自動生成（存在しない場合のみ）
        created, yaml_path = _ensure_providers_yaml(llama_port)
        if created:
            self._cr_err_var.set(f"providers.yaml を生成しました: {yaml_path}")
            self.after(4000, lambda: self._cr_err_var.set(""))
            print(f"[CodeRouter] providers.yaml 生成: {yaml_path}", flush=True)

        self._cr_status = "starting"
        self._update_cr_ui()

        cr_port = self._cr_port  # スレッドに渡すためローカルに保持

        def _run() -> None:
            # shutil.which() をスレッド内で実行（メインスレッドをブロックしない）
            cr_cmd = _find_coderouter_cmd()
            cmd = cr_cmd + ["serve", "--port", str(cr_port)]
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
            try:
                self._cr_proc.terminate()
            except Exception:
                pass
            self._cr_log.append("[coderouter] SIGTERM 送信")
        self._cr_status = "stopped"
        self._cr_proc = None
        self._update_cr_ui()

    # ── ウィンドウ閉時 ───────────────────────────────────────────────────────

    def _on_close(self) -> None:
        """ウィンドウを閉じる際に CodeRouter と全バックエンドを停止する。"""
        # CodeRouter 停止
        if self._cr_proc and self._cr_proc.poll() is None:
            try:
                self._cr_proc.terminate()
            except Exception:
                pass

        # llama.cpp / vllm 停止
        for mp in list(self.processes.values()):
            if mp.proc and mp.proc.poll() is None:
                try:
                    mp.proc.terminate()
                except Exception:
                    pass

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
        # Name が空、または前回ここで自動入力した値のまま（＝手で変更していない）
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
        flags = _suggest_launch_flags(size_gb, hw)
        self._extra_var.set(flags)
        self._hw_var.set(_hw_summary(hw))
        self._set_launch_err("")
        self._set_status(f"推奨値を設定（目安）: {_hw_summary(hw)} → {flags}")

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

        tk.Label(card, text="LAUNCH  llama.cpp / vllm", fg=self.FG2, bg=self.BG2,
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

        lbl("追加オプション", 7, 0)
        self._extra_var = tk.StringVar(value="-ngl 99")
        ttk.Entry(card, textvariable=self._extra_var).grid(
            row=7, column=1, columnspan=2, sticky="ew",
            padx=(0, 6), pady=(6, 0))
        tk.Button(card, text="⚙ 推奨値", fg=self.FG, bg=self.BG3,
                  activebackground=self.BG2, activeforeground=self.FG,
                  relief="flat", bd=0, padx=6, pady=3, cursor="hand2",
                  font=("sans-serif", 9),
                  command=self._suggest_options).grid(
            row=7, column=3, sticky="ew", padx=(0, 10), pady=(6, 0))

        # 起動ボタン
        _btn_wrap = tk.Frame(card, bg=self.ACCENT, bd=0)
        _btn_wrap.grid(row=8, column=0, columnspan=4, sticky="ew", padx=10, pady=8)
        self._launch_btn = tk.Button(
            _btn_wrap, text="▶ llama.cpp / vllm 起動",
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
                 wraplength=400).grid(row=9, column=0, columnspan=4,
                                      sticky="ew", padx=10, pady=(0, 6))

        # アニメーション用（Progressbar 非使用）
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

        # 暫定表示（スレッド完了前）
        self._binary_hint_var.set(f"{binary}  (確認中…)")
        self._binary_hint_lbl.configure(fg=self.FG2)

        def _check() -> None:
            found = _check_binary(binary)
            self.after(0, lambda: self._apply_binary_hint(binary, found, is_custom))

        threading.Thread(target=_check, daemon=True).start()

    def _apply_binary_hint(self, binary: str, found: bool, is_custom: bool) -> None:
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
                "llama.cpp をインストールするか、providers.yaml の\n"
                "launcher.backends.llama\\.cpp.binary にフルパスを設定してください。"
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
            self._set_launch_err("ポートは 1024–65535 の数字で指定してください")
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

        try:
            cmd = _build_cmd(backend, model_path, port, profile_args, extra, binary)
        except ValueError as e:
            self._set_launch_err(str(e))
            return

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

        # ボタンアニメーション開始（Progressbar 非使用）
        self._launch_anim_proc_id = proc_id
        self._launch_btn.configure(state="disabled", cursor="arrow")
        self._launch_anim_tick(proc_id, 0)

        def _run() -> None:
            mp.log_lines.append(f"[launcher] cmd: {' '.join(cmd)}")
            try:
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                )
            except Exception as exc:
                mp.status = "error"
                self._log_queue.put((proc_id, f"_ERR_:{exc}"))
                return

            mp.proc  = p
            mp.pid   = p.pid
            mp.status = "running"
            self._log_queue.put((proc_id, f"_OK_:{name}:{port}"))

            assert p.stdout
            for raw in iter(lambda: p.stdout.read(4096), b""):
                for line in raw.decode("utf-8", errors="replace").splitlines():
                    self._log_queue.put((proc_id, line))
            p.wait()
            mp.returncode = p.returncode
            mp.status = "stopped" if p.returncode == 0 else "error"
            self._log_queue.put(
                (proc_id, f"[launcher] exited (code {p.returncode})"))

        threading.Thread(target=_run, daemon=True).start()

    def _do_stop(self) -> None:
        pid = self.selected_proc_id
        if not pid or pid not in self.processes:
            return
        mp = self.processes[pid]
        if mp.proc and mp.proc.poll() is None:
            mp.status = "stopping"
            try:
                mp.proc.terminate()
            except Exception:
                pass
            mp.log_lines.append("[launcher] SIGTERM sent")
            self._refresh_process_table()

    def _do_kill(self) -> None:
        pid = self.selected_proc_id
        if not pid or pid not in self.processes:
            return
        mp = self.processes[pid]
        if mp.proc and mp.proc.poll() is None:
            try:
                mp.proc.kill()
            except Exception:
                pass
            mp.log_lines.append("[launcher] SIGKILL sent")
            self._refresh_process_table()

    def _do_remove(self) -> None:
        pid = self.selected_proc_id
        if not pid or pid not in self.processes:
            return
        mp = self.processes[pid]
        if mp.status in ("running", "starting"):
            if not messagebox.askyesno("確認", f"{mp.name} は実行中です。強制終了して削除しますか?"):
                return
            try:
                mp.proc.kill()
            except Exception:
                pass
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
        try:
            self._proc_tree.selection_set(proc_id)
        except Exception:
            pass
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
            import traceback; traceback.print_exc()
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

            if line.startswith("_OK_:"):
                parts = line.split(":", 2)
                _, pname, pport = parts
                self._set_status(f"起動: {pname} (PID {mp.pid})")
                self._port_var.set(str(int(pport) + 1))
                self._name_var.set("")
                # アニメーション停止（_launch_anim_tick が次回呼ばれたとき自動停止）
                self._launch_anim_proc_id = None
                self._launch_btn.configure(
                    text="▶ llama.cpp / vllm 起動", state="normal", cursor="hand2"
                )
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
                    text="▶ llama.cpp / vllm 起動", state="normal", cursor="hand2"
                )
                changed = True
                continue

            mp.log_lines.append(line)
            if proc_id == self.selected_proc_id:
                pending_log_lines.append(line)
            changed = True

        # ログをまとめて1回だけ書き込む（行ごとに configure するとUI固まる）
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
            if mp.proc and mp.status in ("running", "starting"):
                rc = mp.proc.poll()
                if rc is not None:
                    mp.returncode = rc
                    mp.status = "stopped" if rc == 0 else "error"
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
