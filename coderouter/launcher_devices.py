"""llama.cpp デバイス検出 + tensor-split 提案 + ベンチスイープ・コアロジック。

GUI版 (launcher_gui.py) と Web版 (ingress/launcher_routes.py) の双方から
共有する純粋ロジック層。実際の起動/停止/readiness はランタイムモデルが
異なる (threading vs asyncio) ため各側で薄く実装し、ここには「データ構造 +
純関数 + best-effort な検出プリミティブ」だけを置く。hardware.py の
5-deps 不変則(標準ライブラリのみ・best-effort・失敗は握りつぶして degrade)
に合わせる。

このモジュールは pydantic を使わず dataclass のみで構成する。standalone
GUI(coderouter パッケージ非同梱)からも import できることが要件で、かつ
検出・提案・展開・解析はいずれも純粋関数なので、外部依存を持たせない。
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import socket
import subprocess  # controlled: fixed argv, no shell
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "KNOWN_BASE_BACKENDS",
    "DeviceProbe",
    "DeviceSelection",
    "LlamaDevice",
    "SweepPlan",
    "SweepState",
    "SweepStep",
    "backend_of",
    "base_backend",
    "build_auto_sweep_configs",
    "build_cross_variant_sweep_configs",
    "build_sweep_steps",
    "detect_llama_devices",
    "foreign_device_ids",
    "group_by_backend",
    "is_port_free",
    "is_valid_backend_name",
    "is_variant",
    "load_latest_results",
    "parse_list_devices",
    "render_bench_command",
    "reset_device_cache",
    "resolve_option_profiles",
    "selectable_devices",
    "suggest_tensor_split",
    "summarize_results",
    "variant_of",
]


# ---------------------------------------------------------------------------
# 2.0 バックエンド名の正規化 (基底名 + バリアント)
# ---------------------------------------------------------------------------

# llama.cpp を CUDA / Vulkan / ROCm 向けに個別ビルドしている環境では、同じ
# "llama.cpp" でもビルドごとに実行ファイルも列挙されるデバイスも違う。これを
# 設定・UI・API の全面で扱うため、バックエンド名に "<基底名>-<バリアント>"
# 形式を認める:
#
#   llama.cpp          基底名 (素のビルド / PATH の llama-server)
#   llama.cpp-cuda     build-cuda/bin/llama-server
#   llama.cpp-vulkan   build-vulkan/bin/llama-server
#   llama.cpp-rocm     build-rocm/bin/llama-server
#
# ★重要: バックエンド名は実行ファイルの選択だけでなく「どのフラグ体系か」
# 「readiness を /health で見るか」「MTP を使えるか」といった挙動の分岐にも
# 使われている。それらの分岐は必ず :func:`base_backend` を通して判定すること。
# 素の文字列比較のままバリアント名が届くと、モデル上書きガードが無効化され
# たり readiness が TCP connect に退行したりと、例外を出さずに壊れる
# (docs/designs/launcher-multi-build.md §4 に全分岐の一覧がある)。
KNOWN_BASE_BACKENDS: tuple[str, ...] = ("llama.cpp", "vllm", "mlx")

# バリアント部分に許す文字。プロバイダ名・プロファイル名・ログに混ざっても
# 壊れない集合に限定し、パス区切りやシェルメタ文字を通さない。
_VARIANT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def base_backend(name: str) -> str:
    """``"llama.cpp-cuda"`` → ``"llama.cpp"``。基底名そのものはそのまま返す。

    既知基底名との**最長一致**で判定する。``name.split("-", 1)[0]`` にしない
    のは意図的で、基底名自体がハイフンを含むようになった日に静かに誤判定する
    のを防ぐため。既知基底名で始まらない名前は加工せずそのまま返し、呼び出し
    側の既存 "Unknown backend" 経路に委ねる。
    """
    best = ""
    for base in KNOWN_BASE_BACKENDS:
        if name == base:
            return base
        if name.startswith(base + "-") and len(base) > len(best):
            best = base
    return best or name


def variant_of(name: str) -> str | None:
    """``"llama.cpp-cuda"`` → ``"cuda"``。基底名そのものなら ``None``。

    ``"llama.cpp-"`` のようにバリアント部が空の場合も ``None`` を返す
    (妥当性判定は :func:`is_valid_backend_name` が担う)。
    """
    base = base_backend(name)
    if base == name or not name.startswith(base + "-"):
        return None
    return name[len(base) + 1 :] or None


def is_variant(name: str) -> bool:
    """``name`` が ``<既知基底名>-<バリアント>`` 形式かどうか。"""
    return variant_of(name) is not None


def is_valid_backend_name(name: str) -> bool:
    """既知基底名そのもの、または既知基底名 + 妥当なバリアントかどうか。

    ``launcher.backends`` のキー検証に使う。``"llamacpp"`` (typo) や
    ``"llama.cpp-CUDA"`` (大文字) / ``"llama.cpp-"`` (空) は False。
    """
    if name in KNOWN_BASE_BACKENDS:
        return True
    variant = variant_of(name)
    return variant is not None and _VARIANT_RE.match(variant) is not None


def resolve_option_profiles[P](
    profiles_by_backend: Mapping[str, Sequence[P]], backend: str
) -> list[P]:
    """バックエンド(バリアント可)に適用する option_profiles を解決する。

    基底名のリスト(継承)を先に、バリアント固有のリストを後に連結する。
    ``name`` が衝突したらバリアント固有が**同じ位置で置き換える** —— 並び順を
    安定させ、末尾に重複を作らないため。``backend`` が基底名そのものなら
    ``profiles_by_backend.get(backend, [])`` と完全に同一の結果になる
    (後方互換)。

    要素は ``.name`` 属性を持つ任意の型でよい。pydantic の
    ``LauncherOptionProfile`` と GUI の dataclass ``OptionProfile`` の双方から
    使うため、この層は pydantic に依存しない (モジュール冒頭の方針どおり)。
    """
    base = base_backend(backend)
    inherited = list(profiles_by_backend.get(base, []))
    if backend == base:
        return inherited
    own = list(profiles_by_backend.get(backend, []))
    if not own:
        return inherited

    by_name = {p.name: p for p in own}  # type: ignore[attr-defined]
    merged: list[P] = []
    replaced: set[str] = set()
    for p in inherited:
        name = p.name  # type: ignore[attr-defined]
        if name in by_name:
            merged.append(by_name[name])  # 継承分と同じ位置で差し替え
            replaced.add(name)
        else:
            merged.append(p)
    merged.extend(
        p for p in own if p.name not in replaced  # type: ignore[attr-defined]
    )
    return merged


# ---------------------------------------------------------------------------
# 2.1 デバイス検出
# ---------------------------------------------------------------------------

# llama.cpp の --list-devices は ggml バックエンドを問わず
#   "<name>: <description> (<total> MiB, <free> MiB free)"
# の共通形式で出力する(ggml_backend_dev_name / _description / メモリ)。
# バックエンド名は CUDA0 / Metal / Vulkan0 / SYCL0 / CPU 等さまざま
# なので id は \S+ で汎用に受ける(CUDA 固定にしない)。行末は $ で
# 固定せず余分な後置トークンにも耐える。
# name(description)は貪欲最小 (.+?) で受け、メモリ括弧は「行末側の
# (<total> MiB, <free> MiB free)」に固定する。これにより description 中の
# 入れ子括弧(例 "AMD Radeon Graphics (RADV GFX1151)")を巻き込まず、
# 最後のメモリ括弧だけを取り出す。
# 例(実機出力):
#   "  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31610 MiB free)"       (Linux/Win CUDA)
#   "  Vulkan2: AMD Radeon Graphics (RADV GFX1151) (114166 MiB, 113602 MiB free)"  (入れ子括弧)
#   "  MTL0: Apple M3 Max (53084 MiB, 53083 MiB free)"                   (macOS Metal, id は MTL0)
#   "  BLAS: Accelerate (0 MiB, 0 MiB free)"                            (0 MiB=選択不可・情報のみ)
#   "  SYCL0: Intel Arc ... (16384 MiB, 16000 MiB free)"                (SYCL)
# id は末尾数字を含む物理バックエンド名(CUDA0/Vulkan1/MTL0/SYCL0/BLAS/CPU)。
# 同一物理 GPU が CUDA0 と Vulkan1 のように複数バックエンドで重複列挙される
# ことがある(backend_of / group_by_backend でグループ化)。
_DEVICE_LINE_RE = re.compile(
    r"^\s+(?P<id>\S+):\s+(?P<name>.+?)\s+"
    r"\((?P<total>\d+)\s*MiB,\s+(?P<free>\d+)\s*MiB\s+free\)"
)
_LIST_DEVICES_TIMEOUT_S = 5.0
_DEVICE_CACHE_TTL_S = 60.0
_MIB_PER_GB = 1024.0  # MiB→GiB 表示用(1024 MiB ≒ 1 GiB)


@dataclass(frozen=True, slots=True)
class LlamaDevice:
    """``llama-server --list-devices`` の 1 行分。"""

    id: str  # 例 "CUDA0" — そのまま --device に渡す名前(--list-devices が正)
    name: str  # 例 "NVIDIA GeForce RTX 5090"
    total_mib: int
    free_mib: int

    @property
    def total_gb(self) -> float:
        return round(self.total_mib / _MIB_PER_GB, 1)

    @property
    def free_gb(self) -> float:
        return round(self.free_mib / _MIB_PER_GB, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "total_mib": self.total_mib,
            "free_mib": self.free_mib,
            "total_gb": self.total_gb,
            "free_gb": self.free_gb,
        }


@dataclass(frozen=True, slots=True)
class DeviceProbe:
    """検出結果。ok=False のとき UI は手入力フォールバックへ。"""

    devices: list[LlamaDevice]
    ok: bool
    error: str | None = None
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "devices": [d.as_dict() for d in self.devices],
        }


def parse_list_devices(output: str) -> list[LlamaDevice]:
    """``--list-devices`` 標準出力をパース。

    マッチしない行は無視する(前後の ``Available devices:`` ヘッダや将来の
    追記、VRAM 表記の無い CPU 行に耐える)。
    """
    devices: list[LlamaDevice] = []
    for line in output.splitlines():
        m = _DEVICE_LINE_RE.match(line)
        if not m:
            continue
        devices.append(
            LlamaDevice(
                id=m["id"],
                name=m["name"].strip(),
                total_mib=int(m["total"]),
                free_mib=int(m["free"]),
            )
        )
    return devices


# ── キャッシュ(hardware.py と同じ RLock+TTL パターン。binary パス別) ──
_cache_lock = threading.RLock()
_cache: dict[str, tuple[float, DeviceProbe]] = {}


def detect_llama_devices(
    binary: str,
    *,
    timeout_s: float = _LIST_DEVICES_TIMEOUT_S,
    use_cache: bool = True,
    runner: Callable[[], subprocess.CompletedProcess[str]] | None = None,
) -> DeviceProbe:
    """``{binary} --list-devices`` を実行してデバイス一覧を返す。

    失敗(バイナリ無し/タイムアウト/非ゼロ終了/パース 0 件)時は
    ``ok=False`` の :class:`DeviceProbe` を返し、呼び出し側は手入力へ
    フォールバックする。``runner`` はテスト用に ``subprocess.run`` を
    差し替えるフック。
    """
    key = str(Path(binary).expanduser())
    now = time.monotonic()
    if use_cache:
        with _cache_lock:
            hit = _cache.get(key)
            if hit and (now - hit[0]) < _DEVICE_CACHE_TTL_S:
                return hit[1]

    def _default_runner() -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # fixed argv, no shell
            [key, "--list-devices"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # ★ Windows cp932 対策
            timeout=timeout_s,
            check=False,
        )

    run = runner or _default_runner
    try:
        out = run()
        raw = (out.stdout or "") + (out.stderr or "")  # 版により stderr 出力
        devices = parse_list_devices(raw)
        if not devices:
            probe = DeviceProbe(
                [], ok=False, error="デバイスを検出できませんでした", raw=raw
            )
        else:
            probe = DeviceProbe(devices, ok=True, raw=raw)
    except FileNotFoundError:
        probe = DeviceProbe([], ok=False, error=f"バイナリが見つかりません: {key}")
    except subprocess.TimeoutExpired:
        probe = DeviceProbe([], ok=False, error="--list-devices がタイムアウトしました")
    except (OSError, subprocess.SubprocessError) as exc:
        probe = DeviceProbe([], ok=False, error=str(exc))

    if use_cache:
        with _cache_lock:
            _cache[key] = (now, probe)
    return probe


def reset_device_cache() -> None:
    """テスト用。検出キャッシュを破棄。"""
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# 2.2 デバイス選択 → CLI
# ---------------------------------------------------------------------------


def _fmt_split(x: float) -> str:
    return f"{x:g}"  # 0.57 / 0.43 のように末尾ゼロを落とす


@dataclass
class DeviceSelection:
    """通常起動 / スイープ 1 構成のデバイス指定。"""

    device_ids: list[str] = field(default_factory=list)  # ["CUDA0","CUDA1"]
    tensor_split: list[float] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.device_ids)

    def to_cli_args(self) -> list[str]:
        """llama.cpp 用 CLI 断片。選択が無ければ空(=現状挙動を完全維持)。"""
        if not self.device_ids:
            return []  # ★ 未選択時は何も足さない
        args = ["--device", ",".join(self.device_ids)]
        if len(self.device_ids) > 1 and self.tensor_split:
            args += ["--tensor-split", ",".join(_fmt_split(x) for x in self.tensor_split)]
        return args


def foreign_device_ids(device_ids: Sequence[str], probe: DeviceProbe) -> list[str]:
    """``device_ids`` のうち **別ビルドの名前空間**に属するものを返す。

    バックエンドバリアントを切り替えるとデバイス ID の名前空間も変わる
    (``CUDA0`` と ``Vulkan0`` は同じ GPU を指さない)。CUDA ビルドで
    ``CUDA0`` を選んだまま Vulkan ビルドで起動すると ``--device CUDA0`` が
    Vulkan ビルドに渡って起動失敗するので、spawn 前に弾くための判定。

    判定は **id の完全一致ではなくバックエンド接頭辞** (:func:`backend_of` —
    ``"CUDA0"`` → ``"CUDA"``) で行う。理由は 2 つある:

    1. 検出したいのは「ビルド違い」= 名前空間そのものの不一致であって、
       同一ビルド内の番号ズレではない。UI はチェックボックスから id を渡す
       ので番号ズレは実際には起きない。
    2. 完全一致にすると ``--list-devices`` の出力形式が将来変わって
       :func:`parse_list_devices` が一部の行だけ取りこぼした場合に、**正しい
       id を誤って拒否してしまう**。接頭辞単位なら 1 行でも同じ接頭辞が
       拾えていれば通る。

    ``probe.ok`` が False (``--list-devices`` 自体が失敗) のときと、デバイス
    を 1 つも拾えなかったときは **常に空リスト** を返す。列挙できない環境で
    機能を殺さないための best-effort 原則 (``hardware.py`` の 5-deps 不変則と
    同じ姿勢)。
    """
    if not probe.ok or not probe.devices:
        return []
    known_prefixes = {backend_of(d.id) for d in probe.devices}
    return [d for d in device_ids if backend_of(d) not in known_prefixes]


# ---------------------------------------------------------------------------
# 2.3 tensor-split 自動提案
# ---------------------------------------------------------------------------


def suggest_tensor_split(
    devices: Sequence[LlamaDevice],
    *,
    by: str = "total",  # "total" | "free"
    decimals: int = 2,
) -> list[float]:
    """VRAM 比の tensor-split を提案。

    合計が丸め後もちょうど 1.0 になるよう最後の要素で辻褄を合わせる
    (llama.cpp 側でも正規化されるが表示の一貫性のため)。単一デバイス
    (Metal 単体・単基 CUDA 等)は ``[]`` を返し tensor-split を提案しない。
    """
    if len(devices) <= 1:
        return []
    caps = [float(d.free_mib if by == "free" else d.total_mib) for d in devices]
    total = sum(caps)
    if total <= 0:
        return []
    props = [round(c / total, decimals) for c in caps]
    props[-1] = round(1.0 - sum(props[:-1]), decimals)  # 丸め誤差を末尾に集約
    return props


# ---------------------------------------------------------------------------
# 2.3b デバイス分類ヘルパ(選択可否 / バックエンドグループ化)
# ---------------------------------------------------------------------------

# 末尾の連番(CUDA0→CUDA / Vulkan12→Vulkan / MTL0→MTL)を落として
# 「バックエンド接頭辞」を得る。数字を持たない BLAS / CPU / Metal は
# そのまま返す。
_BACKEND_SUFFIX_RE = re.compile(r"\d+$")


def selectable_devices(devices: Sequence[LlamaDevice]) -> list[LlamaDevice]:
    """GPU オフロード先として選べるデバイスだけを返す。

    ``total_mib == 0`` のデバイス(macOS の ``BLAS: Accelerate (0 MiB, ...)``
    等)は VRAM を持たず tensor-split やスイープ構成に参加できないため除外
    する。一覧表示からは落とさない(:func:`parse_list_devices` は情報として
    全行を返す)——本関数は「選択・提案・スイープ自動生成」の入力を絞るための
    フィルタ。
    """
    return [d for d in devices if d.total_mib > 0]


def backend_of(device_id: str) -> str:
    """デバイス id からバックエンド接頭辞を返す。

    ``"CUDA0"→"CUDA"`` / ``"Vulkan2"→"Vulkan"`` / ``"MTL0"→"MTL"`` /
    ``"SYCL0"→"SYCL"``。末尾に数字を持たない ``"BLAS"`` / ``"CPU"`` /
    ``"Metal"`` はそのまま返す。同一物理 GPU が CUDA と Vulkan の両方で
    列挙されるケースをバックエンド単位で束ねるための正規化。
    """
    return _BACKEND_SUFFIX_RE.sub("", device_id)


def group_by_backend(
    devices: Sequence[LlamaDevice],
) -> dict[str, list[LlamaDevice]]:
    """デバイスをバックエンド接頭辞ごとにグループ化する(挿入順を保持)。"""
    groups: dict[str, list[LlamaDevice]] = {}
    for d in devices:
        groups.setdefault(backend_of(d.id), []).append(d)
    return groups


def build_auto_sweep_configs(
    devices: Sequence[LlamaDevice],
    *,
    by: str = "total",
) -> list[tuple[str, DeviceSelection]]:
    """検出デバイスからスイープ構成(ラベル + :class:`DeviceSelection`)を自動生成。

    生成規則:
    1. **単体構成**: selectable な各デバイスにつき 1 構成(``"CUDA0 単体"`` 等)。
    2. **バックエンド内複数枚構成**: 同一バックエンド(:func:`backend_of`)に
       selectable が 2 枚以上あるバックエンドのみ、その全枚数を束ねた 1 構成を
       追加し、そのバックエンド内 VRAM 比で tensor-split を自動提案。

    **バックエンド跨ぎの混成構成は自動生成しない**(CUDA と Vulkan、あるいは
    同一物理 GPU が両バックエンドで重複列挙されるケースを混ぜない)。手動での
    任意選択は :class:`DeviceSelection` を直接構築すれば妨げられない。
    ``total_mib == 0`` のデバイス(BLAS 等)は :func:`selectable_devices` で
    除外され、単体構成にもグループにも入らない。
    """
    sel = selectable_devices(devices)
    configs: list[tuple[str, DeviceSelection]] = [
        (f"{d.id} 単体", DeviceSelection(device_ids=[d.id])) for d in sel
    ]
    for backend, members in group_by_backend(sel).items():
        if len(members) < 2:
            continue
        ids = [m.id for m in members]
        split = suggest_tensor_split(members, by=by)
        configs.append(
            (
                f"{backend} x{len(members)}",
                DeviceSelection(device_ids=ids, tensor_split=split),
            )
        )
    return configs


def build_cross_variant_sweep_configs(
    probes: Sequence[tuple[str, Sequence[LlamaDevice]]],
    *,
    by: str = "total",
) -> list[tuple[str, str, DeviceSelection]]:
    """複数ビルド(バリアント)を横断するスイープ構成を生成する。

    ``probes`` は ``[(backend_name, devices), ...]`` —— 各バリアントを
    ``--list-devices`` でプローブした結果。返すのは
    ``[(label, backend_name, DeviceSelection), ...]`` で、ラベルには
    バリアント名を前置する (``"cuda / CUDA0 単体"``)。``{config}`` プレース
    ホルダ経由でベンチ結果 JSON もビルド別に見分けられる。

    ``build_auto_sweep_configs`` を各ビルドに対して呼ぶだけなので、
    「バックエンド跨ぎの混成構成は作らない」という既存規則はビルド内でも
    そのまま維持される。ビルド間の混成 (CUDA ビルドの CUDA0 と Vulkan
    ビルドの Vulkan2 を同時に、など) は **原理的に不可能** ——1 プロセスは
    1 つの実行ファイルでしか動かないので、ここでも作らない。

    これで「同一モデルを CUDA ビルドと Vulkan ビルドで順に起動してベンチし、
    どちらが速いか比較する」が 1 回のスイープで回る。
    """
    out: list[tuple[str, str, DeviceSelection]] = []
    for backend, devices in probes:
        prefix = variant_of(backend) or backend
        for label, selection in build_auto_sweep_configs(devices, by=by):
            out.append((f"{prefix} / {label}", backend, selection))
    return out


# ---------------------------------------------------------------------------
# 2.4 スイープ計画・状態
# ---------------------------------------------------------------------------


class SweepState(StrEnum):
    PENDING = "pending"
    STARTING = "starting"  # サーバ起動→readiness 待ち
    BENCHING = "benching"  # 外部ベンチ実行中
    DONE = "done"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class SweepStep:
    label: str  # "CUDA0 単体" 等(表示 + {config} 置換に使用)
    selection: DeviceSelection
    state: SweepState = SweepState.PENDING
    bench_exit_code: int | None = None
    results_path: str | None = None  # 読めた llmbench results JSON
    summary: dict[str, Any] | None = None  # 抽出済み主要メトリクス
    error: str | None = None
    started_at: float = 0.0
    ended_at: float = 0.0
    # ステップ個別のバックエンド(バリアント横断スイープ用)。None なら
    # ``SweepPlan.backend`` を使う=従来と完全に同じ argv。末尾の任意
    # フィールドなので既存の位置指定コンストラクタ呼び出しは無影響。
    backend: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "device_ids": self.selection.device_ids,
            "tensor_split": self.selection.tensor_split,
            "state": self.state.value,
            "bench_exit_code": self.bench_exit_code,
            "results_path": self.results_path,
            "summary": self.summary,
            "error": self.error,
            "backend": self.backend,
        }


@dataclass
class SweepPlan:
    steps: list[SweepStep]
    model_path: str
    backend: str  # プラン既定。step.backend が None のとき使われる
    port: int
    bench_cmd_template: str
    results_dir: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    extra_args: str = ""


def build_sweep_steps(
    labeled: Sequence[tuple[str, DeviceSelection]]
    | Sequence[tuple[str, DeviceSelection, str | None]],
) -> list[SweepStep]:
    """``(label, selection)`` または ``(label, selection, backend)`` から生成。

    3 要素形はバリアント横断スイープ用 (ステップごとに実行ファイルが違う)。
    2 要素形は従来どおり ``backend=None`` になり、プラン既定で動く。
    """
    steps: list[SweepStep] = []
    for item in labeled:
        lbl, sel = item[0], item[1]
        backend = item[2] if len(item) == 3 else None
        steps.append(SweepStep(label=lbl, selection=sel, backend=backend))
    return steps


# ---------------------------------------------------------------------------
# 2.5 ベンチコマンド展開
# ---------------------------------------------------------------------------

_BENCH_PLACEHOLDERS = ("port", "config", "base_url", "results_dir", "runs")


def render_bench_command(
    template: str,
    *,
    port: int,
    config_label: str,
    results_dir: str | None = None,
    runs: int | None = None,
) -> list[str]:
    """テンプレ内の {port} {config} {base_url} {results_dir} {runs} を置換して
    argv 化。

    ``str.format`` は JSON 波括弧等で誤爆するため単純置換を使う。

    H-2: 置換より **先に** ``shlex.split`` でトークン化し、各トークン内でのみ
    置換する。こうしないと ``{config}`` にスペースを含む値が入ったとき argv が
    増殖し(引数注入)、想定外のフラグを外部ベンチに渡せてしまう。
    """
    mapping = {
        "port": str(port),
        "config": config_label,
        "base_url": f"http://localhost:{port}/v1",
        "results_dir": results_dir or "",
        "runs": str(runs) if runs is not None else "",
    }
    # ★ Windows: shlex の POSIX モードはバックスラッシュをエスケープ文字
    #   として食い潰し `C:\tools\llmbench.exe` を壊す。os.name で分岐して
    #   Windows では posix=False(バックスラッシュを素通し)にする。
    argv: list[str] = []
    for tok in shlex.split(template, posix=(os.name != "nt")):
        for k, v in mapping.items():
            tok = tok.replace("{" + k + "}", v)
        if tok:
            argv.append(tok)
    return argv


# ---------------------------------------------------------------------------
# 2.6 results 解析
# ---------------------------------------------------------------------------


def load_latest_results(
    results_dir: str | Path,
    *,
    since: float,
) -> tuple[str | None, dict[str, Any] | None]:
    """``results_dir`` 内で mtime>=since の最新 ``*.json`` を読む。

    ``(path, summary)`` を返す。読めなければ ``(None, None)``。llmbench の
    スキーマに強く依存せず、既知キーを防御的に拾う。
    """
    base = Path(results_dir).expanduser()
    if not base.is_dir():
        return None, None
    cands = [p for p in base.glob("*.json") if p.stat().st_mtime >= since - 1.0]
    if not cands:
        return None, None
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return str(newest), None
    return str(newest), summarize_results(data)


# llmbench の JSON スキーマは変わりうるので、複数の別名を許容して拾う
_SUMMARY_KEYS = {
    "tokens_per_sec": ("tokens_per_sec", "tok_s", "throughput", "tps"),
    "ttft_ms": ("ttft_ms", "time_to_first_token_ms", "ttft"),
    "latency_ms": ("latency_ms", "avg_latency_ms", "latency"),
    "runs": ("runs", "n_runs", "count"),
}


def summarize_results(data: dict[str, Any]) -> dict[str, Any]:
    """llmbench results JSON から比較用の主要メトリクスを best-effort 抽出。"""
    flat = data.get("summary", data) if isinstance(data, dict) else {}
    out: dict[str, Any] = {}
    for canon, aliases in _SUMMARY_KEYS.items():
        for a in aliases:
            if isinstance(flat, dict) and a in flat:
                out[canon] = flat[a]
                break
    out["_raw_keys"] = sorted(flat.keys()) if isinstance(flat, dict) else []
    return out


# ---------------------------------------------------------------------------
# 2.7 ポート補助
# ---------------------------------------------------------------------------


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """best-effort な空きポート判定(TOCTOU 窓は残る)。"""
    with (
        contextlib.suppress(OSError),
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s,
    ):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    return False
