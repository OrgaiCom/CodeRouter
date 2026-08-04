"""バックエンド名バリアント (``llama.cpp-cuda`` 等) の正規化と分岐の回帰。

設計: docs/designs/launcher-multi-build.md §3 (正規化) / §4 (分岐一覧) /
§12.1・§12.3。

llama.cpp を CUDA / Vulkan / ROCm 向けに個別ビルドしている環境で、
``launcher.backends`` に ``llama.cpp-cuda`` のようなバリアント名を書けるように
した。バックエンド名は実行ファイルの選択だけでなく**挙動の分岐**にも使われて
いるため、バリアント名を取りこぼした分岐は例外を出さずに壊れる。本ファイルは
設計 §4.1 が列挙した fail-open 分岐に 1:1 で対応する回帰テストを置く:

1. ``_MODEL_FLAGS`` — 取りこぼすと H8 モデル上書きガードが無効化 (セキュリティ)
2. ``_backend_ready`` — 取りこぼすと readiness が TCP connect に退行 (正しさ)
3. ``resolve_speculative`` — 取りこぼすと MTP フラグが黙って消える
4. device gating — 取りこぼすと ``--device`` が argv から黙って落ちる
5. ``_suggest_launch_flags`` — else 構造で意図せず正しく動く箇所の固定
"""

from __future__ import annotations

import re

import pytest

from coderouter.ingress.launcher_routes import (
    _assert_no_model_override,
    _backend_ready,
    _build_cmd,
    _resolve_binary,
    _suggest_launch_flags,
)
from coderouter.launcher_devices import (
    KNOWN_BASE_BACKENDS,
    DeviceProbe,
    LlamaDevice,
    base_backend,
    foreign_device_ids,
    is_valid_backend_name,
    is_variant,
    variant_of,
)
from coderouter.launcher_speculative import resolve_speculative

# 実機 (NucBox EVO-X2) で使うバリアント。設計 §1.2 の表と対応。
LLAMA_VARIANTS = ["llama.cpp-cuda", "llama.cpp-vulkan", "llama.cpp-rocm"]


# ---------------------------------------------------------------------------
# §12.1 正規化ヘルパ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", KNOWN_BASE_BACKENDS)
def test_base_backend_passes_through_base_names(base: str) -> None:
    assert base_backend(base) == base
    assert variant_of(base) is None
    assert is_variant(base) is False
    assert is_valid_backend_name(base) is True


@pytest.mark.parametrize(
    ("name", "base", "variant"),
    [
        ("llama.cpp-cuda", "llama.cpp", "cuda"),
        ("llama.cpp-vulkan", "llama.cpp", "vulkan"),
        ("llama.cpp-rocm", "llama.cpp", "rocm"),
        ("llama.cpp-cuda12.4", "llama.cpp", "cuda12.4"),
        ("llama.cpp-metal-debug", "llama.cpp", "metal-debug"),
        ("vllm-rocm", "vllm", "rocm"),
        ("mlx-dev", "mlx", "dev"),
    ],
)
def test_variant_split(name: str, base: str, variant: str) -> None:
    assert base_backend(name) == base
    assert variant_of(name) == variant
    assert is_variant(name) is True
    assert is_valid_backend_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "llamacpp",          # typo — 既知基底名で始まらない
        "llama",             # 部分文字列だが基底名ではない
        "ollama",
        "",
    ],
)
def test_unknown_names_pass_through_unchanged(name: str) -> None:
    """未知名は加工せず返し、呼び出し側の Unknown backend 経路に委ねる。"""
    assert base_backend(name) == name
    assert variant_of(name) is None
    assert is_valid_backend_name(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "llama.cpp-",        # バリアント部が空
        "llama.cpp-CUDA",    # 大文字は許さない
        "llama.cpp--cuda",   # 先頭がハイフン
        "llama.cpp-/etc",    # パス区切り
        "llama.cpp-a b",     # 空白
        "llama.cpp-$(x)",    # シェルメタ文字
    ],
)
def test_invalid_variant_names_rejected(name: str) -> None:
    """設定キーの検証で弾く形。パス区切りやシェルメタ文字を通さない。"""
    assert is_valid_backend_name(name) is False


def test_base_backend_uses_longest_match_not_naive_split() -> None:
    """``split("-", 1)`` 実装では通らないケースを固定する。

    ``base_backend`` は既知基底名との最長一致で判定する契約。素の
    ``name.split("-", 1)[0]`` だと ``"mlx-dev"`` は偶然通るが、基底名自体が
    ハイフンを含むようになった日に静かに誤判定する。ここでは「ハイフンの前が
    既知基底名でなければバリアントとして扱わない」ことを固定して、実装が
    素の分割に退行しないようにする。
    """
    # 素の split("-", 1)[0] なら "not" が返るが、"not" は既知基底名ではない
    # ので加工せず全体を返すのが正しい。
    assert base_backend("not-a-backend") == "not-a-backend"
    assert variant_of("not-a-backend") is None


# ---------------------------------------------------------------------------
# §4.1-1 _MODEL_FLAGS — H8 モデル上書きガード (セキュリティ・最優先)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["llama.cpp", *LLAMA_VARIANTS])
@pytest.mark.parametrize(
    "token",
    ["-m", "--model", "-md", "--model-draft", "--spec-draft-model", "--model=/etc/x"],
)
def test_model_override_guard_active_for_every_variant(
    backend: str, token: str
) -> None:
    """バリアントでもガードが効く。

    ``_MODEL_FLAGS.get(backend, frozenset())`` のままバリアント名が来ると空集合
    が返り、``options`` / ``extra_args`` 経由で ``-m`` を渡して ``model_path``
    を差し替えられてしまう。ここが本機能で最も重要な回帰テスト。
    """
    with pytest.raises(ValueError, match="not allowed"):
        _assert_no_model_override(backend, [token])


def test_model_override_guard_is_fail_closed_for_unknown_backend() -> None:
    """未知バックエンドではガードを緩めず、全 banned 集合の和で弾く。"""
    for token in ["-m", "--model", "-md", "--model-draft", "--spec-draft-model"]:
        with pytest.raises(ValueError, match="not allowed"):
            _assert_no_model_override("totally-unknown", [token])


@pytest.mark.parametrize("backend", ["llama.cpp", *LLAMA_VARIANTS])
def test_model_override_guard_allows_ordinary_flags(backend: str) -> None:
    """正当なフラグは通す(ガードが過剰に効いていないことの確認)。"""
    _assert_no_model_override(backend, ["-ngl", "99", "--ctx-size", "4096"])


@pytest.mark.parametrize("backend", LLAMA_VARIANTS)
def test_build_cmd_rejects_model_override_via_options_and_extra(backend: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _build_cmd(backend, "/m/a.gguf", 18081, {"-m": "/etc/passwd"}, "",
                   binary="/opt/llama-server")
    with pytest.raises(ValueError, match="not allowed"):
        _build_cmd(backend, "/m/a.gguf", 18081, {}, "--model /etc/passwd",
                   binary="/opt/llama-server")


# ---------------------------------------------------------------------------
# §4.1-2 _backend_ready — readiness が /health のまま (正しさ)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["llama.cpp", "vllm", *LLAMA_VARIANTS])
async def test_readiness_uses_health_endpoint_for_variants(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """バリアントでも ``GET /health`` を叩く。

    ここを取りこぼすと素の TCP connect に退行し、モデルのロード完了前に
    provider が登録される —— readiness ゲーティングが直したバグへの逆戻り。
    TCP 経路に落ちたら ``asyncio.open_connection`` が呼ばれるので、それを
    失敗させて「/health が呼ばれたか」だけで判定する。
    """
    called: list[str] = []

    class _Resp:
        status_code = 200

    class _Client:
        def __init__(self, **_kw: object) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def get(self, url: str) -> _Resp:
            called.append(url)
            return _Resp()

    monkeypatch.setattr("coderouter.ingress.launcher_routes.httpx.AsyncClient", _Client)

    assert await _backend_ready(backend, 18081, probe_timeout_s=0.5) is True
    # 127.0.0.1 の literal であること (localhost ではない)。localhost が ::1 に
    # 先に解決される環境 (GitHub の macOS ランナー、Mac 一般) では、IPv4 のみで
    # listen する llama-server に届かず readiness が永久に失敗する。2026-08-04 の
    # macOS CI で実際に踏んだ回帰。
    assert called == ["http://127.0.0.1:18081/health"]


async def test_readiness_still_falls_back_to_tcp_for_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mlx は従来どおり TCP connect (既存挙動の維持)。"""
    called: list[str] = []

    class _Client:
        def __init__(self, **_kw: object) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def get(self, url: str) -> object:  # pragma: no cover - must not run
            called.append(url)
            raise AssertionError("mlx must not use the /health probe")

    monkeypatch.setattr("coderouter.ingress.launcher_routes.httpx.AsyncClient", _Client)
    # 誰も listen していないポート → TCP connect 失敗 → False
    assert await _backend_ready("mlx", 1, probe_timeout_s=0.2) is False
    assert called == []


# ---------------------------------------------------------------------------
# §4.1-3 resolve_speculative — MTP / spec フラグが消えない
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", LLAMA_VARIANTS)
def test_speculative_accepts_variants(backend: str, tmp_path: object) -> None:
    """バリアントでも draft_model_path を受け付ける(拒否されない)。

    ``backend != "llama.cpp"`` のままだと「llama.cpp 以外」と誤判定して
    ValueError を投げるか、spec トークンを黙って落とす。
    """
    draft = f"{tmp_path}/draft.gguf"
    open(draft, "wb").close()
    tokens, _notes = resolve_speculative(
        backend=backend,
        model_path=f"{tmp_path}/main.gguf",
        draft_model_path=draft,
        mtp_mode="auto",
        user_tokens=[],
    )
    assert "--model-draft" in tokens


@pytest.mark.parametrize("backend", ["vllm", "mlx", "vllm-rocm"])
def test_speculative_still_rejects_non_llama_backends(
    backend: str, tmp_path: object
) -> None:
    """llama.cpp 以外(そのバリアント含む)は従来どおり拒否。"""
    draft = f"{tmp_path}/draft.gguf"
    open(draft, "wb").close()
    with pytest.raises(ValueError, match=re.escape("only supported for llama.cpp")):
        resolve_speculative(
            backend=backend,
            model_path=f"{tmp_path}/main.gguf",
            draft_model_path=draft,
            mtp_mode="auto",
            user_tokens=[],
        )


# ---------------------------------------------------------------------------
# §4.1-4 device_args — --device が argv に載る
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["llama.cpp", *LLAMA_VARIANTS])
def test_device_args_land_in_argv_for_variants(backend: str) -> None:
    cmd = _build_cmd(
        backend, "/m/a.gguf", 18081, {}, "",
        binary="/opt/llama-server",
        device_args=["--device", "CUDA0,CUDA1", "--tensor-split", "0.57,0.43"],
    )
    assert cmd[:5] == ["/opt/llama-server", "-m", "/m/a.gguf", "--port", "18081"]
    assert cmd[5:9] == ["--device", "CUDA0,CUDA1", "--tensor-split", "0.57,0.43"]


@pytest.mark.parametrize("backend", ["vllm", "mlx", "vllm-rocm"])
def test_device_args_never_apply_to_non_llama_backends(backend: str) -> None:
    cmd = _build_cmd(
        backend, "/m/model", 18081, {}, "",
        binary="/opt/python", device_args=["--device", "CUDA0"],
    )
    assert "--device" not in cmd


@pytest.mark.parametrize("backend", ["llama.cpp", *LLAMA_VARIANTS])
def test_no_device_selection_keeps_argv_byte_identical(backend: str) -> None:
    """未選択なら ``--device`` を 1 文字も足さない(後方互換の核心)。"""
    cmd = _build_cmd(backend, "/m/a.gguf", 18081, {}, "", binary="/opt/llama-server")
    assert cmd == ["/opt/llama-server", "-m", "/m/a.gguf", "--port", "18081"]


# ---------------------------------------------------------------------------
# §4.2 argv の形 / バイナリ解決
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", LLAMA_VARIANTS)
def test_variant_argv_shape_matches_base_backend(backend: str) -> None:
    """バリアントの argv は基底 llama.cpp と同形 (実行ファイルだけが違う)。"""
    base_cmd = _build_cmd(
        "llama.cpp", "/m/a.gguf", 18081, {"-ngl": 99}, "--no-mmap",
        binary="/opt/build/bin/llama-server",
    )
    var_cmd = _build_cmd(
        backend, "/m/a.gguf", 18081, {"-ngl": 99}, "--no-mmap",
        binary="/opt/build-x/bin/llama-server",
    )
    assert var_cmd[0] == "/opt/build-x/bin/llama-server"
    assert var_cmd[1:] == base_cmd[1:]


def test_unknown_backend_still_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        _build_cmd("llamacpp", "/m/a.gguf", 18081, {}, "", binary="/opt/x")


@pytest.mark.parametrize("backend", LLAMA_VARIANTS)
def test_resolve_binary_prefers_configured_path(backend: str) -> None:
    assert _resolve_binary(backend, "/opt/build-cuda/bin/llama-server") == (
        "/opt/build-cuda/bin/llama-server"
    )


@pytest.mark.parametrize("backend", LLAMA_VARIANTS)
def test_resolve_binary_falls_back_to_base_default_not_backend_name(
    backend: str,
) -> None:
    """設定漏れ時のフォールバックはバックエンド名ではなく基底名の既定。

    実運用ではバリアントは ``binary`` 必須 (config ロード時に検証) なので
    この経路には入らないが、入っても ``llama.cpp-cuda`` という名前を exec
    しようとしないことを固定する。
    """
    assert _resolve_binary(backend, None) == "llama-server"


# ---------------------------------------------------------------------------
# §4.3 意図せず正しく動く箇所の固定
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["llama.cpp", *LLAMA_VARIANTS])
def test_suggest_launch_flags_emits_ngl_for_variants(backend: str) -> None:
    hw = {"cpu_count": 16, "gpu": "cuda", "vram_gb": 32.0, "ram_gb": 64.0}
    flags = _suggest_launch_flags(backend, 8.0, hw)
    assert "-ngl" in flags


@pytest.mark.parametrize("backend", ["vllm", "mlx", "vllm-rocm", "mlx-dev"])
def test_suggest_launch_flags_stays_empty_for_non_llama(backend: str) -> None:
    hw = {"cpu_count": 16, "gpu": "cuda", "vram_gb": 32.0, "ram_gb": 64.0}
    assert _suggest_launch_flags(backend, 8.0, hw) == ""


# ---------------------------------------------------------------------------
# デバイス ID の名前空間チェック (§7.2)
# ---------------------------------------------------------------------------


def _probe(*ids: str) -> DeviceProbe:
    devs = [LlamaDevice(id=i, name=f"GPU {i}", total_mib=24000, free_mib=23000)
            for i in ids]
    return DeviceProbe(devs, ok=True)


def test_foreign_device_ids_detects_cross_variant_mismatch() -> None:
    """CUDA ビルドの ID を Vulkan ビルドで使おうとしたら検出する。"""
    vulkan = _probe("Vulkan0", "Vulkan1", "Vulkan2")
    assert foreign_device_ids(["CUDA0"], vulkan) == ["CUDA0"]
    assert foreign_device_ids(["Vulkan0", "CUDA1"], vulkan) == ["CUDA1"]


def test_foreign_device_ids_accepts_matching_ids() -> None:
    cuda = _probe("CUDA0", "CUDA1")
    assert foreign_device_ids(["CUDA0", "CUDA1"], cuda) == []
    assert foreign_device_ids([], cuda) == []


def test_foreign_device_ids_is_best_effort_when_probe_failed() -> None:
    """プローブ自体が失敗した環境では検証をスキップして機能を殺さない。"""
    failed = DeviceProbe([], ok=False, error="バイナリが見つかりません")
    assert foreign_device_ids(["CUDA0", "whatever"], failed) == []


def test_foreign_device_ids_is_best_effort_when_probe_is_empty() -> None:
    """ok=True でも 1 台も拾えなかったら判定しない(誤検知を避ける)。"""
    empty = DeviceProbe([], ok=True)
    assert foreign_device_ids(["CUDA0"], empty) == []


def test_foreign_device_ids_matches_on_namespace_not_exact_id() -> None:
    """判定はバックエンド接頭辞単位で、id の完全一致ではない。

    同一ビルド内の番号ズレ (``CUDA5``) は通す。``--list-devices`` の出力形式が
    変わって :func:`parse_list_devices` が一部の行を取りこぼしたときに、正しい
    id を誤って拒否しないための緩さ。捕まえたいのは「ビルド違い」であって
    番号違いではない。
    """
    cuda = _probe("CUDA0", "CUDA1")
    assert foreign_device_ids(["CUDA5"], cuda) == []      # 同じ名前空間 → 通す
    assert foreign_device_ids(["Vulkan0"], cuda) == ["Vulkan0"]  # 別名前空間 → 弾く


def test_foreign_device_ids_handles_prefix_only_ids() -> None:
    """末尾数字を持たない id (BLAS / CPU / Metal) も接頭辞として扱える。"""
    metal = _probe("MTL0", "BLAS")
    assert foreign_device_ids(["MTL0", "BLAS"], metal) == []
    assert foreign_device_ids(["CUDA0"], metal) == ["CUDA0"]
