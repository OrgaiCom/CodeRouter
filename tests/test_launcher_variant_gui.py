"""GUI 版 (launcher_gui.py) のバックエンドバリアント対応。

設計: docs/designs/launcher-multi-build.md §7 (UI) / §12.4。

Web 版とバイナリ解決ロジックが別実装なので、同じ分岐が GUI 側でも正規化
されていることを独立に固定する。``tk.Tk()`` / ``LauncherApp`` は生成せず、
Tk 非依存のモジュールレベル関数だけを叩く (既存 GUI テストと同じ方針・
CI の uv Python に tkinter が無いため importorskip)。
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "tkinter", reason="launcher_gui requires the tkinter package (python3-tk)"
)

import launcher_gui as lg

CUDA_BIN = "/opt/llama.cpp/build-cuda/bin/llama-server"
VULKAN_BIN = "/opt/llama.cpp/build-vulkan/bin/llama-server"
PLAIN_BIN = "/opt/llama.cpp/build/bin/llama-server"

LLAMA_VARIANTS = ["llama.cpp-cuda", "llama.cpp-vulkan", "llama.cpp-rocm"]

_VARIANT_YAML = (
    "launcher:\n"
    "  model_dirs: []\n"
    "  backends:\n"
    "    llama.cpp:\n"
    f"      binary: {PLAIN_BIN}\n"
    "    llama.cpp-cuda:\n"
    f"      binary: {CUDA_BIN}\n"
    "    llama.cpp-vulkan:\n"
    f"      binary: {VULKAN_BIN}\n"
)


# ---------------------------------------------------------------------------
# 正規化ヘルパが GUI からも使える (standalone フォールバック含む)
# ---------------------------------------------------------------------------


def test_gui_exposes_normalization_helpers() -> None:
    assert lg.base_backend("llama.cpp-cuda") == "llama.cpp"
    assert lg.variant_of("llama.cpp-cuda") == "cuda"
    assert lg.variant_of("llama.cpp") is None
    assert lg.is_variant("llama.cpp-vulkan") is True
    assert lg.is_valid_backend_name("llama.cpp-rocm") is True
    assert lg.is_valid_backend_name("llamacpp") is False


# ---------------------------------------------------------------------------
# §4.1-2 readiness (GUI 側の _backend_ready)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["llama.cpp", "vllm", *LLAMA_VARIANTS])
def test_gui_readiness_uses_health_for_variants(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """バリアントでも ``GET /health``。TCP connect に退行してはいけない。"""
    called: list[str] = []

    class _Resp:
        status = 200

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def fake_urlopen(req: object, timeout: float = 0.0) -> _Resp:
        called.append(getattr(req, "full_url", str(req)))
        return _Resp()

    monkeypatch.setattr(lg.urllib.request, "urlopen", fake_urlopen)
    assert lg._backend_ready(backend, 18081, probe_timeout_s=0.5) is True
    assert called == ["http://localhost:18081/health"]


def test_gui_readiness_still_tcp_for_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_kw: object) -> None:  # pragma: no cover
        raise AssertionError("mlx must not use the /health probe")

    monkeypatch.setattr(lg.urllib.request, "urlopen", boom)
    # 誰も listen していないポート → TCP connect 失敗 → False
    assert lg._backend_ready("mlx", 1, probe_timeout_s=0.2) is False


# ---------------------------------------------------------------------------
# §4.1-4 / §4.2 argv の形とバイナリ解決
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", LLAMA_VARIANTS)
def test_gui_build_cmd_variant_argv_matches_base(backend: str) -> None:
    base = lg._build_cmd(
        "llama.cpp", "/m/a.gguf", 8080, {"-ngl": 99}, "--no-mmap", PLAIN_BIN, None, None
    )
    var = lg._build_cmd(
        backend, "/m/a.gguf", 8080, {"-ngl": 99}, "--no-mmap", CUDA_BIN, None, None
    )
    assert var[0] == CUDA_BIN
    assert var[1:] == base[1:]


@pytest.mark.parametrize("backend", LLAMA_VARIANTS)
def test_gui_build_cmd_accepts_device_args_for_variants(backend: str) -> None:
    cmd = lg._build_cmd(
        backend, "/m/a.gguf", 8080, {}, "", CUDA_BIN, None,
        ["--device", "CUDA0,CUDA1"],
    )
    assert cmd[5:7] == ["--device", "CUDA0,CUDA1"]


@pytest.mark.parametrize("backend", LLAMA_VARIANTS)
def test_gui_build_cmd_no_device_selection_is_byte_identical(backend: str) -> None:
    cmd = lg._build_cmd(backend, "/m/a.gguf", 8080, {}, "", CUDA_BIN, None, None)
    assert cmd == [CUDA_BIN, "-m", "/m/a.gguf", "--port", "8080"]


def test_gui_build_cmd_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        lg._build_cmd("llamacpp", "/m/a.gguf", 8080, {}, "", "/x", None, None)


# ---------------------------------------------------------------------------
# _load_config — backends にバリアントが読める
# ---------------------------------------------------------------------------


def test_gui_load_config_reads_variant_backends(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text(_VARIANT_YAML)
    cfg = lg._load_config(str(p))
    assert cfg.backends["llama.cpp-cuda"].binary == CUDA_BIN
    assert cfg.backends["llama.cpp-vulkan"].binary == VULKAN_BIN


def test_gui_resolve_binary_picks_variant_path(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text(_VARIANT_YAML)
    cfg = lg._load_config(str(p))
    assert lg._resolve_binary("llama.cpp", cfg) == PLAIN_BIN
    assert lg._resolve_binary("llama.cpp-cuda", cfg) == CUDA_BIN
    assert lg._resolve_binary("llama.cpp-vulkan", cfg) == VULKAN_BIN


def test_gui_resolve_binary_falls_back_to_base_default(tmp_path) -> None:
    """未宣言のバリアントは基底名の既定に落ちる(バックエンド名を exec しない)。"""
    p = tmp_path / "providers.yaml"
    p.write_text("launcher:\n  model_dirs: []\n")
    cfg = lg._load_config(str(p))
    assert lg._resolve_binary("llama.cpp-rocm", cfg) == "llama-server"


# ---------------------------------------------------------------------------
# §6 バックエンド一覧の config 由来化
# ---------------------------------------------------------------------------


def test_gui_backend_names_without_variants_is_unchanged(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text("launcher:\n  model_dirs: []\n")
    cfg = lg._load_config(str(p))
    assert lg._backend_names(cfg) == ["llama.cpp", "vllm", "mlx"]


def test_gui_backend_names_appends_variants(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text(_VARIANT_YAML)
    cfg = lg._load_config(str(p))
    names = lg._backend_names(cfg)
    assert names[:3] == ["llama.cpp", "vllm", "mlx"]
    assert names[3:] == ["llama.cpp-cuda", "llama.cpp-vulkan"]


# ---------------------------------------------------------------------------
# §10 option_profiles の基底名継承
# ---------------------------------------------------------------------------


def test_gui_profiles_inherit_base_and_append_variant(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text(
        "launcher:\n"
        "  model_dirs: []\n"
        "  backends:\n"
        "    llama.cpp-cuda:\n"
        f"      binary: {CUDA_BIN}\n"
        "  option_profiles:\n"
        "    llama.cpp:\n"
        "      - name: 標準\n"
        "        args: {'-ngl': 99}\n"
        "    llama.cpp-cuda:\n"
        "      - name: 5090単体\n"
        "        args: {'-ngl': 99}\n"
    )
    cfg = lg._load_config(str(p))
    assert [x.name for x in lg._profiles_for(cfg, "llama.cpp")] == ["標準"]
    assert [x.name for x in lg._profiles_for(cfg, "llama.cpp-cuda")] == [
        "標準", "5090単体"
    ]
    # 固有プロファイルの無いバリアントも基底分を継承する
    assert [x.name for x in lg._profiles_for(cfg, "llama.cpp-vulkan")] == ["標準"]


def test_gui_profiles_for_base_backend_unchanged(tmp_path) -> None:
    p = tmp_path / "providers.yaml"
    p.write_text(
        "launcher:\n"
        "  option_profiles:\n"
        "    llama.cpp:\n"
        "      - name: 標準\n"
        "        args: {'-ngl': 99}\n"
    )
    cfg = lg._load_config(str(p))
    got = lg._profiles_for(cfg, "llama.cpp")
    assert got == cfg.option_profiles["llama.cpp"]
