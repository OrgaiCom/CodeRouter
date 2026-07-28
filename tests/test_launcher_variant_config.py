"""``launcher.backends`` のバリアント設定の検証と、その効果の回帰。

設計: docs/designs/launcher-multi-build.md §5 (スキーマ) / §6 (一覧の config
由来化) / §10 (option_profiles マージ) / §12.2。

要点:

* バリアントキー (``llama.cpp-cuda``) は ``binary`` 必須。省略を許すと既定名
  フォールバックで PATH の ``llama-server`` が使われ、CUDA ビルドを指定した
  つもりで素のビルドが静かに動く。
* ``backends`` の不正キーはロード時エラー (**破壊的変更** — 従来 typo は黙って
  無視されていた)。
* バリアントを書かない config は API 応答も argv も従来と完全一致。
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from pydantic import ValidationError

from coderouter.config.schemas import (
    CodeRouterConfig,
    LauncherBackendConfig,
    LauncherConfig,
    LauncherOptionProfile,
    LauncherSwapConfig,
    SwapModelSpec,
)
from coderouter.ingress.launcher_routes import _backend_names, _resolve_backends_sync
from coderouter.launcher_devices import resolve_option_profiles

# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------

CUDA_BIN = "/opt/llama.cpp/build-cuda/bin/llama-server"
VULKAN_BIN = "/opt/llama.cpp/build-vulkan/bin/llama-server"
ROCM_BIN = "/opt/llama.cpp/build-rocm/bin/llama-server"
PLAIN_BIN = "/opt/llama.cpp/build/bin/llama-server"


def _launcher(**kw: Any) -> LauncherConfig:
    return LauncherConfig(**kw)


def _variant_backends() -> dict[str, LauncherBackendConfig]:
    """実機構成 (NucBox EVO-X2) 相当の 4 ビルド。"""
    return {
        "llama.cpp": LauncherBackendConfig(binary=PLAIN_BIN),
        "llama.cpp-cuda": LauncherBackendConfig(binary=CUDA_BIN),
        "llama.cpp-vulkan": LauncherBackendConfig(binary=VULKAN_BIN),
        "llama.cpp-rocm": LauncherBackendConfig(binary=ROCM_BIN),
    }


# ---------------------------------------------------------------------------
# §5.2 キー形式と binary 必須
# ---------------------------------------------------------------------------


def test_real_world_variant_config_loads() -> None:
    cfg = _launcher(model_dirs=["/models"], backends=_variant_backends())
    assert set(cfg.backends) == {
        "llama.cpp", "llama.cpp-cuda", "llama.cpp-vulkan", "llama.cpp-rocm",
    }
    assert cfg.backends["llama.cpp-cuda"].binary == CUDA_BIN


def test_variant_requires_binary() -> None:
    """``llama.cpp-cuda: {}`` は素のビルドが静かに動く事故になるので弾く。"""
    with pytest.raises(ValidationError, match="'binary' is required"):
        _launcher(backends={"llama.cpp-cuda": LauncherBackendConfig()})
    with pytest.raises(ValidationError, match="'binary' is required"):
        _launcher(backends={"llama.cpp-cuda": LauncherBackendConfig(binary=None)})


def test_base_backend_binary_stays_optional() -> None:
    """基底名は従来どおり binary 省略可 (PATH 解決)。"""
    cfg = _launcher(backends={"llama.cpp": LauncherBackendConfig()})
    assert cfg.backends["llama.cpp"].binary is None


@pytest.mark.parametrize(
    "bad_key",
    [
        "llamacpp",          # typo — 従来は黙って無視されていた
        "llama.cpp-",        # バリアント部が空
        "llama.cpp-CUDA",    # 大文字
        "ollama",
        "llama.cpp-/etc",    # パス区切り
    ],
)
def test_invalid_backend_keys_fail_fast(bad_key: str) -> None:
    with pytest.raises(ValidationError, match="invalid backend key"):
        _launcher(backends={bad_key: LauncherBackendConfig(binary="/x")})


@pytest.mark.parametrize("base", ["llama.cpp", "vllm", "mlx"])
def test_base_keys_accepted(base: str) -> None:
    _launcher(backends={base: LauncherBackendConfig(binary="/x")})


def test_empty_backends_still_valid() -> None:
    """backends を書かない既存 config はそのまま通る(後方互換)。"""
    assert _launcher().backends == {}


# ---------------------------------------------------------------------------
# §5.3 SwapModelSpec.backend の緩和
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["llama.cpp", "vllm", "mlx"])
def test_swap_spec_accepts_legacy_base_backends(backend: str) -> None:
    """Literal から str に緩めても既存の 3 値はそのまま通る。"""
    spec = SwapModelSpec(name="m", backend=backend, model_path="/models/a.gguf")
    assert spec.backend == backend


def test_swap_spec_accepts_variant() -> None:
    spec = SwapModelSpec(
        name="m", backend="llama.cpp-cuda", model_path="/models/a.gguf"
    )
    assert spec.backend == "llama.cpp-cuda"


@pytest.mark.parametrize("bad", ["llamacpp", "llama.cpp-", "ollama", "llama.cpp-CUDA"])
def test_swap_spec_rejects_invalid_backend(bad: str) -> None:
    with pytest.raises(ValidationError, match="is not valid"):
        SwapModelSpec(name="m", backend=bad, model_path="/models/a.gguf")


def test_swap_variant_must_be_declared_in_backends(tmp_path: Any) -> None:
    """バリアントは実行ファイルパスが backends にしか無いので宣言必須。"""
    model = tmp_path / "a.gguf"
    model.write_bytes(b"x")
    with pytest.raises(ValidationError, match=re.escape("not declared in launcher.backends")):
        _launcher(
            model_dirs=[str(tmp_path)],
            backends={"llama.cpp": LauncherBackendConfig(binary=PLAIN_BIN)},
            swap=LauncherSwapConfig(
                enabled=True,
                models=[
                    SwapModelSpec(
                        name="m", backend="llama.cpp-cuda", model_path=str(model)
                    )
                ],
            ),
        )


def test_swap_variant_declared_is_accepted(tmp_path: Any) -> None:
    model = tmp_path / "a.gguf"
    model.write_bytes(b"x")
    cfg = _launcher(
        model_dirs=[str(tmp_path)],
        backends=_variant_backends(),
        swap=LauncherSwapConfig(
            enabled=True,
            models=[
                SwapModelSpec(
                    name="m", backend="llama.cpp-cuda", model_path=str(model)
                )
            ],
        ),
    )
    assert cfg.swap is not None
    assert cfg.swap.models[0].backend == "llama.cpp-cuda"


def test_swap_base_backend_needs_no_declaration(tmp_path: Any) -> None:
    """基底名は PATH 解決で動くので backends 宣言は不要(従来どおり)。"""
    model = tmp_path / "a.gguf"
    model.write_bytes(b"x")
    _launcher(
        model_dirs=[str(tmp_path)],
        swap=LauncherSwapConfig(
            enabled=True,
            models=[
                SwapModelSpec(name="m", backend="llama.cpp", model_path=str(model))
            ],
        ),
    )


def test_swap_option_profile_can_be_inherited_from_base(tmp_path: Any) -> None:
    """バリアントの swap エントリが基底名のプロファイルを参照できる。"""
    model = tmp_path / "a.gguf"
    model.write_bytes(b"x")
    cfg = _launcher(
        model_dirs=[str(tmp_path)],
        backends=_variant_backends(),
        option_profiles={
            "llama.cpp": [LauncherOptionProfile(name="標準", args={"-ngl": 99})]
        },
        swap=LauncherSwapConfig(
            enabled=True,
            models=[
                SwapModelSpec(
                    name="m", backend="llama.cpp-cuda",
                    model_path=str(model), option_profile="標準",
                )
            ],
        ),
    )
    assert cfg.swap is not None


def test_swap_option_profile_still_validated(tmp_path: Any) -> None:
    model = tmp_path / "a.gguf"
    model.write_bytes(b"x")
    with pytest.raises(ValidationError, match="option_profile"):
        _launcher(
            model_dirs=[str(tmp_path)],
            backends=_variant_backends(),
            option_profiles={
                "llama.cpp": [LauncherOptionProfile(name="標準", args={})]
            },
            swap=LauncherSwapConfig(
                enabled=True,
                models=[
                    SwapModelSpec(
                        name="m", backend="llama.cpp-cuda",
                        model_path=str(model), option_profile="存在しない",
                    )
                ],
            ),
        )


# ---------------------------------------------------------------------------
# §6 バックエンド一覧の config 由来化
# ---------------------------------------------------------------------------


def test_backend_names_without_variants_is_unchanged() -> None:
    """バリアントを書かない利用者の選択肢は従来と完全に同一。"""
    assert _backend_names(None) == ["llama.cpp", "vllm", "mlx"]
    assert _backend_names({}) == ["llama.cpp", "vllm", "mlx"]
    assert _backend_names(
        {"llama.cpp": LauncherBackendConfig(binary=PLAIN_BIN)}
    ) == ["llama.cpp", "vllm", "mlx"]


def test_backend_names_appends_variants_in_declaration_order() -> None:
    names = _backend_names(_variant_backends())
    assert names[:3] == ["llama.cpp", "vllm", "mlx"]     # 基底は常に先頭
    assert names[3:] == ["llama.cpp-cuda", "llama.cpp-vulkan", "llama.cpp-rocm"]


def test_resolve_backends_sync_includes_variants_with_base_and_variant_keys() -> None:
    out = _resolve_backends_sync(_variant_backends())
    assert set(out) == {
        "llama.cpp", "vllm", "mlx",
        "llama.cpp-cuda", "llama.cpp-vulkan", "llama.cpp-rocm",
    }
    cuda = out["llama.cpp-cuda"]
    assert cuda["resolved"] == CUDA_BIN
    assert cuda["base"] == "llama.cpp"
    assert cuda["variant"] == "cuda"
    assert cuda["is_custom"] is True
    assert cuda["default"] == "llama-server"   # 基底名の既定を引く


def test_resolve_backends_sync_keeps_legacy_keys_for_base_backends() -> None:
    """既存キーの構造は不変 (追加のみ) —— 既存クライアントを壊さない。"""
    out = _resolve_backends_sync(None)
    assert list(out) == ["llama.cpp", "vllm", "mlx"]
    for name, info in out.items():
        assert set(info) == {
            "resolved", "configured", "default", "is_custom", "found",
            "base", "variant",
        }
        assert info["base"] == name
        assert info["variant"] is None


# ---------------------------------------------------------------------------
# §10 option_profiles のマージ
# ---------------------------------------------------------------------------


def _p(name: str, **args: Any) -> LauncherOptionProfile:
    return LauncherOptionProfile(name=name, args=args)


def test_profiles_for_base_backend_are_unchanged() -> None:
    profiles = {"llama.cpp": [_p("標準"), _p("速度")]}
    assert resolve_option_profiles(profiles, "llama.cpp") == profiles["llama.cpp"]


def test_variant_inherits_base_profiles() -> None:
    profiles = {"llama.cpp": [_p("標準"), _p("速度")]}
    assert resolve_option_profiles(profiles, "llama.cpp-cuda") == profiles["llama.cpp"]


def test_variant_profiles_are_appended_after_inherited() -> None:
    profiles = {
        "llama.cpp": [_p("標準"), _p("速度")],
        "llama.cpp-cuda": [_p("5090単体")],
    }
    got = [p.name for p in resolve_option_profiles(profiles, "llama.cpp-cuda")]
    assert got == ["標準", "速度", "5090単体"]


def test_variant_profile_replaces_same_name_in_place() -> None:
    """同名は継承分と**同じ位置**で差し替え(末尾に重複を作らない)。"""
    profiles = {
        "llama.cpp": [_p("標準", **{"-ngl": 99}), _p("速度"), _p("省メモリ")],
        "llama.cpp-cuda": [_p("標準", **{"-ngl": 999}), _p("cuda専用")],
    }
    merged = resolve_option_profiles(profiles, "llama.cpp-cuda")
    assert [p.name for p in merged] == ["標準", "速度", "省メモリ", "cuda専用"]
    assert merged[0].args == {"-ngl": 999}   # バリアント側が勝つ


def test_variant_with_no_base_profiles_gets_only_its_own() -> None:
    profiles = {"llama.cpp-cuda": [_p("cuda専用")]}
    assert [p.name for p in resolve_option_profiles(profiles, "llama.cpp-cuda")] == [
        "cuda専用"
    ]


def test_unrelated_backend_profiles_are_not_mixed_in() -> None:
    profiles = {
        "llama.cpp": [_p("llama標準")],
        "vllm": [_p("vllm標準")],
        "llama.cpp-cuda": [_p("cuda")],
    }
    names = [p.name for p in resolve_option_profiles(profiles, "llama.cpp-cuda")]
    assert names == ["llama標準", "cuda"]
    assert "vllm標準" not in names


# ---------------------------------------------------------------------------
# providers.yaml 全体としてのロード (§5.4 の例が通ること)
# ---------------------------------------------------------------------------


def test_full_providers_yaml_shape_loads(tmp_path: Any) -> None:
    model = tmp_path / "Qwen3-30B-A3B-Q4_K_M.gguf"
    model.write_bytes(b"x")
    cfg = CodeRouterConfig.model_validate(
        {
            "providers": [
                {
                    "name": "p",
                    "kind": "openai_compat",
                    "base_url": "http://localhost:1/v1",
                    "model": "",
                }
            ],
            "profiles": [{"name": "default", "providers": ["p"]}],
            "launcher": {
                "model_dirs": [str(tmp_path)],
                "backends": {
                    "llama.cpp": {"binary": PLAIN_BIN},
                    "llama.cpp-cuda": {"binary": CUDA_BIN},
                    "llama.cpp-vulkan": {"binary": VULKAN_BIN},
                    "llama.cpp-rocm": {"binary": ROCM_BIN},
                },
                "option_profiles": {
                    "llama.cpp": [{"name": "標準", "args": {"-ngl": 99}}],
                    "llama.cpp-cuda": [{"name": "5090単体", "args": {"-ngl": 99}}],
                },
                "swap": {
                    "enabled": True,
                    "models": [
                        {
                            "name": "qwen3-30b",
                            "backend": "llama.cpp-cuda",
                            "model_path": str(model),
                            "port": 18081,
                        }
                    ],
                },
            },
        }
    )
    assert cfg.launcher is not None
    assert cfg.launcher.swap is not None
    assert cfg.launcher.swap.models[0].backend == "llama.cpp-cuda"
    merged = resolve_option_profiles(cfg.launcher.option_profiles, "llama.cpp-cuda")
    assert [p.name for p in merged] == ["標準", "5090単体"]
