"""バリアントバックエンドの API 経路の回帰。

設計: docs/designs/launcher-multi-build.md §8 (API) / §7.2 (デバイス連動) /
§12.3。

* ``GET /api/launcher/backends`` にバリアントが出る / 出ない config では従来と
  完全に同一の応答
* ``POST /api/launcher/start`` がバリアントの実行ファイルで spawn する
* 未宣言のバリアント名は **400 でフォールバックしない**
* そのビルドに無いデバイス ID は 400 / プローブ失敗時は通す
* ``GET /api/launcher/option-profiles`` が基底名を継承する
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    LauncherBackendConfig,
    LauncherConfig,
    LauncherOptionProfile,
    ProviderConfig,
)
from coderouter.ingress.app import create_app
from coderouter.launcher_devices import DeviceProbe, LlamaDevice, reset_device_cache
from coderouter.metrics import uninstall_collector

CUDA_BIN = "/opt/llama.cpp/build-cuda/bin/llama-server"
VULKAN_BIN = "/opt/llama.cpp/build-vulkan/bin/llama-server"
PLAIN_BIN = "/opt/llama.cpp/build/bin/llama-server"

# 実機 (NucBox EVO-X2) の --list-devices 出力に対応するデバイス集合。
# CUDA ビルドと Vulkan ビルドで **同じ GPU が別の id で見える** ことが
# デバイス ID 検証の存在理由。
_CUDA_DEVICES = [
    LlamaDevice(id="CUDA0", name="NVIDIA GeForce RTX 5090",
                total_mib=32149, free_mib=31642),
    LlamaDevice(id="CUDA1", name="NVIDIA GeForce RTX 3090",
                total_mib=24123, free_mib=23845),
]
_VULKAN_DEVICES = [
    LlamaDevice(id="Vulkan0", name="NVIDIA GeForce RTX 3090",
                total_mib=24822, free_mib=24332),
    LlamaDevice(id="Vulkan1", name="NVIDIA GeForce RTX 5090",
                total_mib=32607, free_mib=32145),
    LlamaDevice(id="Vulkan2", name="Radeon 8060S Graphics (RADV GFX1151)",
                total_mib=114164, free_mib=113600),
]

_DEVICES_BY_BINARY: dict[str, list[LlamaDevice]] = {
    PLAIN_BIN: [],
    CUDA_BIN: _CUDA_DEVICES,
    VULKAN_BIN: _VULKAN_DEVICES,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(backends: dict[str, LauncherBackendConfig] | None,
                 option_profiles: dict[str, list[LauncherOptionProfile]] | None = None,
                 model_dirs: list[str] | None = None) -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(name="local", base_url="http://localhost:8080/v1",
                           model="qwen-coder", paid=False),
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            model_dirs=model_dirs or ["/tmp"],
            backends=backends or {},
            option_profiles=option_profiles or {},
        ),
    )


def _client(config: CodeRouterConfig,
            monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr("coderouter.ingress.app.load_config", lambda path=None: config)
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


@pytest.fixture
def variant_backends() -> dict[str, LauncherBackendConfig]:
    return {
        "llama.cpp": LauncherBackendConfig(binary=PLAIN_BIN),
        "llama.cpp-cuda": LauncherBackendConfig(binary=CUDA_BIN),
        "llama.cpp-vulkan": LauncherBackendConfig(binary=VULKAN_BIN),
    }


@pytest.fixture
def variant_client(
    variant_backends: dict[str, LauncherBackendConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    yield from _client(_make_config(variant_backends), monkeypatch)


@pytest.fixture
def plain_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """バリアントを一切書かない config —— 従来との一致を見るための対照。"""
    yield from _client(_make_config(None), monkeypatch)


@pytest.fixture(autouse=True)
def _fake_device_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--list-devices`` をバイナリごとに差し替える(実プロセスは起動しない)。"""
    reset_device_cache()

    def fake(binary: str, **_kw: Any) -> DeviceProbe:
        devices = _DEVICES_BY_BINARY.get(binary)
        if devices is None:
            return DeviceProbe([], ok=False, error=f"未知のバイナリ: {binary}")
        return DeviceProbe(list(devices), ok=True)

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes.detect_llama_devices", fake
    )
    yield
    reset_device_cache()


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """spawn される argv を記録して実際のプロセス起動は行わない。"""
    calls: list[list[str]] = []

    class _FakeProc:
        pid = 4242
        returncode = None
        stdout = None
        stderr = None

        async def wait(self) -> int:
            return 0

    async def fake_exec(*cmd: str, **_kw: Any) -> _FakeProc:
        calls.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes.asyncio.create_subprocess_exec", fake_exec
    )
    return calls


@pytest.fixture
def model_file(tmp_path: Any) -> str:
    p = tmp_path / "Qwen3-30B-A3B-Q4_K_M.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


# ---------------------------------------------------------------------------
# GET /api/launcher/backends
# ---------------------------------------------------------------------------


def test_backends_lists_variants(variant_client: TestClient) -> None:
    d = variant_client.get("/api/launcher/backends").json()["backends"]
    assert set(d) == {"llama.cpp", "vllm", "mlx", "llama.cpp-cuda", "llama.cpp-vulkan"}
    assert d["llama.cpp-cuda"]["resolved"] == CUDA_BIN
    assert d["llama.cpp-cuda"]["variant"] == "cuda"
    assert d["llama.cpp-cuda"]["base"] == "llama.cpp"
    assert d["llama.cpp"]["variant"] is None


def test_backends_without_variants_is_exactly_the_legacy_three(
    plain_client: TestClient,
) -> None:
    """バリアントを書かない利用者の応答は従来と同一の 3 キー(後方互換の核)。"""
    d = plain_client.get("/api/launcher/backends").json()["backends"]
    assert list(d) == ["llama.cpp", "vllm", "mlx"]
    for name, info in d.items():
        assert info["base"] == name
        assert info["variant"] is None


# ---------------------------------------------------------------------------
# POST /api/launcher/start — バリアントの実行ファイルで spawn する
# ---------------------------------------------------------------------------


def _start_body(model_file: str, **kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "t", "backend": "llama.cpp", "model_path": model_file,
        "port": 18099, "options": {}, "extra_args": "",
    }
    body.update(kw)
    return body


@pytest.fixture
def model_dirs_client(
    variant_backends: dict[str, LauncherBackendConfig],
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    yield from _client(
        _make_config(variant_backends, model_dirs=[str(tmp_path)]), monkeypatch
    )


def test_start_with_variant_uses_that_builds_binary(
    model_dirs_client: TestClient, spawned: list[list[str]], model_file: str
) -> None:
    r = model_dirs_client.post(
        "/api/launcher/start",
        json=_start_body(model_file, backend="llama.cpp-cuda"),
    )
    assert r.status_code == 200, r.text
    assert spawned, "no process was spawned"
    assert spawned[0][0] == CUDA_BIN
    assert spawned[0][1:5] == ["-m", model_file, "--port", "18099"]


def test_start_with_base_backend_is_unaffected(
    model_dirs_client: TestClient, spawned: list[list[str]], model_file: str
) -> None:
    r = model_dirs_client.post(
        "/api/launcher/start", json=_start_body(model_file, backend="llama.cpp")
    )
    assert r.status_code == 200, r.text
    assert spawned[0][0] == PLAIN_BIN


def test_start_rejects_undeclared_variant_without_falling_back(
    model_dirs_client: TestClient, spawned: list[list[str]], model_file: str
) -> None:
    """未宣言のバリアントは 400。PATH の llama-server に落ちてはいけない。"""
    r = model_dirs_client.post(
        "/api/launcher/start",
        json=_start_body(model_file, backend="llama.cpp-rocm"),
    )
    assert r.status_code == 400
    assert "not declared in launcher.backends" in r.json()["detail"]
    assert spawned == []


def test_start_still_rejects_garbage_backend(
    model_dirs_client: TestClient, model_file: str
) -> None:
    r = model_dirs_client.post(
        "/api/launcher/start", json=_start_body(model_file, backend="llamacpp")
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# §7.2 デバイス ID の名前空間チェック
# ---------------------------------------------------------------------------


def test_start_accepts_device_ids_that_exist_in_that_build(
    model_dirs_client: TestClient, spawned: list[list[str]], model_file: str
) -> None:
    r = model_dirs_client.post(
        "/api/launcher/start",
        json=_start_body(
            model_file, backend="llama.cpp-cuda",
            device_ids=["CUDA0", "CUDA1"], tensor_split=[0.57, 0.43],
        ),
    )
    assert r.status_code == 200, r.text
    argv = spawned[0]
    assert argv[0] == CUDA_BIN
    assert "--device" in argv
    assert argv[argv.index("--device") + 1] == "CUDA0,CUDA1"


def test_start_rejects_device_ids_from_a_different_build(
    model_dirs_client: TestClient, spawned: list[list[str]], model_file: str
) -> None:
    """CUDA ビルドの ID を Vulkan ビルドで使うと 400。

    ここを通すと ``--device CUDA0`` が Vulkan ビルドに渡って llama-server が
    起動失敗する。ビルド切替時にデバイス選択を持ち越す事故の防波堤。
    """
    r = model_dirs_client.post(
        "/api/launcher/start",
        json=_start_body(
            model_file, backend="llama.cpp-vulkan", device_ids=["CUDA0"]
        ),
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "CUDA0" in detail
    assert "build-specific" in detail
    assert spawned == []


def test_start_skips_device_validation_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch, spawned: list[list[str]], model_file: str,
    tmp_path: Any,
) -> None:
    """``--list-devices`` が失敗する環境では検証をスキップして通す。

    best-effort 原則: デバイスを列挙できないマシンで機能を殺さない。
    """
    def failing(binary: str, **_kw: Any) -> DeviceProbe:
        return DeviceProbe([], ok=False, error="バイナリが見つかりません")

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes.detect_llama_devices", failing
    )
    backends = {"llama.cpp-cuda": LauncherBackendConfig(binary=CUDA_BIN)}
    for tc in _client(
        _make_config(backends, model_dirs=[str(tmp_path)]), monkeypatch
    ):
        r = tc.post(
            "/api/launcher/start",
            json=_start_body(
                model_file, backend="llama.cpp-cuda", device_ids=["CUDA0"]
            ),
        )
        assert r.status_code == 200, r.text
        assert spawned[0][0] == CUDA_BIN


def test_start_without_device_ids_never_probes(
    model_dirs_client: TestClient, monkeypatch: pytest.MonkeyPatch,
    spawned: list[list[str]], model_file: str,
) -> None:
    """デバイス未選択なら検証プローブも走らない(既存経路を重くしない)。"""
    calls: list[str] = []

    def counting(binary: str, **_kw: Any) -> DeviceProbe:
        calls.append(binary)
        return DeviceProbe(list(_CUDA_DEVICES), ok=True)

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes.detect_llama_devices", counting
    )
    r = model_dirs_client.post(
        "/api/launcher/start", json=_start_body(model_file, backend="llama.cpp-cuda")
    )
    assert r.status_code == 200, r.text
    assert calls == []


# ---------------------------------------------------------------------------
# GET /api/launcher/devices — バリアントごとに違う一覧
# ---------------------------------------------------------------------------


def test_devices_differ_per_variant(variant_client: TestClient) -> None:
    cuda = variant_client.get(
        "/api/launcher/devices?backend=llama.cpp-cuda"
    ).json()
    vulkan = variant_client.get(
        "/api/launcher/devices?backend=llama.cpp-vulkan"
    ).json()
    assert [d["id"] for d in cuda["devices"]] == ["CUDA0", "CUDA1"]
    assert [d["id"] for d in vulkan["devices"]] == ["Vulkan0", "Vulkan1", "Vulkan2"]


def test_devices_suggests_tensor_split_per_variant(variant_client: TestClient) -> None:
    d = variant_client.get("/api/launcher/devices?backend=llama.cpp-cuda").json()
    assert "CUDA" in d["suggested_tensor_split"]


# ---------------------------------------------------------------------------
# GET /api/launcher/option-profiles — 基底名の継承
# ---------------------------------------------------------------------------


def test_option_profiles_inherit_base_for_variant(
    variant_backends: dict[str, LauncherBackendConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = {
        "llama.cpp": [LauncherOptionProfile(name="標準", args={"-ngl": 99})],
        "llama.cpp-cuda": [LauncherOptionProfile(name="5090単体", args={"-ngl": 99})],
    }
    for tc in _client(_make_config(variant_backends, profiles), monkeypatch):
        d = tc.get("/api/launcher/option-profiles").json()["profiles"]
        assert [p["name"] for p in d["llama.cpp"]] == ["標準"]
        assert [p["name"] for p in d["llama.cpp-cuda"]] == ["標準", "5090単体"]
        # 固有プロファイルを持たないバリアントも基底分を継承する
        assert [p["name"] for p in d["llama.cpp-vulkan"]] == ["標準"]


def test_option_profiles_payload_unchanged_without_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """バリアントなし config の応答は従来と同一(継承で余計なキーを作らない)。"""
    profiles = {
        "llama.cpp": [LauncherOptionProfile(name="標準", args={"-ngl": 99})],
    }
    for tc in _client(_make_config(None, profiles), monkeypatch):
        d = tc.get("/api/launcher/option-profiles").json()["profiles"]
        assert list(d) == ["llama.cpp"]
        assert [p["name"] for p in d["llama.cpp"]] == ["標準"]
