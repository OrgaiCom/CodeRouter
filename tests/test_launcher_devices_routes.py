"""Web (FastAPI) tests for device selection + bench sweep (設計 §4 / §5.4).

サブプロセスは全てモック(実 llama-server / llmbench は起動しない)。
``test_fix_h8_launcher_auth.py`` の ``config`` / ``client`` フィクスチャの
流儀を踏襲する。

カバレッジ:

* ``GET /api/launcher/devices`` — ``detect_llama_devices`` を monkeypatch
  (サンプル DeviceProbe)→ devices + suggested_tensor_split。``ok=false`` 経路。
* ``POST /api/launcher/start`` の ``device_ids`` / ``tensor_split`` →
  ``spawn_process`` を monkeypatch し ``device_args`` を検証。省略時 None
  (後方互換)。vllm では device_args 無視。
* スイープのライフサイクル(開始→状態遷移 DONE / 中断 ABORTED)、二重 start
  409、ポート衝突 400。全て spawn/stop/create_subprocess_exec をモック。
* 認証(H8 パターン): token 設定時 sweep start/abort は X-CodeRouter-Token
  必須、読み取り GET は開放。
* ``_build_cmd`` の device_args 挿入位置とモデル再指定拒否の維持。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    LauncherBenchConfig,
    LauncherConfig,
    ProviderConfig,
)
from coderouter.ingress import launcher_routes
from coderouter.ingress.app import create_app
from coderouter.ingress.launcher_routes import (
    ManagedProcess,
    _build_cmd,
    _registry_for_app,
)
from coderouter.launcher_devices import DeviceProbe, LlamaDevice
from coderouter.metrics import uninstall_collector

# pyproject.toml sets asyncio_mode = "auto" (pytest-asyncio).


# ---------------------------------------------------------------------------
# Fixtures (H8 パターン踏襲)
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local",
                base_url="http://localhost:8080/v1",
                model="qwen-coder",
                paid=False,
            ),
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            model_dirs=["/tmp"],
            # スイープの readiness を短く(abort テストの上限保険)。
            bench=LauncherBenchConfig(runs=2, readiness_timeout_s=5.0),
        ),
    )


@pytest.fixture
def client(
    config: CodeRouterConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    monkeypatch.delenv("CODEROUTER_LAUNCHER_TOKEN", raising=False)
    uninstall_collector()
    app = create_app()
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        uninstall_collector()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


_SAMPLE_DEVICES = [
    LlamaDevice(id="CUDA0", name="NVIDIA GeForce RTX 5090", total_mib=32149, free_mib=31626),
    LlamaDevice(id="CUDA1", name="NVIDIA GeForce RTX 3090", total_mib=24123, free_mib=24000),
]


@pytest.fixture(autouse=True)
def _stub_device_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--list-devices`` を常に ``_SAMPLE_DEVICES`` に固定する。

    ``POST /api/launcher/start`` は device_ids を受け取ると、そのビルドに実在
    する id かをサーバ側で検証する (別ビルドの id を持ち越すと
    ``--device CUDA0`` が Vulkan ビルドに渡って起動失敗するため)。この検証は
    実バイナリの ``--list-devices`` を呼ぶので、スタブしないと **テストが
    実行ホストの GPU 構成に依存する** —— PATH に Vulkan ビルドの
    llama-server がある Linux 機では ``CUDA0`` が弾かれて 400 になり、
    llama-server が無い機では検証がスキップされて 200 になる、という具合に
    結果が変わってしまう。

    個別に ``detect_llama_devices`` を monkeypatch しているテストは、この
    autouse フィクスチャの後に自分で setattr するのでそちらが優先される。
    """
    monkeypatch.setattr(
        launcher_routes,
        "detect_llama_devices",
        lambda binary, **kw: DeviceProbe(list(_SAMPLE_DEVICES), ok=True),
    )


class _AsyncLines:
    """bench 子プロセスの stdout を模した非同期イテレータ。"""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __aiter__(self) -> _AsyncLines:
        self._it = iter(self._lines)
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeBenchProc:
    def __init__(self, returncode: int = 0, lines: list[bytes] | None = None) -> None:
        self.returncode = returncode
        self.stdout = _AsyncLines(lines or [b"[bench] ok"])

    async def wait(self) -> int:
        return self.returncode


class _FakeBackend:
    """spawn_process / stop_process のスタブ(実プロセス無し)。"""

    def __init__(self, behavior: str = "ok") -> None:
        self.behavior = behavior  # "ok" | "hang"
        self.spawn_calls: list[dict[str, Any]] = []
        self.stop_calls: list[str] = []

    async def spawn_process(
        self,
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
        self.spawn_calls.append(
            {"name": name, "port": port, "device_args": device_args}
        )
        proc = ManagedProcess(
            id=f"proc-{len(self.spawn_calls)}",
            name=name,
            backend=backend,
            model_path=model_path,
            port=port,
            options=options or {},
            extra_args=extra_args,
            status="loading",
        )
        _registry_for_app(app).add(proc)
        if self.behavior == "ok":
            proc.status = "running"
            proc.ready.set()
        # "hang": status="loading", ready 未 set(readiness を止める)
        return proc

    async def stop_process(self, app: Any, proc_id: str) -> ManagedProcess:
        self.stop_calls.append(proc_id)
        proc = _registry_for_app(app).get(proc_id)
        proc.stopping = True
        proc.status = "stopped"
        return proc


def _install_backend(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend
) -> None:
    monkeypatch.setattr(launcher_routes, "spawn_process", backend.spawn_process)
    monkeypatch.setattr(launcher_routes, "stop_process", backend.stop_process)

    async def _fake_exec(*args: Any, **kwargs: Any) -> _FakeBenchProc:
        return _FakeBenchProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)


def _wait_sweep(client: TestClient, *, until_done: bool, tries: int = 100) -> dict:
    """/sweep/status を繰り返し叩いて背景タスクを進める。"""
    last: dict = {}
    for _ in range(tries):
        last = client.get("/api/launcher/sweep/status").json()
        if until_done and not last["running"] and last["steps"]:
            return last
        time.sleep(0.02)
    return last


# ---------------------------------------------------------------------------
# GET /api/launcher/devices
# ---------------------------------------------------------------------------


def _prefix(device_id: str) -> str:
    """末尾数字を除いたバックエンド接頭辞(テスト内の跨バックエンド検査用)。"""
    import re

    return re.sub(r"\d+$", "", device_id)


def test_devices_ok_with_suggested_split(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_detect(binary: str, **kwargs: Any) -> DeviceProbe:
        return DeviceProbe(list(_SAMPLE_DEVICES), ok=True)

    monkeypatch.setattr(launcher_routes, "detect_llama_devices", fake_detect)
    d = client.get("/api/launcher/devices").json()
    assert d["ok"] is True
    assert [dev["id"] for dev in d["devices"]] == ["CUDA0", "CUDA1"]
    # バックエンド別提案: 両方 CUDA。32149/(32149+24123)=0.57, 末尾で辻褄 → 0.43
    assert d["suggested_tensor_split"] == {"CUDA": [0.57, 0.43]}
    # auto_configs: 各単体 + CUDA 2 枚 split
    labels = [c["label"] for c in d["auto_configs"]]
    assert "CUDA0 単体" in labels and "CUDA1 単体" in labels
    multi = [c for c in d["auto_configs"] if len(c["device_ids"]) > 1]
    assert len(multi) == 1
    assert multi[0]["device_ids"] == ["CUDA0", "CUDA1"]
    assert multi[0]["tensor_split"] == [0.57, 0.43]


def test_devices_failure_falls_back(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_detect(binary: str, **kwargs: Any) -> DeviceProbe:
        return DeviceProbe([], ok=False, error="バイナリが見つかりません")

    monkeypatch.setattr(launcher_routes, "detect_llama_devices", fake_detect)
    d = client.get("/api/launcher/devices").json()
    assert d["ok"] is False
    assert d["devices"] == []
    assert d["suggested_tensor_split"] == {}
    assert d["auto_configs"] == []
    assert "見つかりません" in d["error"]


def test_devices_mac_metal_with_blas_zero_mib(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mac: MTL0(Apple M3 Max)+ BLAS: Accelerate (0 MiB)。

    BLAS は selectable から除外され、auto_configs は MTL0 単体のみ、
    バックエンド別提案は空(2 枚以上のバックエンドが無い)。
    """
    mac_devices = [
        LlamaDevice(id="MTL0", name="Apple M3 Max", total_mib=49152, free_mib=49152),
        LlamaDevice(id="BLAS", name="Accelerate", total_mib=0, free_mib=0),
    ]

    def fake_detect(binary: str, **kwargs: Any) -> DeviceProbe:
        return DeviceProbe(list(mac_devices), ok=True)

    monkeypatch.setattr(launcher_routes, "detect_llama_devices", fake_detect)
    d = client.get("/api/launcher/devices").json()
    assert d["ok"] is True
    # 表示用 devices には BLAS も残る
    assert [dev["id"] for dev in d["devices"]] == ["MTL0", "BLAS"]
    # 提案なし(単一 selectable)
    assert d["suggested_tensor_split"] == {}
    # auto_configs は MTL0 単体のみ(BLAS は入らない)
    assert d["auto_configs"] == [
        {"label": "MTL0 単体", "device_ids": ["MTL0"], "tensor_split": []}
    ]


def test_devices_cuda_vulkan_mixed_no_cross_backend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CUDA+Vulkan 同時ビルド: 同一物理 GPU が両バックエンドで重複列挙される。

    auto_configs にバックエンド跨ぎ混成は無く、CUDA 2 枚 / Vulkan 3 枚に
    それぞれ split が付与され、提案はバックエンド別に分かれる。
    """
    mixed = [
        LlamaDevice(id="CUDA0", name="NVIDIA RTX 5090", total_mib=32149, free_mib=31000),
        LlamaDevice(id="CUDA1", name="NVIDIA RTX 3090", total_mib=24123, free_mib=24000),
        LlamaDevice(id="Vulkan0", name="NVIDIA RTX 5090", total_mib=32149, free_mib=31000),
        LlamaDevice(id="Vulkan1", name="NVIDIA RTX 3090", total_mib=24123, free_mib=24000),
        LlamaDevice(id="Vulkan2", name="AMD Radeon iGPU", total_mib=114000, free_mib=113000),
    ]

    def fake_detect(binary: str, **kwargs: Any) -> DeviceProbe:
        return DeviceProbe(list(mixed), ok=True)

    monkeypatch.setattr(launcher_routes, "detect_llama_devices", fake_detect)
    d = client.get("/api/launcher/devices").json()

    # バックエンド別提案: CUDA(2 枚)と Vulkan(3 枚)がキーで分かれる
    split = d["suggested_tensor_split"]
    assert set(split.keys()) == {"CUDA", "Vulkan"}
    assert len(split["CUDA"]) == 2
    assert abs(sum(split["CUDA"]) - 1.0) < 1e-9
    assert len(split["Vulkan"]) == 3
    assert abs(sum(split["Vulkan"]) - 1.0) < 1e-9

    # auto_configs: どの構成も単一バックエンド内(跨バックエンド混成なし)
    for c in d["auto_configs"]:
        prefixes = {_prefix(i) for i in c["device_ids"]}
        assert len(prefixes) == 1, c
    # 複数枚構成は CUDA x2 と Vulkan x3 の 2 つだけ
    multi = [c for c in d["auto_configs"] if len(c["device_ids"]) > 1]
    assert {c["label"] for c in multi} == {"CUDA x2", "Vulkan x3"}
    cuda_multi = next(c for c in multi if c["label"] == "CUDA x2")
    assert cuda_multi["device_ids"] == ["CUDA0", "CUDA1"]
    assert len(cuda_multi["tensor_split"]) == 2


def test_devices_refresh_bypasses_cache(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def fake_detect(binary: str, *, use_cache: bool = True, **kw: Any) -> DeviceProbe:
        seen.append(use_cache)
        return DeviceProbe(list(_SAMPLE_DEVICES), ok=True)

    monkeypatch.setattr(launcher_routes, "detect_llama_devices", fake_detect)
    client.get("/api/launcher/devices")
    client.get("/api/launcher/devices?refresh=1")
    assert seen == [True, False]


# ---------------------------------------------------------------------------
# POST /api/launcher/start — device_args
# ---------------------------------------------------------------------------


def test_start_passes_device_args(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    _install_backend(monkeypatch, backend)
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "x",
            "backend": "llama.cpp",
            "model_path": "/tmp/m.gguf",
            "port": 8080,
            "device_ids": ["CUDA0", "CUDA1"],
            "tensor_split": [0.57, 0.43],
        },
    )
    assert resp.status_code == 200, resp.text
    assert backend.spawn_calls[0]["device_args"] == [
        "--device", "CUDA0,CUDA1", "--tensor-split", "0.57,0.43",
    ]


def test_start_without_devices_passes_none(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """デバイス未指定 → device_args=None(argv 後方互換の核心)。"""
    backend = _FakeBackend()
    _install_backend(monkeypatch, backend)
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "x",
            "backend": "llama.cpp",
            "model_path": "/tmp/m.gguf",
            "port": 8080,
        },
    )
    assert resp.status_code == 200, resp.text
    assert backend.spawn_calls[0]["device_args"] is None


def test_start_vllm_ignores_device_ids(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """device 非対応バックエンドでは device_args を組み立てない。"""
    backend = _FakeBackend()
    _install_backend(monkeypatch, backend)
    resp = client.post(
        "/api/launcher/start",
        json={
            "name": "x",
            "backend": "vllm",
            "model_path": "/tmp/m.safetensors",
            "port": 8000,
            "device_ids": ["CUDA0"],
        },
    )
    assert resp.status_code == 200, resp.text
    assert backend.spawn_calls[0]["device_args"] is None


# ---------------------------------------------------------------------------
# Sweep lifecycle
# ---------------------------------------------------------------------------


def _sweep_body(port: int = 18099) -> dict[str, Any]:
    return {
        "backend": "llama.cpp",
        "model_path": "/tmp/m.gguf",
        "port": port,
        "configs": [
            {"label": "CUDA0 単体", "device_ids": ["CUDA0"]},
            {"label": "CUDA1 単体", "device_ids": ["CUDA1"]},
        ],
    }


def test_sweep_runs_to_done(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(behavior="ok")
    _install_backend(monkeypatch, backend)

    start = client.post("/api/launcher/sweep/start", json=_sweep_body())
    assert start.status_code == 200, start.text
    assert start.json()["sweep_id"]
    assert len(start.json()["steps"]) == 2

    final = _wait_sweep(client, until_done=True)
    assert final["running"] is False
    states = [s["state"] for s in final["steps"]]
    assert states == ["done", "done"]
    # 各構成で spawn→stop(ポート解放)が呼ばれる
    assert len(backend.spawn_calls) == 2
    assert len(backend.stop_calls) == 2
    # llama.cpp 単体は --device のみ(tensor-split 無し)
    assert backend.spawn_calls[0]["device_args"] == ["--device", "CUDA0"]


def test_sweep_double_start_conflicts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(behavior="hang")  # 進行中のまま維持
    _install_backend(monkeypatch, backend)

    first = client.post("/api/launcher/sweep/start", json=_sweep_body())
    assert first.status_code == 200, first.text
    second = client.post("/api/launcher/sweep/start", json=_sweep_body(port=18100))
    assert second.status_code == 409, second.text

    # 後片付け: 中断して背景タスクを終わらせる。
    client.post("/api/launcher/sweep/abort")
    _wait_sweep(client, until_done=False, tries=30)


def test_sweep_abort_marks_aborted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(behavior="hang")  # readiness を止める
    _install_backend(monkeypatch, backend)

    start = client.post("/api/launcher/sweep/start", json=_sweep_body())
    assert start.status_code == 200, start.text
    # 数回ポーリングして最初の構成が STARTING に入るのを確認
    _wait_sweep(client, until_done=False, tries=5)

    ab = client.post("/api/launcher/sweep/abort")
    assert ab.status_code == 200, ab.text
    assert ab.json() == {"aborted": True}

    final = _wait_sweep(client, until_done=True)
    assert final["running"] is False
    states = [s["state"] for s in final["steps"]]
    assert states[0] == "aborted"
    assert all(s == "aborted" for s in states)


def test_sweep_port_conflict_returns_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    _install_backend(monkeypatch, backend)
    # レジストリに同ポートの稼働中プロセスを置く → 衝突 400。
    reg = _registry_for_app(client.app)
    reg.add(
        ManagedProcess(
            id="busy",
            name="busy",
            backend="llama.cpp",
            model_path="/tmp/other.gguf",
            port=18099,
            options={},
            extra_args="",
            status="running",
        )
    )
    resp = client.post("/api/launcher/sweep/start", json=_sweep_body(port=18099))
    assert resp.status_code == 400, resp.text
    assert "in use" in resp.json()["detail"]


def test_sweep_status_empty_when_none(client: TestClient) -> None:
    d = client.get("/api/launcher/sweep/status").json()
    assert d == {
        "sweep_id": None,
        "running": False,
        "current_index": -1,
        "steps": [],
    }


def test_sweep_empty_configs_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend()
    _install_backend(monkeypatch, backend)
    body = _sweep_body()
    body["configs"] = []
    resp = client.post("/api/launcher/sweep/start", json=body)
    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# Auth (H8 パターン) — sweep start/abort は token 必須、GET は開放
# ---------------------------------------------------------------------------


def test_sweep_start_requires_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    resp = client.post("/api/launcher/sweep/start", json=_sweep_body())
    assert resp.status_code == 401, resp.text


def test_sweep_abort_requires_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    resp = client.post("/api/launcher/sweep/abort")
    assert resp.status_code == 401, resp.text


def test_sweep_status_open_without_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    resp = client.get("/api/launcher/sweep/status")
    assert resp.status_code == 200, resp.text


def test_devices_open_without_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")

    def fake_detect(binary: str, **kwargs: Any) -> DeviceProbe:
        return DeviceProbe(list(_SAMPLE_DEVICES), ok=True)

    monkeypatch.setattr(launcher_routes, "detect_llama_devices", fake_detect)
    resp = client.get("/api/launcher/devices")
    assert resp.status_code == 200, resp.text


def test_sweep_start_with_correct_token_ok(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEROUTER_LAUNCHER_TOKEN", "s3cret")
    backend = _FakeBackend()
    _install_backend(monkeypatch, backend)
    resp = client.post(
        "/api/launcher/sweep/start",
        json=_sweep_body(),
        headers={"X-CodeRouter-Token": "s3cret"},
    )
    assert resp.status_code == 200, resp.text
    _wait_sweep(client, until_done=True)


# ---------------------------------------------------------------------------
# _build_cmd — device_args 挿入位置 & モデル再指定拒否の維持
# ---------------------------------------------------------------------------


def test_build_cmd_inserts_device_args_after_port() -> None:
    cmd = _build_cmd(
        "llama.cpp",
        "/models/good.gguf",
        8080,
        {},
        "",
        device_args=["--device", "CUDA0,CUDA1", "--tensor-split", "0.57,0.43"],
    )
    assert cmd[:5] == [cmd[0], "-m", "/models/good.gguf", "--port", "8080"]
    assert cmd[5:] == ["--device", "CUDA0,CUDA1", "--tensor-split", "0.57,0.43"]


def test_build_cmd_no_device_args_is_backward_compatible() -> None:
    """device_args 省略/None → 従来の argv と完全一致。"""
    base = _build_cmd("llama.cpp", "/models/good.gguf", 8080, {"--threads": 8}, "-ngl 99")
    with_none = _build_cmd(
        "llama.cpp", "/models/good.gguf", 8080, {"--threads": 8}, "-ngl 99",
        device_args=None,
    )
    assert base == with_none
    assert "--device" not in base


def test_build_cmd_vllm_ignores_device_args() -> None:
    cmd = _build_cmd(
        "vllm", "/models/m.safetensors", 8000, {}, "",
        device_args=["--device", "CUDA0"],
    )
    assert "--device" not in cmd


def test_build_cmd_model_override_still_rejected_with_device_args() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _build_cmd(
            "llama.cpp",
            "/models/good.gguf",
            8080,
            {},
            "-m /models/evil.gguf",
            device_args=["--device", "CUDA0"],
        )
