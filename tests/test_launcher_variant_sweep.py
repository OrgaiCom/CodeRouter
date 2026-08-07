"""ビルド横断ベンチスイープの回帰。

設計: docs/designs/launcher-multi-build.md §9 / §12.3。

目的は「同一モデルを CUDA ビルドと Vulkan ビルドで順に起動してベンチし、
どちらが速いか比較する」を 1 回のスイープで回せること。ステップごとに実行
ファイルが変わるので、``SweepStep.backend`` が正しく spawn に届くことと、
未指定なら従来どおりプラン既定で動くことを固定する。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    LauncherBackendConfig,
    LauncherBenchConfig,
    LauncherBenchPreset,
    LauncherConfig,
    ProviderConfig,
)
from coderouter.ingress.app import create_app
from coderouter.launcher_devices import (
    DeviceSelection,
    LlamaDevice,
    build_auto_sweep_configs,
    build_cross_variant_sweep_configs,
    build_sweep_steps,
)
from coderouter.metrics import uninstall_collector

CUDA_BIN = "/opt/llama.cpp/build-cuda/bin/llama-server"
VULKAN_BIN = "/opt/llama.cpp/build-vulkan/bin/llama-server"
PLAIN_BIN = "/opt/llama.cpp/build/bin/llama-server"

_CUDA = [
    LlamaDevice(id="CUDA0", name="RTX 5090", total_mib=32149, free_mib=31642),
    LlamaDevice(id="CUDA1", name="RTX 3090", total_mib=24123, free_mib=23845),
]
_VULKAN = [
    LlamaDevice(id="Vulkan0", name="RTX 3090", total_mib=24822, free_mib=24332),
    LlamaDevice(id="Vulkan2", name="Radeon 8060S", total_mib=114164, free_mib=113600),
]


# ---------------------------------------------------------------------------
# build_cross_variant_sweep_configs (純ロジック)
# ---------------------------------------------------------------------------


def test_cross_variant_configs_prefix_labels_with_variant() -> None:
    got = build_cross_variant_sweep_configs(
        [("llama.cpp-cuda", _CUDA), ("llama.cpp-vulkan", _VULKAN)]
    )
    labels = [label for label, _b, _s in got]
    assert labels[0].startswith("cuda / ")
    assert any(x.startswith("vulkan / ") for x in labels)
    # 各構成に対応するバックエンドが付く
    backends = {b for _l, b, _s in got}
    assert backends == {"llama.cpp-cuda", "llama.cpp-vulkan"}


def test_cross_variant_configs_match_per_build_auto_configs() -> None:
    """各ビルド分は既存 build_auto_sweep_configs と同じ構成になる。"""
    cross = build_cross_variant_sweep_configs([("llama.cpp-cuda", _CUDA)])
    solo = build_auto_sweep_configs(_CUDA)
    assert len(cross) == len(solo)
    for (clabel, backend, csel), (slabel, ssel) in zip(cross, solo, strict=True):
        assert backend == "llama.cpp-cuda"
        assert clabel == f"cuda / {slabel}"
        assert csel.device_ids == ssel.device_ids
        assert csel.tensor_split == ssel.tensor_split


def test_cross_variant_configs_never_mix_devices_across_builds() -> None:
    """ビルド跨ぎの混成構成は作らない(1 プロセス = 1 実行ファイル)。"""
    got = build_cross_variant_sweep_configs(
        [("llama.cpp-cuda", _CUDA), ("llama.cpp-vulkan", _VULKAN)]
    )
    for _label, _backend, sel in got:
        prefixes = {d.rstrip("0123456789") for d in sel.device_ids}
        assert len(prefixes) <= 1, sel.device_ids


def test_cross_variant_configs_uses_base_name_when_not_a_variant() -> None:
    got = build_cross_variant_sweep_configs([("llama.cpp", _CUDA)])
    assert got[0][0].startswith("llama.cpp / ")


# ---------------------------------------------------------------------------
# build_sweep_steps — 2 要素形 (従来) と 3 要素形 (横断)
# ---------------------------------------------------------------------------


def test_build_sweep_steps_two_tuple_keeps_backend_none() -> None:
    steps = build_sweep_steps([("CUDA0 単体", DeviceSelection(device_ids=["CUDA0"]))])
    assert steps[0].backend is None
    assert steps[0].as_dict()["backend"] is None


def test_build_sweep_steps_three_tuple_carries_backend() -> None:
    steps = build_sweep_steps(
        [("cuda / CUDA0 単体", DeviceSelection(device_ids=["CUDA0"]), "llama.cpp-cuda")]
    )
    assert steps[0].backend == "llama.cpp-cuda"
    assert steps[0].as_dict()["backend"] == "llama.cpp-cuda"


# ---------------------------------------------------------------------------
# POST /api/launcher/sweep/start — ステップごとに実行ファイルが変わる
# ---------------------------------------------------------------------------


@pytest.fixture
def sweep_client(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    config = CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(name="local", base_url="http://localhost:8080/v1",
                           model="m", paid=False)
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            model_dirs=[str(tmp_path)],
            backends={
                "llama.cpp": LauncherBackendConfig(binary=PLAIN_BIN),
                "llama.cpp-cuda": LauncherBackendConfig(binary=CUDA_BIN),
                "llama.cpp-vulkan": LauncherBackendConfig(binary=VULKAN_BIN),
            },
            bench=LauncherBenchConfig(
                runs=1,
                readiness_timeout_s=5.0,
                presets={"noop": LauncherBenchPreset(name="noop", command_template="true")},
                default_preset="noop",
            ),
        ),
    )
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
def model_file(tmp_path: Any) -> str:
    p = tmp_path / "a.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


def _sweep_body(model_file: str, configs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "backend": "llama.cpp",
        "model_path": model_file,
        "port": 18190,
        "configs": configs,
        "bench_preset": "noop",
        "runs": 1,
    }


def test_sweep_rejects_undeclared_variant_in_a_config(
    sweep_client: TestClient, model_file: str
) -> None:
    r = sweep_client.post(
        "/api/launcher/sweep/start",
        json=_sweep_body(
            model_file,
            [{"label": "x", "device_ids": ["ROCm0"], "backend": "llama.cpp-rocm"}],
        ),
    )
    assert r.status_code == 400
    assert "not declared in launcher.backends" in r.json()["detail"]


def test_sweep_accepts_cross_variant_configs_and_reports_backend(
    sweep_client: TestClient, model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """横断構成が受理され、status に各ステップの backend が出る。"""
    # スイープの実行自体は走らせない(spawn を即失敗させて次構成へ進めさせる)。
    async def boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("spawn disabled in test")

    monkeypatch.setattr("coderouter.ingress.launcher_routes.spawn_process", boom)
    r = sweep_client.post(
        "/api/launcher/sweep/start",
        json=_sweep_body(
            model_file,
            [
                {"label": "cuda / CUDA0 単体", "device_ids": ["CUDA0"],
                 "backend": "llama.cpp-cuda"},
                {"label": "vulkan / Vulkan2 単体", "device_ids": ["Vulkan2"],
                 "backend": "llama.cpp-vulkan"},
            ],
        ),
    )
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert [s["backend"] for s in steps] == ["llama.cpp-cuda", "llama.cpp-vulkan"]
    assert [s["label"] for s in steps] == ["cuda / CUDA0 単体", "vulkan / Vulkan2 単体"]


def test_sweep_config_without_backend_stays_on_plan_default(
    sweep_client: TestClient, model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """backend 未指定の構成は従来どおりプラン既定で動く(後方互換)。"""
    async def boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("spawn disabled in test")

    monkeypatch.setattr("coderouter.ingress.launcher_routes.spawn_process", boom)
    r = sweep_client.post(
        "/api/launcher/sweep/start",
        json=_sweep_body(model_file, [{"label": "CUDA0 単体", "device_ids": ["CUDA0"]}]),
    )
    assert r.status_code == 200, r.text
    assert r.json()["steps"][0]["backend"] is None


def test_sweep_step_spawns_with_the_configs_binary(
    sweep_client: TestClient, model_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """横断スイープの各ステップがそのビルドの実行ファイルで spawn される。"""
    seen: list[tuple[str, str]] = []

    async def capture(app: Any, launcher_cfg: Any, **kw: Any) -> Any:
        seen.append((kw["name"], kw["backend"]))
        raise RuntimeError("stop after recording")

    monkeypatch.setattr("coderouter.ingress.launcher_routes.spawn_process", capture)
    r = sweep_client.post(
        "/api/launcher/sweep/start",
        json=_sweep_body(
            model_file,
            [
                {"label": "cuda / CUDA0 単体", "device_ids": ["CUDA0"],
                 "backend": "llama.cpp-cuda"},
                {"label": "vulkan / Vulkan2 単体", "device_ids": ["Vulkan2"],
                 "backend": "llama.cpp-vulkan"},
            ],
        ),
    )
    assert r.status_code == 200, r.text

    # ランナーはバックグラウンドタスク。status が running でなくなるまで待つ。
    for _ in range(200):
        st = sweep_client.get("/api/launcher/sweep/status").json()
        if not st.get("running"):
            break
    backends = [b for _n, b in seen]
    assert backends == ["llama.cpp-cuda", "llama.cpp-vulkan"], seen


def test_sweep_labels_reach_bench_config_placeholder() -> None:
    """ラベルにビルド名が入るので {config} 経由で結果を見分けられる。"""
    from coderouter.launcher_devices import render_bench_command

    argv = render_bench_command(
        "bench --tag {config}", port=18190, config_label="cuda / CUDA0 単体"
    )
    assert "cuda" in " ".join(argv)
    assert not re.search(r"\{config\}", " ".join(argv))
