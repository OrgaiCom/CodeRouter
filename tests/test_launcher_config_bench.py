"""Schemas tests for the bench-sweep config (設計 §4.1 / §5.5).

Covers:

* ``LauncherBenchConfig`` の既定値・``extra="forbid"``・範囲外 runs。
* ``LauncherConfig(bench=...)`` の round-trip と、``bench`` 省略時の完全
  後方互換(従来 YAML 相当の入力が無改変で通過し ``bench is None``)。
* ``StartRequest`` の ``device_ids`` / ``tensor_split`` 既定空
  (既存 Web クライアント不変)。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    LauncherBenchConfig,
    LauncherConfig,
    ProviderConfig,
)
from coderouter.ingress.launcher_routes import StartRequest

# ---------------------------------------------------------------------------
# LauncherBenchConfig
# ---------------------------------------------------------------------------


def test_bench_config_defaults() -> None:
    """未指定なら設計のハードコード既定が入る。"""
    bench = LauncherBenchConfig()
    assert bench.command_template == "llmbench run --model local-openai --runs {runs}"
    assert bench.runs == 5
    assert bench.results_dir is None
    assert bench.readiness_timeout_s == 300.0


def test_bench_config_custom_values() -> None:
    bench = LauncherBenchConfig(
        command_template="llmbench run --base-url {base_url} --runs {runs}",
        runs=20,
        results_dir="/data/results",
        readiness_timeout_s=600.0,
    )
    assert bench.runs == 20
    assert bench.results_dir == "/data/results"
    assert bench.readiness_timeout_s == 600.0


def test_bench_config_rejects_unknown_key() -> None:
    """extra="forbid" — 未知キーは ValidationError。"""
    with pytest.raises(ValidationError):
        LauncherBenchConfig(unknown_field="x")


@pytest.mark.parametrize("runs", [0, 1001, -3])
def test_bench_config_runs_out_of_range(runs: int) -> None:
    with pytest.raises(ValidationError):
        LauncherBenchConfig(runs=runs)


@pytest.mark.parametrize("timeout", [4.0, 3601.0])
def test_bench_config_readiness_timeout_out_of_range(timeout: float) -> None:
    with pytest.raises(ValidationError):
        LauncherBenchConfig(readiness_timeout_s=timeout)


# ---------------------------------------------------------------------------
# LauncherConfig.bench field
# ---------------------------------------------------------------------------


def test_launcher_config_bench_default_none() -> None:
    """bench 省略 → None(スイープ UI はハードコード既定を使う)。"""
    cfg = LauncherConfig(model_dirs=["/tmp/models"])
    assert cfg.bench is None


def test_launcher_config_bench_roundtrip() -> None:
    cfg = LauncherConfig(
        model_dirs=["/tmp/models"],
        bench=LauncherBenchConfig(runs=8, results_dir="results/"),
    )
    assert cfg.bench is not None
    assert cfg.bench.runs == 8
    assert cfg.bench.results_dir == "results/"


def test_launcher_config_bench_from_dict() -> None:
    """dict → model_validate 経由(YAML ロード相当)でも通る。"""
    cfg = LauncherConfig.model_validate(
        {
            "model_dirs": ["/tmp/models"],
            "bench": {"command_template": "mybench {port}", "runs": 3},
        }
    )
    assert cfg.bench is not None
    assert cfg.bench.command_template == "mybench {port}"
    assert cfg.bench.runs == 3


def test_legacy_launcher_yaml_unchanged_is_backward_compatible() -> None:
    """bench を持たない従来 YAML 相当は無改変で通過し bench is None。"""
    legacy = {
        "model_dirs": ["~/models"],
        "backends": {"llama.cpp": {"binary": "~/llama.cpp/build/bin/llama-server"}},
        "option_profiles": {
            "llama.cpp": [{"name": "GPU", "args": {"-ngl": 99}}],
        },
    }
    cfg = LauncherConfig.model_validate(legacy)
    assert cfg.bench is None
    assert cfg.model_dirs == ["~/models"]


def test_coderouter_config_with_bench_launcher() -> None:
    """トップレベル CodeRouterConfig にも問題なく載る。"""
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(name="local", base_url="http://localhost:8080/v1", model="m"),
        ],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            model_dirs=["/tmp"], bench=LauncherBenchConfig(runs=2)
        ),
    )
    assert cfg.launcher is not None
    assert cfg.launcher.bench is not None
    assert cfg.launcher.bench.runs == 2


# ---------------------------------------------------------------------------
# StartRequest device fields (routes 側の pydantic モデル)
# ---------------------------------------------------------------------------


def test_start_request_device_fields_default_empty() -> None:
    """device_ids / tensor_split 未指定 → 空リスト(既存クライアント不変)。"""
    req = StartRequest(
        name="x", backend="llama.cpp", model_path="/m.gguf", port=8080
    )
    assert req.device_ids == []
    assert req.tensor_split == []


def test_start_request_device_fields_populated() -> None:
    req = StartRequest(
        name="x",
        backend="llama.cpp",
        model_path="/m.gguf",
        port=8080,
        device_ids=["CUDA0", "CUDA1"],
        tensor_split=[0.57, 0.43],
    )
    assert req.device_ids == ["CUDA0", "CUDA1"]
    assert req.tensor_split == [0.57, 0.43]
