"""Unit tests for coderouter.launcher_devices — 純関数群(tkinter 不要)。

設計書 §5.1 準拠。サブプロセスは触らず、パース/提案/CLI 変換/ベンチ展開/
results 解析/ポート判定の純粋ロジックのみを検証する。クロスプラットフォーム
(CUDA/Metal/Vulkan/SYCL/VRAM 無し CPU 行)のサンプル出力を網羅する。
"""

from __future__ import annotations

import json
import os
import socket

import pytest

from coderouter.launcher_devices import (
    DeviceSelection,
    LlamaDevice,
    backend_of,
    build_auto_sweep_configs,
    group_by_backend,
    is_port_free,
    load_latest_results,
    parse_list_devices,
    render_bench_command,
    selectable_devices,
    suggest_tensor_split,
    summarize_results,
)

# ---------------------------------------------------------------------------
# サンプル --list-devices 出力(プラットフォーム別)
# ---------------------------------------------------------------------------

CUDA_DUAL_OUTPUT = """\
ggml_backend_cuda_init: found 2 CUDA devices:
Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31626 MiB free)
  CUDA1: NVIDIA GeForce RTX 3090 (24123 MiB, 23800 MiB free)
"""

METAL_OUTPUT = """\
Available devices:
  Metal: Apple M3 Max (49152 MiB, 49152 MiB free)
"""

VULKAN_OUTPUT = """\
  Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 24000 MiB free)
"""

SYCL_OUTPUT = """\
  SYCL0: Intel Arc A770 (16384 MiB, 16000 MiB free)
"""

# VRAM 表記の無い CPU 行 + 正常な CUDA 行の混在
CPU_MIXED_OUTPUT = """\
Available devices:
  CPU: AMD Ryzen 9 7950X 16-Core Processor
  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 24000 MiB free)
"""

# ── 実機出力(coordinator 提供) ──
# Linux, CUDA+Vulkan 同時ビルド。同一物理 GPU が CUDA と Vulkan で重複列挙され、
# description に入れ子括弧 "(RADV GFX1151)" を含む。
REAL_CUDA_VULKAN_OUTPUT = """\
Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31610 MiB free)
  CUDA1: NVIDIA GeForce RTX 3090 (24123 MiB, 23858 MiB free)
  Vulkan0: NVIDIA GeForce RTX 3090 (24822 MiB, 24096 MiB free)
  Vulkan1: NVIDIA GeForce RTX 5090 (32607 MiB, 31610 MiB free)
  Vulkan2: AMD Radeon Graphics (RADV GFX1151) (114166 MiB, 113602 MiB free)
"""

# macOS, M3 Max。id は MTL0(Metal ではない)。BLAS は 0 MiB=選択不可。
REAL_MAC_MTL_OUTPUT = """\
Available devices:
  MTL0: Apple M3 Max (53084 MiB, 53083 MiB free)
  BLAS: Accelerate (0 MiB, 0 MiB free)
"""


# ---------------------------------------------------------------------------
# parse_list_devices
# ---------------------------------------------------------------------------


def test_parse_cuda_dual() -> None:
    devs = parse_list_devices(CUDA_DUAL_OUTPUT)
    assert len(devs) == 2
    assert devs[0] == LlamaDevice(
        id="CUDA0", name="NVIDIA GeForce RTX 5090", total_mib=32149, free_mib=31626
    )
    assert devs[1].id == "CUDA1"
    assert devs[1].name == "NVIDIA GeForce RTX 3090"
    assert devs[1].total_mib == 24123
    assert devs[1].free_mib == 23800


def test_parse_ignores_header_lines() -> None:
    # "ggml_backend_cuda_init:" と "Available devices:" は device 行にならない。
    devs = parse_list_devices(CUDA_DUAL_OUTPUT)
    assert [d.id for d in devs] == ["CUDA0", "CUDA1"]


def test_parse_empty_string() -> None:
    assert parse_list_devices("") == []


def test_parse_whitespace_and_trailing_newline_tolerance() -> None:
    noisy = "\n\n" + CUDA_DUAL_OUTPUT + "\n   \n"
    devs = parse_list_devices(noisy)
    assert len(devs) == 2


def test_parse_mixed_broken_lines_keeps_valid() -> None:
    text = (
        "garbage line without colon\n"
        "  BROKEN: no memory info here\n"
        "  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31626 MiB free)\n"
        "random trailing junk\n"
    )
    devs = parse_list_devices(text)
    assert len(devs) == 1
    assert devs[0].id == "CUDA0"


def test_parse_metal_single() -> None:
    devs = parse_list_devices(METAL_OUTPUT)
    assert len(devs) == 1
    assert devs[0].id == "Metal"
    assert devs[0].name == "Apple M3 Max"
    assert devs[0].total_mib == 49152
    assert devs[0].free_mib == 49152


def test_parse_vulkan() -> None:
    devs = parse_list_devices(VULKAN_OUTPUT)
    assert len(devs) == 1
    assert devs[0].id == "Vulkan0"
    assert devs[0].name == "AMD Radeon RX 7900 XTX"
    assert devs[0].total_mib == 24560


def test_parse_sycl() -> None:
    devs = parse_list_devices(SYCL_OUTPUT)
    assert len(devs) == 1
    assert devs[0].id == "SYCL0"
    assert devs[0].name == "Intel Arc A770"
    assert devs[0].total_mib == 16384


def test_parse_cpu_line_without_vram_is_ignored() -> None:
    # VRAM 表記の無い CPU 行はマッチせず、CUDA 行だけ拾う。
    devs = parse_list_devices(CPU_MIXED_OUTPUT)
    assert [d.id for d in devs] == ["CUDA0"]
    assert all(d.id != "CPU" for d in devs)


def test_parse_real_cuda_vulkan_nested_parens() -> None:
    devs = parse_list_devices(REAL_CUDA_VULKAN_OUTPUT)
    assert [d.id for d in devs] == ["CUDA0", "CUDA1", "Vulkan0", "Vulkan1", "Vulkan2"]
    # 入れ子括弧の description が丸ごと保持され、メモリ括弧は行末側のみ採用。
    v2 = devs[-1]
    assert v2.id == "Vulkan2"
    assert v2.name == "AMD Radeon Graphics (RADV GFX1151)"
    assert v2.total_mib == 114166
    assert v2.free_mib == 113602
    assert devs[0].name == "NVIDIA GeForce RTX 5090"
    assert devs[0].total_mib == 32149


def test_parse_real_mac_mtl_and_blas() -> None:
    devs = parse_list_devices(REAL_MAC_MTL_OUTPUT)
    assert [d.id for d in devs] == ["MTL0", "BLAS"]
    # 実機の Metal id は "MTL0"(旧コメントの "Metal" ではない)。
    assert devs[0].name == "Apple M3 Max"
    assert devs[0].total_mib == 53084
    # BLAS は 0 MiB でも情報としてパースされる(除外は selectable_devices の責務)。
    assert devs[1].id == "BLAS"
    assert devs[1].total_mib == 0
    assert devs[1].free_mib == 0


def test_llamadevice_gb_properties() -> None:
    d = LlamaDevice(id="CUDA0", name="x", total_mib=32149, free_mib=31626)
    assert d.total_gb == round(32149 / 1024.0, 1)
    assert d.free_gb == round(31626 / 1024.0, 1)
    dct = d.as_dict()
    assert dct["id"] == "CUDA0"
    assert dct["total_mib"] == 32149
    assert "total_gb" in dct and "free_gb" in dct


# ---------------------------------------------------------------------------
# suggest_tensor_split
# ---------------------------------------------------------------------------


def _dev(id_: str, total: int, free: int) -> LlamaDevice:
    return LlamaDevice(id=id_, name=id_, total_mib=total, free_mib=free)


def test_suggest_split_5090_3090() -> None:
    devs = [_dev("CUDA0", 32149, 31626), _dev("CUDA1", 24123, 23800)]
    split = suggest_tensor_split(devs, by="total")
    assert split == [0.57, 0.43]
    assert abs(sum(split) - 1.0) < 1e-9


def test_suggest_split_single_device_is_empty() -> None:
    assert suggest_tensor_split([_dev("Metal", 49152, 49152)]) == []


def test_suggest_split_empty_list_is_empty() -> None:
    assert suggest_tensor_split([]) == []


def test_suggest_split_by_free_differs_from_total() -> None:
    devs = [_dev("CUDA0", 20000, 10000), _dev("CUDA1", 10000, 10000)]
    by_total = suggest_tensor_split(devs, by="total")
    by_free = suggest_tensor_split(devs, by="free")
    assert by_total == [0.67, 0.33]
    assert by_free == [0.5, 0.5]


def test_suggest_split_three_devices_sums_to_one() -> None:
    devs = [
        _dev("CUDA0", 10000, 10000),
        _dev("CUDA1", 10000, 10000),
        _dev("CUDA2", 10000, 10000),
    ]
    split = suggest_tensor_split(devs, by="total")
    assert len(split) == 3
    assert abs(sum(split) - 1.0) < 1e-9
    # 丸め誤差が末尾に集約される。
    assert split[-1] == round(1.0 - sum(split[:-1]), 2)


def test_suggest_split_zero_capacity_is_empty() -> None:
    devs = [_dev("CPU0", 0, 0), _dev("CPU1", 0, 0)]
    assert suggest_tensor_split(devs, by="total") == []


# ---------------------------------------------------------------------------
# selectable_devices / backend_of / group_by_backend
# ---------------------------------------------------------------------------


def test_selectable_devices_filters_zero_vram() -> None:
    devs = parse_list_devices(REAL_MAC_MTL_OUTPUT)
    sel = selectable_devices(devs)
    # BLAS (0 MiB) は除外され、MTL0 のみ残る。一覧(devs)からは落ちていない。
    assert [d.id for d in sel] == ["MTL0"]
    assert [d.id for d in devs] == ["MTL0", "BLAS"]


def test_backend_of_strips_trailing_digits() -> None:
    assert backend_of("CUDA0") == "CUDA"
    assert backend_of("Vulkan12") == "Vulkan"
    assert backend_of("MTL0") == "MTL"
    assert backend_of("SYCL0") == "SYCL"
    # 末尾に数字を持たないものはそのまま。
    assert backend_of("BLAS") == "BLAS"
    assert backend_of("CPU") == "CPU"
    assert backend_of("Metal") == "Metal"


def test_group_by_backend_preserves_order() -> None:
    devs = parse_list_devices(REAL_CUDA_VULKAN_OUTPUT)
    groups = group_by_backend(devs)
    assert list(groups.keys()) == ["CUDA", "Vulkan"]
    assert [d.id for d in groups["CUDA"]] == ["CUDA0", "CUDA1"]
    assert [d.id for d in groups["Vulkan"]] == ["Vulkan0", "Vulkan1", "Vulkan2"]


# ---------------------------------------------------------------------------
# build_auto_sweep_configs
# ---------------------------------------------------------------------------


def test_auto_sweep_configs_multi_backend() -> None:
    devs = parse_list_devices(REAL_CUDA_VULKAN_OUTPUT)
    configs = build_auto_sweep_configs(devs)
    labels = [lbl for lbl, _ in configs]
    # 単体 x5(各 selectable)+ バックエンド内複数枚(CUDA x2, Vulkan x3)。
    assert labels == [
        "CUDA0 単体",
        "CUDA1 単体",
        "Vulkan0 単体",
        "Vulkan1 単体",
        "Vulkan2 単体",
        "CUDA x2",
        "Vulkan x3",
    ]
    by_label = dict(configs)
    # CUDA 複数枚は同一バックエンド内のみ、tensor-split は CUDA VRAM 比。
    cuda_multi = by_label["CUDA x2"]
    assert cuda_multi.device_ids == ["CUDA0", "CUDA1"]
    assert cuda_multi.tensor_split == suggest_tensor_split(
        [d for d in devs if d.id in ("CUDA0", "CUDA1")], by="total"
    )
    assert abs(sum(cuda_multi.tensor_split) - 1.0) < 1e-9
    # バックエンド跨ぎ(CUDA+Vulkan 混成)構成は自動生成されない。
    for _, selc in configs:
        backends = {backend_of(i) for i in selc.device_ids}
        assert len(backends) <= 1


def test_auto_sweep_configs_mac_single_gpu_no_multi() -> None:
    devs = parse_list_devices(REAL_MAC_MTL_OUTPUT)
    configs = build_auto_sweep_configs(devs)
    # BLAS 除外 → MTL0 単体のみ。単一 GPU なので複数枚構成も tensor-split も無し。
    assert [lbl for lbl, _ in configs] == ["MTL0 単体"]
    assert configs[0][1].device_ids == ["MTL0"]
    assert configs[0][1].tensor_split == []


def test_auto_sweep_configs_single_backend_pair() -> None:
    devs = [_dev("CUDA0", 32149, 31610), _dev("CUDA1", 24123, 23858)]
    configs = build_auto_sweep_configs(devs)
    assert [lbl for lbl, _ in configs] == ["CUDA0 単体", "CUDA1 単体", "CUDA x2"]
    multi = configs[-1][1]
    assert multi.device_ids == ["CUDA0", "CUDA1"]
    assert multi.tensor_split == [0.57, 0.43]


# ---------------------------------------------------------------------------
# DeviceSelection.to_cli_args
# ---------------------------------------------------------------------------


def test_selection_empty_yields_no_args() -> None:
    # ★ 後方互換の核心: 未選択なら CLI 断片は空。
    assert DeviceSelection().to_cli_args() == []
    assert DeviceSelection(device_ids=[]).active is False


def test_selection_single_device_no_tensor_split() -> None:
    sel = DeviceSelection(device_ids=["CUDA0"])
    assert sel.active is True
    assert sel.to_cli_args() == ["--device", "CUDA0"]


def test_selection_multi_with_split() -> None:
    sel = DeviceSelection(device_ids=["CUDA0", "CUDA1"], tensor_split=[0.57, 0.43])
    assert sel.to_cli_args() == [
        "--device",
        "CUDA0,CUDA1",
        "--tensor-split",
        "0.57,0.43",
    ]


def test_selection_multi_without_split_only_device() -> None:
    sel = DeviceSelection(device_ids=["CUDA0", "CUDA1"])
    assert sel.to_cli_args() == ["--device", "CUDA0,CUDA1"]


def test_selection_single_ignores_provided_split() -> None:
    # 単一デバイスでは split が渡されても付与しない。
    sel = DeviceSelection(device_ids=["CUDA0"], tensor_split=[1.0])
    assert sel.to_cli_args() == ["--device", "CUDA0"]


# ---------------------------------------------------------------------------
# render_bench_command
# ---------------------------------------------------------------------------


def test_render_bench_basic_substitution() -> None:
    argv = render_bench_command(
        "llmbench run --model local-openai --base-url {base_url} --runs {runs}",
        port=8080,
        config_label="CUDA0",
        runs=5,
    )
    assert argv == [
        "llmbench",
        "run",
        "--model",
        "local-openai",
        "--base-url",
        "http://localhost:8080/v1",
        "--runs",
        "5",
    ]


def test_render_bench_config_and_results_dir() -> None:
    argv = render_bench_command(
        "bench --tag {config} --out {results_dir} --port {port}",
        port=9000,
        config_label="dual-gpu",
        results_dir="/tmp/results",
    )
    assert "dual-gpu" in argv
    assert "/tmp/results" in argv
    assert "9000" in argv


def test_render_bench_json_braces_do_not_misfire() -> None:
    # str.format なら {"key": ...} を format field と誤認して KeyError になるが、
    # 単純置換なので例外を出さず {port} だけ置換する(shlex は通常どおり
    # クォートを剥がすため JSON 中の引用符は落ちる = それは想定内)。
    tmpl = 'runner --payload {"model":"x","n":1} --port {port}'
    argv = render_bench_command(tmpl, port=1234, config_label="c")
    assert "1234" in argv  # {port} は置換される
    # JSON 由来の波括弧トークンが未知プレースホルダとして消えずに残る。
    assert any("model" in tok and tok.startswith("{") for tok in argv)


def test_render_bench_runs_none_empty() -> None:
    argv = render_bench_command("bench --runs {runs}", port=1, config_label="c")
    # runs=None → 空文字置換 → shlex で消える。
    assert argv == ["bench", "--runs"]


def test_render_bench_windows_backslash_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    argv = render_bench_command(
        r"C:\tools\llmbench.exe run --port {port}",
        port=8080,
        config_label="c",
    )
    # posix=False でバックスラッシュが素通しされる。
    assert argv[0] == r"C:\tools\llmbench.exe"


def test_render_bench_posix_backslash_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "posix")
    argv = render_bench_command("/usr/bin/llmbench --port {port}", port=8080, config_label="c")
    assert argv[0] == "/usr/bin/llmbench"


def test_render_bench_command_label_cannot_inject_argv() -> None:
    # H-2: split は置換より先。config_label にスペースや追加フラグを混ぜても
    # argv は増殖せず、値は必ず 1 トークンに収まる(引数注入不可)。
    argv = render_bench_command(
        "bench --tag {config} --runs {runs}",
        port=9000,
        config_label="dual --danger extra",
        runs=3,
    )
    assert argv == ["bench", "--tag", "dual --danger extra", "--runs", "3"]
    # 注入されたはずの独立フラグが argv 中に単独トークンとして現れない。
    assert "--danger" not in argv


# ---------------------------------------------------------------------------
# load_latest_results / summarize_results
# ---------------------------------------------------------------------------


def test_summarize_results_canonical_keys() -> None:
    data = {
        "summary": {
            "tokens_per_sec": 42.5,
            "ttft_ms": 120,
            "latency_ms": 900,
            "runs": 5,
            "extra": "ignored",
        }
    }
    out = summarize_results(data)
    assert out["tokens_per_sec"] == 42.5
    assert out["ttft_ms"] == 120
    assert out["latency_ms"] == 900
    assert out["runs"] == 5
    assert "extra" in out["_raw_keys"]


def test_summarize_results_alias_keys() -> None:
    # 別名 (tok_s / throughput / ttft / avg_latency_ms / n_runs) を吸収。
    data = {"tok_s": 30.0, "ttft": 100, "avg_latency_ms": 800, "n_runs": 3}
    out = summarize_results(data)
    assert out["tokens_per_sec"] == 30.0
    assert out["ttft_ms"] == 100
    assert out["latency_ms"] == 800
    assert out["runs"] == 3


def test_summarize_results_throughput_alias() -> None:
    out = summarize_results({"throughput": 55.5})
    assert out["tokens_per_sec"] == 55.5


def test_summarize_results_non_dict() -> None:
    out = summarize_results([1, 2, 3])  # type: ignore[arg-type]
    assert out["_raw_keys"] == []


def test_load_latest_results_picks_newest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text(json.dumps({"tok_s": 10.0}), encoding="utf-8")
    new.write_text(json.dumps({"tok_s": 20.0}), encoding="utf-8")
    # mtime を明示的にずらす。
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    path, summary = load_latest_results(tmp_path, since=0.0)
    assert path is not None and path.endswith("new.json")
    assert summary is not None
    assert summary["tokens_per_sec"] == 20.0


def test_load_latest_results_since_filter(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"tok_s": 1.0}), encoding="utf-8")
    os.utime(f, (1000, 1000))
    # since が mtime よりずっと後 → 除外される(since-1.0 の許容を超える)。
    path, summary = load_latest_results(tmp_path, since=5000.0)
    assert path is None
    assert summary is None


def test_load_latest_results_missing_dir() -> None:
    path, summary = load_latest_results("/nonexistent/dir/xyz", since=0.0)
    assert path is None and summary is None


def test_load_latest_results_broken_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "broken.json"
    f.write_text("{not valid json", encoding="utf-8")
    os.utime(f, (2000, 2000))
    path, summary = load_latest_results(tmp_path, since=0.0)
    assert path is not None and path.endswith("broken.json")
    assert summary is None


# ---------------------------------------------------------------------------
# is_port_free
# ---------------------------------------------------------------------------


def test_is_port_free_true_for_unused() -> None:
    # 未使用ポートを OS に選ばせて解放し、free と判定できることを確認。
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert is_port_free(port) is True


def test_is_port_free_false_when_bound() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.listen(1)
    try:
        assert is_port_free(port) is False
    finally:
        s.close()
