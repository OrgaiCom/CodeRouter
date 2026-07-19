"""Unit tests for coderouter.launcher_devices.detect_llama_devices。

設計書 §5.2 準拠。実 llama-server は起動せず、``runner`` フックに各
プラットフォームのサンプル出力を注入する。キャッシュ(TTL / 破棄)も検証。
"""

from __future__ import annotations

import subprocess

import pytest

from coderouter import launcher_devices as ld
from coderouter.launcher_devices import detect_llama_devices, reset_device_cache

CUDA_DUAL_OUTPUT = """\
Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31626 MiB free)
  CUDA1: NVIDIA GeForce RTX 3090 (24123 MiB, 23800 MiB free)
"""

METAL_OUTPUT = "  Metal: Apple M3 Max (49152 MiB, 49152 MiB free)\n"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """各テスト前後で検出キャッシュを破棄しリークを防ぐ。"""
    reset_device_cache()
    yield
    reset_device_cache()


def _completed(stdout: str = "", stderr: str = "", rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["llama-server", "--list-devices"], returncode=rc, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# 成功系
# ---------------------------------------------------------------------------


def test_detect_success_cuda_dual() -> None:
    probe = detect_llama_devices(
        "llama-server", runner=lambda: _completed(stdout=CUDA_DUAL_OUTPUT)
    )
    assert probe.ok is True
    assert probe.error is None
    assert [d.id for d in probe.devices] == ["CUDA0", "CUDA1"]
    assert probe.devices[0].total_mib == 32149


def test_detect_success_from_stderr() -> None:
    # 一部の版は --list-devices を stderr に出す。
    probe = detect_llama_devices(
        "llama-server", runner=lambda: _completed(stderr=METAL_OUTPUT)
    )
    assert probe.ok is True
    assert probe.devices[0].id == "Metal"


def test_detect_as_dict_shape() -> None:
    probe = detect_llama_devices(
        "llama-server", runner=lambda: _completed(stdout=CUDA_DUAL_OUTPUT)
    )
    d = probe.as_dict()
    assert d["ok"] is True
    assert d["error"] is None
    assert len(d["devices"]) == 2
    assert d["devices"][0]["id"] == "CUDA0"


# ---------------------------------------------------------------------------
# 失敗系
# ---------------------------------------------------------------------------


def test_detect_file_not_found() -> None:
    def boom() -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    probe = detect_llama_devices("/no/such/binary", runner=boom, use_cache=False)
    assert probe.ok is False
    assert probe.error is not None
    assert "見つかりません" in probe.error
    assert probe.devices == []


def test_detect_timeout() -> None:
    def slow() -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="llama-server", timeout=5.0)

    probe = detect_llama_devices("llama-server", runner=slow, use_cache=False)
    assert probe.ok is False
    assert probe.error is not None
    assert "タイムアウト" in probe.error


def test_detect_nonzero_no_devices() -> None:
    # 非ゼロ終了 + パース 0 件 → ok=False。
    probe = detect_llama_devices(
        "llama-server",
        runner=lambda: _completed(stderr="error: unknown flag", rc=1),
        use_cache=False,
    )
    assert probe.ok is False
    assert probe.error is not None
    assert probe.devices == []


def test_detect_os_error() -> None:
    def oserr() -> subprocess.CompletedProcess[str]:
        raise OSError("permission denied")

    probe = detect_llama_devices("llama-server", runner=oserr, use_cache=False)
    assert probe.ok is False
    assert probe.error is not None
    assert "permission denied" in probe.error


# ---------------------------------------------------------------------------
# キャッシュ
# ---------------------------------------------------------------------------


def test_detect_uses_cache_second_call() -> None:
    calls = {"n": 0}

    def counting() -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return _completed(stdout=CUDA_DUAL_OUTPUT)

    first = detect_llama_devices("llama-server", runner=counting)
    second = detect_llama_devices("llama-server", runner=counting)
    assert calls["n"] == 1  # 2 回目はキャッシュヒット
    assert first is second


def test_reset_device_cache_forces_reprobe() -> None:
    calls = {"n": 0}

    def counting() -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return _completed(stdout=CUDA_DUAL_OUTPUT)

    detect_llama_devices("llama-server", runner=counting)
    reset_device_cache()
    detect_llama_devices("llama-server", runner=counting)
    assert calls["n"] == 2


def test_detect_cache_keyed_per_binary() -> None:
    calls = {"n": 0}

    def counting() -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return _completed(stdout=CUDA_DUAL_OUTPUT)

    detect_llama_devices("llama-server-a", runner=counting)
    detect_llama_devices("llama-server-b", runner=counting)
    # binary パスが異なればキャッシュは別枠。
    assert calls["n"] == 2


def test_detect_use_cache_false_always_runs() -> None:
    calls = {"n": 0}

    def counting() -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return _completed(stdout=CUDA_DUAL_OUTPUT)

    detect_llama_devices("llama-server", runner=counting, use_cache=False)
    detect_llama_devices("llama-server", runner=counting, use_cache=False)
    assert calls["n"] == 2


def test_detect_cache_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr(ld.time, "monotonic", lambda: clock["t"])
    calls = {"n": 0}

    def counting() -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return _completed(stdout=CUDA_DUAL_OUTPUT)

    detect_llama_devices("llama-server", runner=counting)
    assert calls["n"] == 1
    # TTL 未経過 → キャッシュ。
    clock["t"] += ld._DEVICE_CACHE_TTL_S - 1.0
    detect_llama_devices("llama-server", runner=counting)
    assert calls["n"] == 1
    # TTL 経過 → 再取得。
    clock["t"] += 2.0
    detect_llama_devices("llama-server", runner=counting)
    assert calls["n"] == 2
