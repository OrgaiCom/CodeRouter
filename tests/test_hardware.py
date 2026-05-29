"""Unit tests for coderouter.hardware."""

from __future__ import annotations

from coderouter import hardware
from coderouter.hardware import (
    HardwareInfo,
    available_budget_gb,
    detect_hardware,
    headroom_gb,
    usable_memory_gb,
)


def test_detect_hardware_returns_info() -> None:
    hardware.reset_cache()
    info = detect_hardware(force_refresh=True)
    assert isinstance(info, HardwareInfo)
    assert info.cpu_count >= 1
    assert info.gpu in {"cuda", "metal", "cpu"}
    assert info.ram_gb >= 0.0
    assert info.vram_gb >= 0.0


def test_detect_hardware_caches() -> None:
    hardware.reset_cache()
    first = detect_hardware(force_refresh=True)
    # Second call (no force) must return the *same* cached object.
    second = detect_hardware()
    assert first is second


def test_reset_cache_forces_new_object() -> None:
    a = detect_hardware(force_refresh=True)
    hardware.reset_cache()
    b = detect_hardware()
    assert a is not b  # new object after reset


def test_usable_memory_cuda_uses_vram() -> None:
    hw = HardwareInfo(ram_gb=64.0, vram_gb=8.0, gpu="cuda", cpu_count=16)
    assert usable_memory_gb(hw) == 8.0


def test_usable_memory_metal_uses_ram() -> None:
    hw = HardwareInfo(ram_gb=16.0, vram_gb=16.0, gpu="metal", cpu_count=10)
    assert usable_memory_gb(hw) == 16.0
    assert hw.unified_memory is True


def test_usable_memory_undetected_is_zero() -> None:
    hw = HardwareInfo(ram_gb=0.0, vram_gb=0.0, gpu="cpu", cpu_count=4)
    assert hw.detected is False
    assert usable_memory_gb(hw) == 0.0


def test_headroom_floor_and_ratio() -> None:
    # Small machine: floor dominates.
    assert headroom_gb(8.0, floor_gb=1.5, ratio=0.15) == 1.5
    # Large machine: ratio dominates.
    assert headroom_gb(64.0, floor_gb=1.5, ratio=0.15) == 64.0 * 0.15


def test_available_budget_subtracts_headroom() -> None:
    hw = HardwareInfo(ram_gb=16.0, vram_gb=16.0, gpu="metal", cpu_count=10)
    budget = available_budget_gb(hw, floor_gb=1.5, ratio=0.15)
    # usable=16, headroom=max(1.5, 2.4)=2.4 → 13.6
    assert abs(budget - 13.6) < 1e-9


def test_available_budget_undetected_is_zero() -> None:
    hw = HardwareInfo(ram_gb=0.0, vram_gb=0.0, gpu="cpu", cpu_count=4)
    assert available_budget_gb(hw) == 0.0


def test_available_budget_never_negative() -> None:
    hw = HardwareInfo(ram_gb=0.5, vram_gb=0.0, gpu="cpu", cpu_count=2)
    assert available_budget_gb(hw, floor_gb=1.5, ratio=0.15) == 0.0
