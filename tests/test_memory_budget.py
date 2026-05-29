"""Unit tests for coderouter.guards.memory_budget (pure fit math)."""

from __future__ import annotations

from coderouter.guards.memory_budget import (
    kv_cache_bytes,
    kv_dim,
    max_num_ctx_for_budget,
    plan_fit,
)

_GB = 1024**3


def test_kv_dim_applies_gqa() -> None:
    # 4096 embd, 32 heads, 8 kv heads → 4096 * 8/32 = 1024
    assert kv_dim(4096, 32, 8) == 1024


def test_kv_dim_no_gqa_metadata_falls_back_to_embd() -> None:
    assert kv_dim(4096, None, None) == 4096
    assert kv_dim(3584, 28, None) == 3584


def test_kv_dim_invalid_kv_heads_falls_back() -> None:
    # kv_heads > heads is invalid → fall back to embd (conservative).
    assert kv_dim(4096, 8, 32) == 4096


def test_kv_cache_bytes_linear_in_ctx() -> None:
    a = kv_cache_bytes(2048, 32, 1024)
    b = kv_cache_bytes(4096, 32, 1024)
    assert b == 2 * a
    # 2 * layers * ctx * width * 2 bytes
    assert a == 2 * 32 * 2048 * 1024 * 2


def test_max_num_ctx_monotonic() -> None:
    small = max_num_ctx_for_budget(1 * _GB, 32, 1024)
    big = max_num_ctx_for_budget(2 * _GB, 32, 1024)
    assert big > small > 0


def test_plan_fit_unknown_when_no_hardware() -> None:
    d = plan_fit(
        available_budget_gb=0.0,
        weights_bytes=4 * _GB,
        requested_num_ctx=8192,
        n_layers=32,
    )
    assert d.action == "unknown"
    assert d.fits is True  # never block when unmeasurable
    assert d.effective_num_ctx == 8192


def test_plan_fit_ok_when_plenty() -> None:
    # 7B Q4 (~4.5GB) on a 24GB budget, modest context → fits as-is.
    d = plan_fit(
        available_budget_gb=24.0,
        weights_bytes=int(4.5 * _GB),
        requested_num_ctx=8192,
        n_layers=32,
        n_embd=4096,
        n_heads=32,
        n_kv_heads=8,
    )
    assert d.action == "ok"
    assert d.effective_num_ctx == 8192
    assert d.fits is True


def test_plan_fit_shrinks_when_tight() -> None:
    # Same model but only ~5.5GB available → must shrink below 8192.
    d = plan_fit(
        available_budget_gb=5.5,
        weights_bytes=int(4.5 * _GB),
        requested_num_ctx=8192,
        n_layers=32,
        n_embd=4096,
        n_heads=32,
        n_kv_heads=8,
        min_num_ctx=2048,
    )
    assert d.action == "shrink"
    assert 2048 <= d.effective_num_ctx < 8192
    assert d.fits is True


def test_plan_fit_insufficient_when_model_too_big() -> None:
    # 14B Q4 (~9GB) requested on an 8GB machine → can't even run min ctx.
    d = plan_fit(
        available_budget_gb=8.0,
        weights_bytes=int(9.0 * _GB),
        requested_num_ctx=8192,
        n_layers=40,
        n_embd=5120,
        n_heads=40,
        n_kv_heads=8,
        min_num_ctx=2048,
    )
    assert d.action == "insufficient"
    assert d.fits is False


def test_plan_fit_effective_never_below_floor() -> None:
    d = plan_fit(
        available_budget_gb=6.0,
        weights_bytes=int(4.5 * _GB),
        requested_num_ctx=32768,
        n_layers=32,
        n_embd=4096,
        n_heads=32,
        n_kv_heads=8,
        min_num_ctx=4096,
    )
    assert d.effective_num_ctx >= 4096


def test_plan_fit_missing_shape_uses_conservative_fallback() -> None:
    # No layer/embd metadata → fallback shape over-counts KV, so a tight
    # budget still yields a valid (possibly shrunk) decision, not a crash.
    d = plan_fit(
        available_budget_gb=10.0,
        weights_bytes=int(4.5 * _GB),
        requested_num_ctx=8192,
        n_layers=None,
        n_embd=None,
    )
    assert d.action in {"ok", "shrink", "insufficient"}
    assert d.effective_num_ctx >= 0
