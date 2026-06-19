"""Phase 1b integration tests for the language-tax vertical.

Covers the wiring that connects the leaf ``language_tax`` module to the
cost calculator, the ``cache-observed`` log line, and the
MetricsCollector aggregation surface that feeds the dashboard.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from coderouter.config.schemas import CostConfig
from coderouter.cost import compute_cost_for_attempt
from coderouter.language_tax import (
    LanguageTaxBreakdown,
    estimate_language_tax_for_request,
)
from coderouter.logging import configure_logging, get_logger, log_cache_observed
from coderouter.metrics.collector import (
    MetricsCollector,
    install_collector,
    uninstall_collector,
)
from coderouter.token_estimation import extract_text_from_anthropic_request

_SONNET = CostConfig(
    input_tokens_per_million=3.00,
    output_tokens_per_million=15.00,
    cache_read_discount=0.10,
    cache_creation_premium=1.25,
)


# ---------------------------------------------------------------------------
# extract_text_from_anthropic_request
# ---------------------------------------------------------------------------


def test_extract_text_str_system_and_dict_messages():
    text = extract_text_from_anthropic_request(
        system="あなたは助手です",
        messages=[
            {"content": "二つの数を足して"},
            {"content": [{"type": "text", "text": "合計を返す"}]},
        ],
    )
    assert "あなたは助手です" in text
    assert "二つの数を足して" in text
    assert "合計を返す" in text


def test_extract_text_list_system_and_skips_nontext_blocks():
    text = extract_text_from_anthropic_request(
        system=[{"type": "text", "text": "SYS"}],
        messages=[
            {"content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "source": {}},  # contributes nothing
                {"type": "tool_use", "name": "x"},
            ]},
        ],
    )
    assert "SYS" in text and "hello" in text
    assert "image" not in text and "tool_use" not in text


def test_extract_text_empty_request():
    assert extract_text_from_anthropic_request(system=None, messages=[]) == ""


# ---------------------------------------------------------------------------
# estimate_language_tax_for_request
# ---------------------------------------------------------------------------


def test_estimate_for_request_no_tokenizer_is_inert():
    b = estimate_language_tax_for_request(
        system="日本語のシステムプロンプト",
        messages=[{"content": "日本語のメッセージ" * 4}],
    )
    assert isinstance(b, LanguageTaxBreakdown)
    assert b.tax_multiplier == 1.0  # no tokenizer -> inert
    assert b.cjk_ratio > 0.9


# ---------------------------------------------------------------------------
# compute_cost_for_attempt threads the language-tax breakdown
# ---------------------------------------------------------------------------


def test_cost_without_language_tax_is_backward_compatible():
    out = compute_cost_for_attempt(
        _SONNET,
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    assert out.language_tax_multiplier == 1.0
    assert out.language_tax_usd == 0.0


def test_cost_with_language_tax_populates_fields():
    lt = LanguageTaxBreakdown(
        char_count=400,
        cjk_ratio=1.0,
        tokens_heuristic=100,
        tokens_accurate=250,
        accurate_available=True,
        tax_multiplier=2.5,
        extra_tokens=150,
    )
    out = compute_cost_for_attempt(
        _SONNET,
        input_tokens=250,
        output_tokens=10,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        language_tax=lt,
    )
    assert out.language_tax_multiplier == pytest.approx(2.5)
    # 150 extra tokens at $3/M
    assert out.language_tax_usd == pytest.approx(150 * 3.0 / 1_000_000)


def test_cost_language_tax_zero_for_local_provider():
    lt = LanguageTaxBreakdown(tax_multiplier=2.0, extra_tokens=150)
    out = compute_cost_for_attempt(
        None,  # local / unpriced
        input_tokens=250,
        output_tokens=10,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        language_tax=lt,
    )
    # None cost_config short-circuits to a zero breakdown.
    assert out.language_tax_usd == 0.0


# ---------------------------------------------------------------------------
# MetricsCollector aggregation of language_tax_usd
# ---------------------------------------------------------------------------


@pytest.fixture
def collector() -> Iterator[MetricsCollector]:
    uninstall_collector()
    configure_logging()
    yield install_collector(ring_size=16)
    uninstall_collector()


def _fire(**extra: Any) -> None:
    log_cache_observed(
        get_logger("test.language_tax"),
        provider=extra.get("provider", "cloud"),
        request_had_cache_control=False,
        outcome="no_cache",
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        input_tokens=extra.get("input_tokens", 0),
        output_tokens=0,
        streaming=False,
        cost_usd=extra.get("cost_usd", 0.0),
        cost_savings_usd=0.0,
        language_tax_usd=extra.get("language_tax_usd", 0.0),
        language_tax_multiplier=extra.get("language_tax_multiplier", 1.0),
    )


def test_collector_aggregates_language_tax(collector: MetricsCollector):
    _fire(provider="cloud", cost_usd=0.01, language_tax_usd=0.003)
    _fire(provider="cloud", cost_usd=0.01, language_tax_usd=0.002)
    snap = collector.snapshot()
    counters = snap["counters"]
    assert counters["language_tax_usd_aggregate"] == pytest.approx(0.005)
    assert counters["language_tax_usd"]["cloud"] == pytest.approx(0.005)


def test_collector_provider_row_has_language_tax(collector: MetricsCollector):
    _fire(provider="cloud", cost_usd=0.01, language_tax_usd=0.004)
    snap = collector.snapshot()
    row = next(p for p in snap["providers"] if p["name"] == "cloud")
    assert row["cost"]["language_tax_usd"] == pytest.approx(0.004)


def test_collector_zero_language_tax_no_entry(collector: MetricsCollector):
    _fire(provider="local", cost_usd=0.0, language_tax_usd=0.0)
    snap = collector.snapshot()
    assert "local" not in snap["counters"]["language_tax_usd"]


def test_collector_reset_clears_language_tax(collector: MetricsCollector):
    _fire(provider="cloud", language_tax_usd=0.01)
    collector.reset()
    snap = collector.snapshot()
    assert snap["counters"]["language_tax_usd_aggregate"] == 0.0
    assert snap["counters"]["language_tax_usd"] == {}


def test_collector_defensive_against_non_float_language_tax(
    collector: MetricsCollector,
):
    get_logger("test.language_tax").info(
        "cache-observed",
        extra={
            "provider": "cloud",
            "language_tax_usd": "not-a-number",
        },
    )
    snap = collector.snapshot()  # must not raise
    assert snap["counters"]["language_tax_usd_aggregate"] == 0.0
