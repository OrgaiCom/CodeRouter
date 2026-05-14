"""Rule suggestion engine for ``coderouter replay --suggest-rules`` (P1-6).

Analyses the request journal statistics produced by
:func:`coderouter.state.replay.summarize_window` and emits a list of
:class:`RuleSuggestion` objects — each containing a plain-English
description, a copy-paste YAML snippet, and the numeric evidence that
drove the recommendation.

Design
------
Pure statistical analysis — no LLM required.  Rules are applied in
priority order; each rule is independently evaluated so multiple
suggestions can fire for the same provider.

Rules (v1.0)
------------

1. **provider_reorder** — If provider B costs less per request than
   provider A *and* B has meaningful traffic, suggest moving B earlier
   in the fallback chain.

2. **enable_prompt_cache** — If a provider has a large average input
   token count (> ``CACHE_INPUT_THRESHOLD``) and a low cache-hit ratio
   (< ``CACHE_HIT_RATIO_THRESHOLD``), suggest enabling
   ``capabilities.prompt_cache: true``.

3. **enable_drift_detection** — If any provider has a non-trivial
   request volume and no drift-detection configuration is visible in
   the stats (proxy: we see the provider at all), emit a reminder to
   set ``drift_detection_action: promote``.

4. **raise_min_window_fill** — If a provider has a low request count
   (< ``SMALL_WINDOW_THRESHOLD``) and drift detection would fire early,
   suggest raising ``drift_detection_sensitivity: low`` to avoid false
   positives.

5. **split_goal_profile** — If there is more than one provider with
   significant traffic and average output tokens differ substantially,
   suggest creating a ``goal`` profile with ``goal_mode: true`` that
   routes to the highest-output provider.

Confidence levels
-----------------
``high``   — clear numeric evidence, low false-positive risk
``medium`` — heuristic, may need operator judgement
``low``    — informational / reminder
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coderouter.state.replay import ProviderSummary, WindowSummary

# ---------------------------------------------------------------------------
# Thresholds (module-level constants for easy tuning)
# ---------------------------------------------------------------------------

# Minimum requests per provider before we emit cost-based suggestions.
_MIN_TRAFFIC: int = 5

# Prompt-cache opportunity: avg input tokens above this → suggest caching.
_CACHE_INPUT_THRESHOLD: int = 2_000

# Prompt-cache opportunity: cache hit ratio below this → suggest enabling.
_CACHE_HIT_RATIO_THRESHOLD: float = 0.10

# Cost reorder: provider B is this fraction cheaper than A → suggest reorder.
_COST_REORDER_THRESHOLD: float = 0.20  # 20% cheaper

# Small-window guard: fewer requests than this → suggest low sensitivity.
_SMALL_WINDOW_THRESHOLD: int = 10

# Goal profile split: relative std-dev of avg output tokens across providers.
_OUTPUT_DIVERGENCE_THRESHOLD: float = 0.40


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RuleSuggestion:
    """One actionable suggestion derived from request journal statistics.

    Attributes
    ----------
    rule:
        Internal rule identifier, e.g. ``"provider_reorder"``.
    title:
        Short human-readable title for the suggestion.
    description:
        Plain-English explanation of what was observed and why the
        change is recommended.
    yaml_snippet:
        Copy-paste YAML fragment showing the recommended change.
        May span multiple lines; always valid YAML in isolation.
    evidence:
        Dict of metric name → value that drove this suggestion.
    confidence:
        ``"high"`` / ``"medium"`` / ``"low"``
    providers_involved:
        Provider names mentioned in this suggestion.
    """

    rule: str
    title: str
    description: str
    yaml_snippet: str
    evidence: dict[str, object] = field(default_factory=dict)
    confidence: str = "medium"
    providers_involved: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------


def _rule_provider_reorder(
    providers: list[ProviderSummary],
) -> list[RuleSuggestion]:
    """Suggest reordering providers by cost-per-request."""
    suggestions: list[RuleSuggestion] = []
    # Only consider providers with meaningful traffic.
    # Include free providers (cost=0) — they are the best candidates.
    active = [p for p in providers if p.request_count >= _MIN_TRAFFIC]
    if len(active) < 2:
        return []

    # Sort by avg cost ascending (cheapest / free first)
    active_by_cost = sorted(active, key=lambda p: p.avg_cost_usd)

    # Compare every pair where the expensive provider costs something.
    for i in range(len(active_by_cost)):
        for j in range(i + 1, len(active_by_cost)):
            cheap = active_by_cost[i]
            expensive = active_by_cost[j]
            if expensive.avg_cost_usd <= 0:
                continue  # both free — no cost advantage to reorder
            saving_pct = (expensive.avg_cost_usd - cheap.avg_cost_usd) / expensive.avg_cost_usd
            if saving_pct >= _COST_REORDER_THRESHOLD:
                suggestions.append(RuleSuggestion(
                    rule="provider_reorder",
                    title=f"Move {cheap.provider!r} before {expensive.provider!r}",
                    description=(
                        f"{cheap.provider!r} costs ${cheap.avg_cost_usd:.4f}/req on average, "
                        f"{saving_pct * 100:.0f}% cheaper than {expensive.provider!r} "
                        f"(${expensive.avg_cost_usd:.4f}/req). "
                        f"Listing the cheaper provider earlier in the fallback chain "
                        f"reduces cost without changing availability."
                    ),
                    yaml_snippet=(
                        f"# In your profile's providers list, move {cheap.provider!r} earlier:\n"
                        f"profiles:\n"
                        f"  - name: default   # or your active profile\n"
                        f"    providers:\n"
                        f"      - {cheap.provider}\n"
                        f"      - {expensive.provider}"
                    ),
                    evidence={
                        "cheap_provider": cheap.provider,
                        "cheap_avg_cost_usd": round(cheap.avg_cost_usd, 6),
                        "expensive_provider": expensive.provider,
                        "expensive_avg_cost_usd": round(expensive.avg_cost_usd, 6),
                        "saving_pct": round(saving_pct * 100, 1),
                    },
                    confidence="high" if saving_pct >= 0.40 else "medium",
                    providers_involved=[cheap.provider, expensive.provider],
                ))
    return suggestions


def _rule_enable_prompt_cache(
    providers: list[ProviderSummary],
) -> list[RuleSuggestion]:
    """Suggest enabling prompt_cache for large-input, low-hit providers."""
    suggestions: list[RuleSuggestion] = []
    for p in providers:
        if p.request_count < _MIN_TRAFFIC:
            continue
        if p.avg_input_tokens < _CACHE_INPUT_THRESHOLD:
            continue
        if p.cache_hit_ratio >= _CACHE_HIT_RATIO_THRESHOLD:
            continue
        suggestions.append(RuleSuggestion(
            rule="enable_prompt_cache",
            title=f"Enable prompt_cache for {p.provider!r}",
            description=(
                f"{p.provider!r} averages {p.avg_input_tokens:.0f} input tokens/req "
                f"but has only a {p.cache_hit_ratio * 100:.1f}% cache-hit ratio. "
                f"Enabling prompt caching can significantly reduce input token costs "
                f"on repeated system prompts (Anthropic models: ~10% cache-read price)."
            ),
            yaml_snippet=(
                f"providers:\n"
                f"  - name: {p.provider}\n"
                f"    capabilities:\n"
                f"      prompt_cache: true"
            ),
            evidence={
                "provider": p.provider,
                "avg_input_tokens": round(p.avg_input_tokens, 0),
                "cache_hit_ratio_pct": round(p.cache_hit_ratio * 100, 1),
                "requests": p.request_count,
            },
            confidence="high" if p.avg_input_tokens > 5_000 else "medium",
            providers_involved=[p.provider],
        ))
    return suggestions


def _rule_enable_drift_detection(
    providers: list[ProviderSummary],
    window_summary: WindowSummary,
) -> list[RuleSuggestion]:
    """Suggest enabling drift detection when there's meaningful traffic."""
    active = [p for p in providers if p.request_count >= _MIN_TRAFFIC * 2]
    if not active:
        return []
    # We can't know if drift detection is already on from stats alone,
    # so this is a "low" confidence reminder for new operators.
    names = ", ".join(f"{p.provider!r}" for p in active)
    return [RuleSuggestion(
        rule="enable_drift_detection",
        title="Consider enabling L4 drift detection",
        description=(
            f"You have {window_summary.total_requests} requests across {names}. "
            f"The L4 drift detector catches quality degradation in long-running "
            f"agent sessions (empty responses, length collapse, tool silence). "
            f"If not already configured, set drift_detection_action: promote to "
            f"auto-demote providers that are silently degrading."
        ),
        yaml_snippet=(
            "# Add to your profile in providers.yaml:\n"
            "profiles:\n"
            "  - name: default\n"
            "    providers: [...]   # your provider list\n"
            "    drift_detection_action: promote\n"
            "    drift_detection_sensitivity: normal\n"
            "    drift_detection_window_size: 20\n"
            "    drift_detection_cooldown_s: 300"
        ),
        evidence={
            "total_requests": window_summary.total_requests,
            "active_providers": [p.provider for p in active],
        },
        confidence="low",
        providers_involved=[p.provider for p in active],
    )]


def _rule_small_window_low_sensitivity(
    providers: list[ProviderSummary],
) -> list[RuleSuggestion]:
    """Suggest low drift sensitivity for providers with small traffic."""
    suggestions: list[RuleSuggestion] = []
    for p in providers:
        if 0 < p.request_count < _SMALL_WINDOW_THRESHOLD:
            suggestions.append(RuleSuggestion(
                rule="low_sensitivity_small_window",
                title=f"Use low drift sensitivity for {p.provider!r} (sparse traffic)",
                description=(
                    f"{p.provider!r} has only {p.request_count} requests in the journal window. "
                    f"With sparse traffic the drift detector's rolling window fills slowly, "
                    f"which can cause false-positives. Setting drift_detection_sensitivity: low "
                    f"requires more evidence before promoting the provider."
                ),
                yaml_snippet=(
                    f"profiles:\n"
                    f"  - name: default\n"
                    f"    drift_detection_sensitivity: low   # was: normal or high\n"
                    f"    drift_detection_window_size: 30    # larger window = more stable"
                ),
                evidence={
                    "provider": p.provider,
                    "request_count": p.request_count,
                    "threshold": _SMALL_WINDOW_THRESHOLD,
                },
                confidence="medium",
                providers_involved=[p.provider],
            ))
    return suggestions


def _rule_goal_profile(
    providers: list[ProviderSummary],
) -> list[RuleSuggestion]:
    """Suggest creating a goal profile when providers differ significantly in output length."""
    import statistics as _stats

    active = [p for p in providers if p.request_count >= _MIN_TRAFFIC and p.avg_output_tokens > 0]
    if len(active) < 2:
        return []

    output_values = [p.avg_output_tokens for p in active]
    mean_out = _stats.mean(output_values)
    if mean_out == 0:
        return []

    stdev_out = _stats.stdev(output_values) if len(output_values) > 1 else 0.0
    rel_stdev = stdev_out / mean_out

    if rel_stdev < _OUTPUT_DIVERGENCE_THRESHOLD:
        return []

    # Highest-output provider is probably best for goal sessions
    best = max(active, key=lambda p: p.avg_output_tokens)
    return [RuleSuggestion(
        rule="goal_profile",
        title=f"Create a 'goal' profile with goal_mode: true → {best.provider!r}",
        description=(
            f"Output token lengths vary significantly across providers "
            f"(relative std-dev {rel_stdev * 100:.0f}%). "
            f"{best.provider!r} produces the most tokens on average "
            f"({best.avg_output_tokens:.0f} tokens/req), making it the "
            f"best candidate for a dedicated goal/agent profile. "
            f"goal_mode: true activates tighter drift thresholds and "
            f"the goal_progress_stall signal for repetition detection."
        ),
        yaml_snippet=(
            "profiles:\n"
            f"  - name: goal\n"
            f"    providers:\n"
            f"      - {best.provider}\n"
            f"    goal_mode: true                   # P1-5: tighter thresholds\n"
            f"    drift_detection_action: promote\n"
            f"    drift_detection_sensitivity: high  # overridden by goal_mode\n"
            f"    drift_detection_window_size: 15\n"
            f"    drift_detection_cooldown_s: 180"
        ),
        evidence={
            "best_provider": best.provider,
            "best_avg_output_tokens": round(best.avg_output_tokens, 0),
            "mean_output_tokens": round(mean_out, 0),
            "relative_stdev_pct": round(rel_stdev * 100, 1),
        },
        confidence="medium",
        providers_involved=[p.provider for p in active],
    )]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def suggest_rules(summary: WindowSummary) -> list[RuleSuggestion]:
    """Analyse a :class:`WindowSummary` and return a list of rule suggestions.

    Parameters
    ----------
    summary:
        Output of :func:`coderouter.state.replay.summarize_window`.

    Returns
    -------
    List of :class:`RuleSuggestion` objects, ordered by confidence
    (``high`` first) then rule name.
    """
    providers = list(summary.providers.values())
    suggestions: list[RuleSuggestion] = []

    suggestions.extend(_rule_provider_reorder(providers))
    suggestions.extend(_rule_enable_prompt_cache(providers))
    suggestions.extend(_rule_enable_drift_detection(providers, summary))
    suggestions.extend(_rule_small_window_low_sensitivity(providers))
    suggestions.extend(_rule_goal_profile(providers))

    _CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: (_CONFIDENCE_ORDER.get(s.confidence, 9), s.rule))
    return suggestions


def format_suggestions(suggestions: list[RuleSuggestion]) -> str:
    """Render suggestions as a human-readable terminal report.

    Returns a plain-text string with section headers, descriptions,
    and copy-paste YAML snippets.
    """
    if not suggestions:
        return "No routing rule suggestions — current configuration looks healthy."

    lines: list[str] = []
    lines.append(f"Found {len(suggestions)} suggestion(s):\n")

    for i, s in enumerate(suggestions, 1):
        conf_badge = {"high": "[HIGH]", "medium": "[MED] ", "low": "[LOW] "}.get(
            s.confidence, "[?]   "
        )
        lines.append(f"  {i}. {conf_badge} {s.title}")
        lines.append(f"     {s.description}")
        if s.evidence:
            evidence_str = ", ".join(f"{k}={v}" for k, v in s.evidence.items())
            lines.append(f"     Evidence: {evidence_str}")
        lines.append("")
        lines.append("     YAML:")
        for yaml_line in s.yaml_snippet.splitlines():
            lines.append(f"       {yaml_line}")
        lines.append("")
        if i < len(suggestions):
            lines.append("  " + "-" * 68)
            lines.append("")

    return "\n".join(lines)


__all__ = [
    "RuleSuggestion",
    "format_suggestions",
    "suggest_rules",
]
