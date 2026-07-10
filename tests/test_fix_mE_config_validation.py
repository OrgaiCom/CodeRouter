"""mE: config-schema validation hardening (M13).

Covers four new fast-fail validators added to
:mod:`coderouter.config.schemas`:

1. ``_check_profile_providers_exist`` — every provider named in a
   profile chain must be declared in ``providers:`` (M13.1).
2. ``_check_names_unique`` — provider and profile names must each be
   unique so ``*_by_name`` can't silently shadow a duplicate (M13.2).
3. ``_check_context_budget_thresholds_ordered`` /
   ``_check_recovery_probe_interval_ordered`` — cross-field threshold
   ordering on :class:`FallbackChain` (M13.3).
4. ``RuleMatcher._exactly_one`` rejecting ``False`` for the boolean
   matchers ``has_tools`` / ``has_image`` (M13.4).

Each validator gets a happy path (the shape the examples/ YAML and the
existing fixtures rely on) plus the failure path it was added to catch.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
    RuleMatcher,
)


def _provider(name: str = "local") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url="http://localhost:8080/v1",
        model="qwen-coder",
    )


def _config(**overrides: object) -> dict[str, object]:
    """Smallest valid CodeRouterConfig kwargs; overrides merge on top."""
    base: dict[str, object] = {
        "allow_paid": False,
        "default_profile": "default",
        "providers": [_provider("local"), _provider("cloud")],
        "profiles": [FallbackChain(name="default", providers=["local", "cloud"])],
    }
    base.update(overrides)
    return base


# ======================================================================
# M13.1: profile providers must exist
# ======================================================================


def test_profile_providers_exist_happy_path() -> None:
    """A profile referencing only declared providers loads cleanly."""
    cfg = CodeRouterConfig(**_config())  # type: ignore[arg-type]
    assert cfg.profile_by_name("default").providers == ["local", "cloud"]


def test_profile_unknown_provider_rejected_at_load() -> None:
    """A typo'd provider name in a chain fast-fails with a pointer.

    Previously this only surfaced at runtime as a
    ``skip-unknown-provider`` warning, and a fully-typo'd chain silently
    had no usable providers until it drained.
    """
    with pytest.raises(ValidationError) as info:
        CodeRouterConfig(
            **_config(  # type: ignore[arg-type]
                profiles=[
                    FallbackChain(name="default", providers=["local", "typpo"]),
                ],
            )
        )
    msg = str(info.value)
    assert "typpo" in msg
    assert "default" in msg  # names the offending profile
    assert "known providers" in msg  # lists valid names for the fix


def test_profile_all_unknown_providers_rejected() -> None:
    """A chain where *every* entry is unknown also fails at load."""
    with pytest.raises(ValidationError, match="ghost"):
        CodeRouterConfig(
            **_config(  # type: ignore[arg-type]
                profiles=[FallbackChain(name="default", providers=["ghost"])],
            )
        )


# ======================================================================
# M13.2: provider / profile name uniqueness
# ======================================================================


def test_unique_names_happy_path() -> None:
    cfg = CodeRouterConfig(**_config())  # type: ignore[arg-type]
    assert {p.name for p in cfg.providers} == {"local", "cloud"}


def test_duplicate_provider_name_rejected() -> None:
    """Two providers named ``local`` — the second would be shadowed by
    ``provider_by_name`` (first-match wins). Reject at load.
    """
    with pytest.raises(ValidationError) as info:
        CodeRouterConfig(
            **_config(  # type: ignore[arg-type]
                providers=[_provider("local"), _provider("local")],
                profiles=[FallbackChain(name="default", providers=["local"])],
            )
        )
    msg = str(info.value)
    assert "duplicate provider name" in msg
    assert "local" in msg


def test_duplicate_profile_name_rejected() -> None:
    """Two profiles named ``default`` — the second would be shadowed by
    ``profile_by_name``. Reject at load.
    """
    with pytest.raises(ValidationError) as info:
        CodeRouterConfig(
            **_config(  # type: ignore[arg-type]
                profiles=[
                    FallbackChain(name="default", providers=["local"]),
                    FallbackChain(name="default", providers=["cloud"]),
                ],
            )
        )
    msg = str(info.value)
    assert "duplicate profile name" in msg
    assert "default" in msg


# ======================================================================
# M13.3: cross-field threshold ordering (FallbackChain)
# ======================================================================


def test_context_budget_thresholds_happy_path() -> None:
    """The default staircase (warn 0.80 <= trim 0.90, target 0.75 < 0.90)
    loads cleanly — this is what the examples/ YAML uses.
    """
    chain = FallbackChain(name="p", providers=["local"])
    assert chain.context_budget_warn_threshold == 0.80
    assert chain.context_budget_trim_threshold == 0.90
    assert chain.context_budget_trim_target == 0.75


def test_context_budget_custom_valid_ordering() -> None:
    """A tighter-but-ordered set (mirrors context-budget-test.yaml) loads."""
    chain = FallbackChain(
        name="p",
        providers=["local"],
        context_budget_warn_threshold=0.50,
        context_budget_trim_threshold=0.70,
        context_budget_trim_target=0.40,
    )
    assert chain.context_budget_trim_target < chain.context_budget_trim_threshold


def test_warn_above_trim_threshold_rejected() -> None:
    """warn_threshold > trim_threshold → the warning could never fire."""
    with pytest.raises(ValidationError) as info:
        FallbackChain(
            name="p",
            providers=["local"],
            context_budget_warn_threshold=0.95,
            context_budget_trim_threshold=0.90,
        )
    msg = str(info.value)
    assert "context_budget_warn_threshold" in msg
    assert "context_budget_trim_threshold" in msg


def test_trim_target_not_below_trim_threshold_rejected() -> None:
    """trim_target >= trim_threshold → trimming never converges.

    ``warn_threshold`` is lowered alongside so the warn-vs-trim ordering
    check passes and this assertion isolates the trim_target rule.
    """
    with pytest.raises(ValidationError) as info:
        FallbackChain(
            name="p",
            providers=["local"],
            context_budget_warn_threshold=0.60,
            context_budget_trim_threshold=0.70,
            context_budget_trim_target=0.80,
        )
    msg = str(info.value)
    assert "context_budget_trim_target" in msg


def test_trim_target_equal_to_trim_threshold_rejected() -> None:
    """Equality is also rejected — strict ``<`` required for convergence."""
    with pytest.raises(ValidationError, match="context_budget_trim_target"):
        FallbackChain(
            name="p",
            providers=["local"],
            context_budget_warn_threshold=0.60,
            context_budget_trim_threshold=0.70,
            context_budget_trim_target=0.70,
        )


def test_recovery_probe_interval_happy_path() -> None:
    """Defaults (initial 30s <= max 300s) load cleanly."""
    chain = FallbackChain(name="p", providers=["local"])
    assert chain.recovery_probe_initial_s <= chain.recovery_probe_max_s


def test_recovery_probe_initial_above_max_rejected() -> None:
    """initial_s > max_s → the backoff ceiling is below its own floor."""
    with pytest.raises(ValidationError) as info:
        FallbackChain(
            name="p",
            providers=["local"],
            recovery_probe_initial_s=400.0,
            recovery_probe_max_s=300.0,
        )
    msg = str(info.value)
    assert "recovery_probe_initial_s" in msg
    assert "recovery_probe_max_s" in msg


# ======================================================================
# v2.x: ``skip`` backend-health action + half-open interval bounds
# ======================================================================


def test_backend_health_action_skip_accepted() -> None:
    """``skip`` is a valid ``backend_health_action`` literal."""
    chain = FallbackChain(
        name="p",
        providers=["local"],
        backend_health_action="skip",
    )
    assert chain.backend_health_action == "skip"
    # Default half-open interval loads cleanly.
    assert chain.backend_health_half_open_s == 30.0


def test_backend_health_half_open_below_floor_rejected() -> None:
    """``backend_health_half_open_s`` below the 5.0s floor is rejected."""
    with pytest.raises(ValidationError) as info:
        FallbackChain(
            name="p",
            providers=["local"],
            backend_health_action="skip",
            backend_health_half_open_s=1.0,
        )
    assert "backend_health_half_open_s" in str(info.value)


def test_backend_health_half_open_above_ceiling_rejected() -> None:
    """``backend_health_half_open_s`` above the 600.0s ceiling is rejected."""
    with pytest.raises(ValidationError) as info:
        FallbackChain(
            name="p",
            providers=["local"],
            backend_health_action="skip",
            backend_health_half_open_s=601.0,
        )
    assert "backend_health_half_open_s" in str(info.value)


# ======================================================================
# M13.4: boolean matcher False is a dead rule → reject at load
# ======================================================================


def test_has_tools_true_accepted() -> None:
    m = RuleMatcher(has_tools=True)
    assert m.has_tools is True


def test_has_image_true_accepted() -> None:
    m = RuleMatcher(has_image=True)
    assert m.has_image is True


def test_has_tools_false_rejected() -> None:
    """``has_tools: False`` is a dead rule (matched with ``is True``), so
    it must be rejected at load rather than silently never firing.
    """
    with pytest.raises(ValidationError) as info:
        RuleMatcher(has_tools=False)
    msg = str(info.value)
    assert "has_tools" in msg


def test_has_image_false_rejected() -> None:
    """Same dead-rule reasoning for ``has_image: False``."""
    with pytest.raises(ValidationError) as info:
        RuleMatcher(has_image=False)
    msg = str(info.value)
    assert "has_image" in msg
