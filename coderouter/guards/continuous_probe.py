"""Continuous health probing (v2.0-I).

Background task that periodically sends minimal 1-token requests to each
configured provider, feeding the results into the L5 backend health
state machine. Detects provider crashes during idle periods (no user
traffic) so the chain resolver knows to skip/demote a dead backend
before the next real request hits it.

Architecture
============

::

    lifespan startup
      └─ asyncio.create_task(probe_loop(...))

    probe_loop:
      while not shutdown:
        sleep(interval_s)
        for provider in providers:
          result = await probe_one(provider)
          backend_health.record_attempt(...)
          emit log + metrics

Design choices
==============

- **1-token completion** rather than ``/api/version`` or ``/api/tags``
  because version endpoints are Ollama-only; a 1-token generate confirms
  the entire model-serving pipeline is operational (model loaded, KV
  allocated, inference works).
- **Sequential** probing (not parallel) to avoid hammering backends and
  to keep the implementation trivially correct without gather/semaphore.
- **No new dependency** — uses httpx (already a runtime dep) + asyncio
  (stdlib).
- **Graceful shutdown** via an ``asyncio.Event`` set by the lifespan
  exit path. The loop checks the event each iteration and breaks cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from coderouter.config.schemas import ProviderConfig
from coderouter.logging import (
    get_logger,
    log_probe_capabilities_drift,
    log_probe_completed,
    log_probe_round_completed,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ProbeResult
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProbeResult:
    """Outcome of a single provider probe."""

    provider: str
    success: bool
    latency_ms: float
    error: str | None = None
    model_name: str | None = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# probe_one: single-provider 1-token probe
# ---------------------------------------------------------------------------


async def probe_one(
    provider: ProviderConfig,
    *,
    timeout_s: float = 10.0,
) -> ProbeResult:
    """Send a minimal 1-token completion request and measure response.

    For ``kind: openai_compat``: POST /v1/chat/completions
    For ``kind: anthropic``: POST /v1/messages

    The request asks for ``max_tokens: 1`` so the probe is as cheap as
    possible (a single output token is generated, exercising the full
    model pipeline without producing meaningful output).

    Never raises — all failures are captured in ProbeResult(success=False).
    """
    import os

    # Local import: keeps the module-level import graph free of the
    # convert ↔ anthropic_native circular dependency (adapters pull in
    # translation.convert, which pulls back in this adapter).
    from coderouter.adapters.anthropic_native import anthropic_messages_url

    start = time.monotonic()
    provider_name = provider.name
    base_url = str(provider.base_url).rstrip("/")

    # Resolve API key from env (same logic as the adapters)
    headers: dict[str, str] = {}
    if provider.api_key_env:
        api_key = os.environ.get(provider.api_key_env, "")
        if api_key:
            if provider.kind == "anthropic":
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            if provider.kind == "anthropic":
                # Use the shared normalizer so the probe hits the exact
                # same endpoint the adapter does — base_url ending in
                # `/v1` must not produce `/v1/v1/messages` (bug H4).
                url = anthropic_messages_url(base_url)
                body: dict[str, Any] = {
                    "model": provider.model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                }
                resp = await client.post(url, json=body, headers=headers)
            else:
                # openai_compat: Ollama, LM Studio, OpenRouter, etc.
                url = f"{base_url}/chat/completions"
                body = {
                    "model": provider.model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                }
                resp = await client.post(url, json=body, headers=headers)

        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code >= 400:
            return ProbeResult(
                provider=provider_name,
                success=False,
                latency_ms=latency_ms,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        # Extract model name from response (for capabilities drift check)
        model_name: str | None = None
        try:
            data = resp.json()
            model_name = data.get("model")
        except Exception:
            pass

        return ProbeResult(
            provider=provider_name,
            success=True,
            latency_ms=latency_ms,
            model_name=model_name,
        )

    except httpx.TimeoutException:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(
            provider=provider_name,
            success=False,
            latency_ms=latency_ms,
            error=f"timeout after {timeout_s}s",
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(
            provider=provider_name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# capabilities drift detection (Phase 3)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DriftReport:
    """Report of a model-name mismatch between config and probe response."""

    provider: str
    configured_model: str
    observed_model: str
    in_registry: bool


def check_probe_drift(
    provider: ProviderConfig,
    observed_model: str | None,
    *,
    registry: Any = None,
) -> DriftReport | None:
    """Compare the probe response model name against the configured model.

    Returns a :class:`DriftReport` when the observed model differs from
    ``provider.model``, or ``None`` when they match (or when no model
    name was returned by the probe). The ``registry`` argument is an
    optional :class:`CapabilityRegistry` instance used to check whether
    the observed model has a known entry — when it doesn't, the report
    sets ``in_registry=False`` as an extra signal for the operator.

    Never raises — a missing registry or lookup error just defaults to
    ``in_registry=True`` (conservative, avoids false positives).
    """
    if not observed_model:
        return None

    configured = provider.model or ""

    # Normalize: some backends return the model with a prefix or
    # formatting variation. We compare case-sensitively but strip
    # whitespace.
    if observed_model.strip() == configured.strip():
        return None

    # Check registry for the observed model
    in_registry = True
    if registry is not None:
        try:
            resolved = registry.lookup(kind=provider.kind, model=observed_model)
            # If every resolved field is None, the model is unknown
            if (
                resolved.thinking is None
                and resolved.tools is None
                and resolved.max_context_tokens is None
                and resolved.claude_code_suitability is None
                and resolved.cache_control is None
            ):
                in_registry = False
        except Exception:
            pass  # defensive — never crash the probe loop

    return DriftReport(
        provider=provider.name,
        configured_model=configured,
        observed_model=observed_model,
        in_registry=in_registry,
    )


# ---------------------------------------------------------------------------
# probe_loop: background task
# ---------------------------------------------------------------------------


async def probe_loop(
    providers: list[ProviderConfig],
    *,
    record_fn: Any = None,
    interval_s: float = 60.0,
    timeout_s: float = 10.0,
    probe_paid: bool = False,
    shutdown_event: asyncio.Event | None = None,
    health_threshold: int = 3,
    registry: Any = None,
) -> None:
    """Run continuous health probes in an infinite loop until shutdown.

    Args:
        providers: list of provider configs to probe.
        record_fn: callable(provider_name, *, success, threshold) that
            feeds the backend health state machine. When None, results
            are only logged (useful for testing).
        interval_s: seconds to sleep between probe rounds.
        timeout_s: per-provider probe timeout.
        probe_paid: if False, providers with ``paid=True`` are skipped.
        shutdown_event: set this event to stop the loop gracefully.
        health_threshold: consecutive-failure threshold passed to record_fn.
        registry: optional CapabilityRegistry for model drift detection.
    """
    _shutdown = shutdown_event or asyncio.Event()

    # Initial delay: let the server finish startup before first probe round.
    try:
        await asyncio.wait_for(_shutdown.wait(), timeout=interval_s)
        return  # shutdown during initial delay
    except TimeoutError:
        pass  # normal: timeout means the delay elapsed without shutdown

    while not _shutdown.is_set():
        probed = 0
        failures = 0

        for provider in providers:
            if _shutdown.is_set():
                break
            if provider.paid and not probe_paid:
                continue

            result = await probe_one(provider, timeout_s=timeout_s)
            probed += 1

            if not result.success:
                failures += 1

            # Feed into backend health state machine
            if record_fn is not None:
                with contextlib.suppress(Exception):
                    record_fn(
                        result.provider,
                        success=result.success,
                        threshold=health_threshold,
                    )

            # Log individual result
            log_probe_completed(
                logger,
                provider=result.provider,
                success=result.success,
                latency_ms=result.latency_ms,
                error=result.error,
                model_name=result.model_name,
            )

            # Check for model-capabilities drift on success
            if result.success and result.model_name:
                drift = check_probe_drift(
                    provider, result.model_name, registry=registry
                )
                if drift is not None:
                    log_probe_capabilities_drift(
                        logger,
                        provider=drift.provider,
                        configured_model=drift.configured_model,
                        observed_model=drift.observed_model,
                        in_registry=drift.in_registry,
                    )

        # Log round summary
        if probed > 0:
            log_probe_round_completed(
                logger,
                providers_probed=probed,
                failures=failures,
            )

        # Wait for next interval or shutdown
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=interval_s)
            break  # shutdown signaled
        except TimeoutError:
            pass  # normal: sleep elapsed, start next round
