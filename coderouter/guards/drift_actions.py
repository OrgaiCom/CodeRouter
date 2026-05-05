"""Drift detection corrective actions (v2.0-G, L4).

Currently the only non-trivial action is ``reload`` — flush the KV cache
on Ollama-shape providers by sending a ``keep_alive=0`` request to unload
the model, forcing a fresh context window on the next request.

The ``promote`` action is handled directly in fallback.py via
``AdaptiveAdjuster.demote()``.

Architecture
============

All functions are **best-effort**: failures are logged but never raised.
The engine continues regardless — the worst case is that the model stays
loaded with its existing (potentially degraded) KV cache and the adaptive
demotion still routes traffic elsewhere until cooldown expires.
"""

from __future__ import annotations

import httpx

from coderouter.config.schemas import ProviderConfig
from coderouter.logging import get_logger, log_drift_reload_attempted

logger = get_logger(__name__)


def _is_ollama_shape(provider_config: ProviderConfig) -> bool:
    """Return True if the provider looks like Ollama (port 11434 or num_ctx declared)."""
    if provider_config.kind != "openai_compat":
        return False
    base_url = str(provider_config.base_url)
    if ":11434" in base_url:
        return True
    extra = provider_config.extra_body or {}
    options = extra.get("options")
    return isinstance(options, dict) and "num_ctx" in options


def _ollama_base_url(provider_config: ProviderConfig) -> str:
    """Derive the Ollama native API base URL from the OpenAI-compat base_url.

    Typical patterns:
      - ``http://localhost:11434/v1`` → ``http://localhost:11434``
      - ``http://host:11434/v1/``    → ``http://host:11434``
    """
    url = str(provider_config.base_url).rstrip("/")
    # Strip the /v1 suffix to get the Ollama native API root
    if url.endswith("/v1"):
        url = url[:-3]
    return url


async def attempt_reload(provider_config: ProviderConfig) -> bool:
    """Attempt to flush the Ollama KV cache by unloading the model.

    Sends ``POST /api/generate`` with ``keep_alive: "0"`` to the Ollama
    native API. This causes Ollama to unload the model from memory; the
    next inference request will reload it with a fresh KV cache.

    Parameters
    ----------
    provider_config:
        The provider's configuration from providers.yaml. Must be
        Ollama-shape (``kind: openai_compat`` + port 11434 or num_ctx).

    Returns
    -------
    True if the unload request succeeded (HTTP 200), False otherwise.
    Non-Ollama providers return False immediately (no-op).
    """
    if not _is_ollama_shape(provider_config):
        logger.debug(
            "drift-reload-skip",
            extra={
                "provider": provider_config.name,
                "reason": "not-ollama-shape",
            },
        )
        return False

    base_url = _ollama_base_url(provider_config)
    model = provider_config.model

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "keep_alive": 0,
                },
            )
        success = resp.status_code == 200
    except (httpx.HTTPError, OSError) as exc:
        logger.debug(
            "drift-reload-http-error",
            extra={
                "provider": provider_config.name,
                "error": str(exc)[:200],
            },
        )
        success = False

    log_drift_reload_attempted(
        logger,
        provider=provider_config.name,
        success=success,
    )
    return success
