"""FastAPI app factory."""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from coderouter import __version__
from coderouter.config import load_config
from coderouter.ingress.anthropic_routes import router as anthropic_router
from coderouter.ingress.dashboard_routes import router as dashboard_router
from coderouter.ingress.metrics_routes import router as metrics_router
from coderouter.ingress.openai_routes import router as openai_router
from coderouter.logging import configure_logging, get_logger
from coderouter.metrics import install_collector
from coderouter.routing import FallbackEngine
from coderouter.routing.capability import check_claude_code_chain_suitability

logger = get_logger(__name__)


def create_app(config_path: str | None = None) -> FastAPI:
    """Build a FastAPI app with routes, engine, and lifespan installed.

    ``config_path`` (optional) is passed through to
    :func:`coderouter.config.load_config`; when ``None`` the loader
    falls through to ``$CODEROUTER_CONFIG`` / ``./providers.yaml``. The
    engine and config are attached to ``app.state`` so route handlers
    can reach them without re-parsing YAML per request.
    """
    configure_logging()
    # v1.5-A: attach the MetricsCollector before the first log line so the
    # startup ``coderouter-startup`` record is already counted. Idempotent,
    # so multiple create_app() calls (tests) don't stack handlers.
    install_collector()
    config = load_config(config_path)
    engine = FallbackEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Log a structured startup line, yield to serve, log shutdown.

        The startup payload captures the effective default profile and
        whether it came from the YAML file or from ``$CODEROUTER_MODE``
        — useful when a shell env is unknowingly overriding the
        committed config.
        """
        # v0.6-A: surface the effective default_profile + where it came from,
        # so operators can tell at a glance whether a shell env is driving the
        # server ("oh, my .envrc set CODEROUTER_MODE") vs the YAML file
        # ("default_profile: coding was committed").
        mode_source = "env" if os.environ.get("CODEROUTER_MODE", "").strip() else "config"
        logger.info(
            "coderouter-startup",
            extra={
                "version": __version__,
                "providers": [p.name for p in config.providers],
                "profiles": [pr.name for pr in config.profiles],
                "allow_paid": config.allow_paid,
                "default_profile": config.default_profile,
                "mode_source": mode_source,
            },
        )
        # v1.7-B: scan ``claude-code-*`` profiles for providers whose
        # registry-resolved ``claude_code_suitability`` is ``degraded``,
        # emitting a structured warn per affected profile. Runs after the
        # startup line so the operator's eye finds startup → warn in
        # chronological order. Non-fatal — the chain still works, just
        # potentially sub-optimally for the agentic harness.
        check_claude_code_chain_suitability(config, logger=logger)

        # v2.0-K: attach persistent state store + audit/request log if configured.
        state_store = None
        audit_handler = None
        request_log_handler = None
        if config.state_dir:
            import logging as _logging
            from pathlib import Path

            from coderouter.state.audit_log import AuditLogHandler
            from coderouter.state.store import StateStore

            state_path = Path(config.state_dir).expanduser()
            state_store = StateStore(state_path / "coderouter.db")
            engine.attach_state_store(state_store)

            # Restore MetricsCollector state from the store.
            from coderouter.metrics import get_collector

            collector = get_collector()
            if collector is not None:
                metrics_state = state_store.get("metrics", "state")
                if metrics_state is not None:
                    with contextlib.suppress(Exception):
                        collector.load_state(metrics_state)  # type: ignore[arg-type]

            logger.info(
                "state-store-attached",
                extra={"state_dir": str(state_path)},
            )

            if config.audit_log == "active":
                audit_handler = AuditLogHandler(
                    state_path / "audit.jsonl",
                    max_bytes=config.audit_log_max_bytes,
                )
                _logging.getLogger().addHandler(audit_handler)
                logger.info(
                    "audit-log-started",
                    extra={
                        "path": str(state_path / "audit.jsonl"),
                        "max_bytes": config.audit_log_max_bytes,
                    },
                )

            if config.request_log == "active":
                from coderouter.state.request_log import RequestLogHandler

                request_log_handler = RequestLogHandler(
                    state_path / "requests.jsonl",
                    max_bytes=config.request_log_max_bytes,
                )
                _logging.getLogger().addHandler(request_log_handler)
                logger.info(
                    "request-log-started",
                    extra={
                        "path": str(state_path / "requests.jsonl"),
                        "max_bytes": config.request_log_max_bytes,
                    },
                )

        # v2.0-I: launch continuous probe background task if configured.
        probe_task = None
        shutdown_event = None
        if config.continuous_probe == "active":
            import asyncio

            from coderouter.guards.continuous_probe import probe_loop
            from coderouter.routing.capability import get_default_registry

            shutdown_event = asyncio.Event()
            probe_task = asyncio.create_task(
                probe_loop(
                    config.providers,
                    record_fn=engine.backend_health.record_attempt,
                    interval_s=config.probe_interval_s,
                    timeout_s=config.probe_timeout_s,
                    probe_paid=config.probe_paid,
                    shutdown_event=shutdown_event,
                    registry=get_default_registry(),
                )
            )
            logger.info(
                "continuous-probe-started",
                extra={
                    "interval_s": config.probe_interval_s,
                    "probe_paid": config.probe_paid,
                    "providers": len(config.providers),
                },
            )

        yield

        # Graceful shutdown of probe task
        if probe_task is not None and shutdown_event is not None:
            shutdown_event.set()
            with contextlib.suppress(Exception):
                await probe_task

        # v2.0-J: graceful shutdown of recovery probe tasks.
        with contextlib.suppress(Exception):
            await engine.shutdown_recovery_probes()

        # v2.0-K: persist state and close audit log on shutdown.
        if state_store is not None:
            with contextlib.suppress(Exception):
                engine.save_all_state()
            # Save MetricsCollector state.
            from coderouter.metrics import get_collector

            collector = get_collector()
            if collector is not None:
                with contextlib.suppress(Exception):
                    state_store.put("metrics", "state", collector.save_state())
            with contextlib.suppress(Exception):
                state_store.close()
        if audit_handler is not None:
            import logging as _logging

            with contextlib.suppress(Exception):
                _logging.getLogger().removeHandler(audit_handler)
                audit_handler.close()
        if request_log_handler is not None:
            import logging as _logging

            with contextlib.suppress(Exception):
                _logging.getLogger().removeHandler(request_log_handler)
                request_log_handler.close()

        logger.info("coderouter-shutdown")

    app = FastAPI(
        title="CodeRouter",
        version=__version__,
        description="Local-first, free-first, fallback-built-in LLM router.",
        lifespan=lifespan,
    )

    # Inject engine + config so route handlers can reach them via app.state
    app.state.engine = engine
    app.state.config = config

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        """Lightweight liveness / config snapshot endpoint.

        Reports the running version plus the effective provider names
        and paid-gate state. Intended for readiness probes and for
        quick operator inspection — does NOT touch upstream providers.
        """
        return {
            "status": "ok",
            "version": __version__,
            "providers": [p.name for p in config.providers],
            "allow_paid": config.allow_paid,
        }

    # Claude Code and similar SDKs probe the base URL with HEAD / or GET /
    # at startup. Return a tiny identifier instead of 404 so those probes
    # succeed cleanly. Non-functional beyond that.
    @app.api_route("/", methods=["GET", "HEAD"])
    async def root() -> dict[str, str]:
        """Minimal identifier for SDK base-URL probes (GET / and HEAD /).

        Claude Code and similar SDKs HEAD/GET the base URL at startup
        to verify reachability. Returning a tiny JSON payload instead
        of 404 keeps those probes from logging scary warnings.
        """
        return {"service": "coderouter", "version": __version__}

    app.include_router(openai_router, prefix="/v1", tags=["openai-compat"])
    app.include_router(anthropic_router, prefix="/v1", tags=["anthropic-compat"])
    # v1.5-A: /metrics.json sits at the root (no /v1 prefix) — metrics are not
    # part of the OpenAI / Anthropic API surface, and Prometheus-style
    # endpoints conventionally live at the root in v1.5-B.
    app.include_router(metrics_router, tags=["metrics"])
    # v1.5-D: single-page HTML view over the same collector snapshot.
    # Same root-level mount as /metrics.json — the dashboard is a UI
    # concern and doesn't belong under the /v1 API surface.
    app.include_router(dashboard_router, tags=["dashboard"])

    return app


# Lazy module-level `app` attribute so `uvicorn coderouter.ingress.app:app …`
# works, but importing this module in tests does NOT immediately load
# providers.yaml. The FastAPI instance is built on first attribute access.
#
# Config path is resolved then — from $CODEROUTER_CONFIG or ./providers.yaml;
# see coderouter.config.loader._candidate_paths for the full search order.
_lazy_app: FastAPI | None = None


def __getattr__(name: str) -> object:
    """PEP 562 module ``__getattr__`` — lazy FastAPI instance on first access.

    Makes ``uvicorn coderouter.ingress.app:app …`` work without having
    ``import coderouter.ingress.app`` load ``providers.yaml`` at import
    time. Tests can import the module without side effects and call
    :func:`create_app` explicitly with a temp config.
    """
    global _lazy_app
    if name == "app":
        if _lazy_app is None:
            _lazy_app = create_app()
        return _lazy_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
