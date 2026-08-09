"""Opt-in token auth for the read-only UI surfaces (v2.14.0).

``/dashboard``, ``/metrics.json`` and ``/metrics`` were the last endpoints
with no authentication story at all. They change nothing, so they were
never part of the H-8 state-changing-endpoint work — but "read-only" is
not "harmless". ``/metrics.json`` returns every provider's name, kind,
paid flag and ``base_url``, plus the profile graph: the full topology of
which models an operator runs and which vendors they pay. On a laptop
that is fine. On a box where the port is reachable by anything else, it
is a free reconnaissance endpoint.

The failure we are deliberately not repeating is codex-router's: its
``/health`` answers *before* the caller-key check and includes the live
session name, so any local process can read what the operator is working
on. Cheap to avoid, easy to ship by accident — the check just has to come
first.

Contract
--------
Identical in shape to :func:`coderouter.ingress.launcher_routes._require_launcher_token`,
so an operator learns it once:

* ``CODEROUTER_METRICS_TOKEN`` unset → the endpoints stay open exactly as
  they were, and a one-time warning is logged. Backwards compatible by
  construction; nobody's Prometheus scrape breaks on upgrade.
* Set → every request must carry a matching ``X-CodeRouter-Token`` header,
  compared with :func:`secrets.compare_digest`.

The token is deliberately **not** accepted as a query parameter. A token
in a URL lands in access logs, in ``Referer`` headers, and in the browser
history — and this codebase now has a log scrubber precisely because
credentials end up in places nobody predicted.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request

from coderouter.logging import get_logger

__all__ = ["METRICS_TOKEN_ENV", "METRICS_TOKEN_HEADER", "require_ui_token"]

logger = get_logger(__name__)

METRICS_TOKEN_ENV = "CODEROUTER_METRICS_TOKEN"
METRICS_TOKEN_HEADER = "X-CodeRouter-Token"

# One warning per process, not per request — a 2-second dashboard poll
# would otherwise bury every other log line.
_warning_emitted = False


def require_ui_token(request: Request) -> None:
    """Enforce the metrics/dashboard token when one is configured.

    Raises ``401`` on mismatch. Returns silently when the env var is
    unset, after logging a single ``metrics-auth-disabled`` line.
    """
    global _warning_emitted
    expected = os.environ.get(METRICS_TOKEN_ENV, "")
    if not expected:
        if not _warning_emitted:
            logger.warning(
                "metrics-auth-disabled",
                extra={
                    "hint": (
                        f"{METRICS_TOKEN_ENV} is not set; /dashboard, "
                        "/metrics.json and /metrics are unauthenticated and "
                        "expose provider names, base_urls and the profile "
                        "graph. Set it when binding to anything but loopback."
                    ),
                },
            )
            _warning_emitted = True
        return
    provided = request.headers.get(METRICS_TOKEN_HEADER, "")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401, detail="Invalid or missing metrics token."
        )


def auth_required() -> bool:
    """Whether a token is configured — safe to embed in the page.

    The dashboard needs to know *that* auth is on so it can prompt; it
    must never receive the token itself. Same split as the v2.13.0 fix
    that stopped ``GET /launcher`` embedding ``CODEROUTER_LAUNCHER_TOKEN``
    straight into the HTML, where ``curl | grep`` recovered it.
    """
    return bool(os.environ.get(METRICS_TOKEN_ENV, ""))
