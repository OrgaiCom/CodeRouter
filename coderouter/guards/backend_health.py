"""Backend health monitor (v1.9-E phase 2, L5).

Passive health state machine for chain providers. Counts consecutive
failures observed via :meth:`record_attempt(provider, success=False)`
and transitions the provider's state through:

    HEALTHY  →  DEGRADED  →  UNHEALTHY
       ▲          │             │
       └──────────┴─────────────┘
              success=True

The engine consults :meth:`is_unhealthy(provider)` at chain-resolve
time and (when the active profile's ``backend_health_action`` is
``demote``) moves UNHEALTHY providers to the back of the chain, so
the chain prefers a known-up backend without ever skipping a
provider entirely. A single subsequent success snaps the provider
back to HEALTHY immediately — no rolling-window inertia, no debounce.

Why state-machine, not rolling window
=====================================

The v1.9-C :class:`coderouter.routing.adaptive.AdaptiveAdjuster`
already handles the **gradient** case (continuous latency / error-
rate observations, rolling window, debounced demotions). L5 handles
the **binary** case: did this backend just crash and start refusing
every request? A hard crash produces a deterministic stream of
identical errors on every retry — a state machine catches it in
``threshold`` attempts without waiting for a rolling window to
saturate.

The two adjusters are orthogonal and compose: the engine consults
the AdaptiveAdjuster's ``compute_effective_order`` first, then the
L5 monitor's UNHEALTHY demotion runs on the result. Either signal
alone is enough to demote a bad provider; both together produce the
expected chain reorder.

Concurrency
===========

Thread-safe via an internal ``RLock``. Reads (``state_for`` /
``is_unhealthy``) and writes (``record_attempt``) hold the lock for
the body of the operation so a chain resolve and a per-attempt
record can't observe a torn state.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

HealthState = Literal["HEALTHY", "DEGRADED", "UNHEALTHY"]
"""Three-class health classification.

- ``HEALTHY``   — initial state; consecutive failure count is 0.
- ``DEGRADED``  — failure count has reached ``threshold``; the
                  provider is still attempted in chain order, but
                  the state-changed log fires for operator visibility.
- ``UNHEALTHY`` — failure count has reached ``2 * threshold``; when
                  the profile's action is ``demote``, the chain
                  resolver moves this provider to the back. A single
                  success resets back to ``HEALTHY`` directly.
"""


@dataclass(frozen=True)
class HealthTransition:
    """The outcome of a state-changing :meth:`record_attempt` call.

    Returned by :meth:`record_attempt` so the engine can decide
    whether to emit a ``backend-health-changed`` log line. The
    engine treats ``None`` returns as "no state change" and stays
    quiet; non-None returns surface a single info line carrying the
    transition.
    """

    provider: str
    old_state: HealthState
    new_state: HealthState
    consecutive_failures: int


@dataclass
class _ProviderHealth:
    state: HealthState = "HEALTHY"
    consecutive_failures: int = 0
    # v2.x (``skip`` action): monotonic timestamps driving the built-in
    # half-open trial. ``unhealthy_since`` is stamped when the provider
    # crosses into UNHEALTHY (diagnostic — "how long has it been down");
    # ``last_half_open_at`` records the last time :meth:`should_skip` let a
    # single trial request through. Both are cleared on any success. These
    # are ``time.monotonic`` values, so they are intentionally NOT persisted
    # across restarts (see :meth:`save_state`).
    unhealthy_since: float | None = None
    last_half_open_at: float | None = None


class BackendHealthMonitor:
    """Per-provider health state machine.

    Public API:

    - :meth:`record_attempt(provider, success)` — fold one observed
      attempt outcome. Returns a :class:`HealthTransition` iff the
      provider's state changed; ``None`` otherwise.
    - :meth:`state_for(provider)` — current state (``HEALTHY``
      default for never-observed providers).
    - :meth:`is_unhealthy(provider)` — convenience predicate the
      engine consults at chain-resolve time.
    - :meth:`reset()` — drop all state. Mainly for tests.

    The threshold parameter is supplied per-call (rather than stored
    on the monitor) so different profiles in the same engine can use
    different thresholds without forcing the monitor to be aware of
    profile resolution. The transition rules are:

      * failure → ``consecutive_failures += 1``
        - if it reaches ``2 * threshold``: state = UNHEALTHY
        - elif it reaches ``threshold``: state = DEGRADED
        - else state unchanged
      * success → ``consecutive_failures = 0``, state = HEALTHY
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._state: dict[str, _ProviderHealth] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_attempt(
        self,
        provider: str,
        *,
        success: bool,
        threshold: int,
    ) -> HealthTransition | None:
        """Fold one observed attempt outcome and return a transition (if any).

        Returns ``None`` when the operation didn't change the
        provider's state — the engine uses this to gate
        ``backend-health-changed`` log emissions (no log spam when
        a HEALTHY provider succeeds repeatedly).
        """
        with self._lock:
            entry = self._state.setdefault(provider, _ProviderHealth())
            old_state = entry.state

            if success:
                entry.consecutive_failures = 0
                # v2.x: a single success snaps back to HEALTHY and clears the
                # half-open bookkeeping so a future crash starts a fresh cycle.
                entry.unhealthy_since = None
                entry.last_half_open_at = None
                if old_state != "HEALTHY":
                    entry.state = "HEALTHY"
                    return HealthTransition(
                        provider=provider,
                        old_state=old_state,
                        new_state="HEALTHY",
                        consecutive_failures=0,
                    )
                return None

            entry.consecutive_failures += 1
            new_state: HealthState
            if entry.consecutive_failures >= 2 * threshold:
                new_state = "UNHEALTHY"
            elif entry.consecutive_failures >= threshold:
                new_state = "DEGRADED"
            else:
                # Below threshold — state unchanged but failure
                # counter is still ticking (caller may transition
                # us on a future call).
                return None

            if new_state == old_state:
                # We've already been at this level (e.g. already
                # UNHEALTHY and failing again) — no transition fired.
                return None

            entry.state = new_state
            if new_state == "UNHEALTHY":
                # v2.x: record the moment we went down (diagnostic). The
                # half-open trial itself is keyed off ``last_half_open_at``,
                # which starts unset so the first resolve after this
                # transition is allowed straight through as a trial.
                entry.unhealthy_since = time.monotonic()
            return HealthTransition(
                provider=provider,
                old_state=old_state,
                new_state=new_state,
                consecutive_failures=entry.consecutive_failures,
            )

    def reset(self) -> None:
        """Drop all per-provider state."""
        with self._lock:
            self._state.clear()

    # ------------------------------------------------------------------
    # Read-only API
    # ------------------------------------------------------------------

    def state_for(self, provider: str) -> HealthState:
        """Return ``provider``'s current health state.

        Never-observed providers default to ``HEALTHY`` — the chain
        resolver doesn't have to special-case "first attempt".
        """
        with self._lock:
            entry = self._state.get(provider)
            if entry is None:
                return "HEALTHY"
            return entry.state

    def is_unhealthy(self, provider: str) -> bool:
        """True iff ``provider``'s current state is ``UNHEALTHY``."""
        return self.state_for(provider) == "UNHEALTHY"

    def should_skip(self, provider: str, *, half_open_interval_s: float) -> bool:
        """v2.x: decide whether to skip ``provider`` at chain-resolve time.

        Drives the ``skip`` backend-health action's built-in half-open
        circuit breaker. Semantics:

        * Not UNHEALTHY → ``False`` (never skip a HEALTHY/DEGRADED provider).
        * UNHEALTHY, but either never trialed (``last_half_open_at is None``)
          or the half-open interval has elapsed since the last trial → stamp
          ``last_half_open_at = now`` and return ``False`` — exactly one
          request is let through as a probe. If it succeeds,
          :meth:`record_attempt` snaps the provider back to HEALTHY; if it
          fails, the counter stays UNHEALTHY and the freshly-stamped trial
          time keeps the provider skipped until the interval elapses again.
        * UNHEALTHY and inside the current interval → ``True`` (skip).

        This is a mutating read: the ``last_half_open_at`` stamp is the
        side effect that makes the half-open trial fire at most once per
        ``half_open_interval_s`` window. Held under the lock so concurrent
        resolves can't both slip a trial through the same window.
        """
        with self._lock:
            entry = self._state.get(provider)
            if entry is None or entry.state != "UNHEALTHY":
                return False
            now = time.monotonic()
            last = entry.last_half_open_at
            if last is None or (now - last) >= half_open_interval_s:
                entry.last_half_open_at = now
                return False
            return True

    # ------------------------------------------------------------------
    # v2.0-K: Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict[str, object]:
        """Export the current per-provider health state for persistence.

        v2.x: only ``state`` and ``consecutive_failures`` are persisted.
        ``unhealthy_since`` / ``last_half_open_at`` are ``time.monotonic``
        values whose zero point is per-process — persisting them across a
        restart would be meaningless (and could compare against a future
        process's clock). They are dropped here and default to ``None`` on
        :meth:`load_state`, which simply means the first resolve after a
        restart is treated as a fresh half-open trial — safe, since a
        successful trial resets the provider anyway.
        """
        with self._lock:
            return {
                name: {
                    "state": entry.state,
                    "consecutive_failures": entry.consecutive_failures,
                }
                for name, entry in self._state.items()
            }

    def load_state(self, state: dict[str, object]) -> None:
        """Restore health state from a previously saved dict."""
        if not isinstance(state, dict):
            return
        with self._lock:
            for name, data in state.items():
                if not isinstance(data, dict):
                    continue
                saved_state = data.get("state", "HEALTHY")
                if saved_state not in ("HEALTHY", "DEGRADED", "UNHEALTHY"):
                    saved_state = "HEALTHY"
                failures = data.get("consecutive_failures", 0)
                if not isinstance(failures, int) or failures < 0:
                    failures = 0
                self._state[name] = _ProviderHealth(
                    state=saved_state,  # type: ignore[arg-type]
                    consecutive_failures=failures,
                )


__all__ = [
    "BackendHealthMonitor",
    "HealthState",
    "HealthTransition",
]
