"""Self-healing routing orchestrator (v2.0-J).

When the L5 :class:`BackendHealthMonitor` transitions a provider to
UNHEALTHY and the profile's ``backend_health_action`` is ``exclude``,
this orchestrator:

1. **Excludes** the provider from the chain (complete removal, not
   just demotion to the back).
2. **Attempts restart** if the provider declares a ``restart_command``
   in ``providers.yaml``.
3. **Schedules recovery probes** with exponential backoff (default
   30 s → 60 s → 120 s → 300 s cap) to detect when the backend
   comes back online.
4. **Restores** the provider to its original chain position on the
   first successful recovery probe.

Architecture
============

::

    BackendHealthMonitor
      └─ transition → UNHEALTHY
           └─ engine calls orchestrator.on_unhealthy(provider)
                ├─ add to _excluded set
                ├─ try restart_command (if configured)
                └─ schedule recovery probe (async)

    recovery_probe_loop:
      while provider in _excluded:
        sleep(backoff interval)
        probe_one(provider)
        if success:
          remove from _excluded
          record_attempt(success=True)  → snap to HEALTHY
          log restore
          break
        else:
          interval = min(interval * 2, max_interval)

    _resolve_chain (engine):
      Pass 4b: if action == "exclude":
        filter out providers in orchestrator._excluded

Design choices
==============

- **Thread-safe** via an internal ``RLock`` (same pattern as
  ``BackendHealthMonitor``). The exclude set and restart lock
  are guarded independently.
- **No new dependency** — subprocess (stdlib) for restart commands,
  asyncio for recovery probe scheduling.
- **Restart is opt-in** — only providers with ``restart_command``
  set get automatic restart. Others rely solely on recovery probes
  (waiting for manual restart by the operator).
- **Double-restart prevention** — a per-provider ``_restart_lock``
  prevents concurrent restart attempts (e.g. two profiles both
  hitting UNHEALTHY on the same provider).
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import threading
import time
from dataclasses import dataclass

from coderouter.config.schemas import ProviderConfig
from coderouter.logging import (
    get_logger,
    log_self_healing_exclude,
    log_self_healing_recovery_probe,
    log_self_healing_restart,
    log_self_healing_restore,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class _ExcludedProvider:
    """Metadata for a provider currently excluded from the chain."""

    provider: str
    excluded_at: float
    profile: str
    consecutive_failures: int


class SelfHealingOrchestrator:
    """Manages provider exclusion, restart, and recovery probing.

    Public API:

    - :meth:`on_unhealthy(provider, ...)` — called by the engine when
      a provider transitions to UNHEALTHY with action ``exclude``.
    - :meth:`on_recovered(provider, ...)` — called when a recovery
      probe succeeds or a regular request succeeds on a previously
      excluded provider.
    - :meth:`is_excluded(provider)` — True iff the provider is
      currently excluded from the chain.
    - :meth:`excluded_providers()` — set of currently excluded
      provider names.
    - :meth:`try_restart(provider_config, ...)` — attempt to restart
      a provider's backend process.
    - :meth:`reset()` — clear all state. Mainly for tests.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._excluded: dict[str, _ExcludedProvider] = {}
        # Per-provider lock to prevent concurrent restart attempts.
        self._restart_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------
    # Exclusion management
    # ------------------------------------------------------------------

    def on_unhealthy(
        self,
        provider: str,
        *,
        profile: str,
        consecutive_failures: int,
    ) -> bool:
        """Mark a provider as excluded from the chain.

        Returns True if the provider was newly excluded (not already
        excluded). Returns False if it was already excluded (idempotent).
        """
        with self._lock:
            if provider in self._excluded:
                return False
            self._excluded[provider] = _ExcludedProvider(
                provider=provider,
                excluded_at=time.monotonic(),
                profile=profile,
                consecutive_failures=consecutive_failures,
            )
        log_self_healing_exclude(
            logger,
            provider=provider,
            profile=profile,
            consecutive_failures=consecutive_failures,
        )
        return True

    def on_recovered(
        self,
        provider: str,
        *,
        profile: str,
    ) -> float | None:
        """Restore a provider to the chain after recovery.

        Returns the duration (seconds) the provider was excluded,
        or None if the provider was not in the excluded set.
        """
        with self._lock:
            entry = self._excluded.pop(provider, None)
            if entry is None:
                return None
        duration = time.monotonic() - entry.excluded_at
        log_self_healing_restore(
            logger,
            provider=provider,
            profile=profile,
            excluded_duration_s=duration,
        )
        return duration

    def is_excluded(self, provider: str) -> bool:
        """True iff the provider is currently excluded from the chain."""
        with self._lock:
            return provider in self._excluded

    def excluded_providers(self) -> set[str]:
        """Return a snapshot of currently excluded provider names."""
        with self._lock:
            return set(self._excluded.keys())

    def excluded_profiles(self) -> dict[str, str]:
        """Return a snapshot mapping excluded provider -> owning profile.

        Used by the engine to re-arm recovery probes after a restart:
        each excluded provider must be probed under the profile whose
        ``backend_health_threshold`` / ``recovery_probe_*`` settings
        drove its exclusion. See :meth:`excluded_providers` for the
        name-only variant.
        """
        with self._lock:
            return {
                name: entry.profile for name, entry in self._excluded.items()
            }

    def reset(self) -> None:
        """Drop all state. Mainly for tests."""
        with self._lock:
            self._excluded.clear()
            self._restart_locks.clear()

    # ------------------------------------------------------------------
    # v2.0-K: Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict[str, object]:
        """Export the current excluded-provider set for persistence."""
        with self._lock:
            return {
                name: {
                    "profile": entry.profile,
                    "consecutive_failures": entry.consecutive_failures,
                }
                for name, entry in self._excluded.items()
            }

    def load_state(self, state: dict[str, object]) -> None:
        """Restore excluded providers from a previously saved dict.

        Re-creates ``_ExcludedProvider`` entries with ``excluded_at``
        set to the current time (the original exclude timestamp is
        lost across restarts — the important thing is that the provider
        *stays* excluded until a recovery probe succeeds).
        """
        if not isinstance(state, dict):
            return

        with self._lock:
            for name, data in state.items():
                if not isinstance(data, dict):
                    continue
                if name in self._excluded:
                    continue  # already excluded
                profile = data.get("profile", "")
                failures = data.get("consecutive_failures", 0)
                if not isinstance(failures, int):
                    failures = 0
                self._excluded[name] = _ExcludedProvider(
                    provider=name,
                    excluded_at=time.monotonic(),
                    profile=str(profile),
                    consecutive_failures=failures,
                )

    # ------------------------------------------------------------------
    # Restart helper
    # ------------------------------------------------------------------

    def try_restart(
        self,
        provider_config: ProviderConfig,
        *,
        timeout_s: float = 30.0,
    ) -> bool:
        """Attempt to restart a provider's backend process.

        Returns True if the restart command succeeded (exit code 0),
        False otherwise (or if no restart_command is configured).

        Thread-safe: only one restart per provider at a time. A
        concurrent call returns False immediately without blocking.
        """
        command = provider_config.restart_command
        if not command:
            return False

        provider = provider_config.name

        # Get or create a per-provider lock.
        with self._lock:
            if provider not in self._restart_locks:
                self._restart_locks[provider] = threading.Lock()
            restart_lock = self._restart_locks[provider]

        # Non-blocking acquire — if another thread is already
        # restarting this provider, we skip silently.
        if not restart_lock.acquire(blocking=False):
            return False

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                timeout=timeout_s,
                text=True,
            )
            success = result.returncode == 0
            error = result.stderr.strip() if not success else None
            log_self_healing_restart(
                logger,
                provider=provider,
                command=command,
                success=success,
                error=error,
            )
            return success
        except subprocess.TimeoutExpired:
            log_self_healing_restart(
                logger,
                provider=provider,
                command=command,
                success=False,
                error=f"timeout after {timeout_s}s",
            )
            return False
        except OSError as exc:
            log_self_healing_restart(
                logger,
                provider=provider,
                command=command,
                success=False,
                error=str(exc),
            )
            return False
        finally:
            restart_lock.release()


# ---------------------------------------------------------------------------
# Recovery probe loop (async, runs as a background task)
# ---------------------------------------------------------------------------


async def recovery_probe_loop(
    provider_config: ProviderConfig,
    *,
    orchestrator: SelfHealingOrchestrator,
    record_fn: object | None = None,
    health_threshold: int = 3,
    initial_interval_s: float = 30.0,
    max_interval_s: float = 300.0,
    restart_timeout_s: float = 30.0,
    probe_timeout_s: float = 10.0,
    shutdown_event: asyncio.Event | None = None,
    profile: str = "",
) -> None:
    """Probe an excluded provider with exponential backoff until recovery.

    This function runs as a long-lived asyncio task, one per excluded
    provider. It terminates when:

    - The provider recovers (probe succeeds) → restores to chain.
    - The shutdown event is set → graceful exit.
    - The provider is no longer excluded (external recovery).

    On first invocation, attempts a restart if configured, then waits
    for the initial interval before the first probe.
    """
    from coderouter.guards.continuous_probe import probe_one

    _shutdown = shutdown_event or asyncio.Event()
    provider_name = provider_config.name
    interval = initial_interval_s

    # Step 1: attempt restart if configured.
    if provider_config.restart_command:
        # Run in executor to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: orchestrator.try_restart(
                provider_config,
                timeout_s=restart_timeout_s,
            ),
        )

    # Step 2: exponential backoff recovery probes.
    while not _shutdown.is_set():
        # Check if still excluded (external code may have restored it).
        if not orchestrator.is_excluded(provider_name):
            return

        # Wait for the current interval (or shutdown).
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=interval)
            return  # shutdown signalled
        except TimeoutError:
            pass  # normal: interval elapsed

        # Still excluded? Probe.
        if not orchestrator.is_excluded(provider_name):
            return

        result = await probe_one(provider_config, timeout_s=probe_timeout_s)

        if result.success:
            # Provider is back! Restore it.
            orchestrator.on_recovered(provider_name, profile=profile)

            # Feed success into the backend health state machine
            # so it snaps back to HEALTHY.
            if record_fn is not None:
                with contextlib.suppress(Exception):
                    record_fn(  # type: ignore[operator]
                        provider_name,
                        success=True,
                        threshold=health_threshold,
                    )

            log_self_healing_recovery_probe(
                logger,
                provider=provider_name,
                success=True,
                next_interval_s=0,
                latency_ms=result.latency_ms,
            )
            return

        # Failed — exponential backoff.
        next_interval = min(interval * 2, max_interval_s)
        log_self_healing_recovery_probe(
            logger,
            provider=provider_name,
            success=False,
            next_interval_s=next_interval,
            latency_ms=result.latency_ms,
        )
        interval = next_interval


__all__ = [
    "SelfHealingOrchestrator",
    "recovery_probe_loop",
]
