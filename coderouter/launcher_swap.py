"""Phase 1 on-demand model swap (llama-swap-equivalent, self-hosted).

See docs/designs/launcher-model-swap.md for the full design. This module
implements :class:`SwapManager` — the coordination layer that turns a
request's ``model`` name into "a launcher-managed backend is running and
registered" before dispatch, with:

- **Thundering-herd collapse**: N concurrent requests for the same
  not-yet-loaded model spawn exactly one process (U1).
- **Readiness hand-off**: waits on the launcher's own
  ``ManagedProcess.ready`` event (set by
  ``coderouter.ingress.launcher_routes._wait_ready_and_register`` once
  it reaches a terminal outcome) rather than re-implementing a health
  check (§10 Q5 — "SwapManager holds no readiness *judgment*, only
  waiting and coordination").
- **Idle TTL unload**: a background sweeper stops processes that have
  had no in-flight lease for ``ttl_seconds``, going through the exact
  same intentional-stop path as the manual Stop button (§10 Q4) so
  launcher auto-restart never fights it.
- **Lease-protected in-flight**: a model with an active request is
  never TTL-evicted, no matter how idle it looks a moment later.

Security (§7): the only thing this module can ever spawn is a model
listed in the static ``launcher.swap.models`` catalog
(:class:`~coderouter.config.schemas.SwapModelSpec`) — a request's
``model`` field is exclusively a *catalog lookup key*, never a path or
command. Spawning always goes through
:func:`coderouter.ingress.launcher_routes.spawn_process`, which in turn
resolves ``model_path`` against ``launcher.model_dirs`` via
``_resolve_within_model_dirs`` — the same traversal guard the manual
Launcher UI uses.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from coderouter.adapters.base import AdapterError
from coderouter.config.schemas import LauncherSwapConfig, ProviderConfig, SwapModelSpec
from coderouter.ingress.launcher_routes import (
    ManagedProcess,
    _registry_for_app,
    _resolve_within_model_dirs,
    spawn_process,
    stop_process,
)
from coderouter.launcher_devices import resolve_option_profiles
from coderouter.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Lease:
    """A held claim on a loaded swap model.

    Returned by :meth:`SwapManager.ensure_loaded`; the caller MUST pass
    it to :meth:`SwapManager.release_lease` exactly once — in a
    ``finally`` so it fires even on error / client-disconnect / stream
    cancellation (§6.6 known-trap #1). ``released`` guards against a
    double-release turning into a double-decrement of ``in_flight``.
    """

    model: str
    released: bool = field(default=False)


@dataclass
class _ModelState:
    """Per-model coordination state (§6.1). One instance per catalog entry."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    error: BaseException | None = None
    proc_id: str | None = None
    last_used: float = field(default_factory=time.monotonic)
    in_flight: int = 0
    status: str = "idle"  # idle | loading | ready | stopping


def _pick_ephemeral_port() -> int:
    """Ask the OS for a free port. Best-effort only (§6.6 known-trap #4):

    there is an unavoidable TOCTOU window between this call's bind+close
    returning a "free" port and the child actually binding that same
    port a moment later — nothing stops another process (or another
    concurrent swap spawn) from grabbing it first. Retrying on a fresh
    port when the child fails to come up
    (``LauncherSwapConfig.port_retry_attempts``, default 2 additional
    attempts) narrows the *impact* of losing that race but does NOT
    close the window itself — a genuinely race-free allocation would
    need the OS to hand the child an already-bound socket fd, which the
    current spawn path (argv ``--port N``) does not support. Fixed
    ``port:`` in the catalog entry avoids the race entirely (§10 Q2,
    recommended) and remains the only way to eliminate it outright.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SwapManager:
    """Coordinates on-demand spawn / readiness wait / idle TTL unload.

    One instance lives at ``app.state.swap`` and is wired into
    :class:`~coderouter.routing.fallback.FallbackEngine` via
    ``attach_swap_manager`` so the dispatch entry points
    (``generate`` / ``stream`` / ``generate_anthropic`` /
    ``stream_anthropic``) can call :meth:`ensure_loaded` before chain
    resolution.
    """

    def __init__(
        self, app: Any, config: LauncherSwapConfig, launcher_cfg: Any
    ) -> None:
        self._app = app
        self._config = config
        self._launcher_cfg = launcher_cfg
        self._catalog: dict[str, SwapModelSpec] = {m.name: m for m in config.models}
        # Review fix M-2/M-3: the dispatch hook keys on the RESOLVED
        # profile (and chain membership), not on the request's model
        # name — these indexes serve those lookups.
        self._by_profile: dict[str, SwapModelSpec] = {
            m.profile_name: m for m in config.models
        }
        self._by_provider: dict[str, SwapModelSpec] = {
            m.provider_name: m for m in config.models
        }
        self._states: dict[str, _ModelState] = {}
        self._ttl_seconds = config.ttl_seconds
        self._readiness_timeout_s = config.readiness_timeout_s
        self._sweep_interval_s = config.sweep_interval_s
        self._sweeper_task: asyncio.Task[None] | None = None
        # [Unreleased] per-model TTL override: the sweeper must run
        # whenever EITHER the global TTL is enabled OR at least one
        # catalog entry sets its own ``ttl_seconds`` (even if the global
        # value is None/disabled) — see _effective_ttl / start().
        self._any_ttl_configured = self._ttl_seconds is not None or any(
            m.ttl_seconds is not None for m in config.models
        )

    # ------------------------------------------------------------------
    # Catalog matching
    # ------------------------------------------------------------------

    def match(self, model: str) -> SwapModelSpec | None:
        """Return the catalog entry ``model`` resolves to, or ``None``.

        Exact ``name`` match first, then each entry's optional
        ``model_pattern`` (``re.fullmatch``) — mirrors
        ``SwapModelSpec.model_pattern``'s docstring.
        """
        spec = self._catalog.get(model)
        if spec is not None:
            return spec
        for spec in self._catalog.values():
            if spec.model_pattern and re.fullmatch(spec.model_pattern, model):
                return spec
        return None

    def spec_for_profile(self, profile_name: str | None) -> SwapModelSpec | None:
        """Catalog entry whose DEDICATED profile is ``profile_name``, or None.

        Review fix M-2/M-3: the dispatch hook keys on the resolved
        profile — a request routed to ``launcher-swap-<name>`` gets a
        lease no matter what its ``model`` field says (M-3), and a
        request routed anywhere else never spawns just because its
        model name happens to match the catalog (M-2).
        """
        if profile_name is None:
            return None
        return self._by_profile.get(profile_name)

    def spec_for_provider(self, provider_name: str) -> SwapModelSpec | None:
        """Catalog entry registered under ``provider_name``, or None.

        Lets the dispatch hook also serve chains that explicitly list a
        swap provider (``launcher-swap-<name>``) among other providers.
        """
        return self._by_provider.get(provider_name)

    # ------------------------------------------------------------------
    # Public coordination API (§4.2)
    # ------------------------------------------------------------------

    async def ensure_loaded(self, model: str) -> Lease:
        """Ensure ``model``'s backend is running + registered; return a lease.

        Raises ``KeyError`` if ``model`` isn't in the catalog (callers
        should check :meth:`match` first — the dispatch hook does) and
        a retryable :class:`~coderouter.adapters.base.AdapterError` if
        spawn or readiness fails (never a permanent "poison" — the next
        call retries from scratch, §6.4).
        """
        spec = self._catalog.get(model)
        if spec is None:
            spec = self.match(model)
        if spec is None:
            raise KeyError(f"{model!r} is not in the swap catalog")
        return await self._ensure_loaded_spec(spec)

    async def release_lease(self, lease: Lease) -> None:
        """Release a lease acquired by :meth:`ensure_loaded`. Idempotent."""
        if lease.released:
            return
        lease.released = True
        state = self._states.get(lease.model)
        if state is None:
            return
        async with state.lock:
            state.in_flight = max(0, state.in_flight - 1)
            state.last_used = time.monotonic()

    def touch(self, model: str) -> None:
        """Update ``last_used`` without touching the lease count."""
        state = self._states.get(model)
        if state is not None:
            state.last_used = time.monotonic()

    async def unload(self, model: str, *, reason: str) -> None:
        """Stop ``model``'s backend if loaded. Used by :meth:`sweep_once`."""
        spec = self._catalog.get(model)
        state = self._states.get(model)
        if spec is None or state is None:
            return
        async with state.lock:
            await self._unload_locked(state, spec, reason)

    def _effective_ttl(self, spec: SwapModelSpec) -> float | None:
        """``spec.ttl_seconds`` if set, else the global ``self._ttl_seconds``.

        [Unreleased] per-model TTL override (docs/backends/launcher.md
        "launcher.swap fields" table). ``None`` at either level keeps
        its usual meaning ("TTL disabled") — a spec-level override only
        takes effect when it's not ``None``.
        """
        return spec.ttl_seconds if spec.ttl_seconds is not None else self._ttl_seconds

    async def sweep_once(self) -> None:
        """One TTL sweep pass: stop every idle, un-leased, expired model.

        Takes each model's own lock independently (never more than one
        per-model lock at a time — §6.6 known-trap #7) so a slow unload
        for model A never delays the sweep of model B. Each model's TTL
        is resolved individually via :meth:`_effective_ttl` — a model
        whose global TTL is disabled can still expire on its own
        override, and vice versa.
        """
        if not self._any_ttl_configured:
            return
        now = time.monotonic()
        for spec in list(self._catalog.values()):
            ttl = self._effective_ttl(spec)
            if ttl is None:
                continue
            state = self._states.get(spec.name)
            if state is None:
                continue
            async with state.lock:
                if state.status != "ready" or state.in_flight > 0:
                    continue
                if (now - state.last_used) < ttl:
                    continue
                await self._unload_locked(state, spec, reason="ttl")

    # ------------------------------------------------------------------
    # Background sweeper lifecycle (wired from ingress/app.py's lifespan)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the TTL sweeper background task.

        No-op when TTL is disabled both globally AND on every catalog
        entry (``self._any_ttl_configured`` — [Unreleased] per-model
        override means a per-model ``ttl_seconds`` alone is enough to
        need the sweeper, even with the global TTL off).
        """
        if not self._any_ttl_configured or self._sweeper_task is not None:
            return
        self._sweeper_task = asyncio.create_task(self._sweeper_loop())

    async def stop(self) -> None:
        """Cancel the sweeper task. Does NOT stop already-loaded processes —

        ``shutdown_launcher`` (ingress/launcher_routes.py) already tears
        down every ``ManagedProcess``, swap-spawned ones included
        (§6.6 known-trap #9).
        """
        task = self._sweeper_task
        self._sweeper_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _sweeper_loop(self) -> None:
        while True:
            await asyncio.sleep(self._sweep_interval_s)
            with contextlib.suppress(Exception):
                await self.sweep_once()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _state_for(self, name: str) -> _ModelState:
        state = self._states.get(name)
        if state is None:
            state = _ModelState()
            self._states[name] = state
        return state

    def _get_proc(self, proc_id: str) -> ManagedProcess | None:
        try:
            return _registry_for_app(self._app).get(proc_id)
        except KeyError:
            return None

    def _engine(self) -> Any | None:
        return getattr(self._app.state, "engine", None)

    def _remove_from_registry(self, proc_id: str) -> None:
        """Drop a swap-managed ``ManagedProcess`` from the launcher registry.

        [Unreleased] registry-litter fix: ``stop_process`` only sets
        ``status="stopped"`` — it never removes the entry, because for
        MANUALLY started processes the stopped row doubles as visible
        history in the /launcher UI (with logs and an explicit ✕ delete
        button). Swap-managed processes have no such contract: every
        failed readiness attempt (times ``1 + port_retry_attempts``) and
        every TTL unload would otherwise leave a permanent "stopped" row
        accumulating unboundedly in ``GET /api/launcher/processes``. So
        SwapManager removes ITS OWN processes right after it stops them,
        guarded on ``swap_managed`` so a manual process can never be
        swept up by mistake. Crash leftovers (a swap process that died
        on its own, status "error"/"stopped" without SwapManager
        stopping it) are deliberately NOT removed here — their log tail
        is the only crash forensics an operator has; the ✕ button still
        applies.
        """
        reg = _registry_for_app(self._app)
        try:
            proc = reg.get(proc_id)
        except KeyError:
            return
        if not proc.swap_managed:
            return
        with contextlib.suppress(KeyError):
            reg.remove(proc_id)

    def _resolve_options(self, spec: SwapModelSpec) -> dict[str, Any]:
        """Resolve ``spec.option_profile`` into the launcher options dict.

        ``spec.backend`` may be a variant (``llama.cpp-cuda``), in which case
        the base backend's presets are inherited — the same merge the manual
        UI applies, so a swap entry can reference a shared preset without
        duplicating it under every variant key.
        """
        if not spec.option_profile or self._launcher_cfg is None:
            return {}
        profiles = resolve_option_profiles(
            self._launcher_cfg.option_profiles, spec.backend
        )
        for p in profiles:
            if p.name == spec.option_profile:
                return dict(p.args)
        return {}

    def _to_adapter_error(self, model: str, exc: BaseException | None) -> AdapterError:
        detail = str(exc) if exc is not None else "unknown error"
        return AdapterError(
            f"swap: model {model!r} failed to load: {detail}",
            provider=f"launcher-swap-{model}",
            retryable=True,
        )

    async def _ensure_loaded_spec(self, spec: SwapModelSpec) -> Lease:
        state = self._state_for(spec.name)
        async with state.lock:
            if state.status == "ready" and self._proc_running(state.proc_id):
                state.last_used = time.monotonic()
                state.in_flight += 1
                return Lease(spec.name)

            # idle (or a stale "ready" whose process died under us) ->
            # (re)spawn. Concurrent callers for the SAME model queue on
            # this very asyncio.Lock; once we release it below they
            # re-enter this method and take the fast "ready" path above,
            # so at most one spawn happens no matter how many requests
            # arrive at once (U1). This holds the lock for the *entire*
            # spawn + readiness wait — a deliberate simplification versus
            # the design's fork-only-then-Event-release sketch (§6.2);
            # see the implementation report for the rationale. It never
            # blocks any OTHER model's dispatch or sweep, since every
            # lock here is strictly per-model.
            state.status = "loading"
            state.error = None
            proc, exc = await self._spawn_with_retry(spec)
            if proc is None:
                state.status = "idle"
                state.proc_id = None
                state.error = exc
                raise self._to_adapter_error(spec.name, exc)
            state.status = "ready"
            state.proc_id = proc.id
            state.error = None
            state.last_used = time.monotonic()
            state.in_flight += 1
            return Lease(spec.name)

    def _proc_running(self, proc_id: str | None) -> bool:
        if proc_id is None:
            return False
        proc = self._get_proc(proc_id)
        return proc is not None and proc.status == "running"

    async def _spawn_with_retry(
        self, spec: SwapModelSpec
    ) -> tuple[ManagedProcess | None, Exception | None]:
        """Spawn + wait for readiness.

        §10 Q2: when ``spec.port`` is unset, up to
        ``1 + LauncherSwapConfig.port_retry_attempts`` attempts total
        (default: 1 initial + 2 retries = 3), each on a freshly picked
        ephemeral port via :func:`_pick_ephemeral_port` — see that
        function's docstring for the residual pick-then-bind TOCTOU
        window this does NOT close, only bounds the impact of. Fixed
        ports never retry — a second attempt on the same port would
        just collide again.
        """
        attempts = (
            1 if spec.port is not None else 1 + self._config.port_retry_attempts
        )
        last_exc: Exception | None = None
        for _attempt in range(attempts):
            port = spec.port if spec.port is not None else _pick_ephemeral_port()
            try:
                proc = await self._spawn(spec, port)
            except (ValueError, FileNotFoundError, OSError) as exc:
                last_exc = exc
                continue
            if await self._await_ready(proc):
                self._register(spec, port)
                return proc, None
            last_exc = RuntimeError(
                f"swap model {spec.name!r} did not become ready "
                f"(status={proc.status!r}, port={port})"
            )
            with contextlib.suppress(Exception):
                await stop_process(self._app, proc.id)
            # [Unreleased] registry-litter fix: without this, every failed
            # readiness attempt leaves one "stopped" row in the registry
            # forever (1 + port_retry_attempts rows per failed load).
            self._remove_from_registry(proc.id)
        return None, last_exc

    async def _spawn(self, spec: SwapModelSpec, port: int) -> ManagedProcess:
        model_dirs = (
            self._launcher_cfg.model_dirs if self._launcher_cfg is not None else []
        )
        # M14-style traversal guard (§7): model_path is static catalog
        # config, but re-validated at spawn time exactly like the manual
        # /api/launcher/start UI.
        resolved = _resolve_within_model_dirs(spec.model_path, model_dirs)
        return await spawn_process(
            self._app,
            self._launcher_cfg,
            name=spec.provider_name,
            backend=spec.backend,
            model_path=str(resolved),
            port=port,
            options=self._resolve_options(spec),
            extra_args=spec.extra_args,
            draft_model_path=spec.draft_model_path,
            mtp_mode=spec.mtp_mode,
            # H-1: suppress the generic 'launcher-<backend>-<port>'
            # registration (SwapManager registers/deregisters its own
            # provider). H-2: exempt from launcher auto-restart (crash
            # recovery = next-request re-spawn under the per-model lock).
            swap_managed=True,
            # [Unreleased]: catalog model name, surfaced by
            # GET /api/launcher/processes as "swap_model" so the /launcher
            # UI can show which swap catalog entry a process backs.
            swap_model=spec.name,
        )

    async def _await_ready(self, proc: ManagedProcess) -> bool:
        """Wait on the launcher's own readiness signal (§10 Q5) — no polling,
        no re-implemented health check."""
        try:
            await asyncio.wait_for(
                proc.ready.wait(), timeout=self._readiness_timeout_s
            )
        except TimeoutError:
            return False
        return proc.status == "running"

    def _register(self, spec: SwapModelSpec, port: int) -> None:
        """Register the dedicated single-model profile (§4.4 "provider同期").

        Distinct from (and additional to) the generic auto-registration
        ``_wait_ready_and_register`` already performed into the shared
        "launcher" profile under a port-based name — this is
        SwapManager's OWN registration under ``spec.provider_name`` /
        ``spec.profile_name`` so the auto-injected auto_router rule
        (§10 Q7) has somewhere correct to route.
        """
        engine = self._engine()
        if engine is None:
            return
        provider_cfg = ProviderConfig(
            name=spec.provider_name,
            base_url=f"http://localhost:{port}/v1",
            model="",
            timeout_s=120.0,
        )
        with contextlib.suppress(Exception):
            engine.register_provider(provider_cfg, profile_name=spec.profile_name)

    async def _deregister(self, spec: SwapModelSpec) -> None:
        engine = self._engine()
        if engine is None:
            return
        with contextlib.suppress(Exception):
            await engine.deregister_provider(
                spec.provider_name, profile_name=spec.profile_name
            )

    async def _unload_locked(
        self, state: _ModelState, spec: SwapModelSpec, reason: str
    ) -> None:
        """Stop + deregister. MUST be called with ``state.lock`` held.

        §10 Q4: goes through :func:`stop_process` — the same intentional-
        stop path as the manual Stop button (sets
        ``ManagedProcess.stopping = True`` before signalling) — so
        launcher auto-restart (when the operator has it on) never treats
        a TTL unload as a crash to heal.
        """
        if state.status not in ("ready", "loading"):
            return
        proc_id = state.proc_id
        state.status = "stopping"
        if proc_id is not None:
            with contextlib.suppress(Exception):
                await stop_process(self._app, proc_id)
            await self._deregister(spec)
            # [Unreleased] registry-litter fix: a TTL-unloaded swap process
            # is not "history" the way a manually stopped one is — leaving
            # it would grow the registry by one row per load/unload cycle.
            self._remove_from_registry(proc_id)
        logger.info(
            "swap-unload",
            extra={"model": spec.name, "reason": reason, "proc_id": proc_id},
        )
        state.status = "idle"
        state.proc_id = None
        state.error = None
