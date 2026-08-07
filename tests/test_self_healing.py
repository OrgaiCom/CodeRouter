"""v2.0-J: self-healing orchestrator tests.

Four test groups:

- **Orchestrator (pure)**: exclude/restore/idempotency/reset.
- **Restart helper**: subprocess mock, timeout, double-restart prevention.
- **Recovery probe loop**: backoff, success restore, shutdown.
- **Engine integration**: exclude chain filtering, spawn trigger.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import AdapterError, ProviderCallOverrides
from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.guards.self_healing import SelfHealingOrchestrator, recovery_probe_loop
from coderouter.routing import FallbackEngine
from coderouter.translation.anthropic import (
    AnthropicRequest,
    AnthropicResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)

# ----------------------------------------------------------------------
# Group 1: Orchestrator (pure)
# ----------------------------------------------------------------------


def test_on_unhealthy_excludes_provider() -> None:
    """First call excludes the provider and returns True."""
    orch = SelfHealingOrchestrator()
    result = orch.on_unhealthy("p1", profile="default", consecutive_failures=6)
    assert result is True
    assert orch.is_excluded("p1")
    assert "p1" in orch.excluded_providers()


def test_on_unhealthy_idempotent() -> None:
    """Second call for the same provider returns False (already excluded)."""
    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=6)
    result = orch.on_unhealthy("p1", profile="default", consecutive_failures=7)
    assert result is False


def test_on_recovered_restores_provider() -> None:
    """Recovery removes provider from excluded set and returns duration."""
    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=6)
    assert orch.is_excluded("p1")

    duration = orch.on_recovered("p1", profile="default")
    assert duration is not None
    assert duration >= 0
    assert not orch.is_excluded("p1")


def test_on_recovered_returns_none_for_non_excluded() -> None:
    """Recovery on a non-excluded provider returns None."""
    orch = SelfHealingOrchestrator()
    result = orch.on_recovered("p1", profile="default")
    assert result is None


def test_excluded_providers_snapshot() -> None:
    """excluded_providers() returns a snapshot set."""
    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=6)
    orch.on_unhealthy("p2", profile="default", consecutive_failures=6)
    assert orch.excluded_providers() == {"p1", "p2"}

    orch.on_recovered("p1", profile="default")
    assert orch.excluded_providers() == {"p2"}


def test_reset_clears_all_state() -> None:
    """reset() removes all excluded providers."""
    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=6)
    orch.reset()
    assert not orch.is_excluded("p1")
    assert orch.excluded_providers() == set()


# ----------------------------------------------------------------------
# Group 2: Restart helper
# ----------------------------------------------------------------------


def test_restart_no_command_returns_false() -> None:
    """Provider without restart_command → skip (False)."""
    orch = SelfHealingOrchestrator()
    pc = ProviderConfig(
        name="p1",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
    )
    assert orch.try_restart(pc) is False


def test_restart_success() -> None:
    """Successful restart command returns True."""
    orch = SelfHealingOrchestrator()
    pc = ProviderConfig(
        name="p1",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
        restart_command="echo ok",
    )
    assert orch.try_restart(pc, timeout_s=5.0) is True


def test_restart_failure_returns_false() -> None:
    """Failed restart command (non-zero exit) returns False.

    Uses ``false`` (the coreutils binary) rather than ``exit 1``: under
    the v2.13.0 argv dispatch there is no shell, so ``exit`` — a shell
    builtin with no ``/bin/exit`` binary — would raise FileNotFoundError
    instead of exercising the "command ran and returned non-zero" path
    this test is about. ``/usr/bin/false`` execs and exits 1 for real.
    """
    orch = SelfHealingOrchestrator()
    pc = ProviderConfig(
        name="p1",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
        restart_command="false",
    )
    assert orch.try_restart(pc, timeout_s=5.0) is False


def test_restart_timeout() -> None:
    """Timeout on restart command returns False."""
    orch = SelfHealingOrchestrator()
    pc = ProviderConfig(
        name="p1",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
        restart_command="sleep 60",
    )
    assert orch.try_restart(pc, timeout_s=0.1) is False


def test_restart_double_prevention() -> None:
    """Concurrent restart attempts on the same provider — second skips."""
    import threading

    orch = SelfHealingOrchestrator()
    pc = ProviderConfig(
        name="p1",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
        restart_command="sleep 2",
    )

    results: list[bool] = []

    def restart():
        results.append(orch.try_restart(pc, timeout_s=5.0))

    t1 = threading.Thread(target=restart)
    t2 = threading.Thread(target=restart)
    t1.start()
    # Small delay so t1 grabs the lock first.
    time.sleep(0.1)
    t2.start()
    t1.join()
    t2.join()

    # One should succeed (or timeout), the other should skip (False).
    assert False in results


# ----------------------------------------------------------------------
# Group 2b: v2.13.0 argv dispatch (shell=False) hardening
# ----------------------------------------------------------------------


class _FakeCompleted:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr


def _pc(restart_command: str) -> ProviderConfig:
    return ProviderConfig(
        name="p1",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
        restart_command=restart_command,
    )


def test_restart_command_is_not_shell_interpreted(tmp_path) -> None:
    """A ``;``-chained command must not run as a shell pipeline.

    Under shell=False, ``echo x; touch <marker>`` is refused outright (it
    carries a shell metacharacter), so neither ``<marker>`` nor a file
    literally named ``<marker>;`` is ever created.
    """
    orch = SelfHealingOrchestrator()
    marker = tmp_path / "marker"
    pc = _pc(f"echo x; touch {marker}")

    assert orch.try_restart(pc, timeout_s=5.0) is False
    assert not marker.exists()
    assert not (tmp_path / f"{marker.name};").exists()
    # No stray files at all should have been produced in tmp_path.
    assert list(tmp_path.iterdir()) == []


def test_restart_command_shell_metachars_refused() -> None:
    """Shell metacharacters → refused, subprocess.run never called."""
    orch = SelfHealingOrchestrator()
    pc = _pc("pkill x && x")

    captured: dict[str, object] = {}

    def _record(_logger, **kwargs: object) -> None:
        captured.update(kwargs)

    with patch("coderouter.guards.self_healing.subprocess.run") as mock_run, patch(
        "coderouter.guards.self_healing.log_self_healing_restart", _record
    ):
        result = orch.try_restart(pc, timeout_s=5.0)

    assert result is False
    mock_run.assert_not_called()
    assert captured["success"] is False
    assert "shell syntax" in str(captured["error"])


def test_restart_command_quoted_argument_stays_one_token() -> None:
    """A quoted argument survives as a single argv token."""
    orch = SelfHealingOrchestrator()
    pc = _pc('/bin/echo "a b"')

    with patch(
        "coderouter.guards.self_healing.subprocess.run",
        return_value=_FakeCompleted(returncode=0),
    ) as mock_run:
        result = orch.try_restart(pc, timeout_s=5.0)

    assert result is True
    argv = mock_run.call_args.args[0]
    assert argv == ["/bin/echo", "a b"]
    assert mock_run.call_args.kwargs["shell"] is False


def test_restart_command_sh_c_escape_hatch_works() -> None:
    """``/bin/sh -c '...'`` is the documented escape hatch and runs for real."""
    orch = SelfHealingOrchestrator()
    pc = _pc("/bin/sh -c 'echo ok'")
    assert orch.try_restart(pc, timeout_s=5.0) is True


def test_restart_command_unbalanced_quotes_returns_false() -> None:
    """A value shlex cannot parse → False with an 'unparsable' error."""
    orch = SelfHealingOrchestrator()
    pc = _pc('a "b')

    captured: dict[str, object] = {}

    def _record(_logger, **kwargs: object) -> None:
        captured.update(kwargs)

    with patch(
        "coderouter.guards.self_healing.log_self_healing_restart", _record
    ):
        result = orch.try_restart(pc, timeout_s=5.0)

    assert result is False
    assert "unparsable" in str(captured["error"])


def test_restart_command_whitespace_only_returns_false() -> None:
    """A command that parses to an empty argv → False with an 'empty' error."""
    orch = SelfHealingOrchestrator()
    pc = _pc("   ")

    captured: dict[str, object] = {}

    def _record(_logger, **kwargs: object) -> None:
        captured.update(kwargs)

    with patch(
        "coderouter.guards.self_healing.log_self_healing_restart", _record
    ):
        result = orch.try_restart(pc, timeout_s=5.0)

    assert result is False
    assert "empty" in str(captured["error"])


def test_restart_command_missing_binary_returns_false() -> None:
    """A non-existent binary execs and fails (OSError) → False, not a crash."""
    orch = SelfHealingOrchestrator()
    pc = _pc("/nonexistent/xyz")
    assert orch.try_restart(pc, timeout_s=5.0) is False


def test_restart_lock_released_on_parse_failure() -> None:
    """A parse failure must not leave the restart lock wedged.

    The reject/parse checks run before the lock is taken, so a later valid
    restart on the same provider must still be able to proceed.
    """
    orch = SelfHealingOrchestrator()

    assert orch.try_restart(_pc('a "b'), timeout_s=5.0) is False

    with patch(
        "coderouter.guards.self_healing.subprocess.run",
        return_value=_FakeCompleted(returncode=0),
    ):
        assert orch.try_restart(_pc("echo ok"), timeout_s=5.0) is True


# ----------------------------------------------------------------------
# Group 3: Recovery probe loop
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_probe_success_restores() -> None:
    """Recovery probe succeeds → provider restored, loop exits."""
    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=6)

    pc = ProviderConfig(
        name="p1",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
    )

    record_calls: list[dict] = []

    def mock_record(provider, *, success, threshold):
        record_calls.append(
            {"provider": provider, "success": success, "threshold": threshold}
        )

    # Mock probe_one to succeed immediately.
    # probe_one is imported inside recovery_probe_loop's body, so we
    # patch the canonical location in continuous_probe.
    with patch("coderouter.guards.continuous_probe.probe_one") as mock_probe:
        from coderouter.guards.continuous_probe import ProbeResult

        mock_probe.return_value = ProbeResult(
            provider="p1", success=True, latency_ms=5.0
        )

        await recovery_probe_loop(
            pc,
            orchestrator=orch,
            record_fn=mock_record,
            health_threshold=3,
            initial_interval_s=0.1,  # fast for tests
            max_interval_s=0.5,
            probe_timeout_s=5.0,
            profile="default",
        )

    assert not orch.is_excluded("p1")
    assert any(r["success"] for r in record_calls)


@pytest.mark.asyncio
async def test_recovery_probe_shutdown() -> None:
    """Shutdown event stops the recovery loop."""
    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=6)

    pc = ProviderConfig(
        name="p1",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
    )

    shutdown = asyncio.Event()

    async def stop_soon():
        await asyncio.sleep(0.2)
        shutdown.set()

    with patch("coderouter.guards.continuous_probe.probe_one") as mock_probe:
        from coderouter.guards.continuous_probe import ProbeResult

        mock_probe.return_value = ProbeResult(
            provider="p1", success=False, latency_ms=5.0, error="connection refused"
        )

        task = asyncio.create_task(stop_soon())
        await recovery_probe_loop(
            pc,
            orchestrator=orch,
            initial_interval_s=0.1,
            max_interval_s=0.5,
            shutdown_event=shutdown,
            profile="default",
        )
        await task

    # Provider still excluded (no recovery happened).
    assert orch.is_excluded("p1")


@pytest.mark.asyncio
async def test_recovery_probe_backoff() -> None:
    """Failed probes cause interval to double (exponential backoff)."""
    orch = SelfHealingOrchestrator()
    orch.on_unhealthy("p1", profile="default", consecutive_failures=6)

    pc = ProviderConfig(
        name="p1",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
    )

    probe_count = 0
    shutdown = asyncio.Event()

    with patch("coderouter.guards.continuous_probe.probe_one") as mock_probe:
        from coderouter.guards.continuous_probe import ProbeResult

        async def fake_probe(provider_config, *, timeout_s=10.0):
            nonlocal probe_count
            probe_count += 1
            # Succeed on third probe to break the loop.
            if probe_count >= 3:
                return ProbeResult(
                    provider="p1", success=True, latency_ms=5.0
                )
            return ProbeResult(
                provider="p1", success=False, latency_ms=5.0, error="refused"
            )

        mock_probe.side_effect = fake_probe

        await recovery_probe_loop(
            pc,
            orchestrator=orch,
            initial_interval_s=0.05,
            max_interval_s=1.0,
            profile="default",
            shutdown_event=shutdown,
        )

    assert probe_count == 3
    assert not orch.is_excluded("p1")


# ----------------------------------------------------------------------
# Group 4: Engine integration — exclude chain filtering
# ----------------------------------------------------------------------


class _AlwaysFailAdapter(AnthropicAdapter):
    """Test double: every call raises a non-OOM AdapterError."""

    async def healthcheck(self) -> bool:
        return True

    async def generate_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AnthropicResponse:
        raise AdapterError(
            "500 internal server error",
            provider=self.name,
            status_code=500,
            retryable=True,
        )

    async def stream_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        if False:
            yield


class _HealthyAdapter(AnthropicAdapter):
    """Test double: returns a trivial successful response."""

    async def healthcheck(self) -> bool:
        return True

    async def generate_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AnthropicResponse:
        return AnthropicResponse(
            id="msg_healthy",
            model=self.config.model,
            content=[{"type": "text", "text": "ok"}],
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=1, output_tokens=1),
            coderouter_provider=self.name,
        )


def _make_provider(name: str) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        model="test",
    )


def _make_engine_with_exclude(
    provider_names: list[str],
    *,
    threshold: int = 2,
) -> FallbackEngine:
    """Build a FallbackEngine with backend_health_action='exclude'."""
    providers = [_make_provider(n) for n in provider_names]
    profile = FallbackChain(
        name="default",
        providers=provider_names,
        backend_health_action="exclude",
        backend_health_threshold=threshold,
    )
    config = CodeRouterConfig(
        providers=providers,
        profiles=[profile],
        default_profile="default",
    )
    return FallbackEngine(config)


def test_exclude_removes_from_chain() -> None:
    """When orchestrator has an excluded provider, _resolve_chain filters it."""
    engine = _make_engine_with_exclude(["p1", "p2"])

    # Manually exclude p1.
    engine.self_healing.on_unhealthy("p1", profile="default", consecutive_failures=6)

    # Resolve chain — p1 should be absent.
    chain = engine._resolve_chain("default")
    names = [a.name for a in chain]
    assert "p1" not in names
    assert "p2" in names


def test_exclude_all_providers_returns_empty() -> None:
    """When all providers are excluded, chain is empty (NoProvidersAvailableError)."""
    engine = _make_engine_with_exclude(["p1"])

    engine.self_healing.on_unhealthy("p1", profile="default", consecutive_failures=6)

    chain = engine._resolve_chain("default")
    assert len(chain) == 0


def test_restore_brings_provider_back() -> None:
    """After recovery, the provider reappears in the chain."""
    engine = _make_engine_with_exclude(["p1", "p2"])

    engine.self_healing.on_unhealthy("p1", profile="default", consecutive_failures=6)
    chain = engine._resolve_chain("default")
    assert "p1" not in [a.name for a in chain]

    # Restore.
    engine.self_healing.on_recovered("p1", profile="default")
    chain = engine._resolve_chain("default")
    assert "p1" in [a.name for a in chain]


@pytest.mark.asyncio
async def test_engine_triggers_exclude_on_unhealthy() -> None:
    """Engine calls self_healing.on_unhealthy when transition to UNHEALTHY + exclude."""
    engine = _make_engine_with_exclude(["p1", "p2"], threshold=2)

    # Drive p1 to UNHEALTHY (4 consecutive failures = 2x threshold).
    for _ in range(4):
        engine._observe_provider_failure(
            "p1",
            profile="default",
            exc=AdapterError("fail", provider="p1", status_code=500, retryable=True),
        )

    assert engine.self_healing.is_excluded("p1")


def test_demote_action_does_not_trigger_exclude() -> None:
    """backend_health_action='demote' should NOT trigger self-healing exclude."""
    providers = [_make_provider("p1"), _make_provider("p2")]
    profile = FallbackChain(
        name="default",
        providers=["p1", "p2"],
        backend_health_action="demote",
        backend_health_threshold=2,
    )
    config = CodeRouterConfig(
        providers=providers,
        profiles=[profile],
        default_profile="default",
    )
    engine = FallbackEngine(config)

    for _ in range(4):
        engine._observe_provider_failure(
            "p1",
            profile="default",
            exc=AdapterError("fail", provider="p1", status_code=500, retryable=True),
        )

    # p1 should be UNHEALTHY but NOT excluded (demote, not exclude).
    assert engine.backend_health.is_unhealthy("p1")
    assert not engine.self_healing.is_excluded("p1")

    # Chain should have p1 at the back (demoted), not removed.
    chain = engine._resolve_chain("default")
    names = [a.name for a in chain]
    assert "p1" in names
    assert names[-1] == "p1"
