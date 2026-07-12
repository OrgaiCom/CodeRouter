"""Regression tests from the adversarial review of launcher swap Phase 1.

Adopted from /tmp/review_tests/{test_review_findings.py,test_review_e2e.py}
with the assertion direction REVERSED: the originals passed by reproducing
the bugs; these pass only when the fixes hold.

Findings covered (see the review for full write-ups):

* C-1 (CRITICAL) — enabling swap under ``default_profile: auto`` with no
  explicit ``auto_router:`` block must MERGE the bundled ruleset into the
  synthesized block, not silently replace it.
* H-1 (HIGH)     — a swap-spawned backend must not leak a generic
  ``launcher-<backend>-<port>`` provider (+ adapter) that TTL unload
  never cleans up.
* H-2 (HIGH)     — launcher auto-restart must never double-supervise a
  swap-managed process (port fight with SwapManager's re-spawn).
* M-1 (MEDIUM)   — ``FallbackChain.providers`` keeps ``min_length=1``
  for user-declared chains (swap placeholders bypass via
  ``model_construct``).
* M-2 (MEDIUM)   — a request whose model name matches the catalog but
  routes to a different profile must not spawn.
* M-3 (MEDIUM)   — a request routed to a swap dedicated profile must
  hold a lease even when its model name doesn't match the catalog.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    LauncherBackendConfig,
    LauncherConfig,
    LauncherSwapConfig,
    ProviderConfig,
    SwapModelSpec,
)
from coderouter.ingress.launcher_routes import ManagedProcess, _attempt_restart
from coderouter.routing.auto_router import classify

# Reuse the stub backend + client helpers from the main swap test module
# (same directory, imported as a plain module by pytest's rootdir insert).
from tests.test_launcher_swap import (
    _SWAP_BACKEND_BODY,
    _client_with_config,
    _poll,
    _write_script,
)

# pyproject.toml sets asyncio_mode = "auto" (pytest-asyncio).


# ---------------------------------------------------------------------------
# Shared fixtures (mirrors the review reproduction helpers)
# ---------------------------------------------------------------------------


def _spec(**ov: Any) -> SwapModelSpec:
    base: dict[str, Any] = dict(
        name="qwen-coder-14b", backend="llama.cpp",
        model_path="/tmp/m1.gguf", port=19400,
    )
    base.update(ov)
    return SwapModelSpec(**base)


def _bundled_profiles() -> list[FallbackChain]:
    # multi/coding/writing must exist for default_profile: auto (bundled).
    return [
        FallbackChain(name="multi", providers=["local"]),
        FallbackChain(name="coding", providers=["local"]),
        FallbackChain(name="writing", providers=["local"]),
    ]


def _providers() -> list[ProviderConfig]:
    return [ProviderConfig(name="local", base_url="http://localhost:8080/v1", model="q")]


CODE_BODY = {
    "messages": [
        {"role": "user", "content": "```python\n" + "x=1\n" * 40 + "```"}
    ]
}
IMAGE_BODY = {
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]}
    ]
}
SWAP_MODEL_BODY = {
    "model": "qwen-coder-14b",
    "messages": [{"role": "user", "content": "hi"}],
}


def _cfg(script: Path, *, port: int, ttl_seconds: float | None) -> CodeRouterConfig:
    swap_cfg = LauncherSwapConfig(
        enabled=True,
        ttl_seconds=ttl_seconds,
        readiness_timeout_s=5.0,
        sweep_interval_s=1.0,
        models=[SwapModelSpec(
            name="swap-model", backend="llama.cpp",
            model_path="/tmp/does-not-need-to-exist.gguf", port=port, mtp_mode="off",
        )],
    )
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[ProviderConfig(name="local", base_url="http://localhost:8080/v1", model="q")],
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            model_dirs=["/tmp"],
            backends={"llama.cpp": LauncherBackendConfig(binary=str(script))},
            readiness_timeout_s=5.0,
            readiness_poll_interval_s=0.2,
            swap=swap_cfg,
        ),
    )


# ---------------------------------------------------------------------------
# C-1: bundled auto_router must survive swap enablement
# ---------------------------------------------------------------------------


def test_baseline_bundled_routes_code_to_coding_without_swap() -> None:
    """Control: no swap -> bundled rules route a code-fence request to 'coding'."""
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="auto",
        providers=_providers(),
        profiles=_bundled_profiles(),
    )
    assert cfg.auto_router is None  # bundled path
    assert classify(CODE_BODY, cfg) == "coding"
    assert classify(IMAGE_BODY, cfg) == "multi"


def test_c1_enabling_swap_preserves_bundled_auto_router() -> None:
    """C-1 FIXED: enabling launcher.swap with default_profile=auto and NO
    explicit auto_router block synthesizes a block carrying the BUNDLED
    rules plus the swap rules — bundled classification keeps working, and
    the swap model routes to its dedicated profile.
    """
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="auto",
        providers=_providers(),
        profiles=_bundled_profiles(),
        launcher=LauncherConfig(
            model_dirs=["/tmp"],
            swap=LauncherSwapConfig(enabled=True, models=[_spec()]),
        ),
    )
    assert cfg.auto_router is not None
    ids = [r.id for r in cfg.auto_router.rules]
    # Swap rules first (coordinator-reviewed order): an exact model-name
    # match is a stronger signal than the bundled content heuristics and,
    # being exact, cannot affect requests that don't name a swap model.
    assert ids == [
        "swap:qwen-coder-14b",
        "builtin:image-attachment",
        "builtin:code-fence-dense",
    ]
    # Bundled fallthrough preserved too.
    assert cfg.auto_router.default_rule_profile == "writing"

    # Bundled classification is intact...
    assert classify(CODE_BODY, cfg) == "coding"
    assert classify(IMAGE_BODY, cfg) == "multi"
    # ...and the swap model still resolves to its dedicated profile.
    assert classify(SWAP_MODEL_BODY, cfg) == "launcher-swap-qwen-coder-14b"


def test_c1_swap_model_name_beats_code_fence_heuristic() -> None:
    """Swap-first ordering: a request that explicitly names a swap model
    routes to the swap profile even when its body is code-fence-dense —
    the exact-match rule outranks the bundled content heuristic (which
    would otherwise hijack the request to 'coding')."""
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="auto",
        providers=_providers(),
        profiles=_bundled_profiles(),
        launcher=LauncherConfig(
            model_dirs=["/tmp"],
            swap=LauncherSwapConfig(enabled=True, models=[_spec()]),
        ),
    )
    dense_code_with_swap_model = {
        "model": "qwen-coder-14b",
        "messages": CODE_BODY["messages"],
    }
    # Control: the same body WITHOUT the swap model name still classifies
    # as code (the bundled heuristic is alive, not shadowed).
    assert classify(CODE_BODY, cfg) == "coding"
    # Explicitly-named swap model wins over the code-fence heuristic.
    assert classify(dense_code_with_swap_model, cfg) == "launcher-swap-qwen-coder-14b"


def test_c1_user_auto_router_still_appended_not_merged_with_bundled() -> None:
    """When the user DID declare auto_router, behavior is unchanged from
    Phase 1: swap rules are appended after the user's rules; the bundled
    set is not resurrected (an explicit block replaces bundled — the
    pre-existing documented semantics)."""
    from coderouter.config.schemas import AutoRouterConfig, AutoRouteRule, RuleMatcher

    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="auto",
        providers=_providers(),
        profiles=_bundled_profiles(),
        auto_router=AutoRouterConfig(
            rules=[AutoRouteRule(
                id="user:mine", profile="coding", match=RuleMatcher(has_tools=True),
            )],
            default_rule_profile="writing",
        ),
        launcher=LauncherConfig(
            model_dirs=["/tmp"],
            swap=LauncherSwapConfig(enabled=True, models=[_spec()]),
        ),
    )
    assert [r.id for r in cfg.auto_router.rules] == [
        "user:mine", "swap:qwen-coder-14b",
    ]


def test_swap_disabled_zero_impact_on_auto_router() -> None:
    """Control: swap disabled/absent -> auto_router untouched."""
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="auto",
        providers=_providers(),
        profiles=_bundled_profiles(),
        launcher=LauncherConfig(
            model_dirs=["/tmp"],
            swap=LauncherSwapConfig(enabled=False, models=[_spec()]),
        ),
    )
    assert cfg.auto_router is None
    assert classify(CODE_BODY, cfg) == "coding"


# ---------------------------------------------------------------------------
# M-1: empty user-declared chains fail fast again
# ---------------------------------------------------------------------------


def test_m1_empty_user_profile_chain_fails_at_load() -> None:
    """M-1 FIXED: min_length=1 is back — a user typo that empties a chain
    fails at construction, not with a 502 at request time. Swap
    placeholder profiles bypass this via model_construct only."""
    with pytest.raises(ValidationError):
        FallbackChain(name="oops", providers=[])

    # The same empty chain fed to a whole config (as raw dicts, the YAML
    # loading shape) fails at load too.
    with pytest.raises(ValidationError):
        CodeRouterConfig.model_validate({
            "allow_paid": False,
            "default_profile": "oops",
            "providers": [{
                "name": "local",
                "base_url": "http://localhost:8080/v1",
                "model": "q",
            }],
            "profiles": [{"name": "oops", "providers": []}],
        })


def test_m1_swap_placeholder_profile_still_injected_empty() -> None:
    """The swap placeholder (the one legitimate empty chain) still works."""
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=_providers(),
        profiles=[FallbackChain(name="default", providers=["local"])],
        launcher=LauncherConfig(
            model_dirs=["/tmp"],
            swap=LauncherSwapConfig(enabled=True, models=[_spec()]),
        ),
    )
    assert cfg.profile_by_name("launcher-swap-qwen-coder-14b").providers == []


# ---------------------------------------------------------------------------
# H-2: auto-restart never touches swap-managed processes
# ---------------------------------------------------------------------------


async def test_h2_attempt_restart_skips_swap_managed_process() -> None:
    """H-2 FIXED: even with auto_restart enabled and budget available,
    a swap-managed process is never respawned by the launcher — crash
    recovery is SwapManager's next-request re-spawn, single-supervisor."""
    proc = ManagedProcess(
        id="r1", name="launcher-swap-m1", backend="llama.cpp",
        model_path="/m.gguf", port=19405, options={}, extra_args="",
        status="error", cmd=["fake-server", "--port", "19405"],
        swap_managed=True,
    )
    cfg = SimpleNamespace(
        auto_restart=True,
        auto_restart_max_attempts=3,
        auto_restart_backoff_s=0.001,
        auto_restart_backoff_max_s=0.001,
    )
    assert await _attempt_restart(proc, cfg) is False
    assert proc.restart_count == 0
    assert not any("auto-restart" in ln for ln in proc.log_tail)


async def test_h2_attempt_restart_still_works_for_manual_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the H-2 guard must not break auto-restart for manual procs."""

    class _FakeRestartProc:
        def __init__(self) -> None:
            self.pid = 4242
            self.stdout = None
            self.stderr = None
            self.returncode = None

        async def wait(self) -> int:
            return 0

    async def _fake_exec(*_a: object, **_k: object) -> _FakeRestartProc:
        return _FakeRestartProc()

    monkeypatch.setattr(
        "coderouter.ingress.launcher_routes.asyncio.create_subprocess_exec",
        _fake_exec,
    )
    proc = ManagedProcess(
        id="r2", name="manual", backend="llama.cpp",
        model_path="/m.gguf", port=19406, options={}, extra_args="",
        status="error", cmd=["fake-server", "--port", "19406"],
        swap_managed=False,
    )
    cfg = SimpleNamespace(
        auto_restart=True,
        auto_restart_max_attempts=3,
        auto_restart_backoff_s=0.001,
        auto_restart_backoff_max_s=0.001,
    )
    assert await _attempt_restart(proc, cfg) is True
    assert proc.restart_count == 1


# ---------------------------------------------------------------------------
# L-1: model_path outside model_dirs fails at load
# ---------------------------------------------------------------------------


def test_l1_swap_model_path_outside_model_dirs_fails_at_load() -> None:
    with pytest.raises(ValidationError, match="not under any configured"):
        LauncherConfig(
            model_dirs=["/tmp/models-only"],
            swap=LauncherSwapConfig(
                enabled=True, models=[_spec(model_path="/etc/passwd")],
            ),
        )


def test_l1_swap_without_model_dirs_fails_at_load() -> None:
    with pytest.raises(ValidationError, match="model_dirs is empty"):
        LauncherConfig(
            model_dirs=[],
            swap=LauncherSwapConfig(enabled=True, models=[_spec()]),
        )


def test_l1_disabled_swap_skips_model_path_check() -> None:
    cfg = LauncherConfig(
        model_dirs=[],
        swap=LauncherSwapConfig(enabled=False, models=[_spec(model_path="/etc/passwd")]),
    )
    assert cfg.swap is not None  # loads clean — disabled block is zero-impact


# ---------------------------------------------------------------------------
# H-1 / M-2 / M-3: e2e with a real stub backend
# ---------------------------------------------------------------------------


def test_h1_ttl_unload_leaves_no_generic_launcher_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-1 FIXED: a swap-spawned backend never registers the generic
    'launcher-<backend>-<port>' provider, so TTL unload leaves neither a
    dead-port provider nor a stale 'launcher' profile entry behind."""
    port = 19410
    script = _write_script(tmp_path / "swap-backend", _SWAP_BACKEND_BODY)
    cfg = _cfg(script, port=port, ttl_seconds=0.0)
    with _client_with_config(cfg, monkeypatch) as tc:
        engine = tc.app.state.engine
        resp = tc.post("/v1/chat/completions", json={
            "model": "swap-model",
            "messages": [{"role": "user", "content": "hi"}],
            "profile": "launcher-swap-swap-model",
        })
        assert resp.status_code == 200, resp.text

        generic_name = f"launcher-llamacpp-{port}"
        # The generic provider must never appear — registration suppressed
        # at spawn (ManagedProcess.swap_managed), not cleaned up later.
        assert not any(p.name == generic_name for p in engine.config.providers)

        # Wait for TTL sweep to unload the swap process.
        def _unloaded() -> bool:
            procs = tc.get("/api/launcher/processes").json()["processes"]
            e = next((p for p in procs if p["name"] == "launcher-swap-swap-model"), None)
            return e is not None and e["status"] == "stopped"
        assert _poll(_unloaded)

        # After unload: swap provider deregistered, generic never existed,
        # no 'launcher' profile pollution, no leaked adapter.
        assert not any(
            p.name == "launcher-swap-swap-model" for p in engine.config.providers
        )
        assert not any(p.name == generic_name for p in engine.config.providers)
        assert generic_name not in engine._adapters
        with contextlib.suppress(KeyError):
            launcher_chain = engine.config.profile_by_name("launcher").providers
            assert generic_name not in launcher_chain


def test_m2_model_name_match_does_not_spawn_when_routing_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-2 FIXED: a request that names a swap model but explicitly routes
    to a different profile must NOT spawn the backend."""
    port = 19411
    script = _write_script(tmp_path / "swap-backend", _SWAP_BACKEND_BODY)
    cfg = _cfg(script, port=port, ttl_seconds=None)
    with _client_with_config(cfg, monkeypatch) as tc:
        resp = tc.post("/v1/chat/completions", json={
            "model": "swap-model",
            "messages": [{"role": "user", "content": "hi"}],
            "profile": "default",
        })
        # 'local' provider is not a real server -> 502; the key assertion
        # is that no swap backend was spawned for a default-profile request.
        assert resp.status_code == 502
        time.sleep(0.3)  # generous window for a wrongly-firing spawn
        procs = tc.get("/api/launcher/processes").json()["processes"]
        assert not any(p["name"] == "launcher-swap-swap-model" for p in procs), (
            "swap backend was spawned even though the request routed to the "
            "'default' profile"
        )


def test_m3_swap_profile_with_alien_model_name_gets_lease_and_responds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-3 FIXED: a request routed to the swap dedicated profile acquires
    the lease (and spawns) even when its model field doesn't match the
    catalog — previously it dispatched lease-less and could be TTL-evicted
    mid-flight."""
    port = 19412
    script = _write_script(tmp_path / "swap-backend", _SWAP_BACKEND_BODY)
    # Hostile TTL: 0 seconds + 1s sweep — only the lease protects the call.
    cfg = _cfg(script, port=port, ttl_seconds=0.0)
    with _client_with_config(cfg, monkeypatch) as tc:
        swap_manager = tc.app.state.swap
        resp = tc.post("/v1/chat/completions", json={
            "model": "an-alias-not-in-the-catalog",
            "messages": [{"role": "user", "content": "hi"}],
            "profile": "launcher-swap-swap-model",
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "hi from swap"
        # The lease cycle actually ran: in_flight came back to 0.
        state = swap_manager._states["swap-model"]
        assert state.in_flight == 0
