"""Tests for the [Unreleased]/planned v2.12.0 breaking-change warnings.

Two independent warnings, both added ahead of non-behavior-changing
deprecations so operators get a diagnostic instead of a silent break
when the changes land:

  * ``coderouter/config/loader.py``: implicit CWD ``providers.yaml``
    discovery is slated to become opt-in
    (``CODEROUTER_ALLOW_CWD_CONFIG``).
  * ``coderouter/config/schemas.py``: ``ProviderConfig.restart_command``
    is currently dispatched via ``subprocess.run(shell=True)`` and is
    slated to switch to ``shlex.split`` + ``shell=False``.

Neither change is implemented here — these tests only cover the warning.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import coderouter.config.loader as loader_module
from coderouter.config.loader import load_config
from coderouter.config.schemas import (
    Capabilities,
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)


@pytest.fixture(autouse=True)
def _reset_cwd_warning_once_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the process-wide "already warned" guard before each test.

    ``_warn_if_cwd_config`` intentionally emits its warning only once per
    process (see loader.py) — a real deployment loads config once at
    startup. Tests, however, run many independent scenarios in one pytest
    process, so each test needs its own clean slate to observe (or not
    observe) the warning on its own terms.
    """
    monkeypatch.setattr(loader_module, "_cwd_config_warning_emitted", False)


def _minimal_config_yaml(**provider_overrides: object) -> str:
    provider_kwargs: dict[str, object] = dict(
        name="local",
        base_url="http://localhost:8080/v1",
        model="qwen-coder",
        paid=False,
        capabilities=Capabilities(),
    )
    provider_kwargs.update(provider_overrides)
    cfg = CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[ProviderConfig(**provider_kwargs)],
        profiles=[FallbackChain(name="default", providers=["local"])],
    )
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)


# ---------------------------------------------------------------------------
# loader.py: implicit CWD providers.yaml discovery
# ---------------------------------------------------------------------------


def test_cwd_config_emits_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    (cwd_dir / "providers.yaml").write_text(_minimal_config_yaml(), encoding="utf-8")
    monkeypatch.chdir(cwd_dir)

    with caplog.at_level("WARNING", logger="coderouter.config.loader"):
        cfg = load_config(None)

    assert isinstance(cfg, CodeRouterConfig)
    warnings = [r for r in caplog.records if r.message == "cwd-config-loaded"]
    assert len(warnings) == 1
    assert str(cwd_dir / "providers.yaml") in warnings[0].path  # type: ignore[attr-defined]


def test_cwd_warning_emitted_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    (cwd_dir / "providers.yaml").write_text(_minimal_config_yaml(), encoding="utf-8")
    monkeypatch.chdir(cwd_dir)

    with caplog.at_level("WARNING", logger="coderouter.config.loader"):
        load_config(None)
        load_config(None)

    warnings = [r for r in caplog.records if r.message == "cwd-config-loaded"]
    assert len(warnings) == 1


def test_explicit_config_path_does_not_warn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    # The explicit path happens to be a file named providers.yaml sitting in
    # CWD — a coincidence, not implicit discovery — and must stay quiet.
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    explicit_path = cwd_dir / "providers.yaml"
    explicit_path.write_text(_minimal_config_yaml(), encoding="utf-8")
    monkeypatch.chdir(cwd_dir)

    with caplog.at_level("WARNING", logger="coderouter.config.loader"):
        cfg = load_config(explicit_path)

    assert isinstance(cfg, CodeRouterConfig)
    warnings = [r for r in caplog.records if r.message == "cwd-config-loaded"]
    assert warnings == []


def test_explicit_env_config_path_does_not_warn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    config_path = cwd_dir / "providers.yaml"
    config_path.write_text(_minimal_config_yaml(), encoding="utf-8")
    monkeypatch.chdir(cwd_dir)
    monkeypatch.setenv("CODEROUTER_CONFIG", str(config_path))

    with caplog.at_level("WARNING", logger="coderouter.config.loader"):
        cfg = load_config(None)

    assert isinstance(cfg, CodeRouterConfig)
    warnings = [r for r in caplog.records if r.message == "cwd-config-loaded"]
    assert warnings == []


def test_home_config_does_not_warn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    coderouter_dir = home_dir / ".coderouter"
    coderouter_dir.mkdir()
    (coderouter_dir / "providers.yaml").write_text(_minimal_config_yaml(), encoding="utf-8")

    # CWD has no providers.yaml of its own, so the search must fall through
    # to ~/.coderouter/providers.yaml.
    cwd_dir = tmp_path / "cwd_without_config"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    with caplog.at_level("WARNING", logger="coderouter.config.loader"):
        cfg = load_config(None)

    assert isinstance(cfg, CodeRouterConfig)
    warnings = [r for r in caplog.records if r.message == "cwd-config-loaded"]
    assert warnings == []


# ---------------------------------------------------------------------------
# schemas.py: ProviderConfig.restart_command shell-syntax warning
# ---------------------------------------------------------------------------


def test_restart_command_with_shell_metacharacters_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="coderouter.config.schemas"):
        ProviderConfig(
            name="local",
            base_url="http://localhost:8080/v1",
            model="qwen-coder",
            restart_command="pkill ollama && ollama serve",
        )

    warnings = [r for r in caplog.records if r.message == "restart-command-shell-syntax"]
    assert len(warnings) == 1
    assert warnings[0].provider == "local"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "restart_command",
    [
        "~/bin/restart.sh",
        "OLLAMA_HOST=0.0.0.0 ollama serve",
        "pkill x && x",
        "a | b",
        "a; b",
        "a > out.log",
    ],
)
def test_restart_command_problematic_forms_warn(
    restart_command: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="coderouter.config.schemas"):
        ProviderConfig(
            name="local",
            base_url="http://localhost:8080/v1",
            model="qwen-coder",
            restart_command=restart_command,
        )
    warnings = [r for r in caplog.records if r.message == "restart-command-shell-syntax"]
    assert len(warnings) == 1


def test_restart_command_plain_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="coderouter.config.schemas"):
        ProviderConfig(
            name="local",
            base_url="http://localhost:8080/v1",
            model="qwen-coder",
            restart_command="ollama serve",
        )
    warnings = [r for r in caplog.records if r.message == "restart-command-shell-syntax"]
    assert warnings == []


def test_restart_command_unset_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="coderouter.config.schemas"):
        ProviderConfig(
            name="local",
            base_url="http://localhost:8080/v1",
            model="qwen-coder",
        )
    warnings = [r for r in caplog.records if r.message == "restart-command-shell-syntax"]
    assert warnings == []


def test_restart_command_warning_does_not_raise() -> None:
    # Must not raise even though the value would misbehave under the
    # planned shell=False dispatch — this validator only warns.
    cfg = ProviderConfig(
        name="local",
        base_url="http://localhost:8080/v1",
        model="qwen-coder",
        restart_command="pkill ollama && ollama serve",
    )
    assert cfg.restart_command == "pkill ollama && ollama serve"
