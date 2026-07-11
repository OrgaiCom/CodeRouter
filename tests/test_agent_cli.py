"""Config-schema validation tests for ``kind="agent_cli"`` providers.

Phase 2b of the agent_cli plugin extraction
(``docs/designs/agent-cli-plugin-extraction.md`` §4.4 案(b), §4.5, §7)
moved the adapter-behavior tests (argv construction, output parsing,
subprocess stubbing, env isolation, timeouts, recursion limits, and the
TestClient E2E flows — ~82 of the original 91 tests) to the
``coderouter-plugin-agents`` plugin package, alongside the
``AgentCliAdapter`` implementation itself
(``coderouter/adapters/agent_cli.py``, kept in-core only as a
backward-compatible copy through Phase 2c — see
``coderouter/adapters/registry.py``'s ``agent-cli-in-core-deprecated``
log).

What stays in Core, and stays here, is the schema contract: ``kind:
agent_cli`` and the ``agent_cli:`` sub-config (``AgentCliConfig``) are
recognized by ``ProviderConfig`` regardless of whether the adapter
plugin is installed, because Core owns config validation (fail-fast at
config-load time, ``extra="forbid"``). These 9 tests are that contract's
regression guard.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coderouter.config.schemas import AgentCliConfig, ProviderConfig


def test_schema_agent_cli_required_for_kind() -> None:
    with pytest.raises(ValidationError, match="agent_cli sub-config is required"):
        ProviderConfig(name="x", kind="agent_cli", model="opus")


def test_schema_base_url_optional_for_agent_cli() -> None:
    p = ProviderConfig(
        name="x",
        kind="agent_cli",
        model="opus",
        agent_cli=AgentCliConfig(agent="claude"),
    )
    assert p.base_url is None


def test_schema_base_url_required_for_openai_compat() -> None:
    with pytest.raises(ValidationError, match="base_url is required"):
        ProviderConfig(name="x", kind="openai_compat", model="m")


def test_schema_command_defaults_to_agent() -> None:
    cfg = AgentCliConfig(agent="claude")
    assert cfg.command == "claude"


def test_schema_command_defaults_to_agy_for_antigravity() -> None:
    # antigravity is the one agent whose binary name differs from the
    # ``agent`` value (product "Antigravity CLI", command "agy").
    cfg = AgentCliConfig(agent="antigravity")
    assert cfg.command == "agy"


def test_schema_command_still_defaults_to_agent_for_others() -> None:
    for agent in ("claude", "codex", "grok"):
        cfg = AgentCliConfig(agent=agent)  # type: ignore[arg-type]
        assert cfg.command == agent


def test_schema_antigravity_accepted_by_literal() -> None:
    cfg = AgentCliConfig(agent="antigravity")
    assert cfg.agent == "antigravity"


def test_schema_antigravity_explicit_command_not_overridden() -> None:
    cfg = AgentCliConfig(agent="antigravity", command="/opt/bin/agy")
    assert cfg.command == "/opt/bin/agy"


def test_schema_write_conflict_rejected() -> None:
    with pytest.raises(ValidationError, match="conflicts with"):
        AgentCliConfig(agent="claude", allow_file_writes=True, sandbox_mode="read_only")
