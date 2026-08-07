"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from coderouter.config.schemas import (
    Capabilities,
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.token_estimation import (
    get_include_tool_content,
    set_include_tool_content,
)


@pytest.fixture(autouse=True)
def _restore_token_estimation_scope() -> Iterator[None]:
    """Undo any test that flipped the H-5 tool-content escape hatch.

    ``token_estimation_include_tool_content`` is published into a module
    global by ``CodeRouterConfig``'s validator, so merely *constructing*
    a config with the key set to false — in any test module — silently
    changes what every later test's estimator returns. That is an
    order-dependent flake waiting to happen: a test asserting the guard
    sees tool_result content fails only when it runs after an opt-out
    test. Autouse here (rather than in one module) because the leak
    crosses module boundaries.
    """
    previous = get_include_tool_content()
    yield
    set_include_tool_content(previous)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe env vars that the loader picks up so tests are deterministic."""
    for var in (
        "ALLOW_PAID",
        "CODEROUTER_ALLOW_CWD_CONFIG",
        "CODEROUTER_CONFIG",
        "CODEROUTER_MODE",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def basic_config() -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="local",
                base_url="http://localhost:8080/v1",
                model="qwen-coder",
                paid=False,
                capabilities=Capabilities(),
            ),
            ProviderConfig(
                name="free-cloud",
                base_url="https://openrouter.ai/api/v1",
                model="qwen/qwen-2.5-coder-32b-instruct:free",
                api_key_env="OPENROUTER_API_KEY",
                paid=False,
                capabilities=Capabilities(),
            ),
            ProviderConfig(
                name="paid-cloud",
                base_url="https://openrouter.ai/api/v1",
                model="anthropic/claude-sonnet-4",
                api_key_env="OPENROUTER_API_KEY",
                paid=True,
                capabilities=Capabilities(tools=True),
            ),
        ],
        profiles=[
            FallbackChain(
                name="default",
                providers=["local", "free-cloud", "paid-cloud"],
            ),
            FallbackChain(name="free-only", providers=["local", "free-cloud"]),
        ],
    )


@pytest.fixture
def yaml_config_path(tmp_path: Path, basic_config: CodeRouterConfig) -> Path:
    """Write basic_config out as YAML and return the path."""
    file = tmp_path / "providers.yaml"
    file.write_text(
        yaml.safe_dump(
            basic_config.model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return file
