"""Tests for CLI-session credential borrowing (v2.14.0).

The feature exists so a subscription-authenticated provider can be an
ordinary ``openai_compat`` entry instead of a ``kind: agent_cli`` island.
So the tests that matter are: the token actually reaches the outbound
header, a stale one triggers exactly one refresh, and a broken session
file degrades to "unauthenticated request" rather than a crash — because
that is what lets the fallback chain do its job.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from coderouter.config.schemas import ProviderConfig
from coderouter.credentials import (
    CredentialError,
    resolve_provider_credential,
    session_path_is_sane,
)
from coderouter.secret_redaction import clear_secrets, redact

TOKEN = "session-token-abcdefghijklmnopqrstuvwxyz"
FRESH = "fresh-token-0123456789abcdefghijklmno"


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_secrets()
    yield
    clear_secrets()


@pytest.fixture()
def home(tmp_path: Path, monkeypatch) -> Path:
    """A fake $HOME so the under-home path guard can be exercised for real."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _session(path: Path, token: str, *, expires_at: float | None = None, **extra) -> Path:
    payload: dict[str, object] = {"access_token": token, **extra}
    if expires_at is not None:
        payload["expires_at"] = expires_at
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _provider(**credential) -> ProviderConfig:
    return ProviderConfig(
        name="kimi-sub",
        kind="openai_compat",
        base_url="https://api.moonshot.cn/v1",
        model="k2",
        credential=credential,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_api_key_env_and_credential_are_mutually_exclusive() -> None:
    """Two sources on one provider is a question that must not reach a request."""
    with pytest.raises(ValueError, match="not both"):
        ProviderConfig(
            name="x",
            base_url="https://a.example/v1",
            model="m",
            api_key_env="SOME_KEY",
            credential={"source": "env", "env": "OTHER_KEY"},
        )


def test_cli_session_requires_a_path() -> None:
    with pytest.raises(ValueError, match=r"requires credential\.path"):
        ProviderConfig(
            name="x",
            base_url="https://a.example/v1",
            model="m",
            credential={"source": "cli_session"},
        )


def test_env_source_requires_a_var_name() -> None:
    with pytest.raises(ValueError, match=r"requires credential\.env"):
        ProviderConfig(
            name="x",
            base_url="https://a.example/v1",
            model="m",
            credential={"source": "env"},
        )


def test_a_session_path_outside_home_is_rejected(home: Path) -> None:
    assert session_path_is_sane(str(home / ".kimi" / "auth.json")) is True
    assert session_path_is_sane("/etc/shadow") is False


def test_refresh_command_must_be_a_list(home: Path) -> None:
    """No string form exists, so there is no shell to have to refuse."""
    with pytest.raises(ValueError):
        _provider(
            source="cli_session",
            path=str(home / "s.json"),
            refresh={"command": "grok models"},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_a_token_on_disk_is_returned(home: Path) -> None:
    _session(home / "s.json", TOKEN)
    provider = _provider(source="cli_session", path=str(home / "s.json"))
    assert resolve_provider_credential(provider) == TOKEN


def test_a_nested_field_is_reachable_by_dotted_path(home: Path) -> None:
    path = home / "s.json"
    path.write_text(json.dumps({"tokens": {"access": TOKEN}}), encoding="utf-8")
    provider = _provider(
        source="cli_session", path=str(path), field="tokens.access"
    )
    assert resolve_provider_credential(provider) == TOKEN


def test_the_resolved_token_is_registered_with_the_scrubber(home: Path) -> None:
    """A borrowed token is a credential like any other."""
    _session(home / "s.json", TOKEN)
    resolve_provider_credential(_provider(source="cli_session", path=str(home / "s.json")))
    assert TOKEN not in redact(f"upstream said {TOKEN} expired")


def test_env_source_still_works(monkeypatch) -> None:
    monkeypatch.setenv("CR_ENV_SOURCE_KEY", TOKEN)
    provider = ProviderConfig(
        name="x",
        base_url="https://a.example/v1",
        model="m",
        credential={"source": "env", "env": "CR_ENV_SOURCE_KEY"},
    )
    assert resolve_provider_credential(provider) == TOKEN


def test_a_provider_with_neither_resolves_to_none() -> None:
    provider = ProviderConfig(name="local", base_url="http://localhost:8080/v1", model="m")
    assert resolve_provider_credential(provider) is None


# ---------------------------------------------------------------------------
# Degradation — a broken session must not crash the request path
# ---------------------------------------------------------------------------


def test_a_missing_session_file_yields_none_not_an_exception(home: Path) -> None:
    """Unauthenticated + upstream 401 lets the fallback chain move on."""
    provider = _provider(source="cli_session", path=str(home / "never-written.json"))
    assert resolve_provider_credential(provider) is None


def test_malformed_json_yields_none(home: Path) -> None:
    path = home / "s.json"
    path.write_text("{not json", encoding="utf-8")
    assert resolve_provider_credential(_provider(source="cli_session", path=str(path))) is None


def test_a_non_string_token_yields_none(home: Path) -> None:
    path = home / "s.json"
    path.write_text(json.dumps({"access_token": {"nested": "oops"}}), encoding="utf-8")
    assert resolve_provider_credential(_provider(source="cli_session", path=str(path))) is None


def test_an_unknown_source_is_a_load_error_not_a_runtime_one() -> None:
    with pytest.raises(ValueError):
        _provider(source="carrier-pigeon", path="~/x.json")


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


def _spy_refresh(monkeypatch, *, rewrite: Path | None = None, token: str = FRESH):
    """Replace the subprocess call with a spy that rewrites the session file."""
    calls: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = b""
        stderr = b""

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(command))
        if rewrite is not None:
            _session(rewrite, token, expires_at=time.time() + 3600)
        return _Completed()

    monkeypatch.setattr("coderouter.credentials.subprocess.run", fake_run)
    return calls


def test_a_fresh_token_does_not_trigger_a_refresh(home: Path, monkeypatch) -> None:
    _session(home / "s.json", TOKEN, expires_at=time.time() + 86400)
    calls = _spy_refresh(monkeypatch)
    provider = _provider(
        source="cli_session",
        path=str(home / "s.json"),
        refresh={"command": ["kimi", "auth", "status"]},
    )
    assert resolve_provider_credential(provider) == TOKEN
    assert calls == []


def test_an_expiring_token_triggers_exactly_one_refresh(home: Path, monkeypatch) -> None:
    path = _session(home / "s.json", TOKEN, expires_at=time.time() + 30)
    calls = _spy_refresh(monkeypatch, rewrite=path)
    provider = _provider(
        source="cli_session",
        path=str(path),
        refresh={"command": ["kimi", "auth", "status"], "min_lead_s": 300},
    )
    assert resolve_provider_credential(provider) == FRESH
    assert calls == [["kimi", "auth", "status"]]


def test_refresh_happens_early_not_at_expiry(home: Path, monkeypatch) -> None:
    """min_lead_s is a floor: 4 minutes left with a 5-minute lead refreshes."""
    path = _session(home / "s.json", TOKEN, expires_at=time.time() + 240)
    calls = _spy_refresh(monkeypatch, rewrite=path)
    provider = _provider(
        source="cli_session",
        path=str(path),
        refresh={"command": ["grok", "models"], "min_lead_s": 300},
    )
    resolve_provider_credential(provider)
    assert len(calls) == 1


def test_a_missing_token_triggers_a_refresh(home: Path) -> None:
    """No token at all is the strongest possible signal to go get one."""
    from coderouter.credentials import _needs_refresh, _SessionRead

    assert _needs_refresh(_SessionRead(None, None), early_ratio=0.5, min_lead_s=300)


def test_no_expiry_information_means_no_refresh(home: Path, monkeypatch) -> None:
    """Without an expiry the upstream 401 is the signal, not a guess."""
    _session(home / "s.json", TOKEN)
    calls = _spy_refresh(monkeypatch)
    provider = _provider(
        source="cli_session",
        path=str(home / "s.json"),
        refresh={"command": ["kimi", "auth", "status"]},
    )
    assert resolve_provider_credential(provider) == TOKEN
    assert calls == []


def test_a_failing_refresh_still_returns_the_stale_token(home: Path, monkeypatch) -> None:
    """Stale-but-present beats nothing; the upstream error is clearer anyway."""
    _session(home / "s.json", TOKEN, expires_at=time.time() + 10)

    def boom(command, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("grok: command not found")

    monkeypatch.setattr("coderouter.credentials.subprocess.run", boom)
    provider = _provider(
        source="cli_session",
        path=str(home / "s.json"),
        refresh={"command": ["grok", "models"]},
    )
    assert resolve_provider_credential(provider) == TOKEN


def test_refresh_stderr_is_never_logged(home: Path, monkeypatch, caplog) -> None:
    """A failing auth CLI is the most likely thing to print a device code."""
    _session(home / "s.json", TOKEN, expires_at=time.time() + 10)

    class _Failed:
        returncode = 1
        stdout = b""
        stderr = b"visit https://auth.example/device?code=SUPERSECRETCODE"

    monkeypatch.setattr(
        "coderouter.credentials.subprocess.run", lambda command, **kw: _Failed()
    )
    provider = _provider(
        source="cli_session",
        path=str(home / "s.json"),
        refresh={"command": ["grok", "models"]},
    )
    with caplog.at_level("WARNING"):
        resolve_provider_credential(provider)
    assert "SUPERSECRETCODE" not in caplog.text


def test_milliseconds_and_seconds_expiries_are_both_understood() -> None:
    from coderouter.credentials import _as_epoch_seconds

    assert _as_epoch_seconds(1_800_000_000) == pytest.approx(1_800_000_000)
    assert _as_epoch_seconds(1_800_000_000_000) == pytest.approx(1_800_000_000)
    assert _as_epoch_seconds("1800000000") == pytest.approx(1_800_000_000)
    assert _as_epoch_seconds("never") is None
    assert _as_epoch_seconds(None) is None


# ---------------------------------------------------------------------------
# The point of the whole feature: it reaches the outbound header
# ---------------------------------------------------------------------------


def test_the_borrowed_token_lands_in_the_authorization_header(home: Path) -> None:
    from coderouter.adapters.openai_compat import OpenAICompatAdapter

    _session(home / "s.json", TOKEN)
    provider = _provider(source="cli_session", path=str(home / "s.json"))
    headers = OpenAICompatAdapter(provider)._headers()
    assert headers["Authorization"] == f"Bearer {TOKEN}"


def test_the_borrowed_token_lands_in_the_anthropic_header(home: Path) -> None:
    from coderouter.adapters.anthropic_native import AnthropicAdapter

    _session(home / "s.json", TOKEN)
    provider = ProviderConfig(
        name="claude-sub",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-x",
        credential={"source": "cli_session", "path": str(home / "s.json")},
    )
    headers = AnthropicAdapter(provider)._headers()
    assert headers["x-api-key"] == TOKEN


def test_credential_error_is_still_a_named_type() -> None:
    """Callers outside this module should be able to catch it specifically."""
    assert issubclass(CredentialError, RuntimeError)
