"""Tests for the credential registry + log redaction (v2.14.0).

The point of these tests is not that the functions exist — it is that a
registered credential cannot survive a trip through the logging stack.
So the core cases drive real ``logging`` objects (a LogRecord through a
real Filter, a real handler writing to a real file) rather than calling
:func:`redact` directly, which would prove nothing about the wiring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from coderouter.config.loader import resolve_api_key
from coderouter.logging import JsonLineFormatter, configure_logging
from coderouter.secret_redaction import (
    SecretRedactingFilter,
    check_secret_hygiene,
    clear_secrets,
    exit_code_for_secret_report,
    install_secret_filter,
    redact,
    register_config_secrets,
    register_secret,
    registered_labels,
    scan_logs_for_secrets,
    self_test,
)

SECRET = "sk-test-abcdefghijklmnopqrstuvwxyz012345"


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Every test starts and ends with an empty process-global registry."""
    clear_secrets()
    yield
    clear_secrets()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_register_secret_accepts_a_realistic_key() -> None:
    assert register_secret(SECRET, "TEST_KEY") is True
    assert registered_labels() == ("TEST_KEY",)


@pytest.mark.parametrize("value", ["", "   ", "short", "1234567", None, 12345])
def test_register_secret_rejects_short_or_non_string_values(value: object) -> None:
    """The length floor is what stops the scrubber eating ordinary prose."""
    assert register_secret(value, "TEST_KEY") is False  # type: ignore[arg-type]
    assert registered_labels() == ()


def test_register_secret_is_idempotent_and_keeps_the_first_label() -> None:
    register_secret(SECRET, "FIRST_LABEL")
    register_secret(SECRET, "SECOND_LABEL")
    assert registered_labels() == ("FIRST_LABEL",)


def test_resolve_api_key_registers_what_it_hands_out(monkeypatch) -> None:
    """The resolver is the choke point — every key must arm the scrubber."""
    monkeypatch.setenv("CR_TEST_KEY", SECRET)
    assert resolve_api_key("CR_TEST_KEY") == SECRET
    assert "CR_TEST_KEY" in registered_labels()
    assert redact(f"upstream rejected {SECRET}") == "upstream rejected [redacted:CR_TEST_KEY]"


def test_resolve_api_key_registers_nothing_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("CR_TEST_KEY", raising=False)
    assert resolve_api_key("CR_TEST_KEY") is None
    assert registered_labels() == ()


# ---------------------------------------------------------------------------
# redact()
# ---------------------------------------------------------------------------


def test_redact_replaces_registered_value_everywhere_in_the_string() -> None:
    register_secret(SECRET, "K")
    out = redact(f"{SECRET} then again {SECRET}")
    assert SECRET not in out
    assert out.count("[redacted:K]") == 2


def test_redact_leaves_ordinary_text_untouched() -> None:
    register_secret(SECRET, "K")
    prose = "provider ollama returned 200 in 1234ms for model qwen3:8b"
    assert redact(prose) == prose


@pytest.mark.parametrize(
    ("raw", "must_not_contain"),
    [
        ("key=sk-proj-AAAAAAAAAAAAAAAAAAAA", "sk-proj-AAAAAAAAAAAAAAAAAAAA"),
        ("token ghp_AAAAAAAAAAAAAAAAAAAA", "ghp_AAAAAAAAAAAAAAAAAAAA"),
        ("AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAA", "AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAA"),
        ("Authorization: Bearer abcdefghijklmnopqrst", "abcdefghijklmnopqrst"),
        ("https://h/v1?api_key=abcdefghijkl&z=1", "abcdefghijkl"),
        ("https://user:hunter2hunter2@proxy/v1", "hunter2hunter2"),
    ],
)
def test_backstop_patterns_catch_unregistered_credentials(
    raw: str, must_not_contain: str
) -> None:
    """A key that never passed through our resolver still gets masked."""
    assert must_not_contain not in redact(raw)


def test_redact_is_a_fixed_point() -> None:
    """Re-scrubbing an already-scrubbed line must not corrupt it further."""
    register_secret(SECRET, "K")
    once = redact(f"value {SECRET}")
    assert redact(once) == once


# ---------------------------------------------------------------------------
# The filter, driven through real logging objects
# ---------------------------------------------------------------------------


def _record(msg: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="coderouter.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_filter_scrubs_the_message() -> None:
    register_secret(SECRET, "K")
    record = _record(f"calling with {SECRET}")
    assert SecretRedactingFilter().filter(record) is True
    assert SECRET not in str(record.msg)


def test_filter_scrubs_nested_extra_payloads() -> None:
    """``extra={...}`` is where this codebase puts its structured detail."""
    register_secret(SECRET, "K")
    record = _record(
        "provider-failed",
        provider="cloud",
        detail={"headers": {"authorization": f"Bearer {SECRET}"}},
        chain=[f"tried {SECRET}", "then local"],
    )
    SecretRedactingFilter().filter(record)
    assert SECRET not in json.dumps(record.detail)  # type: ignore[attr-defined]
    assert SECRET not in json.dumps(record.chain)  # type: ignore[attr-defined]


def test_filter_scrubs_printf_style_args() -> None:
    register_secret(SECRET, "K")
    record = logging.LogRecord(
        name="coderouter.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="upstream rejected: %s",
        args=(f"bad key {SECRET}",),
        exc_info=None,
    )
    SecretRedactingFilter().filter(record)
    assert SECRET not in record.getMessage()


def test_filter_never_drops_a_record() -> None:
    """This is plumbing, not a gate — a clean record still passes."""
    assert SecretRedactingFilter().filter(_record("nothing to see")) is True


def test_install_secret_filter_is_idempotent() -> None:
    handler = logging.StreamHandler()
    install_secret_filter(handler)
    install_secret_filter(handler)
    assert sum(isinstance(f, SecretRedactingFilter) for f in handler.filters) == 1


def test_self_test_detects_a_working_filter() -> None:
    assert self_test() is True


# ---------------------------------------------------------------------------
# End-to-end: a secret must not reach a formatted log line or a JSONL sink
# ---------------------------------------------------------------------------


def test_configured_stderr_handler_scrubs_the_rendered_line(capsys) -> None:
    """configure_logging() must install the filter, not just the formatter."""
    register_secret(SECRET, "K")
    configure_logging("INFO")
    logging.getLogger("coderouter.test").info(
        "provider-failed", extra={"body": f"invalid key {SECRET}"}
    )
    err = capsys.readouterr().err
    assert SECRET not in err
    assert "[redacted:K]" in err


def test_request_log_handler_writes_a_scrubbed_line(tmp_path: Path) -> None:
    """The JSONL sink persists, so an unscrubbed key there outlives the run."""
    from coderouter.state.request_log import RequestLogHandler

    register_secret(SECRET, "K")
    handler = RequestLogHandler(tmp_path / "requests.jsonl", flush_every_n=1)
    try:
        record = _record(
            "request-completed",
            provider="cloud",
            model="claude-x",
            detail=f"auth header was {SECRET}",
        )
        # Route through the handler exactly as logging would.
        for f in handler.filters:
            f.filter(record)
        handler.emit(record)
        handler.flush()
    finally:
        handler.close()
    written = (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    assert SECRET not in written


def test_json_formatter_output_is_scrubbed_after_the_filter() -> None:
    register_secret(SECRET, "K")
    record = _record("evt", provider=f"weird-{SECRET}")
    SecretRedactingFilter().filter(record)
    payload = json.loads(JsonLineFormatter().format(record))
    assert SECRET not in json.dumps(payload)


# ---------------------------------------------------------------------------
# scan_logs_for_secrets / check_secret_hygiene
# ---------------------------------------------------------------------------


def test_scan_finds_a_secret_written_before_redaction_existed(tmp_path: Path) -> None:
    register_secret(SECRET, "OLD_KEY")
    (tmp_path / "requests.jsonl").write_text(
        f'{{"msg":"ok"}}\n{{"msg":"bad {SECRET}"}}\n', encoding="utf-8"
    )
    hits = scan_logs_for_secrets(tmp_path)
    assert [(p.name, label, line) for p, label, line in hits] == [
        ("requests.jsonl", "OLD_KEY", 2)
    ]


def test_scan_also_covers_the_rotated_backup(tmp_path: Path) -> None:
    """Rotation keeps a .1 sibling — a leak there is just as exposed."""
    register_secret(SECRET, "OLD_KEY")
    (tmp_path / "audit.jsonl.1").write_text(f"{SECRET}\n", encoding="utf-8")
    assert [p.name for p, _, _ in scan_logs_for_secrets(tmp_path)] == ["audit.jsonl.1"]


def test_scan_returns_nothing_when_no_secret_is_registered(tmp_path: Path) -> None:
    (tmp_path / "requests.jsonl").write_text(f"{SECRET}\n", encoding="utf-8")
    assert scan_logs_for_secrets(tmp_path) == []


class _Provider:
    """Minimal duck-typed stand-in — the suite only reads three fields."""

    def __init__(self, name: str, base_url: str, api_key_env: str | None = None) -> None:
        self.name = name
        self.base_url = base_url
        self.api_key_env = api_key_env


class _Config:
    """Duck-typed config: ``check_secret_hygiene`` uses getattr throughout."""

    def __init__(self, providers: list[_Provider], state_dir: str | None = None) -> None:
        self.providers = providers
        self.state_dir = state_dir


def _config(providers: list[_Provider], state_dir: str | None = None) -> _Config:
    return _Config(providers, state_dir)


def test_hygiene_is_clean_for_a_keyless_local_config() -> None:
    config = _config([_Provider("local", "http://localhost:8080/v1")])
    report = check_secret_hygiene(config)
    assert exit_code_for_secret_report(report) == 0


def test_hygiene_flags_a_credential_pasted_into_base_url() -> None:
    config = _config(
        [_Provider("leaky", "https://api.example.com/v1?api_key=abcdefghijkl")]
    )
    report = check_secret_hygiene(config)
    assert exit_code_for_secret_report(report) == 1
    embedded = next(c for c in report.checks if c.name == "config-embedded-credentials")
    assert embedded.verdict == "error"
    assert "leaky" in embedded.detail


def test_hygiene_warns_when_a_declared_env_var_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("CR_UNSET_KEY", raising=False)
    config = _config([_Provider("cloud", "https://api.example.com", "CR_UNSET_KEY")])
    report = check_secret_hygiene(config)
    assert exit_code_for_secret_report(report) == 2


def test_hygiene_reports_a_leak_already_on_disk(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CR_LIVE_KEY", SECRET)
    (tmp_path / "requests.jsonl").write_text(f"leaked {SECRET}\n", encoding="utf-8")
    config = _config(
        [_Provider("cloud", "https://api.example.com", "CR_LIVE_KEY")],
        state_dir=str(tmp_path),
    )
    report = check_secret_hygiene(config)
    scan = next(c for c in report.checks if c.name == "written-log-scan")
    assert scan.verdict == "error"
    assert "requests.jsonl:1" in scan.detail
    assert exit_code_for_secret_report(report) == 1
    # The report itself must not quote the credential it found.
    assert SECRET not in scan.detail


def test_register_config_secrets_returns_labels_not_values(monkeypatch) -> None:
    monkeypatch.setenv("CR_LIVE_KEY", SECRET)
    config = _config([_Provider("cloud", "https://api.example.com", "CR_LIVE_KEY")])
    labels = register_config_secrets(config)
    assert labels == ["CR_LIVE_KEY"]
    assert SECRET not in "".join(labels)
