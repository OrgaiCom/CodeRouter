"""Tests for the external-bind startup warning (v2.7.5).

Binding beyond loopback while ``CODEROUTER_ALLOWED_HOSTS`` is unset means
the v2.7.0 Host-validation guard will 403 every LAN request — a trap a real
user hit when upgrading from 2.6 ("worked with --host 0.0.0.0 before").
The CLI now prints a warning for exactly that combination, and stays quiet
for every coherent configuration.
"""

from __future__ import annotations

import pytest

from coderouter.cli import _external_bind_warning


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]"])
def test_loopback_bind_never_warns(host: str) -> None:
    assert _external_bind_warning(host, None) is None
    assert _external_bind_warning(host, "192.168.1.5") is None


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "myhost"])
def test_external_bind_without_allowed_hosts_warns(host: str) -> None:
    warning = _external_bind_warning(host, None)
    assert warning is not None
    assert "CODEROUTER_ALLOWED_HOSTS" in warning
    assert "403" in warning
    # The warning must carry the security caveat, not just the fix.
    assert "authentication" in warning


def test_external_bind_with_allowed_hosts_is_quiet() -> None:
    assert _external_bind_warning("0.0.0.0", "133.129.112.25") is None
    assert _external_bind_warning("0.0.0.0", "a.example, b.example") is None


def test_blank_allowed_hosts_still_warns() -> None:
    # An empty/whitespace env var does not actually allow anything.
    assert _external_bind_warning("0.0.0.0", "") is not None
    assert _external_bind_warning("0.0.0.0", "   ") is not None
