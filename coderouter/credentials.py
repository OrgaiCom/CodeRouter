"""Credential resolution, including CLI-session borrowing (v2.14.0).

Until now a provider's credential could only be an env var. That is fine
for API keys and useless for the thing an increasing number of vendors
actually sell: a subscription, authenticated by an OAuth token that their
own CLI already fetched and wrote to disk.

CodeRouter's existing answer to those was ``kind: agent_cli`` — spawn the
vendor's CLI once per request and read its stdout. It works, and it
throws away almost everything CodeRouter is for. A one-shot ``exec`` has
no streaming, no tool-call repair, no context-budget guard, and it does
not sit in a fallback chain the way an HTTP provider does. The vendor's
subscription ends up as an island.

The alternative, which is what codex-router does for Kimi and Grok, is to
skip the process entirely: **read the token the CLI already wrote, and
make the HTTP call ourselves.** The provider then becomes an ordinary
``openai_compat`` entry, and every routing feature applies to it —
"use the subscription until it 429s, then fall through to a free cloud
model, then to local llama.cpp" becomes an ordinary chain.

    - name: kimi-sub
      kind: openai_compat
      base_url: https://api.moonshot.cn/v1
      credential:
        source: cli_session
        path: ~/.kimi-code/credentials/kimi-code.json
        field: access_token
        expiry_field: expires_at
        refresh:
          command: ["kimi", "auth", "status"]

Refresh: delegate, don't reimplement
------------------------------------
Refreshing is the part that rots. Every vendor has its own endpoint,
client id, rotation policy and error shape, and all of them change. So
this module does not implement OAuth. It runs the vendor's own CLI and
re-reads the file — codex-router's approach for Grok, and the reason that
path has survived contact with a moving vendor while its bespoke Kimi
refresh has not.

The command is an argv **list**, dispatched with ``shell=False``. That is
the same trust decision v2.13.0 made for ``restart_command``: a string
that goes through a shell turns a config file into arbitrary code
execution, and half-executing one (``touch a; b`` under ``shell=False``
creates a file literally named ``a;`` and reports success) is worse than
refusing. Here there is no string form to refuse — the schema only
accepts a list.

Concurrency
-----------
Two workers hitting an expired token would otherwise both shell out.
Refresh is single-flighted in-process by a per-path lock, and serialised
across processes by an advisory ``flock`` on a sidecar file where the
platform has one. After acquiring either lock the file is re-read before
deciding to refresh, because the holder of the lock may already have done
the work.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coderouter.logging import get_logger
from coderouter.secret_redaction import register_secret

__all__ = [
    "CredentialError",
    "resolve_provider_credential",
]

logger = get_logger(__name__)

# One lock per session file, created on demand. Keyed by the resolved
# absolute path so two providers pointing at the same file share it.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class CredentialError(RuntimeError):
    """A credential was configured but could not be produced."""


@dataclass(frozen=True)
class _SessionRead:
    token: str | None
    expires_at: float | None


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


def _dotted_get(payload: Any, dotted: str) -> Any:
    """Fetch ``a.b.c`` out of nested mappings.

    Vendors nest differently (``access_token`` at the root for Kimi, under
    ``tokens`` for others), so the field is a path rather than a key. A
    missing segment returns ``None`` rather than raising — an absent token
    is a normal state (the operator has not logged in yet) and the caller
    turns it into a clear message.
    """
    current = payload
    for segment in dotted.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _as_epoch_seconds(value: Any) -> float | None:
    """Normalise an expiry field to epoch seconds.

    Accepts seconds and milliseconds, because both are common and telling
    them apart is unambiguous: any plausible expiry in seconds is under
    1e11, and any in milliseconds is over it. A value we cannot read at
    all yields ``None``, which is treated as "no expiry information" —
    the token is used and the upstream 401 becomes the signal instead.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds / 1000.0 if seconds > 1e11 else seconds
    if isinstance(value, str):
        try:
            return _as_epoch_seconds(float(value))
        except ValueError:
            return None
    return None


def _read_session(path: Path, *, field: str, expiry_field: str) -> _SessionRead:
    """Read the token (and its expiry, if present) out of a session file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CredentialError(f"cannot read session file {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CredentialError(f"session file {path} is not valid JSON: {exc}") from exc

    token = _dotted_get(payload, field)
    if token is not None and not isinstance(token, str):
        raise CredentialError(
            f"session file {path}: field {field!r} is "
            f"{type(token).__name__}, expected a string"
        )
    expires_at = _as_epoch_seconds(_dotted_get(payload, expiry_field))
    return _SessionRead(token=token or None, expires_at=expires_at)


def _needs_refresh(read: _SessionRead, *, early_ratio: float, min_lead_s: float) -> bool:
    """Whether to refresh now.

    Refresh early rather than on expiry: a token that dies mid-request
    costs a retry and a confusing upstream error. ``early_ratio`` is a
    fraction of the *remaining* lifetime as seen right now, floored by
    ``min_lead_s`` — so a long-lived token still refreshes several minutes
    before it lapses rather than at the last second.
    """
    if read.token is None:
        return True
    if read.expires_at is None:
        return False
    remaining = read.expires_at - time.time()
    return remaining <= max(min_lead_s, remaining * early_ratio)


def _run_refresh(command: list[str], *, timeout_s: float, path: Path) -> None:
    """Run the vendor CLI so it rotates the token, then let the caller re-read.

    ``shell=False`` with an argv list: the config file names a program, it
    does not get to compose a shell command. Failures are logged and
    swallowed — a stale-but-present token is still worth trying, and the
    upstream 401 is a better error for the operator than a refresh
    traceback that hides it.
    """
    try:
        completed = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "credential-refresh-failed",
            extra={"path": str(path), "command": command[0], "error": str(exc)},
        )
        return
    if completed.returncode != 0:
        # stderr is deliberately NOT logged: a failing auth CLI is exactly
        # the thing most likely to print a token or a device code.
        logger.warning(
            "credential-refresh-nonzero",
            extra={
                "path": str(path),
                "command": command[0],
                "returncode": completed.returncode,
            },
        )


def _flock(handle: Any, *, exclusive: bool = True) -> Any:
    """Best-effort advisory lock; a no-op where the platform has none."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        return None
    fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    return fcntl


def _resolve_cli_session(spec: Any, provider_name: str) -> str | None:
    """Read (and if needed refresh) a token a vendor CLI wrote to disk."""
    path = Path(str(spec.path)).expanduser()
    field = spec.field
    expiry_field = spec.expiry_field

    read = _read_session(path, field=field, expiry_field=expiry_field)
    refresh = spec.refresh
    if refresh is None or not _needs_refresh(
        read, early_ratio=refresh.early_ratio, min_lead_s=refresh.min_lead_s
    ):
        return _register(read.token, provider_name)

    with _lock_for(str(path)):
        # Re-read under the in-process lock: another worker may have
        # refreshed while we waited.
        read = _read_session(path, field=field, expiry_field=expiry_field)
        if not _needs_refresh(
            read, early_ratio=refresh.early_ratio, min_lead_s=refresh.min_lead_s
        ):
            return _register(read.token, provider_name)

        lock_path = path.with_name(path.name + ".coderouter-lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "a+") as handle:
                _flock(handle)
                # And once more across processes, same reason.
                read = _read_session(path, field=field, expiry_field=expiry_field)
                if _needs_refresh(
                    read,
                    early_ratio=refresh.early_ratio,
                    min_lead_s=refresh.min_lead_s,
                ):
                    logger.info(
                        "credential-refresh-started",
                        extra={"provider": provider_name, "command": refresh.command[0]},
                    )
                    _run_refresh(
                        list(refresh.command), timeout_s=refresh.timeout_s, path=path
                    )
                    read = _read_session(path, field=field, expiry_field=expiry_field)
        except OSError as exc:
            logger.warning(
                "credential-lock-failed",
                extra={"path": str(lock_path), "error": str(exc)},
            )

    return _register(read.token, provider_name)


def _register(token: str | None, provider_name: str) -> str | None:
    """Arm the log scrubber before the token goes anywhere near a header."""
    if token:
        register_secret(token, f"{provider_name}-session")
    return token


def resolve_provider_credential(config: Any) -> str | None:
    """Return the credential for ``config``, or ``None`` when there is none.

    Order of precedence is deliberately not "try everything":
    ``credential`` and ``api_key_env`` are mutually exclusive at the
    schema level, so exactly one path can apply. A resolver that silently
    fell back between sources would make "which credential is this request
    actually using?" unanswerable, which is the question an operator asks
    at exactly the wrong moment.
    """
    from coderouter.config.loader import resolve_api_key

    spec = getattr(config, "credential", None)
    if spec is None:
        return resolve_api_key(getattr(config, "api_key_env", None))

    name = str(getattr(config, "name", "?"))
    if spec.source == "env":
        return resolve_api_key(spec.env)
    if spec.source == "cli_session":
        try:
            return _resolve_cli_session(spec, name)
        except CredentialError as exc:
            # A missing or malformed session file is an operator problem
            # (not logged in yet, vendor changed its layout), not a crash.
            # Returning None sends the request unauthenticated, the upstream
            # 401 fires, and the fallback chain moves on.
            logger.warning(
                "credential-unavailable",
                extra={"provider": name, "source": spec.source, "error": str(exc)},
            )
            return None
    raise CredentialError(f"provider {name}: unknown credential source {spec.source!r}")


def session_path_is_sane(path: str) -> bool:
    """Reject a session path that reaches outside the user's home.

    ``providers.yaml`` is operator-owned config, at the same trust level as
    ``launcher.backends[*].binary`` — but v2.13.0 exists because that
    assumption broke once already when a file could be picked up from the
    working directory. A vendor CLI writes under ``$HOME``; nothing
    legitimate needs ``/etc/shadow``.
    """
    try:
        resolved = Path(path).expanduser().resolve()
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        return False
    return resolved == home or home in resolved.parents or os.name == "nt"
