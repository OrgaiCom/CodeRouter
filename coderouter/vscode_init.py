"""v2.10-A: ``coderouter vscode-init`` — one-shot VSCode workspace scaffolder.

Writes ``.vscode/settings.json`` with ``terminal.integrated.env.*`` so a
Claude Code session launched from VSCode's integrated terminal
auto-points at CodeRouter. Optionally emits a direnv ``.envrc`` in the
same run.

The design contract:

* stdlib only — no new runtime deps (keeps the strict-5 rule in
  ``pyproject.toml`` intact).
* Never overwrite unrelated keys in an existing ``settings.json``. Only
  ``terminal.integrated.env.osx`` / ``.linux`` / ``.windows`` are
  touched, and inside them only the CodeRouter-managed env vars.
* When an existing target value differs from the desired one, refuse
  without ``--force`` and print a unified diff so the operator decides.
* ``--dry-run`` is byte-identical to the write path minus the actual
  ``os.replace`` at the end.
* All writes are atomic (tmp file + ``os.replace``) so a partial write
  cannot corrupt an existing ``settings.json``. This matters more here
  than in most CodeRouter modules because the file we're editing is
  workspace-critical: a broken JSON blob would silently break every
  VSCode terminal env until a human debugged it.

Files touched (opt-in per flag):

* ``{target}/.vscode/settings.json`` — always
* ``{target}/.envrc``                 — ``--with-envrc``

Non-Claude-Code extensions (Cline / Roo / Kilo / Continue) are handled
by the cheat sheet printed to stdout at end of run and the fuller
``docs/guides/vscode.md`` — this CLI deliberately does NOT reach into
those extensions' settings, because each ships its own config schema
that changes with its own release cadence.

CLI plumbing lives in :mod:`coderouter.cli` (``_run_vscode_init``); this
module is import-clean so tests can exercise ``run_vscode_init``
directly without argparse in the way.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path

# v2.10-A: The README, quickstart, and every backend guide teach 8088 as
# the CodeRouter port. ``coderouter serve`` still defaults to 4000 (a
# pre-README default that predates the docs unification); vscode-init
# sides with the docs so that the settings.json it writes matches every
# tutorial the user is likely to be reading. Operators who deviate can
# pass ``--port`` explicitly.
DEFAULT_PORT = 8088

# Bogus token — CodeRouter's chat ingress does not validate ANTHROPIC_AUTH_TOKEN
# (real API keys are managed on the ``providers.yaml`` side). Using a
# fixed sentinel here means we never accidentally embed the operator's
# real token into a workspace file.
DEFAULT_TOKEN = "dummy"  # noqa: S105 — intentional placeholder, not a secret

# VSCode reads terminal env from a separate key per host OS. Writing all
# three means the same ``.vscode/`` folder works whether the developer
# opens it on macOS, Linux, or WSL/Windows without a second scaffolder
# run. The keys are stable across VSCode versions (documented public
# API since 1.x).
_TERMINAL_ENV_KEYS: tuple[str, ...] = (
    "terminal.integrated.env.osx",
    "terminal.integrated.env.linux",
    "terminal.integrated.env.windows",
)

# Env vars this scaffolder owns inside the terminal-env blocks. Anything
# NOT in this set that already exists under one of the terminal-env
# keys is preserved verbatim on merge (e.g. the user's PATH tweak).
_MANAGED_ENV_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "CODEROUTER_MODE",
)


@dataclass(frozen=True)
class FileOutcome:
    """The outcome of touching (or planning to touch) one file.

    ``action`` values:

    * ``created``   — file didn't exist, we wrote it
    * ``updated``   — file existed, we changed it
    * ``unchanged`` — file already had the desired content
    * ``conflict``  — existing content differs and ``--force`` was
                      not set; the file was NOT written
    * ``skipped``   — hard error prevented processing (unparseable
                      JSON, wrong type, permission denied); NOT written
    """

    path: Path
    action: str
    diff: str = ""
    reason: str = ""


@dataclass(frozen=True)
class VSCodeInitResult:
    outcomes: list[FileOutcome]

    @property
    def has_conflicts(self) -> bool:
        return any(o.action == "conflict" for o in self.outcomes)

    @property
    def has_skipped(self) -> bool:
        return any(o.action == "skipped" for o in self.outcomes)


def run_vscode_init(
    target: Path,
    *,
    port: int = DEFAULT_PORT,
    profile: str | None = None,
    with_envrc: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> VSCodeInitResult:
    """Scaffold VSCode workspace settings for CodeRouter.

    Parameters
    ----------
    target:
        Workspace root — the folder that VSCode opens. ``.vscode/`` and
        (if ``with_envrc``) ``.envrc`` are placed under this directory.
        Must already exist as a directory; we do not create the
        workspace root itself.
    port:
        CodeRouter port for ``ANTHROPIC_BASE_URL``. Defaults to 8088
        (docs-standard); override to match a non-default ``serve``.
    profile:
        Optional ``CODEROUTER_MODE`` value to preload. When set, the
        VSCode terminal gets ``CODEROUTER_MODE=<profile>`` too.
    with_envrc:
        When true, also write ``.envrc`` (direnv). Idempotent and
        conflict-aware just like ``settings.json``.
    dry_run:
        When true, compute all the outcomes and diffs but skip the
        actual ``os.replace``. Guarantees byte-parity with the write
        path — see the module docstring for the invariant.
    force:
        When true, existing conflicting values are overwritten. Without
        this flag, a conflict is reported with a diff and the file is
        left untouched.

    Raises
    ------
    FileNotFoundError:
        The target directory does not exist. All other issues (bad
        JSON, permission denied, existing conflicts) are captured as
        ``FileOutcome`` entries with the appropriate action.
    """
    target = Path(target).expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"target directory does not exist: {target}")

    base_url = f"http://localhost:{port}"
    outcomes: list[FileOutcome] = []

    # 1. .vscode/settings.json (always).
    outcomes.append(
        _apply_settings_json(
            target / ".vscode" / "settings.json",
            base_url=base_url,
            profile=profile,
            force=force,
            dry_run=dry_run,
        )
    )

    # 2. .envrc (opt-in via --with-envrc).
    if with_envrc:
        outcomes.append(
            _apply_envrc(
                target / ".envrc",
                base_url=base_url,
                profile=profile,
                force=force,
                dry_run=dry_run,
            )
        )

    return VSCodeInitResult(outcomes=outcomes)


# ---------------------------------------------------------------------------
# settings.json
# ---------------------------------------------------------------------------


def _apply_settings_json(
    path: Path,
    *,
    base_url: str,
    profile: str | None,
    force: bool,
    dry_run: bool,
) -> FileOutcome:
    desired_env: dict[str, str] = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": DEFAULT_TOKEN,
    }
    if profile:
        desired_env["CODEROUTER_MODE"] = profile

    old_text = ""
    if path.exists():
        try:
            old_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return FileOutcome(
                path=path,
                action="skipped",
                reason=f"cannot read existing file: {exc}",
            )
        try:
            existing = json.loads(old_text) if old_text.strip() else {}
        except json.JSONDecodeError as exc:
            return FileOutcome(
                path=path,
                action="skipped",
                reason=(
                    "existing settings.json is not valid JSON "
                    f"({exc.msg} at line {exc.lineno}). Fix or move it "
                    "aside and re-run."
                ),
            )
        if not isinstance(existing, dict):
            return FileOutcome(
                path=path,
                action="skipped",
                reason=(
                    "existing settings.json is not a JSON object "
                    f"(got {type(existing).__name__}). Fix or move it "
                    "aside and re-run."
                ),
            )
    else:
        existing = {}

    merged, conflicts = _merge_terminal_env(existing, desired_env, force=force)

    if conflicts and not force:
        # Report the diff between existing and what --force WOULD write
        # so the operator can decide whether the change is desired.
        force_merged, _ = _merge_terminal_env(existing, desired_env, force=True)
        diff = _json_diff(existing, force_merged, path)
        return FileOutcome(
            path=path,
            action="conflict",
            diff=diff,
            reason=(
                "existing settings.json has different value(s) for "
                f"{', '.join(sorted(conflicts))}. Re-run with --force to overwrite."
            ),
        )

    new_text = _dump_json(merged)
    if new_text == old_text:
        return FileOutcome(path=path, action="unchanged")

    diff = _text_diff(old_text, new_text, path)
    if not dry_run:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, new_text)
        except OSError as exc:
            return FileOutcome(
                path=path,
                action="skipped",
                reason=f"write failed: {exc}",
                diff=diff,
            )

    return FileOutcome(
        path=path,
        action="updated" if old_text else "created",
        diff=diff,
    )


def _merge_terminal_env(
    existing: dict,
    desired_env: dict[str, str],
    *,
    force: bool,
) -> tuple[dict, set[str]]:
    """Merge our env vars into a settings.json dict.

    Preserves unrelated top-level keys (``editor.fontSize`` etc.) AND
    unrelated env keys inside the three ``terminal.integrated.env.*``
    blocks. When ``force`` is false, an existing conflicting value for
    one of our managed keys is left in place and its dotted path
    (e.g. ``terminal.integrated.env.osx.ANTHROPIC_BASE_URL``) is added
    to the returned conflict set.

    Returns
    -------
    merged:
        The merged dict, safe to serialize with :func:`_dump_json`.
    conflicts:
        Set of dotted paths whose existing value differs from the
        desired one and were NOT overwritten (because ``force=False``).
    """
    # Deep copy via json round-trip — the input dict may share nested
    # references with the caller and we must not mutate it.
    merged = json.loads(json.dumps(existing))
    conflicts: set[str] = set()

    for os_key in _TERMINAL_ENV_KEYS:
        sub = merged.get(os_key)
        if not isinstance(sub, dict):
            # Overwrite any non-dict value at this key — VSCode would
            # ignore it anyway. Preserve nothing since the previous
            # value was already broken.
            sub = {}

        for env_key, env_val in desired_env.items():
            if env_key in sub and sub[env_key] != env_val and not force:
                conflicts.add(f"{os_key}.{env_key}")
                # Leave sub[env_key] unchanged.
                continue
            sub[env_key] = env_val

        merged[os_key] = sub

    return merged, conflicts


# ---------------------------------------------------------------------------
# .envrc
# ---------------------------------------------------------------------------


def _apply_envrc(
    path: Path,
    *,
    base_url: str,
    profile: str | None,
    force: bool,
    dry_run: bool,
) -> FileOutcome:
    new_text = _render_envrc(base_url=base_url, profile=profile)
    old_text = ""
    if path.exists():
        try:
            old_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return FileOutcome(
                path=path,
                action="skipped",
                reason=f"cannot read existing file: {exc}",
            )

    if new_text == old_text:
        return FileOutcome(path=path, action="unchanged")

    diff = _text_diff(old_text, new_text, path)

    if old_text and not force:
        return FileOutcome(
            path=path,
            action="conflict",
            diff=diff,
            reason=(
                ".envrc already exists and its contents differ. "
                "Re-run with --force to overwrite. "
                "(Tip: keep secrets in a separate .envrc.local and "
                "``source_env_if_exists`` from .envrc.)"
            ),
        )

    if not dry_run:
        try:
            _atomic_write(path, new_text)
        except OSError as exc:
            return FileOutcome(
                path=path,
                action="skipped",
                reason=f"write failed: {exc}",
                diff=diff,
            )

    return FileOutcome(
        path=path,
        action="updated" if old_text else "created",
        diff=diff,
    )


def _render_envrc(*, base_url: str, profile: str | None) -> str:
    """Build the .envrc contents. Kept split for test targeting."""
    lines = [
        "# Auto-generated by `coderouter vscode-init`. Safe to edit.",
        "# direnv will export these when you `cd` into this directory.",
        "# Run `direnv allow` once after generation.",
        "",
        f'export ANTHROPIC_BASE_URL="{base_url}"',
        f'export ANTHROPIC_AUTH_TOKEN="{DEFAULT_TOKEN}"',
    ]
    if profile:
        lines.append(f'export CODEROUTER_MODE="{profile}"')
    lines.append("")  # trailing newline
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def format_result(result: VSCodeInitResult, *, dry_run: bool, port: int) -> str:
    """Human-readable rendering of the outcomes for CLI stdout.

    The output has three sections:

    1. Per-file outcome lines (with unified diffs where relevant).
    2. A cheat sheet for the extensions this scaffolder does NOT wire
       (Cline / Roo / Kilo / Continue) — the user copy-pastes.
    3. A trailing hint about the fuller ``docs/guides/vscode.md``.

    ``dry_run`` toggles the "(dry-run — nothing written)" reminder.
    """
    lines: list[str] = []
    marker_for = {
        "created": "  new ",
        "updated": " diff ",
        "unchanged": " nop  ",
        "conflict": "CONFLICT",
        "skipped": " SKIP ",
    }
    for o in result.outcomes:
        marker = marker_for.get(o.action, f" {o.action} ")
        lines.append(f"[{marker}] {o.path}")
        if o.reason:
            lines.append(f"          → {o.reason}")
        if o.diff and o.action in ("created", "updated", "conflict"):
            lines.append("")
            lines.extend(o.diff.rstrip("\n").splitlines())
            lines.append("")

    if dry_run:
        lines.append("")
        lines.append("(dry-run — no files were written. Re-run without --dry-run to apply.)")

    lines.append("")
    lines.append(_render_cheat_sheet(port))
    return "\n".join(lines)


def _render_cheat_sheet(port: int) -> str:
    """Emit the copy-paste snippets shown after every ``vscode-init`` run.

    The Continue.dev snippet is built with :func:`json.dumps` rather
    than hand-formatted string concatenation to guarantee the output
    is syntactically valid JSON. An earlier hand-formatted version
    accidentally leaked an extra ``}`` from mismatched f-string /
    plain-string brace escaping — the JSON round-trip closes that
    class of bug and the test in ``tests/test_vscode_init.py`` pins
    the invariant.
    """
    continue_snippet = _render_continue_snippet(port)
    return (
        "─── Other extensions (copy/paste) ─────────────────────────────\n"
        "\n"
        "  Cline / Roo Code / Kilo Code:\n"
        "    API Provider: OpenAI Compatible\n"
        f"    Base URL:     http://localhost:{port}/v1\n"
        "    API Key:      dummy   (any non-empty value)\n"
        "    Model ID:     anything (CodeRouter routes via default_profile)\n"
        "\n"
        "  Continue.dev — add to the models array in ~/.continue/config.json:\n"
        f"{continue_snippet}\n"
        "\n"
        "Full guide: docs/guides/vscode.md in the CodeRouter repo.\n"
    )


def _render_continue_snippet(port: int) -> str:
    """Return one indented pretty-printed JSON block for Continue.dev.

    Split out from :func:`_render_cheat_sheet` so tests can
    ``json.loads`` the exact bytes users copy-paste and verify the
    snippet is valid.
    """
    model_entry: dict[str, str] = {
        "title": "CodeRouter",
        "provider": "openai",
        "model": "any-id",
        "apiBase": f"http://localhost:{port}/v1",
        "apiKey": "dummy",
    }
    pretty = json.dumps(model_entry, indent=2, ensure_ascii=False)
    # Indent every line by 4 spaces so it aligns under the "─── Other
    # extensions" header exactly like the Cline block above.
    return "\n".join("    " + line for line in pretty.splitlines())


def exit_code_for(result: VSCodeInitResult) -> int:
    """Map the outcome set to a shell exit code.

    * 0 — every file is clean or a no-op.
    * 2 — at least one conflict (needs ``--force``); nothing was
          written for the conflicting file(s), but other files may
          have been.
    * 1 — a hard error prevented processing (unparseable JSON, write
          failure). Treated as more severe than a conflict.
    """
    if result.has_skipped:
        return 1
    if result.has_conflicts:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------


def _dump_json(obj: object) -> str:
    """Deterministic pretty JSON for settings.json.

    ``indent=2`` matches VSCode's own formatter default, ``ensure_ascii=False``
    keeps CJK env values (e.g. a profile name) readable in the file.
    A trailing newline is appended so POSIX tools don't complain.
    """
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp + os.replace).

    A partial write cannot corrupt the target file: the tmp file lives
    on the same filesystem (same parent directory) so ``os.replace`` is
    an atomic inode swap on POSIX. On Windows ``os.replace`` is also
    atomic since Python 3.3.

    The tmp file uses the parent + ``<name>.tmp`` pattern rather than
    ``Path.with_suffix`` because ``.envrc`` has no stem/suffix in the
    ``Path`` sense and ``with_suffix`` would silently misbehave.
    """
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _text_diff(old: str, new: str, path: Path) -> str:
    """Unified diff between two text buffers, labelled by file name."""
    return "".join(
        unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            n=3,
        )
    )


def _json_diff(old: dict, new: dict, path: Path) -> str:
    """Diff between two JSON dicts, formatted as ``_dump_json`` would."""
    old_text = _dump_json(old) if old else ""
    new_text = _dump_json(new)
    return _text_diff(old_text, new_text, path)
