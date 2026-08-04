"""Regression tests for H-7: a circular import broke standalone imports.

Import chain (pre-fix):
    translation/__init__.py -> convert.py -> adapters/__init__.py
      -> registry.py -> anthropic_native.py -> translation.convert (partial)

`anthropic_native.py` imported `stream_anthropic_to_chat_chunks` /
`to_anthropic_request` / `to_chat_response` from `coderouter.translation.convert`
at module scope. If `coderouter.translation` (or one of its submodules) was
the *first* thing imported in a process, Python would still be mid-way
through initializing `coderouter.translation.convert` when the import chain
looped back into it via `coderouter.adapters`, and the partially-initialized
module wouldn't yet have those three names bound -> ImportError.

These tests MUST run in a fresh subprocess per import: importing something
in the current test process first (as most of this suite does, thanks to
`coderouter.adapters` typically being imported earlier via other test
modules / alphabetical collection order) would already have the module in
`sys.modules`, hiding the bug.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Repo root: the directory containing the `coderouter` package. Resolved
# relative to this file so the test passes regardless of the cwd pytest
# was invoked from (bare `pytest tests/test_import_hygiene.py`, `pytest`
# from a different directory, etc.).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_import(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh Python subprocess with the repo root on sys.path.

    Inherits the parent environment (PATH, VIRTUAL_ENV, SSL cert vars, etc.)
    rather than replacing it, so this works whether pytest itself is running
    under a uv-managed venv, a plain venv, or the system interpreter —  only
    PYTHONPATH is adjusted, and any existing PYTHONPATH is preserved after it
    so an already-correct path isn't clobbered.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_REPO_ROOT) if not existing else f"{_REPO_ROOT}{os.pathsep}{existing}"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"import failed (rc={result.returncode})\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_translation_package_imports_standalone() -> None:
    """`import coderouter.translation` must succeed with nothing preloaded."""
    result = _run_import("import coderouter.translation")
    _assert_ok(result)


def test_translation_submodules_import_standalone() -> None:
    """Each translation submodule must be importable on its own, in isolation."""
    for module in (
        "coderouter.translation.anthropic",
        "coderouter.translation.convert",
        "coderouter.translation.tool_repair",
    ):
        result = _run_import(f"import {module}")
        _assert_ok(result)


def test_adapters_imports_standalone() -> None:
    """Regression guard: `coderouter.adapters` alone must keep working too."""
    result = _run_import("import coderouter.adapters")
    _assert_ok(result)


def test_guards_and_routing_import_standalone() -> None:
    """Other modules that sit in/near the same import graph must not regress."""
    for module in (
        "coderouter.guards.context_budget",
        "coderouter.routing.fallback",
    ):
        result = _run_import(f"import {module}")
        _assert_ok(result)


def test_no_toplevel_convert_import_in_anthropic_native() -> None:
    """Regression guard: the fix must stay a lazy/function-local import.

    If someone "helpfully" hoists the import back to module scope, this
    test fails immediately instead of waiting for the subprocess-import
    tests above to catch it indirectly.
    """
    import inspect

    import coderouter.adapters.anthropic_native as mod

    source = inspect.getsource(mod)
    # Only look at the module up to the first class/def statement — i.e.
    # the top-level import block — so a *local* import inside a function
    # body (which legitimately contains this exact text) doesn't trip the
    # check.
    header, _, _ = source.partition("\nclass ")
    assert "from coderouter.translation.convert import" not in header, (
        "found a top-level import of coderouter.translation.convert in "
        "anthropic_native.py — this reintroduces the H-7 circular import"
    )
