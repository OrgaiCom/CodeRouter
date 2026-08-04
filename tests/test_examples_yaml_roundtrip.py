"""Guard the bundled ``examples/*.yaml`` against silent reformatting.

``coderouter doctor --apply`` edits providers.yaml through ruamel.yaml.
Any input style ruamel does not reproduce byte-for-byte becomes an
unrequested diff in the operator's file — and, one run later, in the
``.bak`` that is supposed to hold their original. The examples are the
files most likely to be hit: the README tells users to copy one and
edit it, so a style that survives review here is a style that ships to
everybody.

These tests fix the shipped examples as the reference corpus. If a
future example lands with a >80-column quoted scalar, an unusual quote
style, or anything else the round-trip mangles, it fails here rather
than in a user's config.

Known and deliberately tolerated deviation
------------------------------------------
ruamel re-emits an explicit ``key: null`` as an empty scalar
(``key:``). The two are identical to every YAML parser, and ruamel
offers no per-node way to remember which spelling the source used —
the only lever is a global ``None`` representer, which would flip the
*other* population (files written with the empty-scalar spelling) to
``null`` instead. We deliberately did not install one: with the write
gated on "did a merge change a value?"
(:func:`coderouter.doctor_apply.apply_doctor_patches`), an untouched
file is never rewritten, so neither spelling can be silently converted.
Two shipped examples carry explicit nulls today —
``providers.yaml`` (5 lines) and ``providers.llamacpp-vllm.yaml``
(2 lines).

:func:`_normalize_explicit_null` encodes exactly that one rewrite and
nothing else, so the assertion below stays strict about every other
kind of drift — line folding, comment loss, quote loss, reindentation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import pytest

try:
    import ruamel.yaml  # noqa: F401

    _RUAMEL_AVAILABLE = True
except ImportError:
    _RUAMEL_AVAILABLE = False

_requires_ruamel = pytest.mark.skipif(
    not _RUAMEL_AVAILABLE,
    reason=(
        "ruamel.yaml not installed (required by `coderouter doctor "
        "--check-model --apply`). Install with: "
        "uv pip install 'ruamel.yaml>=0.18.6' or uv sync --extra dev"
    ),
)

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
_EXAMPLE_FILES = sorted(_EXAMPLES_DIR.glob("*.yaml"))

# ``  key: null   # comment`` → the key, then the value column blanked.
# ruamel keeps the comment in its original column, so we pad by the
# width of the removed token to compare like for like.
_EXPLICIT_NULL_RE = re.compile(r"^(\s*[^\s#][^:]*:) null(?=(?:\s|$))")


def _normalize_explicit_null(raw: str) -> str:
    """Apply ruamel's ``null`` → empty-scalar rewrite, and only that."""
    out: list[str] = []
    for line in raw.splitlines(keepends=True):
        match = _EXPLICIT_NULL_RE.match(line)
        if match:
            rest = line[match.end() :]
            if rest.strip():
                # A trailing comment follows — keep its column.
                line = match.group(1) + " " * len(" null") + rest
            else:
                line = match.group(1) + rest.lstrip(" ")
        out.append(line)
    return "".join(out)


def test_examples_dir_is_not_empty() -> None:
    """Sanity: a glob that silently matches nothing would make every
    parametrized test below vacuously pass."""
    assert _EXAMPLE_FILES, f"no example YAML found under {_EXAMPLES_DIR}"


@_requires_ruamel
@pytest.mark.parametrize("example", _EXAMPLE_FILES, ids=lambda p: p.name)
def test_examples_providers_yaml_round_trips_byte_identical(example: Path) -> None:
    """load → dump through the apply helpers reproduces the file exactly.

    "Exactly" modulo the documented ``null`` spelling (see module
    docstring); every other byte, including comments, blank lines,
    quote style and long-line layout, must survive untouched.
    """
    from coderouter.doctor_apply import (
        _dump_yaml_with_comments,
        _load_yaml_with_comments,
    )

    doc, raw = _load_yaml_with_comments(example)
    dumped = _dump_yaml_with_comments(doc)

    assert dumped == _normalize_explicit_null(raw)


@_requires_ruamel
@pytest.mark.parametrize("example", _EXAMPLE_FILES, ids=lambda p: p.name)
def test_no_op_apply_leaves_example_byte_identical(
    example: Path, tmp_path: Path
) -> None:
    """The user-visible guarantee: a no-op ``--apply`` writes nothing.

    Stronger than the round-trip check above because it also covers the
    tolerated ``null`` deviation — those two files must come through an
    ``--apply`` untouched, which is the property the ``.bak`` chain
    depends on.
    """
    from coderouter.doctor_apply import apply_doctor_patches

    original = example.read_text(encoding="utf-8")
    target = tmp_path / "providers.yaml"
    target.write_text(original, encoding="utf-8")

    class _Result:
        # A patch naming a provider that does not exist in this example
        # merges to a guaranteed no-op, whatever the example contains.
        name = "tool_calls"
        target_file = "providers.yaml"
        suggested_patch = (
            "providers:\n"
            "  - name: __definitely_not_a_real_provider__\n"
            "    capabilities:\n"
            "      tools: true\n"
        )

    class _Report:
        results: ClassVar[list[object]] = [_Result()]

    result = apply_doctor_patches(
        report=_Report(), config_path=target, write=True
    )

    assert result.written is False
    assert target.read_text(encoding="utf-8") == original
    assert not target.with_suffix(".yaml.bak").exists()
