"""Minimal, dependency-free GGUF header introspection (low-memory track).

Why self-written
================

To right-size ``num_ctx`` *before* dispatch we need a model's layer
count and embedding width so the KV-cache footprint can be estimated.
That data lives in the GGUF metadata header. Rather than add the
official ``gguf`` package (and its ``numpy`` transitive dep) we read
only the handful of header fields we need with the standard library —
preserving the 5-deps invariant.

The GGUF binary layout we parse (little-endian):

  magic      : 4 bytes  == b"GGUF"
  version    : uint32   (2 or 3 supported)
  tensor_cnt : uint64   (ignored — we never read tensor data)
  kv_count   : uint64   (number of metadata key/value pairs)
  kv_pairs   : kv_count repetitions of:
      key        : gguf-string (uint64 length + UTF-8 bytes)
      value_type : uint32  (see _GGUF_TYPE_*)
      value      : type-dependent

We walk the KV pairs, capturing only the keys we care about, and skip
the rest (including arbitrarily nested arrays) without materialising
them.

Security
========

The parser treats the file as **untrusted input**:

  * Every string length and array element count is clamped against
    :data:`_MAX_STR_BYTES` / :data:`_MAX_ARRAY_LEN` so a corrupt or
    hostile header cannot trigger a multi-GB allocation (DoS).
  * Reads past EOF raise :class:`GGUFParseError`, never an unbounded
    loop. This includes ``seek`` past EOF: every skip (array element
    count x element size, or a lone scalar) is checked against the
    file's actual size *before* the seek happens, so a truncated file
    that declares a huge array count fails fast instead of spinning a
    multi-million-iteration loop over bytes that were never written.
  * Array nesting depth is capped at :data:`_MAX_ARRAY_DEPTH`. Skipping
    a value uses an explicit work-stack (``_skip_value``), not Python
    recursion, so there is no ``RecursionError`` failure mode to begin
    with — the depth cap is a belt-and-suspenders DoS guard (a
    pathological file can otherwise spend 12 bytes per nesting level to
    force millions of stack-machine iterations), not a stack-overflow
    guard. Every llama.cpp-written GGUF array (tokenizer tokens/scores/
    merges, etc.) nests exactly one level deep.
  * No ``mmap``, no tensor payload read, no code execution path — we
    only seek/read a small prefix.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

# ---------------------------------------------------------------------------
# Constants / format
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"

# GGUF metadata value type tags.
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

# Fixed-width scalar (struct format, size) by type tag.
_SCALAR: dict[int, tuple[str, int]] = {
    _GGUF_TYPE_UINT8: ("<B", 1),
    _GGUF_TYPE_INT8: ("<b", 1),
    _GGUF_TYPE_UINT16: ("<H", 2),
    _GGUF_TYPE_INT16: ("<h", 2),
    _GGUF_TYPE_UINT32: ("<I", 4),
    _GGUF_TYPE_INT32: ("<i", 4),
    _GGUF_TYPE_FLOAT32: ("<f", 4),
    _GGUF_TYPE_BOOL: ("<?", 1),
    _GGUF_TYPE_UINT64: ("<Q", 8),
    _GGUF_TYPE_INT64: ("<q", 8),
    _GGUF_TYPE_FLOAT64: ("<d", 8),
}

# Defensive clamps against hostile / corrupt headers.
_MAX_STR_BYTES: int = 1 << 20  # 1 MiB key/value string ceiling
_MAX_ARRAY_LEN: int = 1 << 24  # element-count ceiling for arrays
_MAX_KV_PAIRS: int = 1 << 20  # metadata pair ceiling
# Array-of-array nesting ceiling. Every array we've ever seen llama.cpp
# emit (tokenizer.ggml.tokens / token_type / scores / merges, ...) nests
# exactly one level deep, and this repo's synthetic test fixtures never
# exceed that either — real headers are nowhere near this limit. It exists
# purely as a DoS guard: each extra nesting level costs an attacker only
# 12 bytes (elem_type u32 + count u64), so an unbounded depth lets a
# ~12 KiB file blow the interpreter's C stack. ``_skip_value`` walks an
# explicit work-stack rather than recursing, so this cap is enforced
# directly (no reliance on hitting Python's recursion limit).
_MAX_ARRAY_DEPTH: int = 8

# Human-readable names for the GGUF ``general.file_type`` enum (subset).
_FILE_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
}


class GGUFParseError(Exception):
    """Raised when a file is not a parseable GGUF header."""


@dataclass(frozen=True, slots=True)
class GGUFInfo:
    """The subset of GGUF metadata needed for memory accounting."""

    architecture: str | None
    n_layers: int | None
    n_embd: int | None
    n_heads: int | None
    n_kv_heads: int | None
    file_type: int | None
    file_size_bytes: int
    n_nextn: int | None = None

    @property
    def supports_mtp(self) -> bool:
        """True when the GGUF embeds Multi-Token-Prediction (nextn) layers.

        Derived from ``{arch}.nextn_predict_layers`` — a positive count means
        the main model carries MTP/nextn tensors and can drive llama.cpp's
        ``--spec-type draft-mtp`` without a separate draft gguf.
        """
        return bool(self.n_nextn and self.n_nextn > 0)

    @property
    def quant_name(self) -> str | None:
        """Human-readable quantization label, or None if unknown."""
        if self.file_type is None:
            return None
        return _FILE_TYPE_NAMES.get(self.file_type, f"type{self.file_type}")

    @property
    def weights_bytes(self) -> int:
        """Approximate on-disk weight size — the file size is the best
        proxy (GGUF is almost entirely tensor data)."""
        return self.file_size_bytes


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------


def _read_exact(fh: BinaryIO, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise GGUFParseError(f"unexpected EOF (wanted {n} bytes, got {len(data)})")
    return data


def _read_scalar(fh: BinaryIO, type_tag: int) -> object:
    fmt_size = _SCALAR.get(type_tag)
    if fmt_size is None:
        raise GGUFParseError(f"unknown scalar type tag {type_tag}")
    fmt, size = fmt_size
    return struct.unpack(fmt, _read_exact(fh, size))[0]


def _read_u32(fh: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(fh, 4))[0]


def _read_u64(fh: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(fh, 8))[0]


def _read_gguf_string(fh: BinaryIO) -> str:
    length = _read_u64(fh)
    if length > _MAX_STR_BYTES:
        raise GGUFParseError(f"string length {length} exceeds cap")
    return _read_exact(fh, length).decode("utf-8", errors="replace")


def _seek_checked(fh: BinaryIO, nbytes: int, file_size: int) -> None:
    """Skip ``nbytes`` forward, refusing to seek past the file's real end.

    ``seek`` never fails on a plain file even when the target offset is
    past EOF — the OS happily "succeeds" and a subsequent read just
    returns fewer bytes than expected. That silence is what let a
    49-byte file with a declared 16M-element array parse "successfully"
    (see module docstring). Checking the target offset against the
    ``file_size`` captured once via ``stat()`` at the top of
    :func:`read_gguf_metadata` turns that into an immediate, cheap
    :class:`GGUFParseError` instead of either an unbounded loop or a
    silently-wrong result.
    """
    pos = fh.tell()
    if pos + nbytes > file_size:
        raise GGUFParseError(
            f"value extends past EOF (offset {pos} + {nbytes} bytes "
            f"> file size {file_size})"
        )
    fh.seek(nbytes, 1)


def _skip_value(fh: BinaryIO, type_tag: int, file_size: int, *, depth: int = 0) -> None:
    """Consume a metadata value of ``type_tag`` without retaining it.

    Iterative (explicit work-stack), not recursive: a metadata value can
    be an array of arrays, and a corrupt/hostile file can nest that
    arbitrarily deep for only 12 bytes per level (``elem_type`` u32 +
    ``count`` u64). Recursing one Python frame per level let a
    ~12 KiB file blow the interpreter's stack with an uncaught
    ``RecursionError`` (see module docstring) — the C-level exception
    that walks straight through :class:`GGUFParseError`'s ``except``
    clauses. Doing the walk with a stack of ``(type_tag, remaining,
    depth)`` frames instead makes ``RecursionError`` structurally
    impossible here, and :data:`_MAX_ARRAY_DEPTH` still bounds how deep
    a legitimate-looking nested array is allowed to go (DoS guard,
    independent of the recursion fix).

    Arrays of fixed-width scalars (the common case: token ids, scores,
    ``token_type``, ...) are skipped with a single bounds-checked
    ``seek`` for the whole array rather than one iteration per element —
    this is what turns an 8s parse of a 150K-element array into a
    sub-millisecond one.
    """
    # Each frame: (type_tag_to_process, how_many_times_left, nesting_depth).
    stack: list[tuple[int, int, int]] = [(type_tag, 1, depth)]
    while stack:
        tag, remaining, d = stack.pop()
        if remaining > 1:
            stack.append((tag, remaining - 1, d))

        if tag == _GGUF_TYPE_STRING:
            _read_gguf_string(fh)
            continue

        if tag == _GGUF_TYPE_ARRAY:
            if d >= _MAX_ARRAY_DEPTH:
                raise GGUFParseError(
                    f"array nesting depth exceeds cap ({_MAX_ARRAY_DEPTH}); "
                    "refusing to descend further"
                )
            elem_type = _read_u32(fh)
            count = _read_u64(fh)
            if count > _MAX_ARRAY_LEN:
                raise GGUFParseError(f"array length {count} exceeds cap")
            fmt_size = _SCALAR.get(elem_type)
            if fmt_size is not None:
                # Fixed-width elements: one bounds-checked bulk seek,
                # not a per-element loop.
                _seek_checked(fh, count * fmt_size[1], file_size)
            elif elem_type in (_GGUF_TYPE_STRING, _GGUF_TYPE_ARRAY):
                if count > 0:
                    stack.append((elem_type, count, d + 1))
            else:
                raise GGUFParseError(f"unknown value type tag {elem_type}")
            continue

        # Lone scalar (only reachable if _skip_value is ever invoked
        # directly on a scalar type_tag; kept for defense in depth).
        fmt_size = _SCALAR.get(tag)
        if fmt_size is None:
            raise GGUFParseError(f"unknown value type tag {tag}")
        _seek_checked(fh, fmt_size[1], file_size)


def _read_scalar_value(fh: BinaryIO, type_tag: int, file_size: int) -> object:
    """Read (and return) a value, skipping arrays/strings we don't need."""
    if type_tag == _GGUF_TYPE_STRING:
        return _read_gguf_string(fh)
    if type_tag == _GGUF_TYPE_ARRAY:
        _skip_value(fh, type_tag, file_size)
        return None
    return _read_scalar(fh, type_tag)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Suffixes of the arch-prefixed keys we capture (e.g. "llama.block_count").
_KEY_BLOCK_COUNT = ".block_count"
_KEY_EMBED_LEN = ".embedding_length"
_KEY_HEAD_COUNT = ".attention.head_count"
_KEY_HEAD_COUNT_KV = ".attention.head_count_kv"
# Number of Multi-Token-Prediction (nextn) layers embedded in the main model
# (e.g. ``glm4moe.nextn_predict_layers``). A positive value means the GGUF can
# drive llama.cpp speculative decoding via ``--spec-type draft-mtp`` alone.
_KEY_NEXTN = ".nextn_predict_layers"


def read_gguf_metadata(path: str | Path) -> GGUFInfo:
    """Parse the GGUF header at ``path`` and return a :class:`GGUFInfo`.

    Raises :class:`GGUFParseError` if the file is missing, too short,
    or not a GGUF container. Captures only the keys needed for memory
    accounting; everything else is skipped.
    """
    p = Path(path)
    try:
        file_size = p.stat().st_size
    except OSError as exc:  # missing / unreadable
        raise GGUFParseError(f"cannot stat {path}: {exc}") from exc

    arch: str | None = None
    n_layers: int | None = None
    n_embd: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    file_type: int | None = None
    n_nextn: int | None = None

    try:
        fh = p.open("rb")
    except OSError as exc:
        # TOCTOU: the file can vanish (or become unreadable) between the
        # stat() above and this open() — e.g. a concurrent model-directory
        # scan racing a download/cleanup. Convert to GGUFParseError so
        # callers only need to catch one exception type.
        raise GGUFParseError(f"cannot open {path}: {exc}") from exc

    with fh:
        magic = fh.read(4)
        if magic != _GGUF_MAGIC:
            raise GGUFParseError(f"bad magic {magic!r} (not a GGUF file)")
        version = _read_u32(fh)
        if version not in (2, 3):
            raise GGUFParseError(f"unsupported GGUF version {version}")
        _read_u64(fh)  # tensor_count: advance cursor, not needed
        kv_count = _read_u64(fh)
        if kv_count > _MAX_KV_PAIRS:
            raise GGUFParseError(f"kv_count {kv_count} exceeds cap")

        for _ in range(kv_count):
            key = _read_gguf_string(fh)
            value_type = _read_u32(fh)
            value = _read_scalar_value(fh, value_type, file_size)

            if key == "general.architecture" and isinstance(value, str):
                arch = value
            elif key == "general.file_type" and isinstance(value, int):
                file_type = value
            elif key.endswith(_KEY_BLOCK_COUNT) and isinstance(value, int):
                n_layers = value
            elif key.endswith(_KEY_EMBED_LEN) and isinstance(value, int):
                n_embd = value
            elif key.endswith(_KEY_HEAD_COUNT_KV) and isinstance(value, int):
                n_kv_heads = value
            elif key.endswith(_KEY_HEAD_COUNT) and isinstance(value, int):
                n_heads = value
            elif key.endswith(_KEY_NEXTN) and isinstance(value, int):
                n_nextn = value

    return GGUFInfo(
        architecture=arch,
        n_layers=n_layers,
        n_embd=n_embd,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        file_type=file_type,
        file_size_bytes=file_size,
        n_nextn=n_nextn,
    )


def try_read_gguf_metadata(path: str | Path) -> GGUFInfo | None:
    """Like :func:`read_gguf_metadata` but returns None on any parse
    failure — convenient for best-effort advisory paths.

    Catches ``OSError`` in addition to :class:`GGUFParseError` because a
    TOCTOU race can still surface a raw ``OSError`` from ``open()`` in
    principle, and ``RecursionError`` / ``MemoryError`` as defense in
    depth: with the iterative :func:`_skip_value` walk and the depth cap
    above, neither should actually be reachable from this module
    anymore, but a caller in a best-effort advisory path (e.g. scanning
    every ``.gguf`` in a model directory) should never be taken down by
    one regardless.
    """
    try:
        return read_gguf_metadata(path)
    except (GGUFParseError, OSError, RecursionError, MemoryError):
        return None


__all__ = [
    "GGUFInfo",
    "GGUFParseError",
    "read_gguf_metadata",
    "try_read_gguf_metadata",
]
