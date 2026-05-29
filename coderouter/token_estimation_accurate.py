"""Optional precision token counting (low-memory accuracy track).

The core estimator in :mod:`coderouter.token_estimation` uses a
``char/4`` heuristic that under-counts CJK text badly — which is
exactly the failure mode that makes the memory-budget guard either
OOM (under-count) or over-trim (over-count). This module offers an
opt-in precise backend without breaking the 5-deps invariant.

Design
======

* **Optional dependency.** ``tokenizers`` (HuggingFace, Rust core) is
  declared under the ``accuracy`` extra, *not* a core dependency. It is
  imported lazily; if absent, every function falls back to the char/4
  heuristic. Callers always get an ``int``.

* **Local files only — no network.** We load tokenizers exclusively via
  ``Tokenizer.from_file(<local tokenizer.json>)``. We never call
  ``from_pretrained`` or anything that contacts the HuggingFace Hub, so
  this module performs **zero network I/O** and cannot be steered into
  downloading arbitrary content.

* **No pickle / no torch.** ``tokenizers`` reads JSON only; we never
  import ``torch`` or ``transformers`` (avoids the pickle-deserialization
  RCE surface).

A loaded tokenizer is cached per resolved path so repeated requests
don't re-parse ``tokenizer.json``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from coderouter.token_estimation import CHARS_PER_TOKEN_HEURISTIC

# ---------------------------------------------------------------------------
# Lazy backend detection
# ---------------------------------------------------------------------------

_backend_lock = threading.RLock()
_tokenizer_cache: dict[str, Any] = {}
_accuracy_available: bool | None = None


def is_accuracy_available() -> bool:
    """True iff the optional ``tokenizers`` backend can be imported.

    Result is memoised. Never raises — a missing package simply
    returns False (callers fall back to the heuristic).
    """
    global _accuracy_available
    if _accuracy_available is not None:
        return _accuracy_available
    with _backend_lock:
        if _accuracy_available is None:
            try:
                import tokenizers  # noqa: F401  (probe only)

                _accuracy_available = True
            except Exception:  # pragma: no cover - import failure path
                _accuracy_available = False
        return _accuracy_available


def _load_tokenizer(tokenizer_path: str | Path) -> Any | None:
    """Load and cache a tokenizer from a **local** ``tokenizer.json``.

    Returns None if the backend is unavailable, the path is missing,
    or the file fails to parse. Strictly local — never touches the Hub.
    """
    if not is_accuracy_available():
        return None
    p = Path(tokenizer_path)
    key = str(p.resolve()) if p.exists() else str(p)
    with _backend_lock:
        if key in _tokenizer_cache:
            return _tokenizer_cache[key]
        if not p.is_file():
            _tokenizer_cache[key] = None
            return None
        try:
            from tokenizers import Tokenizer  # local import

            tok = Tokenizer.from_file(str(p))  # local file only, no network
        except Exception:
            tok = None
        _tokenizer_cache[key] = tok
        return tok


def reset_cache() -> None:
    """Clear the tokenizer cache and backend probe. Mainly for tests."""
    global _accuracy_available
    with _backend_lock:
        _tokenizer_cache.clear()
        _accuracy_available = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _heuristic(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN_HEURISTIC


def count_tokens(text: str, *, tokenizer_path: str | Path | None = None) -> int:
    """Count tokens in ``text``.

    Uses the precise ``tokenizers`` backend when ``tokenizer_path``
    points at a readable local ``tokenizer.json`` *and* the optional
    dependency is installed; otherwise falls back to the char/4
    heuristic. Always returns a non-negative ``int`` and never raises
    on backend problems.
    """
    if not text:
        return 0
    if tokenizer_path is not None:
        tok = _load_tokenizer(tokenizer_path)
        if tok is not None:
            try:
                return len(tok.encode(text).ids)
            except Exception:  # pragma: no cover - encode failure path
                return _heuristic(text)
    return _heuristic(text)


__all__ = [
    "count_tokens",
    "is_accuracy_available",
    "reset_cache",
]
