"""Speculative-decoding / MTP flag resolution for the llama.cpp launcher.

Why this module exists
======================

llama.cpp's ``llama-server`` (2026) folds Multi-Token Prediction (MTP) into
its speculative-decoding framework. Instead of asking the operator to hand-
assemble the right ``--spec-type`` / ``--model-draft`` incantation, the
launcher exposes two high-level knobs — ``draft_model_path`` and
``mtp_mode`` — and derives the concrete CLI tokens here.

Both launchers (the FastAPI web UI in
:mod:`coderouter.ingress.launcher_routes` and the tkinter desktop app in
``launcher_gui.py``) call :func:`resolve_speculative` so the decision logic
lives in exactly one place.

Decision summary (llama.cpp only)
---------------------------------

* ``mtp_mode="off"`` — never emit speculative flags (reproduces the historical
  command exactly). Combining it with ``draft_model_path`` is a conflict.
* Explicit ``draft_model_path`` — validate it, then pick ``draft-mtp`` when the
  filename looks MTP-ish (``/mtp/i``) or ``draft-simple`` otherwise.
* ``mtp_mode="auto"`` (the default) — inspect the main GGUF: if it embeds nextn
  layers (``{arch}.nextn_predict_layers > 0``) use ``--spec-type draft-mtp``
  with no separate draft model; otherwise scan the *same directory* for a
  small companion draft/MTP gguf and wire it up if one is found.

The functions here are pure stdlib + :mod:`coderouter.gguf_introspect`; they
perform filesystem reads but never spawn processes and hold no FastAPI
dependency, which keeps them trivially unit-testable.

Security
--------

Every resolved path is placed into an argv list by the caller (never a shell
string), and ``draft_model_path`` must resolve to an existing regular file —
so this module adds no new shell-interpolation or path-probing surface beyond
reading GGUF headers the operator already pointed us at.
"""

from __future__ import annotations

import re
from pathlib import Path

from coderouter.gguf_introspect import try_read_gguf_metadata

__all__ = ["find_draft_companion", "resolve_speculative"]

_LLAMA_CPP = "llama.cpp"

# Companion candidates must be meaningfully smaller than the main model — a
# draft / MTP head is a fraction of the target's weights. Guards against
# mistaking a second full model in the same folder for a draft.
_COMPANION_MAX_SIZE_RATIO = 0.5

# Minimum stripped-prefix length before a shared-prefix match is trusted
# (avoids matching two unrelated models that merely share a short token).
_MIN_PREFIX_LEN = 3

# Trailing GGUF shard marker, e.g. "-00001-of-00003".
_SHARD_RE = re.compile(r"[-._]\d{5}-of-\d{5}$", re.IGNORECASE)

# Trailing quantisation token, e.g. "-Q4_K_M", ".IQ3_XXS", "-F16", "-BF16".
_QUANT_RE = re.compile(
    r"[-._]?(i?q\d[_a-z0-9]*|bf16|fp?16|fp?32)$",
    re.IGNORECASE,
)

# Filename hint that the companion carries MTP tensors (→ draft-mtp).
_MTP_NAME_RE = re.compile(r"mtp", re.IGNORECASE)


def _strip_quant_suffix(stem: str) -> str:
    """Return ``stem`` with a trailing shard marker and quant token removed.

    Used to derive a model's "family prefix" so a companion sitting next to
    ``Foo-7B-Q4_K_M.gguf`` (prefix ``Foo-7B``) can be matched even though its
    own quant/shape suffix differs.
    """
    s = _SHARD_RE.sub("", stem)
    s = _QUANT_RE.sub("", s)
    return s


def _has_spec_type(user_tokens: list[str]) -> bool:
    """True if the operator already supplied ``--spec-type`` in extra args."""
    return any(
        tok == "--spec-type" or tok.startswith("--spec-type=")
        for tok in user_tokens
    )


def _has_split_mode_tensor(user_tokens: list[str]) -> bool:
    """True if the tokens request ``--split-mode tensor`` (either form).

    llama.cpp issue #24309: a nextn-embedded model combined with tensor split
    crashes, so callers warn (but do not block) when this pairs with emitted
    speculative flags.
    """
    for i, tok in enumerate(user_tokens):
        if tok == "--split-mode=tensor":
            return True
        if (
            tok == "--split-mode"
            and i + 1 < len(user_tokens)
            and user_tokens[i + 1] == "tensor"
        ):
            return True
    return False


def _spec_type_for_name(path: Path) -> str:
    """Pick ``draft-mtp`` for MTP-named files, ``draft-simple`` otherwise."""
    return "draft-mtp" if _MTP_NAME_RE.search(path.name) else "draft-simple"


def find_draft_companion(main_path: str | Path) -> Path | None:
    """Scan the directory of ``main_path`` for a companion draft/MTP gguf.

    A candidate qualifies when it is a ``*.gguf`` other than the main file,
    is smaller than :data:`_COMPANION_MAX_SIZE_RATIO` of the main file, and
    either:

    * has an ``mtp`` or ``draft`` hint in its filename, or
    * shares the main file's family prefix (main filename with its shard /
      quant suffix stripped).

    If a candidate's GGUF ``architecture`` is readable and differs from the
    main model's, it is dropped (tokenizer/vocab must match for speculation);
    unreadable metadata is kept on a best-effort basis.

    Candidates are ranked by name hint (``mtp`` > ``draft`` > none), then by
    whether they share the prefix, then by smallest size. Returns the best
    match, or ``None`` when the directory holds no suitable companion.
    """
    main = Path(main_path).expanduser()
    directory = main.parent
    try:
        main_size = main.stat().st_size
    except OSError:
        return None
    if main_size <= 0:
        return None

    main_prefix = _strip_quant_suffix(main.stem)
    main_meta = try_read_gguf_metadata(main)
    main_arch = main_meta.architecture if main_meta else None

    try:
        entries = list(directory.iterdir())
    except OSError:
        return None

    # Each ranked candidate: (name_hint, shares_prefix, -size) sorted so that
    # the strongest hint / prefix / smallest file sorts first.
    ranked: list[tuple[int, int, int, Path]] = []
    for cand in entries:
        if cand == main:
            continue
        if cand.suffix.lower() != ".gguf":
            continue
        try:
            if not cand.is_file():
                continue
            size = cand.stat().st_size
        except OSError:
            continue
        if size <= 0 or size >= main_size * _COMPANION_MAX_SIZE_RATIO:
            continue

        name_lower = cand.name.lower()
        has_mtp = "mtp" in name_lower
        has_draft = "draft" in name_lower
        shares_prefix = (
            len(main_prefix) >= _MIN_PREFIX_LEN
            and cand.stem.lower().startswith(main_prefix.lower())
        )
        if not (has_mtp or has_draft or shares_prefix):
            continue

        # Reject a candidate whose architecture is readable and mismatches the
        # main model — a speculator must share the target's vocabulary.
        cand_meta = try_read_gguf_metadata(cand)
        if (
            cand_meta is not None
            and cand_meta.architecture is not None
            and main_arch is not None
            and cand_meta.architecture != main_arch
        ):
            continue

        name_hint = 2 if has_mtp else (1 if has_draft else 0)
        ranked.append((name_hint, int(shares_prefix), size, cand))

    if not ranked:
        return None
    # Highest hint, then prefix match, then smallest size.
    ranked.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return ranked[0][3]


def resolve_speculative(
    backend: str,
    model_path: str,
    draft_model_path: str | None,
    mtp_mode: str,
    user_tokens: list[str],
) -> tuple[list[str], list[str]]:
    """Resolve speculative-decoding / MTP CLI tokens for a launch.

    Parameters
    ----------
    backend:
        Target backend id. Only ``"llama.cpp"`` supports speculation; other
        backends must not receive ``draft_model_path`` / ``mtp_mode="off"``.
    model_path:
        The vetted main model path (already set via the launcher's dedicated
        field). Inspected for embedded nextn layers / same-folder companions.
    draft_model_path:
        Optional explicit companion draft or MTP gguf. ``None`` means "not
        supplied".
    mtp_mode:
        ``"auto"`` (default behaviour — detect) or ``"off"`` (never emit
        speculative flags).
    user_tokens:
        The flat list of option/extra-args tokens already destined for the
        command line. Used to defer to an operator-supplied ``--spec-type``
        and to warn about the tensor-split crash.

    Returns
    -------
    ``(spec_tokens, notes)`` where ``spec_tokens`` is the list of CLI tokens to
    append (empty when none apply) and ``notes`` is a list of human-readable
    strings describing what was decided (surfaced in the process log).

    Raises
    ------
    ValueError
        On misuse — the web API maps this to HTTP 400:

        * ``draft_model_path`` / ``mtp_mode="off"`` on a non-llama.cpp backend,
        * ``mtp_mode="off"`` combined with an explicit ``draft_model_path``,
        * an explicit ``draft_model_path`` that does not resolve to a file.
    """
    notes: list[str] = []

    # 1. Non-llama.cpp backends do not support speculation. Reject explicit
    #    use; otherwise (auto default, no draft) stay a silent no-op.
    if backend != _LLAMA_CPP:
        if draft_model_path or mtp_mode == "off":
            raise ValueError(
                "draft_model_path / mtp_mode are only supported for llama.cpp"
            )
        return [], notes

    # 2. Explicit opt-out. Never emit flags; conflicting draft path is an error.
    if mtp_mode == "off":
        if draft_model_path:
            raise ValueError(
                "mtp_mode='off' conflicts with an explicit draft_model_path"
            )
        return [], notes

    # 3. Operator already drives speculation via extra args → defer entirely.
    if _has_spec_type(user_tokens):
        notes.append(
            "spec flags supplied via extra args; auto MTP detection skipped"
        )
        return [], notes

    spec_tokens: list[str] = []

    # 4. Explicit companion draft / MTP gguf.
    if draft_model_path:
        draft = Path(draft_model_path).expanduser()
        if not draft.is_file():
            raise ValueError(
                f"draft_model_path does not exist or is not a file: {draft}"
            )
        spec_type = _spec_type_for_name(draft)
        spec_tokens = ["--spec-type", spec_type, "--model-draft", str(draft)]
        notes.append(
            f"MTP: explicit draft model {draft.name} → --spec-type {spec_type}"
        )
    # 5. Auto detection (only when the main path is an existing .gguf).
    else:
        main = Path(model_path).expanduser()
        if main.suffix.lower() != ".gguf" or not main.is_file():
            notes.append(
                "MTP auto: main model is not an existing .gguf; "
                "skipping speculative detection"
            )
        else:
            info = try_read_gguf_metadata(main)
            if info is not None and info.n_nextn and info.n_nextn > 0:
                # 5a. Main gguf embeds nextn/MTP tensors — no draft model needed.
                spec_tokens = ["--spec-type", "draft-mtp"]
                notes.append(
                    f"MTP: nextn layers ({info.n_nextn}) detected in main "
                    "gguf → --spec-type draft-mtp"
                )
            else:
                # 5b. Look for a companion gguf next to the main model.
                companion = find_draft_companion(main)
                if companion is not None:
                    spec_type = _spec_type_for_name(companion)
                    spec_tokens = [
                        "--spec-type",
                        spec_type,
                        "--model-draft",
                        str(companion),
                    ]
                    notes.append(
                        f"MTP: companion gguf {companion.name} found next to "
                        f"main → --spec-type {spec_type}"
                    )
                else:
                    # 5c. Nothing found — start without speculative decoding.
                    notes.append(
                        f"MTP/draft gguf not found next to {main.name}; "
                        "starting without speculative decoding"
                    )

    # 6. Warn (do not block) about the nextn + tensor-split crash.
    if spec_tokens and _has_split_mode_tensor(user_tokens):
        notes.append(
            "WARNING: --split-mode tensor with speculative/nextn is known to "
            "crash llama.cpp (issue #24309); consider --split-mode layer"
        )

    return spec_tokens, notes
