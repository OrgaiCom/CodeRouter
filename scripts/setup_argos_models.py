#!/usr/bin/env python3
"""Setup Argos Translate models for translation layer.

Downloads (if needed) and verifies translate-ja_en-1_1.argosmodel and
translate-en_ja-1_1.argosmodel, then optionally installs into Argos cache
or model_dir.

Offline verification: checks SHA256 and direct model availability.
Usage:
    python scripts/setup_argos_models.py --model-dir ./models/argos --verify-only
    python scripts/setup_argos_models.py --model-dir ./models/argos

Design: doc/翻訳層設計書.md §8.2, §10
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Known model file names
MODELS = ["translate-ja_en-1_1.argosmodel", "translate-en_ja-1_1.argosmodel"]

# SHA256 verification map — fill after downloading from official Argos source.
# Procedure:
#   1. Download .argosmodel from https://www.argosopentech.com/argospm/index/ or
#      https://github.com/argosopentech/argos-translate/releases
#   2. Compute: python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('f').read_bytes()).hexdigest())"
#   3. Paste the hash here and commit. CI must run without --skip-hash-check.
# TODO(K-1): Fill EXPECTED_SHA256 before v2.16.0 release.
# BLOCKING: CI MUST run `python scripts/setup_argos_models.py --verify-only`
# WITHOUT --skip-hash-check. The warn-only path is for local dev only.
# Steps:
#   1. Download models from official Argos source (see procedure above)
#   2. Compute SHA256 and fill the dict below
#   3. Remove --skip-hash-check from any CI invocations
#   4. Verify: grep -r 'skip-hash-check' .github/ must return empty
# See doc/翻訳層設計書.md §10. Review gate: doc/review-2026-09-02.md R-1.
EXPECTED_SHA256: dict[str, str] = {
    # "translate-ja_en-1_1.argosmodel": "<fill after download>",
    # "translate-en_ja-1_1.argosmodel": "<fill after download>",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_hash_configured() -> bool:
    return bool(EXPECTED_SHA256) and any(v and not v.startswith("<fill") for v in EXPECTED_SHA256.values())


def verify_model_file(path: Path, skip_hash: bool = False) -> bool:
    if not path.is_file():
        print(f"[error] not found: {path}", file=sys.stderr)
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[ok] found {path.name} ({size_mb:.1f} MB)")
    if not skip_hash and path.name in EXPECTED_SHA256:
        expected = EXPECTED_SHA256[path.name]
        if expected and not expected.startswith("<fill"):
            actual = sha256_of(path)
            if actual != expected:
                print(f"[error] SHA256 mismatch for {path.name}: expected {expected}, got {actual}", file=sys.stderr)
                return False
            print(f"[ok] SHA256 verified: {path.name}")
            return True
    # No valid expected hash configured
    if not skip_hash:
        if not _is_hash_configured():
            print(
                f"[warn] EXPECTED_SHA256 not configured for {path.name} -- TODO(K-1) blocking before v2.16.0",
                file=sys.stderr,
            )
            print("  Run: python -c \"import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('"
                + path.name + "').read_bytes()).hexdigest())\" and fill EXPECTED_SHA256",
                file=sys.stderr,
            )
            print("  CI must fail until hash is filled; local dev may use --skip-hash-check", file=sys.stderr)
        else:
            print(f"[warn] no expected SHA256 for {path.name}, skipping hash check (use --skip-hash-check to suppress)", file=sys.stderr)
        actual = sha256_of(path)
        print(f"  actual SHA256: {actual}")
    return True


def argos_direct_available() -> bool:
    """Check if Argos direct models are loadable (requires argostranslate installed)."""
    try:
        from argostranslate import translate  # type: ignore[import-untyped]
    except ImportError:
        print("[warn] argostranslate not installed -- skipping direct model check (pip install coderouter-cli[translation])")
        return False
    try:
        ja_en = translate.get_translation_from_codes("ja", "en")  # type: ignore[attr-defined]
        en_ja = translate.get_translation_from_codes("en", "ja")  # type: ignore[attr-defined]
        if ja_en is None or en_ja is None:
            print("[error] Direct JA<->EN models not found in Argos package index", file=sys.stderr)
            print("  Ensure .argosmodel files are installed via `argospm install` or `argos-translate --install`", file=sys.stderr)
            return False
        # Check direct codes
        if getattr(ja_en, "from_code", "ja") != "ja" or getattr(ja_en, "to_code", "en") != "en":
            print("[error] JA->EN is pivot, not direct", file=sys.stderr)
            return False
        if getattr(en_ja, "from_code", "en") != "en" or getattr(en_ja, "to_code", "ja") != "ja":
            print("[error] EN->JA is pivot, not direct", file=sys.stderr)
            return False
        print("[ok] Argos direct models available: ja→en, en→ja")
        return True
    except Exception as exc:
        print(f"[error] Argos check failed: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup Argos JA<->EN models (verify .argosmodel hash and Argos availability)")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory containing .argosmodel files (optional, default: Argos cache). No auto-install; use argospm install manually.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify files and Argos index; do not attempt install (verify-only is the safe default)")
    parser.add_argument("--skip-hash-check", action="store_true", help="Skip SHA256 verification (temporary until EXPECTED_SHA256 is filled; do not use in CI)")
    parser.add_argument("--require-hash", action="store_true", help="Fail if EXPECTED_SHA256 not configured (CI gate; ensures TODO K-1 is resolved)")
    args = parser.parse_args()

    # CI gate: --require-hash ensures release blocker is not bypassed
    if args.require_hash and not _is_hash_configured():
        print("[error] EXPECTED_SHA256 not configured -- TODO(K-1) must be resolved before release", file=sys.stderr)
        print("  Fill EXPECTED_SHA256 per procedure in file header, then remove --skip-hash-check from CI", file=sys.stderr)
        return 1
    if not _is_hash_configured() and not args.skip_hash_check:
        print("[warn] EXPECTED_SHA256 empty -- BLOCKING before v2.16.0 (doc/review-2026-09-02 R-1)", file=sys.stderr)
        print("  CI should run with --require-hash to enforce failure; grep -r 'skip-hash-check' .github/ must be empty", file=sys.stderr)

    ok = True
    if args.model_dir:
        md = Path(args.model_dir).expanduser()
        print(f"Checking model_dir: {md}")
        for name in MODELS:
            ok &= verify_model_file(md / name, skip_hash=args.skip_hash_check)
    else:
        print("No --model-dir given, checking Argos package index")

    if args.verify_only:
        ok &= argos_direct_available()
        return 0 if ok else 1

    # Verify Argos availability after potential install
    if not argos_direct_available():
        print("[info] If models are in --model-dir, install them via:", file=sys.stderr)
        print("  argospm install <path/to/*.argosmodel>  or  python -m argostranslate.package --install", file=sys.stderr)
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
