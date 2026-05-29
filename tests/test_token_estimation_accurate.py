"""Unit tests for coderouter.token_estimation_accurate (optional backend)."""

from __future__ import annotations

from pathlib import Path

from coderouter import token_estimation_accurate as tea
from coderouter.token_estimation import CHARS_PER_TOKEN_HEURISTIC


def setup_function() -> None:
    tea.reset_cache()


def test_is_accuracy_available_returns_bool() -> None:
    assert isinstance(tea.is_accuracy_available(), bool)


def test_empty_text_is_zero() -> None:
    assert tea.count_tokens("") == 0


def test_no_path_uses_heuristic() -> None:
    text = "x" * 40
    assert tea.count_tokens(text) == 40 // CHARS_PER_TOKEN_HEURISTIC


def test_missing_tokenizer_path_falls_back(tmp_path: Path) -> None:
    text = "y" * 80
    missing = tmp_path / "nope" / "tokenizer.json"
    assert tea.count_tokens(text, tokenizer_path=missing) == (
        80 // CHARS_PER_TOKEN_HEURISTIC
    )


def test_invalid_tokenizer_file_falls_back(tmp_path: Path) -> None:
    bad = tmp_path / "tokenizer.json"
    bad.write_text("{ not valid tokenizer json")
    text = "z" * 100
    # Whether or not the backend is installed, a bad file must degrade
    # to the heuristic rather than raise.
    assert tea.count_tokens(text, tokenizer_path=bad) == (
        100 // CHARS_PER_TOKEN_HEURISTIC
    )


def test_precise_backend_when_available(tmp_path: Path) -> None:
    """If `tokenizers` is installed, a real tokenizer.json must count
    precisely; otherwise this test verifies the graceful fallback."""
    if not tea.is_accuracy_available():
        # Backend absent → fallback path already covered above. No-op.
        return
    from tokenizers import Tokenizer, models, pre_tokenizers

    # Build a trivial whitespace word-level tokenizer and persist it.
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    from tokenizers.trainers import WordLevelTrainer

    trainer = WordLevelTrainer(special_tokens=["[UNK]"])
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog")
    tok.train([str(corpus)], trainer)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))

    text = "the quick brown fox"
    n = tea.count_tokens(text, tokenizer_path=tok_path)
    assert n == 4  # four whitespace-separated tokens
