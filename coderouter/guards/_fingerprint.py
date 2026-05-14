"""Response fingerprinting for goal_progress_stall detection (P1-4).

A "fingerprint" is a compact, order-independent signature of the *content*
of an assistant response — independent of surface variation (filler phrases,
minor rewordings).  Two responses with the same fingerprint are considered
semantically repetitive for stall-detection purposes.

Algorithm
---------
1. Normalise: lowercase, strip punctuation, collapse whitespace.
2. Extract the N most-frequent content words (excluding a small stop-list).
3. Sort alphabetically, join with '|', SHA-256 → 12-hex prefix.

The 12-hex prefix gives 281 trillion distinct values — collision probability
across any 20-response window is negligible (< 1 in 10^15).

Why top-N content words instead of full hash?
----------------------------------------------
A verbatim hash would fail to catch "I cannot do X. Let me try Y" vs
"Let me try Y as I cannot do X" — same stall, different hash.  By
extracting the dominant vocabulary we get useful fuzzy equality without
the overhead of embedding models.

Usage
-----
    from coderouter.guards._fingerprint import fingerprint_response

    fp = fingerprint_response(response_text)
    obs = ResponseObservation(..., response_fingerprint=fp)
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# ---------------------------------------------------------------------------
# Stop-word list (English + common LLM filler)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        # English function words
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "it", "its", "be",
        "was", "are", "were", "been", "has", "have", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "shall",
        "this", "that", "these", "those", "i", "you", "he", "she", "we",
        "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "their", "our", "what", "which", "who", "how", "when", "where",
        "why", "not", "no", "so", "up", "out", "into", "about", "than",
        "then", "there", "here", "also", "just", "can", "get", "all",
        # Common LLM assistant filler
        "certainly", "sure", "absolutely", "great", "happy", "help",
        "please", "let", "know", "feel", "free", "answer", "question",
        "response", "following", "based", "provide", "using",
    }
)

# ---------------------------------------------------------------------------
# Number of top content words to include in the fingerprint
# ---------------------------------------------------------------------------

_TOP_N: int = 12


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fingerprint_response(text: str, *, top_n: int = _TOP_N) -> str:
    """Return a 12-hex fingerprint string for *text*.

    Parameters
    ----------
    text:
        Raw assistant response text (plain text, not JSON).
    top_n:
        Number of most-frequent content words to include in the signature.
        Defaults to ``_TOP_N`` (12).  Lower values are more fuzzy; higher
        values are more precise.

    Returns
    -------
    A 12-character lowercase hexadecimal string, e.g. ``"a3f7b2c091de"``.
    Returns ``""`` for empty / whitespace-only input.
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode normalisation + lowercase
    normalised = unicodedata.normalize("NFKC", text).lower()

    # 2. Strip punctuation / digits, collapse whitespace
    normalised = re.sub(r"[^\w\s]", " ", normalised)
    normalised = re.sub(r"\d+", " ", normalised)
    normalised = re.sub(r"\s+", " ", normalised).strip()

    # 3. Tokenise and filter stop words (also skip very short tokens)
    tokens = [w for w in normalised.split() if len(w) > 2 and w not in _STOP_WORDS]

    if not tokens:
        return ""

    # 4. Count frequencies, take top-N
    freq: dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1

    top_words = sorted(freq, key=lambda w: (-freq[w], w))[:top_n]

    # 5. Sort alphabetically → stable join → hash
    signature = "|".join(sorted(top_words))
    digest = hashlib.sha256(signature.encode()).hexdigest()
    return digest[:12]
