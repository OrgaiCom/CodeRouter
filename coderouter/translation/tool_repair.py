"""Tool-call repair: extract tool invocations that a model wrote as plain text.

Background
----------
Small coding models (e.g. qwen2.5-coder) sometimes respond to a tool-bearing
prompt by *writing* a JSON object describing the tool call in the assistant
message body instead of populating the structured `tool_calls` field. The
downstream Anthropic/OpenAI clients then see regular text and never execute
the tool.

This module scans assistant text for such embedded tool-call JSON and pulls
it back into the OpenAI-shape `tool_calls` list, so the rest of the
translation pipeline (`to_anthropic_response`, stream event emitter) can
produce real `tool_use` content blocks.

Recognised shapes
-----------------
1. Fenced code blocks:
    ```json
    {"name": "Bash", "arguments": {"command": "pwd"}}
    ```
   (the language tag is optional: ``` ...``` also works.)
2. Bare JSON objects embedded in text:
    "Let me check the current directory. {\"name\":\"Bash\",\"arguments\":{}}"
3. Multiple JSON objects in sequence (for multi-call turns).
4. XML-flavoured forms (R3), seen with qwen2.5-coder at default temperature:
    <echo message="probe"/>                        (attribute form)
    <tool>{"name": "echo", "arguments": {...}}</tool>   (wrapper form)
    <echo>{"message": "probe"}</echo>              (named-wrapper form)
5. Nested-XML name-attribute forms (R4a), where the tool name lives in a
   ``name`` attribute rather than the tag itself (L2 default-temp residual):
    <tools><function name="echo" arguments='{...}'/></tools>   (container form)
    <function name="echo" arguments='{...}'/>       (bare call-tag form)
    <tool name="read_file" args='{...}'/>           (args alias)
   The container/call tags are a fixed known set; the ``name`` attribute must
   be allow-listed and the ``arguments``/``args`` value is delegated to R1.
6. JSON envelope forms (R4b), where the model echoes a response wrapper:
    {"tool_calls": [{"name": "echo", "arguments": {...}}, ...]}  (OpenAI list)
    {"function_call": {"name": ..., "arguments": "<JSON string>"}}  (legacy)
   The envelope is unwrapped and each inner object is run through the same
   shape + allow-list validation; the legacy ``arguments`` string is
   double-parsed. Works both fenced and bare.
7. Call-syntax forms (R4c), the "name + parens + args" family, recognised
   ONLY inside a fenced block or on its own standalone line (never inline in
   prose):
    print(default_api.echo(message="probe"))        (Gemma tool_code idiom)
    echo(message="probe")                           (python kwargs)
    echo(message: 'demo')                           (colon-separated kwargs)
    write_note({"path": "a", "text": "b"})          (single JSON-object arg)
   Guards: the function name must be allow-listed, the argument list must
   parse completely (a broken inner JSON is left alone rather than executed
   with corrupt args), and an explanatory example cue preceding the fence
   suppresses extraction.

Each candidate is accepted only if it parses to one of:
    {"name": <str>, "arguments": <dict | str>}          # direct shape
    {"function": {"name": <str>, "arguments": ...}}     # OpenAI shape
    ... plus the R2 key aliases (tool / tool_name / parameters / input / args).

Lenient JSON (R1)
-----------------
When strict ``json.loads`` fails, a second, tolerant pass is attempted for the
malformed forms small models actually emit:
    - double-braced objects  ``{{...}}``  ->  ``{...}``  (L2 default-temp form)
    - trailing commas
    - single-quoted strings / keys (Python-repr dicts)
    - unquoted object keys
The lenient parse is a stdlib-only tokenizer; the resulting dict is still run
through the exact same shape + allow-list validation as a strict parse.

False-positive discipline
-------------------------
If `allowed_tool_names` is provided, the tool `name` must be in that set;
otherwise any tool-shaped JSON is accepted. Passing the allow-list is
strongly recommended.

In addition, two context guards keep genuine documentation / source code from
being turned into executable tool calls, even when it contains an allow-listed
name:
    - programming-language code fences (```python, ```js, ...) are protected:
      their interior is never scanned by the bare-JSON or XML scanners.
    - bare JSON / XML immediately introduced by an explanatory cue
      ("for example", "you would write", "is documented as", ...) is treated
      as prose and left in place.
These guards are conservative: they only ever *suppress* an extraction, so a
name that is not in the allow-list is still never repaired.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import uuid
from array import array
from typing import Any

__all__ = ["deduplicate_tool_calls", "repair_tool_calls_in_text"]

# Plain stdlib logger (same idiom as coderouter/translation/anthropic.py).
# This module is deliberately import-free of the rest of the package:
# benchmarks/tool-repair/run_offline.py loads it straight from its source
# path, with no `coderouter` package on sys.path.
logger = logging.getLogger(__name__)

# Hard ceiling on the text handed to the bare-JSON brace scanners. The
# scanners themselves are linear, but every candidate they emit still costs a
# JSON parse attempt plus a prose-cue check, and an assistant message this
# large is never a tool call anyway. Above the limit the bare-JSON step is
# skipped outright; the fenced / XML / R4a / R4c paths are regex-driven and
# linear, so they keep running.
_MAX_BARE_SCAN_CHARS = 262_144


# ------------------------------------------------------------------
# R2: shape detection + normalisation (with key aliases)
# ------------------------------------------------------------------

# Accepted aliases for the tool-name key and the arguments key. Order does not
# matter for detection, but ambiguity (more than one distinct alias present)
# is treated as "not a tool call" and left untouched (safe side).
_NAME_KEYS = ("name", "tool", "tool_name")
_ARG_KEYS = ("arguments", "parameters", "input", "args")


def _pick_unique(obj: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    """Return (matched_key, value) if exactly one of ``keys`` is present.

    If zero keys are present, returns (None, None). If more than one key is
    present the result is ambiguous -> returns ("<ambiguous>", None) so callers
    can decline to repair.
    """
    present = [k for k in keys if k in obj]
    if not present:
        return None, None
    if len(present) > 1:
        return "<ambiguous>", None
    k = present[0]
    return k, obj[k]


def _looks_like_tool_call(obj: Any, allowed: set[str] | None) -> tuple[str, Any] | None:
    """Return (name, arguments) if obj looks like a tool call, else None.

    Recognises:
      - direct shape with name-key alias + args-key alias
      - OpenAI ``{"function": {...}}`` wrapper (recursed into)
    Ambiguous objects (multiple name aliases or multiple arg aliases) are
    rejected.
    """
    if not isinstance(obj, dict):
        return None

    # OpenAI function shape: {"function": {"name": "...", "arguments": ...}}
    fn = obj.get("function")
    if isinstance(fn, dict):
        inner = _looks_like_tool_call(fn, allowed)
        if inner is not None:
            return inner
        # fall through: maybe the outer dict itself is tool-shaped

    name_key, name = _pick_unique(obj, _NAME_KEYS)
    if name_key == "<ambiguous>":
        return None
    arg_key, args = _pick_unique(obj, _ARG_KEYS)
    if arg_key == "<ambiguous>":
        return None

    if not isinstance(name, str) or name_key is None:
        return None
    if arg_key is None:
        # A tool-shaped object must carry an arguments container to be a call;
        # a lone {"name": "..."} is not distinguishable from ordinary data.
        return None
    if allowed is not None and name not in allowed:
        return None
    return name, args


# R4b: JSON envelope keys whose value carries the actual tool call(s).
#   {"tool_calls": [ {call}, {call}, ... ]}   -> a list of calls
#   {"function_call": {call}}                 -> a single call
# These are the OpenAI response-envelope shapes a model sometimes echoes back
# into the text body verbatim. The envelope wrapper is unwrapped and each inner
# object is validated by the same _looks_like_tool_call predicate.
_ENVELOPE_LIST_KEY = "tool_calls"
_ENVELOPE_SINGLE_KEY = "function_call"


def _expand_tool_call_envelope(
    obj: Any, allowed: set[str] | None
) -> list[tuple[str, Any]] | None:
    """Unwrap an R4b JSON envelope into a list of (name, arguments) tuples.

    Recognises exactly two response-envelope wrappers:
      - ``{"tool_calls": [ ... ]}``    (OpenAI tool_calls list)
      - ``{"function_call": { ... }}`` (OpenAI legacy single call)

    The dict must carry the envelope key and nothing else that would make it
    ambiguous with a plain tool-call object (i.e. it must NOT itself already
    look like a direct call). Every inner element must resolve to an
    allow-listed call, otherwise the whole envelope is declined (all-or-nothing
    keeps false positives at zero — a half-recognised wrapper is suspicious).

    Returns the list of calls, or None if ``obj`` is not an envelope. An empty
    envelope (no valid inner calls) also returns None.
    """
    if not isinstance(obj, dict):
        return None
    # If the object is itself a direct tool call, it is not an envelope; let the
    # ordinary single-call path handle it (avoids double extraction).
    if _looks_like_tool_call(obj, allowed) is not None:
        return None

    if _ENVELOPE_LIST_KEY in obj:
        raw = obj[_ENVELOPE_LIST_KEY]
        if not isinstance(raw, list) or not raw:
            return None
        calls: list[tuple[str, Any]] = []
        for item in raw:
            hit = _looks_like_tool_call(item, allowed)
            if hit is None:
                return None  # all-or-nothing
            calls.append(hit)
        return calls or None

    if _ENVELOPE_SINGLE_KEY in obj:
        inner = obj[_ENVELOPE_SINGLE_KEY]
        hit = _looks_like_tool_call(inner, allowed)
        if hit is None:
            return None
        name, args = hit
        # Legacy shape carries arguments as a JSON *string*; double-parse it so
        # the normaliser emits a proper arguments object where possible. If the
        # inner string is not valid JSON, keep it verbatim (bug-for-bug parity
        # with bare_json_04's string-arguments behaviour).
        if isinstance(args, str):
            with contextlib.suppress(ValueError):
                args = _parse_json_object(args)
        return [(name, args)]

    return None


def _normalise_to_openai_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    """Build an OpenAI-shape tool_calls entry."""
    if isinstance(arguments, str):
        args_str = arguments
    elif isinstance(arguments, dict):
        args_str = json.dumps(arguments, ensure_ascii=False)
    else:
        # list / None / anything else — fall back to serialising what we got.
        args_str = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


# ------------------------------------------------------------------
# R1: lenient JSON parsing (only tried after strict json.loads fails)
# ------------------------------------------------------------------


def _strip_double_braces(s: str) -> str:
    """Collapse a fully double-braced object ``{{...}}`` to ``{...}``.

    Only collapses when the whole (stripped) string is wrapped in exactly one
    extra layer of braces on both ends — the L2 default-temp failure form
    ``{{"name": ...}}``. A single wrap is removed; deeper nesting is left as-is
    (the tokenizer below handles the resulting single-braced object).
    """
    t = s.strip()
    if t.startswith("{{") and t.endswith("}}"):
        return t[1:-1].strip()
    return s


def _lenient_json_loads(raw: str) -> Any:
    """Best-effort parse of near-JSON emitted by small models.

    Handles, in a single tolerant tokenizer pass:
      - single-quoted strings  ('...')  -> double-quoted
      - unquoted object keys             -> quoted
      - trailing commas                  -> dropped
      - (double braces are stripped by the caller before this runs)

    Returns the parsed object, or raises ValueError if the tokenizer cannot
    produce syntactically valid JSON. Never executes the input.
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty")

    out: list[str] = []
    i = 0
    n = len(s)
    # Stack tracks whether the current container is an object ('{') so we can
    # know when a bareword is a key (needs quoting) vs a value.
    while i < n:
        c = s[i]

        # --- string literals: normalise quote char, copy verbatim otherwise ---
        if c == '"' or c == "'":
            quote = c
            j = i + 1
            buf = ['"']
            while j < n:
                cj = s[j]
                if cj == "\\":
                    # keep escape pair verbatim (but a backslash-escaped single
                    # quote inside a single-quoted string becomes a plain ').
                    if j + 1 < n:
                        nxt = s[j + 1]
                        if quote == "'" and nxt == "'":
                            buf.append("'")
                            j += 2
                            continue
                        buf.append(cj)
                        buf.append(nxt)
                        j += 2
                        continue
                    buf.append("\\")
                    j += 1
                    continue
                if cj == quote:
                    j += 1
                    break
                if cj == '"' and quote == "'":
                    # a double quote inside a single-quoted string must be
                    # escaped in the JSON output.
                    buf.append('\\"')
                    j += 1
                    continue
                buf.append(cj)
                j += 1
            else:
                raise ValueError("unterminated string")
            buf.append('"')
            out.append("".join(buf))
            i = j
            continue

        # --- unquoted key/identifier: quote it ---
        if c.isalpha() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] in "_-."):
                j += 1
            word = s[i:j]
            lowered = word.lower()
            if lowered in ("true", "false", "null"):
                out.append(lowered)
            else:
                # bareword -> treat as a quoted key/string token
                out.append('"' + word + '"')
            i = j
            continue

        # --- trailing comma: drop a comma that is followed only by ws then } or ] ---
        if c == ",":
            k = i + 1
            while k < n and s[k] in " \t\r\n":
                k += 1
            if k < n and s[k] in "}]":
                i += 1  # skip the comma
                continue
            out.append(",")
            i += 1
            continue

        out.append(c)
        i += 1

    candidate = "".join(out)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"lenient parse failed: {exc}") from exc


def _parse_json_object(body: str) -> Any:
    """Strict json.loads, then a lenient fallback. Raises ValueError on failure."""
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    return _lenient_json_loads(_strip_double_braces(body))


# ------------------------------------------------------------------
# Scanners: fenced code blocks, then balanced braces in remaining text
# ------------------------------------------------------------------

# Match ```json ... ``` or ``` ... ``` with anything after the fence tag line.
# Group 1 captures the (optional) language tag, group 2 the body.
_FENCED_RE = re.compile(
    r"```([\w+-]*)[ \t]*\r?\n(.*?)\r?\n?```",
    re.DOTALL,
)

# Language tags that denote *real source code* (not a serialised tool call).
# A fence carrying one of these is protected: its interior is never scanned by
# the bare-JSON / XML scanners, so an allow-listed name written as a code
# literal inside such a fence is not turned into a tool call.
_CODE_FENCE_LANGS = frozenset(
    {
        "python",
        "py",
        "python3",
        "js",
        "javascript",
        "ts",
        "typescript",
        "jsx",
        "tsx",
        "go",
        "golang",
        "rust",
        "rs",
        "java",
        "c",
        "cpp",
        "c++",
        "cs",
        "csharp",
        "ruby",
        "rb",
        "php",
        "sh",
        "bash",
        "shell",
        "zsh",
        "sql",
        "html",
        "css",
        "yaml",
        "yml",
        "toml",
        "ini",
        "kotlin",
        "swift",
        "scala",
        "lua",
        "perl",
        "r",
        "dart",
        "elixir",
        "haskell",
    }
)


def _extract_tool_call_fenced_blocks(
    text: str,
    allowed: set[str] | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Pull tool-call-shaped ```...``` blocks out of text.

    Only fenced blocks whose body parses (strictly or leniently) to a
    recognised tool-call shape are removed from the text and returned as
    normalised OpenAI tool_calls. Any other fenced block (prose, real source
    code, non-tool JSON) is preserved verbatim in the returned text — the
    removal decision and the extraction decision use the exact same predicate,
    so a fenced block is never dropped without also being surfaced as a tool
    call (bug H2).

    Fences carrying a *programming-language* tag (``python``, ``js`` ...) are
    protected: their body is never parsed as a tool call, even if it happens to
    contain a tool-shaped literal. This is the code-fence false-positive guard.

    Returns (text_without_tool_fences, tool_calls).
    """
    tool_calls: list[dict[str, Any]] = []

    def _repair(match: re.Match[str]) -> str:
        lang = (match.group(1) or "").strip().lower()
        body = match.group(2).strip()
        if lang in _CODE_FENCE_LANGS:
            return match.group(0)  # protected source-code fence
        # R4c: call-syntax family (name(...)) inside a non-code fence. Only
        # attempted when the body is a single call line and its name is
        # allow-listed with fully-parseable arguments; an example cue on the
        # line(s) preceding the fence suppresses it.
        if not body.startswith("{") and allowed is not None:
            if not _preceded_by_prose_cue_before_fence(text, match.start()):
                call_hits = _extract_call_syntax_lines(body, allowed)
                if call_hits:
                    for name, args in call_hits:
                        tool_calls.append(_normalise_to_openai_tool_call(name, args))
                    return ""
            return match.group(0)  # keep other non-JSON fenced blocks (code)
        if not body.startswith("{"):
            return match.group(0)  # keep non-JSON fenced blocks (e.g. code)
        try:
            obj = _parse_json_object(body)
        except ValueError:
            return match.group(0)  # keep unparseable fenced blocks
        # R4b: response-envelope wrappers ({"tool_calls": [...]}, ...).
        envelope = _expand_tool_call_envelope(obj, allowed)
        if envelope is not None:
            for name, args in envelope:
                tool_calls.append(_normalise_to_openai_tool_call(name, args))
            return ""
        hit = _looks_like_tool_call(obj, allowed)
        if hit is None:
            return match.group(0)  # keep non-tool-call JSON blocks
        name, args = hit
        tool_calls.append(_normalise_to_openai_tool_call(name, args))
        return ""  # remove only recognised tool-call blocks

    cleaned = _FENCED_RE.sub(_repair, text)
    return cleaned, tool_calls


def _protected_code_spans(text: str) -> list[tuple[int, int]]:
    """Byte spans of programming-language code fences to exclude from scanning.

    Returns (start, end) index pairs covering the *entire* fenced block
    (fence markers included) for fences whose language tag is a real
    programming language. Used to shield code from the bare-JSON / XML scanners.
    """
    spans: list[tuple[int, int]] = []
    for m in _FENCED_RE.finditer(text):
        lang = (m.group(1) or "").strip().lower()
        if lang in _CODE_FENCE_LANGS:
            spans.append((m.start(), m.end()))
    return spans


def _all_fence_spans(text: str) -> list[tuple[int, int]]:
    """Byte spans of *every* fenced block (any language tag or none).

    Used by the standalone-line R4c scanner: a call line that survives inside a
    fence was already offered to the fenced R4c path (and, if suppressed by an
    example cue, must stay suppressed), so the standalone scanner must never
    reach inside any fence.
    """
    return [(m.start(), m.end()) for m in _FENCED_RE.finditer(text)]


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


# ------------------------------------------------------------------
# Brace scanning (bare-JSON candidate discovery)
#
# Both scanners below answer the same question for every ``{`` in the text:
# "does a *fresh* scan started right here reach a balanced close?".  The
# obvious implementation re-scans to end-of-text once per ``{``, which is
# O(k*n) for k unclosed braces — a 48 KB assistant message full of stray
# braces took ~20 s, and because ``to_anthropic_response`` is called from the
# request path that stalled the whole server.
#
# Instead we precompute, in ONE right-to-left sweep, a table that answers
# "starting at offset j outside any string, where is the first ``}`` that is
# still at brace depth 0?".  The closing brace of a ``{`` at offset q is then
# simply ``table[q + 1]``, so the driver below is a plain left-to-right walk
# and the whole scan is linear in the length of the text.
#
# The table is defined by a right-to-left recurrence over the scanner's state
# machine (outside a string / inside a string / previous char was a
# backslash).  Only the "outside a string" row needs random access — the ``{``
# case has to jump over the nested object it just resolved — so the
# in-string rows are carried as rolling scalars instead of arrays.
# ------------------------------------------------------------------


def _scan_object_spans(text: str, close_table: array[int]) -> list[tuple[int, int, str]]:
    """Walk ``text`` left to right emitting balanced ``{...}`` spans.

    ``close_table`` comes from :func:`_strict_close_table` or
    :func:`_lenient_close_table` and supplies, for each offset, the index of
    the first depth-0 ``}`` at or after it (``-1`` when there is none). The
    driver is shared; the flavour lives entirely in the table.

    Returns (start, end_exclusive, substring) triples. See the two callers'
    docstrings for the (deliberate, load-bearing) return-value contract.
    """
    out: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        start = text.find("{", i)
        if start < 0:
            break
        close = close_table[start + 1]
        if close < 0:
            # Never closes — drop this `{` and resume at the next one. The
            # restart deliberately begins from a clean "outside a string"
            # state (see the contract note in the callers).
            i = start + 1
            continue
        out.append((start, close + 1, text[start : close + 1]))
        i = close + 1
    return out


def _strict_close_table(text: str) -> array[int]:
    """Depth-0 close-brace lookup table with *strict* JSON string rules.

    ``table[j]`` is the offset of the first ``}`` that a scanner starting at
    ``j`` — outside any string literal, at brace depth 0 — would reach while
    still at depth 0, or ``-1`` if it runs off the end. Only ``"`` opens a
    string; ``\\`` escapes the next character inside one.

    Built in a single right-to-left sweep, so building plus scanning is
    O(len(text)) regardless of how many unbalanced braces the text contains.
    """
    n = len(text)
    outside = array("i", [-1]) * (n + 1)
    in_str = -1  # answer when the scan *enters* offset j+1 inside a "..."
    escaped = -1  # ... inside a "..." with the previous char a backslash
    for j in range(n - 1, -1, -1):
        c = text[j]
        nxt_outside = outside[j + 1]
        nxt_in_str = in_str
        nxt_escaped = escaped
        # Escaped char: consumed verbatim, then we are back inside the string.
        escaped = nxt_in_str
        # Inside a string literal.
        if c == "\\":
            in_str = nxt_escaped
        elif c == '"':
            in_str = nxt_outside
        else:
            in_str = nxt_in_str
        # Outside any string literal.
        if c == '"':
            outside[j] = nxt_in_str
        elif c == "{":
            inner = nxt_outside  # close of the nested object opening at j
            outside[j] = -1 if inner < 0 else outside[inner + 1]
        elif c == "}":
            outside[j] = j
        else:
            outside[j] = nxt_outside
    return outside


def _lenient_close_table(text: str) -> array[int]:
    """Depth-0 close-brace lookup table tolerating single-quoted strings.

    Same contract as :func:`_strict_close_table`, but BOTH ``'`` and ``"``
    open a string literal (and only the matching quote closes it), which is
    what the lenient pass needs to survive Python-repr / single-quoted
    malformed objects. Five scanner states instead of three; again only the
    "outside a string" row needs to be materialised as an array.
    """
    n = len(text)
    outside = array("i", [-1]) * (n + 1)
    in_sq = in_dq = esc_sq = esc_dq = -1
    for j in range(n - 1, -1, -1):
        c = text[j]
        nxt_outside = outside[j + 1]
        nxt_in_sq, nxt_in_dq = in_sq, in_dq
        nxt_esc_sq, nxt_esc_dq = esc_sq, esc_dq
        esc_sq, esc_dq = nxt_in_sq, nxt_in_dq
        if c == "\\":
            in_sq, in_dq = nxt_esc_sq, nxt_esc_dq
        else:
            in_sq = nxt_outside if c == "'" else nxt_in_sq
            in_dq = nxt_outside if c == '"' else nxt_in_dq
        if c == "'":
            outside[j] = nxt_in_sq
        elif c == '"':
            outside[j] = nxt_in_dq
        elif c == "{":
            inner = nxt_outside
            outside[j] = -1 if inner < 0 else outside[inner + 1]
        elif c == "}":
            outside[j] = j
        else:
            outside[j] = nxt_outside
    return outside


def _find_balanced_json_objects(text: str) -> list[tuple[int, int, str]]:
    """Find top-level `{...}` JSON substrings by a brace-counter scan.

    Returns a list of (start, end_exclusive, substring). Handles escape
    sequences and string literals so braces inside JSON strings do not
    confuse the counter. Malformed (unclosed) candidates are skipped.

    Note: this deliberately matches strict JSON strings (double-quoted) for
    brace balancing. Single-quoted / unquoted malformed objects are found by
    :func:`_find_candidate_object_spans` instead, which brace-balances without
    assuming JSON string syntax.

    Return-value contract (load-bearing — downstream recall depends on all
    three; do NOT "simplify" this into a single global stack pass):

    1. **Top-level objects only.** Once a ``{`` closes, the scan resumes after
       its ``}``, so objects nested inside a *closed* object are not returned.
       ``'{"a": {"b": 1}}'`` yields the outer object alone.
    2. **Nested objects surface when the outer never closes.** An unbalanced
       ``{`` is dropped and the scan restarts at the next ``{``, so
       ``'{"a": {"b":1}'`` yields ``[(6, 13, '{"b":1}')]`` — the inner
       object is still repairable even though the outer one is broken.
    3. **Every restart begins with a fresh string state.** The scan between
       candidates does not track quotes, so a stray/unbalanced quote before a
       ``{`` cannot swallow it. This is what keeps prose apostrophes
       (``"I'll call the tool now."``) and quoted braces (``'{ "z{}" '`` ->
       ``[(4, 6, '{}')]``) from hiding a real call. A single global pass that
       carried string state across candidates loses both.
    """
    if "{" not in text:
        return []
    return _scan_object_spans(text, _strict_close_table(text))


def _find_candidate_object_spans(text: str) -> list[tuple[int, int, str]]:
    """Brace-balance ``{...}`` spans, tolerating single/double quoted strings.

    Unlike :func:`_find_balanced_json_objects`, this counts braces while
    respecting BOTH ``'`` and ``"`` string delimiters, so it can locate
    Python-repr / single-quoted malformed objects for the lenient parser.
    Returns (start, end_exclusive, substring).

    The same three-point return-value contract documented on
    :func:`_find_balanced_json_objects` applies here: top-level objects only,
    nested objects surface when the outer never closes, and every restart
    begins from a fresh string state. Point 3 matters most on this flavour —
    ``'`` opens a string here, so carrying state across candidates would let
    an ordinary prose apostrophe swallow every following object.
    """
    if "{" not in text:
        return []
    return _scan_object_spans(text, _lenient_close_table(text))


# ------------------------------------------------------------------
# Prose-cue guard: bare JSON introduced as an *example* is not a call
# ------------------------------------------------------------------

# If one of these phrases appears immediately before a bare JSON object (within
# a short window, on the same clause), the object is being *described* rather
# than emitted as a call. Conservative: only suppresses, never forces a repair.
#
# The vocabulary is deliberately broad — documentation / example / "here is the
# format" framings all mark the following block as *illustrative*. It stays
# clear of tokens that appear in genuine descriptive prose which nonetheless
# precedes a real call (e.g. "The `echo` function echoes back the provided
# message." for python_call_04), so words like ``function`` / ``message`` are
# NOT cues.
_PROSE_CUE_RE = re.compile(
    r"(?:"
    r"for\s+example|for\s+instance|e\.?g\.?|such\s+as|"
    r"you\s+(?:would|could|can|might)\s+write|"
    r"you\s+would\s+(?:use|call)|"
    r"(?:is|are)\s+documented(?:\s+as)?|documented(?:\s+like)?|"
    r"looks?\s+like|written\s+as|"
    r"as\s+follows|"
    r"(?:calling\s+)?convention|"
    r"below\s+is|"
    r"(?:this|the|that)\s+format|the\s+format\s+is|"
    r"sample|payload|"
    r"syntax|signature|"
    r"look(?:s|ing)?\s+in|"
    r"例えば|たとえば|のように書"
    r")"
    r"[^\n{]{0,80}$",
    re.IGNORECASE,
)


def _preceded_by_prose_cue(text: str, start: int) -> bool:
    """True if a documentation/example cue precedes position ``start``.

    Two windows are checked so a cue on the *previous* line still guards a call
    that a model placed on its own line under an introductory sentence
    (``"Here's a sample payload:\n{...}"`` — negative_19 / negative_22):

      1. the current same-clause lead-in — the object's own line, explicitly
         sliced from the last newline, which catches inline "..., e.g. {...}"
         framings; and
      2. the immediately preceding non-blank line, matched whole, which catches
         "Here's how it looks in this format:" one line above the object.

    Design note (colon-terminated lead-ins): a legitimate call announcement such
    as ``"I'll call it now:"`` ends in a cue-adjacent colon and, when it carries
    a cue word (e.g. "sample", "payload", "format"), is deliberately suppressed
    together with the illustrative ones. Under the FP-0%-first principle this
    trade-off is accepted on purpose: losing the occasional genuine "here it
    comes" preamble is preferable to executing a described example as a real
    call. (The bare cue guard here still keys off the *vocabulary*, so a
    cue-free colon lead-in like "Here's the tool call:" stays repairable; the
    unconditional colon rule lives only on the call-syntax fence path.)

    Conservative: only ever suppresses, never forces a repair.
    """
    prefix = text[:start]
    # (1) same-clause lead-in. Slice the object's *own* line explicitly from the
    #     last newline so the cue anchor is measured against the current line's
    #     real end, not the whole-string end. Anchoring on ``prefix[-200:]``
    #     alone breaks when the object sits at a line start (prefix ends "\n\n"):
    #     the ``[^\n{]`` tail then collapses to zero width and the anchor can no
    #     longer reach a cue that lives before the newline. Isolating the current
    #     line keeps window (1) strictly same-line, leaving cross-line cues to
    #     window (2).
    nl = prefix.rfind("\n")
    current_line = prefix[nl + 1 :] if nl != -1 else prefix
    if _PROSE_CUE_RE.search(current_line[-200:]):
        return True
    # (2) the nearest non-blank line above the object.
    lines = prefix.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    # Drop the current (partial) line the object sits on — but ONLY when it
    # actually exists. When ``prefix`` ends in a newline the object is at a line
    # start, so ``splitlines()`` already excludes any current-line residue and
    # the last surviving entry *is* the lead-in line; popping it unconditionally
    # would discard the cue line itself (negative_19 / negative_23 off-by-one).
    if lines and not prefix.endswith("\n"):
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        lead = lines[-1].strip()
        if _PROSE_CUE_RE.search(lead[-200:]):
            return True
    return False


def _preceded_by_prose_cue_before_fence(text: str, fence_start: int) -> bool:
    """True if the last non-blank line before a fence is an example cue.

    The bare-JSON cue guard (:func:`_preceded_by_prose_cue`) only looks at the
    immediate same-clause lead-in, which does not span the blank line(s) that
    separate an introductory sentence from a following ```...``` fence. R4c
    call-syntax fences are commonly introduced on a *separate* line
    ("For example, you would write:\\n\\n```..."), so this looks back across
    blank lines to the nearest non-empty line and applies the same cue regex.

    This is the discriminator between negative_15 (example-cue + fenced call ->
    suppress) and python_call_04 (descriptive prose + fenced call -> repair).
    Conservative: only ever suppresses, never forces a repair.
    """
    prefix = text[:fence_start]
    lines = prefix.splitlines()
    # Drop the (possibly empty) trailing fragment and any blank separator lines
    # so we land on the last line carrying actual prose.
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return False
    lead = lines[-1].strip()
    # A lead-in line ending in a colon ("... as follows:", "... this format:",
    # a long "For example, ...:") is an introduction to an illustrative block,
    # so a call-syntax fence that follows it is being *shown*, not invoked. This
    # is a general signal that does not depend on the cue vocabulary and so is
    # robust to paraphrase and to arbitrarily long lead-in lines (negative_18 /
    # 20 / 21). It is applied ONLY on this call-syntax fence path — bare JSON /
    # JSON-envelope fences are introduced by legitimate colon lead-ins too
    # ("Here's the tool call:", "First:") and must stay repairable.
    if lead.endswith((":", "：")):  # noqa: RUF001 — full-width colon (U+FF1A) is intentional
        return True
    # Otherwise fall back to the cue vocabulary anywhere in the lead-in line.
    return bool(_PROSE_CUE_RE.search(lead[-200:]))


# ------------------------------------------------------------------
# R4c: call-syntax family (name + parens + args)
# ------------------------------------------------------------------

# Head of a call line: an optional ``print(`` wrapper, an optional
# ``default_api.`` prefix (Gemma tool_code idiom), then the tool name and its
# opening paren. Anchored at the start of a (stripped) line so an inline
# mid-prose call is never a candidate.
_CALL_HEAD_RE = re.compile(
    r"^(?P<print>print\s*\(\s*)?"
    r"(?:default_api\s*\.\s*)?"
    r"(?P<name>[A-Za-z_]\w*)\s*\("
)

# A single ``key = value`` / ``key : value`` kwargs pair.
_CALL_KWARG_RE = re.compile(r"^(?P<key>[A-Za-z_]\w*)\s*(?:=|:)\s*(?P<val>.*)$", re.DOTALL)


def _find_matching_paren(s: str, open_idx: str | int) -> int:
    """Index of the ``)`` matching the ``(`` at ``open_idx``, or -1.

    Balances parentheses while respecting single/double quoted strings (so a
    paren inside a string literal does not close the call).
    """
    depth = 0
    i = int(open_idx)
    n = len(s)
    quote: str | None = None
    escape = False
    while i < n:
        c = s[i]
        if escape:
            escape = False
        elif quote is not None:
            if c == "\\":
                escape = True
            elif c == quote:
                quote = None
        else:
            if c == '"' or c == "'":
                quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _split_top_level_commas(body: str) -> list[str] | None:
    """Split ``body`` on top-level commas, respecting quotes and brackets.

    Returns the list of segments, or None if the string/bracket nesting is
    unbalanced (which means the arguments are corrupt and must not be repaired).
    """
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    quote: str | None = None
    escape = False
    for c in body:
        if escape:
            cur.append(c)
            escape = False
            continue
        if quote is not None:
            cur.append(c)
            if c == "\\":
                escape = True
            elif c == quote:
                quote = None
            continue
        if c == '"' or c == "'":
            quote = c
            cur.append(c)
            continue
        if c in "([{":
            depth += 1
            cur.append(c)
            continue
        if c in ")]}":
            depth -= 1
            cur.append(c)
            continue
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(c)
    if quote is not None or depth != 0:
        return None
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_call_value(raw: str) -> tuple[bool, Any]:
    """Parse a single kwargs value. Returns (ok, value).

    Tries strict JSON, then the lenient JSON pipe (which normalises single
    quotes, unquoted words, etc.). ``ok`` is False when the value cannot be
    parsed at all, so a corrupt argument fails the whole call (guard b).
    """
    v = raw.strip()
    if not v:
        return False, None
    try:
        return True, json.loads(v)
    except json.JSONDecodeError:
        pass
    try:
        return True, _lenient_json_loads(v)
    except ValueError:
        return False, None


def _parse_call_args(body: str, name: str) -> tuple[str, Any] | None:
    """Parse a call's argument list into (name, arguments) or None.

    Handles all three R4c argument styles with a single splitter:
      - ``key="value"`` / ``key=value`` (Python kwargs)
      - ``key: 'value'``               (colon-separated, Ruby/Swift flavour)
      - ``({...})`` a single JSON-object positional argument (JS flavour)

    Every argument must parse completely; a broken inner value causes the whole
    call to be declined (guard b — a form-only fix with corrupt args is more
    harmful than no repair).
    """
    body = body.strip()
    if not body:
        # Zero-arg call: valid, empty arguments object.
        return name, {}

    # Single JSON-object positional argument: write_note({...}).
    if body.startswith("{"):
        try:
            obj = _parse_json_object(body)
        except ValueError:
            return None
        if not isinstance(obj, dict):
            return None
        return name, obj

    segments = _split_top_level_commas(body)
    if not segments:
        return None
    out: dict[str, Any] = {}
    for seg in segments:
        m = _CALL_KWARG_RE.match(seg.strip())
        if not m:
            return None  # positional args / unparseable -> decline
        ok, val = _parse_call_value(m.group("val"))
        if not ok:
            return None
        out[m.group("key")] = val
    return name, out


def _extract_one_call_line(line: str, allowed: set[str]) -> tuple[str, Any] | None:
    """Extract a single call-syntax invocation from one standalone line.

    The line must be *entirely* a call (optionally wrapped in ``print(...)``);
    trailing garbage after the closing paren disqualifies it, so an inline
    call embedded in a sentence is never matched. The name must be allow-listed
    and the arguments must parse fully.
    """
    stripped = line.strip()
    m = _CALL_HEAD_RE.match(stripped)
    if not m:
        return None
    name = m.group("name")
    if name not in allowed:
        return None
    open_idx = m.end() - 1
    close_idx = _find_matching_paren(stripped, open_idx)
    if close_idx < 0:
        return None
    inner = stripped[open_idx + 1 : close_idx]
    rest = stripped[close_idx + 1 :].strip()
    if m.group("print"):
        # A print(...) wrapper must be closed by exactly one trailing ')'.
        if not rest.startswith(")"):
            return None
        rest = rest[1:].strip()
    if rest:
        return None  # trailing garbage -> not a clean standalone call
    return _parse_call_args(inner, name)


def _extract_call_syntax_lines(
    block: str, allowed: set[str]
) -> list[tuple[str, Any]]:
    """Extract call-syntax invocations from the non-blank lines of ``block``.

    Used for fence interiors (one or more call lines). Every non-blank line
    must resolve to an allow-listed call with fully-parseable arguments; if any
    line fails, the whole block is declined (all-or-nothing keeps a mixed
    code/call fence from being partially executed).
    """
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return []
    hits: list[tuple[str, Any]] = []
    for ln in lines:
        hit = _extract_one_call_line(ln, allowed)
        if hit is None:
            return []
        hits.append(hit)
    return hits


def _extract_r4c_standalone_lines(
    text: str,
    allowed: set[str],
    protected: list[tuple[int, int]],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract R4c call-syntax invocations that stand alone on their own line.

    Complements the fenced R4c path: a call like ``write_note({...})`` may
    appear bare (no fence) as the whole response or on its own line. Each
    candidate line must be *entirely* a call (``_extract_one_call_line``
    enforces the no-trailing-garbage rule), so an inline call embedded in a
    sentence is never a candidate. Lines inside a protected code fence, or
    introduced by an example cue, are skipped.

    Returns (text_with_call_lines_removed, tool_calls).
    """
    if "(" not in text:
        return text, []
    tool_calls: list[dict[str, Any]] = []
    out_lines: list[str] = []
    pos = 0
    changed = False
    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        if _in_spans(line_start, protected):
            out_lines.append(line)
            continue
        if _preceded_by_prose_cue(text, line_start):
            out_lines.append(line)
            continue
        hit = _extract_one_call_line(stripped, allowed)
        if hit is None:
            out_lines.append(line)
            continue
        name, args = hit
        tool_calls.append(_normalise_to_openai_tool_call(name, args))
        changed = True
        # Drop the call line entirely (keep a trailing newline structure sane).
    if not changed:
        return text, []
    return "".join(out_lines), tool_calls


# ------------------------------------------------------------------
# R3: XML-flavoured tool-call forms
# ------------------------------------------------------------------

# Tags that are known NOT to be tool calls: reasoning wrappers and common HTML.
_NON_TOOL_TAGS = frozenset(
    {
        "think",
        "thinking",
        "thought",
        "reasoning",
        "scratchpad",
        "reflection",
        "answer",
        "final",
        "p",
        "div",
        "span",
        "ul",
        "ol",
        "li",
        "a",
        "b",
        "i",
        "em",
        "strong",
        "code",
        "pre",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "tr",
        "td",
        "th",
        "img",
        "button",
        "form",
        "input",
        "label",
        "section",
        "article",
        "header",
        "footer",
        "nav",
        "tool",  # handled separately as a generic wrapper (body carries name)
    }
)

# Generic wrapper tags whose *body* carries the actual tool-call JSON.
_WRAPPER_TAGS = frozenset({"tool", "tool_call", "function_call", "invoke"})

# <tag attr="v" .../>  or  <tag attr="v" ...>  (self-closing or open; we only
# treat the self-closing / immediately-closed form as an attribute call).
_XML_SELFCLOSE_RE = re.compile(
    r"<([A-Za-z_][\w.-]*)((?:\s+[\w.:-]+\s*=\s*\"[^\"]*\")*)\s*/>",
)
# <tag> ... </tag>  wrapper.
_XML_WRAPPER_RE = re.compile(
    r"<([A-Za-z_][\w.-]*)\s*>(.*?)</\1\s*>",
    re.DOTALL,
)
_XML_ATTR_RE = re.compile(r"([\w.:-]+)\s*=\s*\"([^\"]*)\"")

# R4a: nested-XML name-attribute forms.
# Known call/container tags whose ``name`` attribute (not the tag itself)
# carries the tool name. Restricting to this fixed set — rather than treating
# any tag with a ``name`` attribute as a call — keeps arbitrary markup (e.g.
# <input name="q"/>) from being mistaken for a tool call.
_R4A_CALL_TAGS = frozenset({"function", "tool", "function_call", "invoke", "tool_call"})
# Container tags that merely wrap one or more call tags; scanned through, never
# themselves a call.
_R4A_CONTAINER_TAGS = frozenset({"tools", "tool_calls", "function_calls", "invoke"})
# A self-closing call tag: <function name="..." arguments='...'/>. Attribute
# values may be single- OR double-quoted; the ``arguments`` value is captured
# raw and delegated to the JSON pipe (R1/R2). ``\1`` on the quote char keeps the
# value greedy up to the matching closing quote.
_R4A_ATTR_RE = re.compile(
    r"""([\w.:-]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')""",
)
_R4A_SELFCLOSE_RE = re.compile(
    r"""<([A-Za-z_][\w.-]*)((?:\s+[\w.:-]+\s*=\s*(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'))+)\s*/>""",
)
# R4a attribute keys that hold the tool name / the arguments payload.
_R4A_NAME_ATTRS = ("name",)
_R4A_ARG_ATTRS = ("arguments", "args", "parameters", "input")


def _r4a_parse_attrs(attr_blob: str) -> dict[str, str]:
    """Parse an XML attribute blob into {key: value}, unescaping quoted values.

    A double-quoted value may carry backslash-escaped inner quotes
    (``arguments="{\\"command\\": ...}"``); those are unescaped so the value is
    valid JSON before it reaches the parse pipe. Single-quoted values are taken
    verbatim (their inner double quotes are already literal JSON).
    """
    out: dict[str, str] = {}
    for m in _R4A_ATTR_RE.finditer(attr_blob):
        key = m.group(1)
        if m.group(2) is not None:  # double-quoted
            val = m.group(2).replace('\\"', '"').replace("\\\\", "\\")
        else:  # single-quoted
            val = m.group(3) or ""
        out[key] = val
    return out


def _r4a_build_call(
    attrs: dict[str, str], allowed: set[str]
) -> tuple[str, Any] | None:
    """Turn parsed R4a attributes into (name, arguments) if valid, else None."""
    name = None
    for k in _R4A_NAME_ATTRS:
        if k in attrs:
            name = attrs[k]
            break
    if not name or name not in allowed:
        return None
    args: Any = {}
    for k in _R4A_ARG_ATTRS:
        if k in attrs:
            raw = attrs[k].strip()
            if not raw:
                args = {}
                break
            try:
                args = _parse_json_object(raw)
            except ValueError:
                return None  # malformed arguments -> decline (safe side)
            break
    return name, args


def _extract_r4a_nested_xml(
    text: str,
    allowed: set[str] | None,
    guard: list[tuple[int, int]],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract R4a nested-XML name-attribute tool calls.

    Handles both the bare call tag (``<function name=.../>``) and the same tag
    inside a container (``<tools>...</tools>``). Containers are transparent —
    the scanner simply finds every allow-listed call tag, wherever it sits, and
    removes it; a lone empty ``<tools></tools>`` shell left behind is trimmed by
    residue cleanup. Reasoning-block / example-cue guards are honoured.

    Returns (text_with_calls_removed, tool_calls).
    """
    if allowed is None or "<" not in text:
        return text, []
    tool_calls: list[dict[str, Any]] = []
    removals: list[tuple[int, int]] = []
    for m in _R4A_SELFCLOSE_RE.finditer(text):
        tag = m.group(1).lower()
        if tag not in _R4A_CALL_TAGS:
            continue
        if _in_spans(m.start(), guard) or _preceded_by_prose_cue(text, m.start()):
            continue
        attrs = _r4a_parse_attrs(m.group(2) or "")
        hit = _r4a_build_call(attrs, allowed)
        if hit is None:
            continue
        name, args = hit
        tool_calls.append(_normalise_to_openai_tool_call(name, args))
        removals.append((m.start(), m.end()))

    if not removals:
        return text, []
    # Also sweep up now-empty container shells around removed calls.
    cleaned = text
    removals.sort()
    for s, e in reversed(removals):
        cleaned = cleaned[:s] + cleaned[e:]
    for ctag in _R4A_CONTAINER_TAGS:
        cleaned = re.sub(
            rf"<{ctag}\s*>\s*</{ctag}\s*>", "", cleaned, flags=re.IGNORECASE
        )
    return cleaned, tool_calls


def _known_non_tool_tag_ranges(text: str) -> list[tuple[int, int]]:
    """Ranges covered by <think>...</think> (and similar) reasoning blocks.

    XML extraction must not reach inside these, so a tool-shaped tag mentioned
    in reasoning is never executed.
    """
    ranges: list[tuple[int, int]] = []
    for tag in ("think", "thinking", "thought", "reasoning", "scratchpad"):
        for m in re.finditer(
            rf"<{tag}\b[^>]*>.*?</{tag}\s*>", text, re.DOTALL | re.IGNORECASE
        ):
            ranges.append((m.start(), m.end()))
    return ranges


def _extract_xml_tool_calls(
    text: str,
    allowed: set[str] | None,
    protected: list[tuple[int, int]],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract R3 XML-flavoured tool calls, honouring guards.

    Returns (text_with_xml_calls_removed, tool_calls). Only forms whose tag /
    inner name resolve to an allow-listed tool are removed.
    """
    if allowed is None:
        # Without an allow-list, XML extraction is too risky (any <tag> could
        # look like a call). Skip R3 entirely — keeps false positives at zero.
        return text, []
    if "<" not in text:
        return text, []

    tool_calls: list[dict[str, Any]] = []
    reasoning = _known_non_tool_tag_ranges(text)
    guard = protected + reasoning
    removals: list[tuple[int, int]] = []

    def _blocked(pos: int) -> bool:
        return _in_spans(pos, guard) or _preceded_by_prose_cue(text, pos)

    # --- wrapper form: <tool>{JSON}</tool>  and  <name>{JSON}</name> ---
    for m in _XML_WRAPPER_RE.finditer(text):
        tag = m.group(1)
        inner = m.group(2).strip()
        tag_l = tag.lower()
        if _blocked(m.start()):
            continue
        if tag_l in _WRAPPER_TAGS:
            # Body carries the whole tool-call object.
            if not inner.startswith("{"):
                continue
            try:
                obj = _parse_json_object(inner)
            except ValueError:
                continue
            hit = _looks_like_tool_call(obj, allowed)
            if hit is None:
                continue
            name, args = hit
            tool_calls.append(_normalise_to_openai_tool_call(name, args))
            removals.append((m.start(), m.end()))
            continue
        # Named-wrapper: the tag itself is the tool name, body is the arguments.
        if tag_l in _NON_TOOL_TAGS:
            continue
        if tag not in allowed:
            continue
        if not inner.startswith("{"):
            continue
        try:
            args_obj = _parse_json_object(inner)
        except ValueError:
            continue
        if not isinstance(args_obj, dict):
            continue
        tool_calls.append(_normalise_to_openai_tool_call(tag, args_obj))
        removals.append((m.start(), m.end()))

    # --- self-closing attribute form: <echo message="probe"/> ---
    for m in _XML_SELFCLOSE_RE.finditer(text):
        tag = m.group(1)
        if tag.lower() in _NON_TOOL_TAGS:
            continue
        if tag not in allowed:
            continue
        if _blocked(m.start()):
            continue
        # Don't double-extract something already inside a wrapper removal.
        if _in_spans(m.start(), removals):
            continue
        attrs = dict(_XML_ATTR_RE.findall(m.group(2) or ""))
        tool_calls.append(_normalise_to_openai_tool_call(tag, attrs))
        removals.append((m.start(), m.end()))

    if not removals:
        return text, []

    # Remove matched spans back-to-front; merge overlaps first.
    removals.sort()
    merged: list[tuple[int, int]] = []
    for s, e in removals:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    cleaned = text
    for s, e in reversed(merged):
        cleaned = cleaned[:s] + cleaned[e:]
    return cleaned, tool_calls


# ------------------------------------------------------------------
# Residue cleanup (trap #2)
# ------------------------------------------------------------------

# Empty fenced blocks left after a tool-call body was removed.
_EMPTY_FENCE_RE = re.compile(r"```[\w+-]*[ \t]*\r?\n?[ \t]*\r?\n?```")
# Array skeletons like "[, ]" or "[ , , ]" left after removing array elements.
_EMPTY_ARRAY_RE = re.compile(r"\[\s*(?:,\s*)+\]")


def _cleanup_residue(text: str) -> str:
    """Remove cosmetic leftovers from extraction (empty fences, ``[,]``).

    Only *empty* fenced blocks (a fence pair with nothing between them) are
    removed — a lone closing ``` that still belongs to an intact, preserved
    code fence is left alone, so protected source blocks stay valid.
    """
    text = _EMPTY_FENCE_RE.sub("", text)
    text = _EMPTY_ARRAY_RE.sub("", text)
    # Collapse a lone remaining empty array pair.
    text = re.sub(r"\[\s*\]", "", text)
    return text


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def repair_tool_calls_in_text(
    text: str,
    allowed_tool_names: list[str] | set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract embedded tool-call JSON from assistant text.

    Returns:
        (cleaned_text, tool_calls)
          cleaned_text  : the input with recognised tool-call JSON removed,
                          stripped of surrounding whitespace.
          tool_calls    : OpenAI-shape tool_calls entries, in the order they
                          appeared in the original text. Each entry has a
                          freshly minted `id` (the source JSON did not carry one).

    If nothing repairable is found, returns (text, []).
    """
    if not isinstance(text, str) or not text:
        return text, []

    allowed: set[str] | None = None if allowed_tool_names is None else set(allowed_tool_names)

    extracted: list[dict[str, Any]] = []

    # 1. Pull tool-call-shaped fenced code blocks out first — they're the most
    #    common shape when a chat-tuned model explains what it's doing. Fenced
    #    blocks that are NOT tool calls (prose, real source code, plain JSON
    #    examples) are left in place so we never drop legitimate content (H2).
    #    Programming-language fences are protected outright.
    cleaned, fenced_tool_calls = _extract_tool_call_fenced_blocks(text, allowed)
    extracted.extend(fenced_tool_calls)

    # 2. R3: XML-flavoured forms (only when an allow-list constrains names).
    #    Runs on the fence-stripped text; protected code fences and reasoning
    #    blocks are shielded.
    protected = _protected_code_spans(cleaned)
    cleaned, xml_tool_calls = _extract_xml_tool_calls(cleaned, allowed, protected)
    extracted.extend(xml_tool_calls)

    # 2b. R4a: nested-XML name-attribute forms (<function name=.../> etc.).
    #     Same guards as R3: protected code fences + reasoning blocks are
    #     shielded, and an example cue immediately before a call tag suppresses
    #     it.
    protected = _protected_code_spans(cleaned)
    r4a_guard = protected + _known_non_tool_tag_ranges(cleaned)
    cleaned, r4a_tool_calls = _extract_r4a_nested_xml(cleaned, allowed, r4a_guard)
    extracted.extend(r4a_tool_calls)

    # 2c. R4c: standalone call-syntax lines outside any fence
    #     (e.g. ``write_note({...})`` alone on a line). Fenced call-syntax was
    #     already handled in step 1. Only whole-line calls with an allow-listed
    #     name and fully-parseable arguments are taken; inline / mid-prose forms
    #     are never candidates.
    if allowed is not None:
        cleaned, r4c_tool_calls = _extract_r4c_standalone_lines(
            cleaned, allowed, _all_fence_spans(cleaned)
        )
        extracted.extend(r4c_tool_calls)

    # 3. Scan remaining text for bare JSON objects (strict then lenient).
    #    Skip anything inside a protected code fence or introduced by a prose
    #    example cue. Walk back-to-front so slicing removals don't shift the
    #    indices of earlier matches.
    #    Oversized inputs skip this step entirely (see _MAX_BARE_SCAN_CHARS):
    #    the per-candidate JSON parsing / cue checking is the expensive part
    #    and a message that large is not a tool call. Steps 1/2/2b/2c above
    #    are regex-driven and linear, so they are NOT skipped.
    protected = _protected_code_spans(cleaned)
    spans_to_remove: list[tuple[int, int]] = []
    repaired_from_bare: list[dict[str, Any]] = []

    strict_spans: list[tuple[int, int, str]] = []
    lenient_spans: list[tuple[int, int, str]] = []
    if len(cleaned) > _MAX_BARE_SCAN_CHARS:
        logger.warning("tool-repair-input-too-large", extra={"chars": len(cleaned)})
    else:
        strict_spans = _find_balanced_json_objects(cleaned)
        lenient_spans = _find_candidate_object_spans(cleaned)

    # First pass: strict-JSON balanced objects.
    for start, end, substr in strict_spans:
        if _in_spans(start, protected):
            continue
        if _preceded_by_prose_cue(cleaned, start):
            continue
        try:
            obj = json.loads(substr)
        except json.JSONDecodeError:
            continue
        # R4b: response-envelope wrappers ({"tool_calls": [...]}, ...) expand to
        # one or more inner calls before the direct-shape check.
        envelope = _expand_tool_call_envelope(obj, allowed)
        if envelope is not None:
            for name, args in envelope:
                repaired_from_bare.append(_normalise_to_openai_tool_call(name, args))
            spans_to_remove.append((start, end))
            continue
        hit = _looks_like_tool_call(obj, allowed)
        if hit is None:
            continue
        name, args = hit
        repaired_from_bare.append(_normalise_to_openai_tool_call(name, args))
        spans_to_remove.append((start, end))

    # Second pass: lenient objects (single-quoted / unquoted / trailing comma /
    # double-brace) that strict parsing missed. Skip spans already claimed.
    for start, end, substr in lenient_spans:
        if _in_spans(start, spans_to_remove):
            continue
        if _in_spans(start, protected):
            continue
        if _preceded_by_prose_cue(cleaned, start):
            continue
        # Skip if strict JSON already parses this (handled in first pass).
        try:
            json.loads(substr)
            continue  # strict-parseable -> either already handled or non-tool
        except json.JSONDecodeError:
            pass
        try:
            obj = _lenient_json_loads(_strip_double_braces(substr))
        except ValueError:
            continue
        hit = _looks_like_tool_call(obj, allowed)
        if hit is None:
            continue
        name, args = hit
        repaired_from_bare.append(_normalise_to_openai_tool_call(name, args))
        spans_to_remove.append((start, end))

    # Remove the matched spans from the text back-to-front (merge overlaps).
    spans_to_remove.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans_to_remove:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    for start, end in reversed(merged):
        cleaned = cleaned[:start] + cleaned[end:]

    extracted.extend(repaired_from_bare)

    # Residue cleanup (empty fences, [,], stray backticks) then whitespace.
    cleaned = _cleanup_residue(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # v2.2: deduplicate tool calls within the same response.
    # Small models sometimes output the same tool-call JSON 2-3 times
    # in a single turn. We keep the first occurrence only.
    extracted = deduplicate_tool_calls(extracted)

    return cleaned, extracted


# ------------------------------------------------------------------
# Deduplication (v2.2)
# ------------------------------------------------------------------


def deduplicate_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate tool calls sharing the same (name, arguments).

    Preserves order — the first occurrence wins. Each entry is expected
    to be in OpenAI tool_calls shape (``{"function": {"name": ...,
    "arguments": ...}, ...}``). Entries that lack the expected
    structure are kept unconditionally (conservative fallback).

    This is separate from L3 tool-loop detection (which operates across
    turns in the conversation history). Deduplication operates within a
    single assistant response where the model outputted the same JSON
    tool-call block multiple times.
    """
    if len(tool_calls) <= 1:
        return tool_calls

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function")
        if not isinstance(func, dict):
            # Not in expected shape — keep unconditionally.
            deduped.append(tc)
            continue
        key = (func.get("name", ""), func.get("arguments", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(tc)
    return deduped
