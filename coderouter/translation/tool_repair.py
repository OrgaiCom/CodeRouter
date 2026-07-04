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

import json
import re
import uuid
from typing import Any

__all__ = ["deduplicate_tool_calls", "repair_tool_calls_in_text"]


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
        if not body.startswith("{"):
            return match.group(0)  # keep non-JSON fenced blocks (e.g. code)
        try:
            obj = _parse_json_object(body)
        except ValueError:
            return match.group(0)  # keep unparseable fenced blocks
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


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _find_balanced_json_objects(text: str) -> list[tuple[int, int, str]]:
    """Find top-level `{...}` JSON substrings by a brace-counter scan.

    Returns a list of (start, end_exclusive, substring). Handles escape
    sequences and string literals so braces inside JSON strings do not
    confuse the counter. Malformed (unclosed) candidates are skipped.

    Note: this deliberately matches strict JSON strings (double-quoted) for
    brace balancing. Single-quoted / unquoted malformed objects are found by
    :func:`_find_candidate_object_spans` instead, which brace-balances without
    assuming JSON string syntax.
    """
    out: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        # Scan forward to find a balanced close.
        depth = 0
        j = i
        in_str = False
        escape = False
        while j < n:
            c = text[j]
            if escape:
                escape = False
            elif in_str:
                if c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        out.append((i, j + 1, text[i : j + 1]))
                        i = j + 1
                        break
            j += 1
        else:
            # Ran off the end without closing — skip this `{` and move on.
            i += 1
            continue
    return out


def _find_candidate_object_spans(text: str) -> list[tuple[int, int, str]]:
    """Brace-balance ``{...}`` spans, tolerating single/double quoted strings.

    Unlike :func:`_find_balanced_json_objects`, this counts braces while
    respecting BOTH ``'`` and ``"`` string delimiters, so it can locate
    Python-repr / single-quoted malformed objects for the lenient parser.
    Returns (start, end_exclusive, substring).
    """
    out: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        quote: str | None = None
        escape = False
        while j < n:
            c = text[j]
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
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        out.append((i, j + 1, text[i : j + 1]))
                        i = j + 1
                        break
            j += 1
        else:
            i += 1
            continue
    return out


# ------------------------------------------------------------------
# Prose-cue guard: bare JSON introduced as an *example* is not a call
# ------------------------------------------------------------------

# If one of these phrases appears immediately before a bare JSON object (within
# a short window, on the same clause), the object is being *described* rather
# than emitted as a call. Conservative: only suppresses, never forces a repair.
_PROSE_CUE_RE = re.compile(
    r"(?:"
    r"for\s+example|for\s+instance|e\.?g\.?|such\s+as|"
    r"you\s+(?:would|could|can|might)\s+write|"
    r"you\s+would\s+(?:use|call)|"
    r"(?:is|are)\s+documented(?:\s+as)?|documented\s+like|"
    r"looks?\s+like\s+this|written\s+as|the\s+format\s+is|"
    r"syntax\s+is|例えば|たとえば|のように書"
    r")"
    r"[^\n{]{0,40}$",
    re.IGNORECASE,
)


def _preceded_by_prose_cue(text: str, start: int) -> bool:
    """True if a documentation/example cue immediately precedes position ``start``."""
    prefix = text[:start]
    # Look only within the current line/clause (last ~80 chars, no newline jump
    # further than the immediate lead-in).
    tail = prefix[-80:]
    return bool(_PROSE_CUE_RE.search(tail))


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

    # 3. Scan remaining text for bare JSON objects (strict then lenient).
    #    Skip anything inside a protected code fence or introduced by a prose
    #    example cue. Walk back-to-front so slicing removals don't shift the
    #    indices of earlier matches.
    protected = _protected_code_spans(cleaned)
    spans_to_remove: list[tuple[int, int]] = []
    repaired_from_bare: list[dict[str, Any]] = []

    # First pass: strict-JSON balanced objects.
    for start, end, substr in _find_balanced_json_objects(cleaned):
        if _in_spans(start, protected):
            continue
        if _preceded_by_prose_cue(cleaned, start):
            continue
        try:
            obj = json.loads(substr)
        except json.JSONDecodeError:
            continue
        hit = _looks_like_tool_call(obj, allowed)
        if hit is None:
            continue
        name, args = hit
        repaired_from_bare.append(_normalise_to_openai_tool_call(name, args))
        spans_to_remove.append((start, end))

    # Second pass: lenient objects (single-quoted / unquoted / trailing comma /
    # double-brace) that strict parsing missed. Skip spans already claimed.
    for start, end, substr in _find_candidate_object_spans(cleaned):
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
