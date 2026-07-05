"""Unit tests for the R4 tool-call repair enhancements.

R4 extends the lenient repairer (R1/R2/R3) with three shape families measured
as gaps in the offline bench corpus:

    R4a  nested-XML name-attribute forms — the tool name lives in a ``name``
         attribute rather than the tag itself
         (``<tools><function name="echo" arguments='{...}'/></tools>``).
    R4b  JSON response-envelope forms the model echoes back verbatim
         (``{"tool_calls": [...]}`` and the legacy ``{"function_call": {...}}``).
    R4c  the call-syntax family — ``name(args)`` written inside a fence or on a
         standalone line, in kwargs / colon / single-JSON-object styles, with
         optional ``print(...)`` / ``default_api.`` wrappers.

As with ``test_tool_repair_lenient.py`` the false-positive guards are the
safety-critical half: every family carries positive, negative, and boundary
cases, and the out-of-allow-list / example-cue / inline-prose guards must keep
false positives at zero.
"""

from __future__ import annotations

import json

from coderouter.translation.tool_repair import repair_tool_calls_in_text


def _names(calls: list[dict]) -> list[str]:
    return [c["function"]["name"] for c in calls]


def _args(call: dict) -> dict:
    return json.loads(call["function"]["arguments"])


# ======================================================================
# R4a — nested-XML name-attribute forms
# ======================================================================


def test_r4a_container_single_function() -> None:
    text = '<tools><function name="echo" arguments=\'{"message": "demo"}\'/></tools>'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "demo"}
    assert "<function" not in cleaned
    assert "<tools>" not in cleaned  # empty container shell swept up


def test_r4a_double_quoted_escaped_arguments() -> None:
    """The ``arguments`` value is a double-quoted, backslash-escaped JSON blob."""
    text = '<tools><function name="Bash" arguments="{\\"command\\": \\"ls -la\\"}"/></tools>'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]
    assert _args(calls[0]) == {"command": "ls -la"}


def test_r4a_bare_function_tag_with_prose() -> None:
    text = 'I\'ll run it now.\n<function name="echo" arguments=\'{"message": "probe"}\'/>'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}
    assert "I'll run it now." in cleaned


def test_r4a_tool_tag_args_alias() -> None:
    text = '<tool name="read_file" args=\'{"path": "src/main.py"}\'/>'
    _, calls = repair_tool_calls_in_text(text, ["read_file"])
    assert _names(calls) == ["read_file"]
    assert _args(calls[0]) == {"path": "src/main.py"}


def test_r4a_multiple_functions_in_container() -> None:
    text = (
        "<tools>\n"
        '<function name="echo" arguments=\'{"message": "one"}\'/>\n'
        '<function name="echo" arguments=\'{"message": "two"}\'/>\n'
        "</tools>"
    )
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo", "echo"]
    assert [_args(c) for c in calls] == [{"message": "one"}, {"message": "two"}]


def test_r4a_name_not_in_allowlist_is_ignored() -> None:
    text = '<function name="DeleteEverything" arguments=\'{"path": "/"}\'/>'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert cleaned == text


def test_r4a_unknown_call_tag_is_ignored() -> None:
    """Only the fixed call-tag set carries a name attribute; <input> does not."""
    text = '<input name="echo" value="x"/>'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert cleaned == text


def test_r4a_malformed_arguments_declined() -> None:
    """A broken arguments blob is left alone rather than executed corrupt."""
    text = '<function name="echo" arguments=\'{"message": broken\'/>'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []


def test_r4a_inside_think_block_not_extracted() -> None:
    text = (
        '<think>I could emit <function name="echo" arguments=\'{"message": "x"}\'/> '
        "but the user only asked a question.</think>\nNo action needed."
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "<function" in cleaned  # untouched inside reasoning


def test_r4a_example_cue_suppresses() -> None:
    text = (
        "The XML tool syntax is documented like this: write "
        '<function name="echo" arguments=\'{"message": "x"}\'/> where the name '
        "attribute holds the tool."
    )
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []


def test_r4a_paraphrased_doc_cue_suppresses() -> None:
    """negative_22: a paraphrased example lead-in ("look in this format:") on the
    line above a name-attribute tag must suppress it (cross-line cue guard)."""
    text = (
        "Here's how function calls look in this format:\n"
        '<function name="echo" arguments=\'{"message":"x"}\'/>'
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "<function" in cleaned


def test_r4a_no_allowlist_skips() -> None:
    text = '<function name="echo" arguments=\'{"message": "x"}\'/>'
    cleaned, calls = repair_tool_calls_in_text(text)
    assert calls == []
    assert cleaned == text


# ----------------------------------------------------------------------
# R4a — cross-line cue guard: blank-line variants (0 / 1 / 2 blank lines)
#
# Regression for the window(1) anchor bug and the window(2) lines.pop()
# off-by-one: a documentation cue on the line *above* an XML name-attribute
# tag must suppress it regardless of how many blank lines separate them.
# ----------------------------------------------------------------------

_XML_DOC_TAG = '<function name="echo" arguments=\'{"message":"x"}\'/>'


def test_r4a_doc_cue_zero_blank_lines_suppresses() -> None:
    """Cue line directly above the tag (no blank separator)."""
    text = "Here's how function calls look in this format:\n" + _XML_DOC_TAG
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "<function" in cleaned


def test_r4a_doc_cue_one_blank_line_suppresses() -> None:
    """negative_24: cue line, one blank line, then the tag."""
    text = "Here's how function calls look in this format:\n\n" + _XML_DOC_TAG
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "<function" in cleaned


def test_r4a_doc_cue_two_blank_lines_suppresses() -> None:
    """Cue line, two blank lines, then the tag — window(2) must still reach it."""
    text = "Here's how function calls look in this format:\n\n\n" + _XML_DOC_TAG
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "<function" in cleaned


def test_r4a_no_cue_with_blank_lines_still_repairs() -> None:
    """Control positive: a bare name-attribute tag under blank-line-separated
    *non-cue* prose is a real call and must still be repaired (proves the
    blank-line fix narrowed the window, it did not blanket-suppress)."""
    text = "I'll run the tool for you.\n\n" + _XML_DOC_TAG
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "x"}


# ======================================================================
# R4b — JSON response-envelope forms
# ======================================================================


def test_r4b_tool_calls_array_bare() -> None:
    text = '{"tool_calls": [{"name": "echo", "arguments": {"message": "probe"}}]}'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}


def test_r4b_tool_calls_array_multiple() -> None:
    text = (
        '{"tool_calls": [{"name": "echo", "arguments": {"message": "a"}}, '
        '{"name": "echo", "arguments": {"message": "b"}}]}'
    )
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo", "echo"]
    assert [_args(c) for c in calls] == [{"message": "a"}, {"message": "b"}]


def test_r4b_function_call_legacy_string_arguments() -> None:
    """Legacy ``function_call`` carries arguments as a JSON *string* -> double parse."""
    text = '{"function_call": {"name": "echo", "arguments": "{\\"message\\": \\"probe\\"}"}}'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}


def test_r4b_fenced_envelope_two_calls() -> None:
    text = (
        "```json\n"
        '{"tool_calls": [{"name": "echo", "arguments": {"message": "a"}}, '
        '{"name": "echo", "arguments": {"message": "b"}}]}\n'
        "```"
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo", "echo"]
    assert cleaned == ""


def test_r4b_top_level_array_still_works() -> None:
    """A bare top-level array of calls (no envelope key) is handled by the scanner."""
    text = '[{"name": "echo", "arguments": {"message": "probe"}}]'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]


def test_r4b_envelope_disallowed_name_declined() -> None:
    text = '{"tool_calls": [{"name": "Nuke", "arguments": {"path": "/"}}]}'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert cleaned == text


def test_r4b_envelope_all_or_nothing() -> None:
    """If any inner call is out-of-allowlist, the whole envelope is declined."""
    text = (
        '{"tool_calls": [{"name": "echo", "arguments": {}}, '
        '{"name": "Nuke", "arguments": {}}]}'
    )
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []


def test_r4b_empty_envelope_declined() -> None:
    text = '{"tool_calls": []}'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []


def test_r4b_sample_payload_cue_suppresses() -> None:
    """negative_19: an envelope shown as a "sample response payload" example, on
    the line below the cue, must not be repaired (cross-line cue guard)."""
    text = (
        "Here's a sample response payload you might receive:\n"
        '{"tool_calls": [{"name": "echo", "arguments": {"message": "example"}}]}'
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert '"tool_calls"' in cleaned


# ----------------------------------------------------------------------
# R4b — cross-line cue guard: blank-line variants (0 / 1 / 2 blank lines)
#
# The envelope-path mirror of the R4a blank-line regression: a "sample
# payload" cue above a {"tool_calls": [...]} envelope must suppress it for
# any blank-line count, while a cue-free blank-line-separated envelope
# stays repairable.
# ----------------------------------------------------------------------

_ENVELOPE = '{"tool_calls": [{"name": "echo", "arguments": {"message": "example"}}]}'


def test_r4b_payload_cue_zero_blank_lines_suppresses() -> None:
    text = "Here's a sample response payload you might receive:\n" + _ENVELOPE
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert '"tool_calls"' in cleaned


def test_r4b_payload_cue_one_blank_line_suppresses() -> None:
    """negative_23: cue line, one blank line, then the envelope."""
    text = "Here's a sample response payload you might receive:\n\n" + _ENVELOPE
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert '"tool_calls"' in cleaned


def test_r4b_payload_cue_two_blank_lines_suppresses() -> None:
    text = "Here's a sample response payload you might receive:\n\n\n" + _ENVELOPE
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert '"tool_calls"' in cleaned


def test_r4b_no_cue_with_blank_lines_still_repairs() -> None:
    """Control positive: an envelope under blank-line-separated *non-cue* prose
    is a real multi-call turn and must still be repaired."""
    text = "Let me do that for you.\n\n" + _ENVELOPE
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "example"}


# ======================================================================
# R4c — call-syntax family
# ======================================================================


def test_r4c_tool_code_default_api_wrapper() -> None:
    text = '```tool_code\nprint(default_api.echo(message="probe"))\n```'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}
    assert cleaned == ""


def test_r4c_tool_code_plain_kwargs() -> None:
    text = '```tool_code\necho(message="probe")\n```'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}


def test_r4c_untagged_fence_kwargs() -> None:
    text = '```\necho(message="probe")\n```'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}


def test_r4c_colon_style_with_description() -> None:
    """Colon-separated kwargs, descriptive (non-cue) prose before the fence."""
    text = "The `echo` function echoes back the provided message.\n\n```\necho(message: 'demo')\n```"
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "demo"}


def test_r4c_json_object_arg_standalone() -> None:
    text = 'write_note({"path": "notes/a.txt", "text": "hello world"})'
    _, calls = repair_tool_calls_in_text(text, ["write_note"])
    assert _names(calls) == ["write_note"]
    assert _args(calls[0]) == {"path": "notes/a.txt", "text": "hello world"}


def test_r4c_multiple_kwargs() -> None:
    text = '```tool_code\nWrite(file_path="a.txt", content="hi")\n```'
    _, calls = repair_tool_calls_in_text(text, ["Write"])
    assert _names(calls) == ["Write"]
    assert _args(calls[0]) == {"file_path": "a.txt", "content": "hi"}


def test_r4c_corrupted_inner_json_declined() -> None:
    """A call whose inner JSON is unparseable must NOT be repaired (guard b)."""
    text = (
        'write_note({"path":"notes/日本語.txt","text":"line1\\n'
        'define("quotes", 1234) and a comma, plus {braces}\\n"})'
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["write_note"])
    assert calls == []
    assert cleaned == text


def test_r4c_bare_python_call_in_prose_declined() -> None:
    """Bare call syntax inline in prose is out of scope — must not repair."""
    text = 'You can invoke it directly, e.g. echo(message="probe") returns the same string.'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert cleaned == text


def test_r4c_command_style_declined() -> None:
    text = 'Run this:\necho --message "probe"'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []


def test_r4c_example_cue_before_fence_declined() -> None:
    """The negative_15 boundary: example cue + fenced call -> suppress."""
    text = (
        "For example, you would write:\n\n```\necho(message: 'demo')\n```\n\n"
        "but don't actually run it now — this is just the syntax."
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "echo(message" in cleaned  # fence left in place


def test_r4c_below_convention_cue_declined() -> None:
    """negative_17: "Below is the calling convention." (period-terminated) is a
    documentation framing recognised by the widened cue vocabulary."""
    text = "Below is the calling convention.\n\n```\necho(message: 'demo')\n```"
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "echo(message" in cleaned


def test_r4c_as_follows_colon_leadin_declined() -> None:
    """negative_18: a colon-terminated lead-in ("... is as follows:") marks the
    following call-syntax fence as illustrative."""
    text = 'The calling convention is as follows:\n\n```\necho(message="demo")\n```'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "echo(message" in cleaned


def test_r4c_this_format_colon_leadin_declined() -> None:
    """negative_20: "... uses this format:" — colon lead-in + "this format" cue."""
    text = "The button config uses this format:\n\n```\necho(message: 'demo')\n```"
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "echo(message" in cleaned


def test_r4c_long_leadin_colon_declined() -> None:
    """negative_21: an example lead-in longer than the cue window still suppresses,
    because the colon-terminated-lead-in rule is window-independent."""
    text = (
        "For example, here is a very very very very very very very very long "
        "lead-in sentence that goes on and on before finally showing the call:"
        "\n\n```\necho(message=\"probe\")\n```"
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "echo(message" in cleaned


def test_r4c_colon_leadin_does_not_block_real_json_call() -> None:
    """Guard scope: the colon-lead-in rule is confined to call-syntax fences, so a
    genuine JSON tool call under a colon intro ("Here's the tool call:") still
    repairs (mirrors bench bare_json_05 / multiple_calls_03)."""
    text = 'Here is the tool call:\n{"name": "echo", "arguments": {"message": "hi"}}'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "hi"}


def test_r4c_inline_prose_call_declined() -> None:
    """A call embedded mid-sentence (not standalone, not fenced) is never a candidate."""
    text = (
        "The echo(message) signature takes one string; calling "
        'echo(message="hi") simply returns "hi" unchanged.'
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert cleaned == text


def test_r4c_name_not_in_allowlist_declined() -> None:
    text = '```tool_code\nDeleteEverything(path="/")\n```'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert cleaned == text


def test_r4c_python_fence_is_protected() -> None:
    """A real ```python fence containing a call must be left as source code."""
    text = '```python\necho(message="probe")\n```'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "```python" in cleaned


def test_r4c_no_allowlist_skips() -> None:
    text = '```tool_code\necho(message="probe")\n```'
    cleaned, calls = repair_tool_calls_in_text(text)
    assert calls == []
    assert cleaned == text


# ======================================================================
# Integration / cross-family regression sanity
# ======================================================================


def test_r4_mixed_families_in_one_response() -> None:
    """A nested-XML call, an envelope, and a fenced call-syntax call together."""
    text = (
        '<function name="echo" arguments=\'{"message": "x"}\'/>\n'
        '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}\n'
        '```tool_code\nWrite(file_path="a.txt", content="hi")\n```'
    )
    _, calls = repair_tool_calls_in_text(text, ["echo", "Bash", "Write"])
    assert sorted(_names(calls)) == ["Bash", "Write", "echo"]


def test_r4_out_of_allowlist_always_wins() -> None:
    """Across every new family, a disallowed name is never repaired."""
    for text in (
        '<function name="Nuke" arguments=\'{}\'/>',
        '{"tool_calls": [{"name": "Nuke", "arguments": {}}]}',
        '{"function_call": {"name": "Nuke", "arguments": "{}"}}',
        '```tool_code\nNuke(path="/")\n```',
        'Nuke({"path": "/"})',
    ):
        _, calls = repair_tool_calls_in_text(text, ["echo", "Bash"])
        assert calls == [], text
