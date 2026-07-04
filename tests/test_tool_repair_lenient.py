"""Unit tests for the R1/R2/R3 lenient tool-call repair enhancements.

These cover the malformed-JSON, key-alias, and XML-flavoured forms that small
coding models (qwen2.5-coder etc.) emit at non-zero temperature, alongside the
false-positive guards that keep documentation / source code from being turned
into executable tool calls. The strict-JSON happy paths live in
``test_tool_repair.py``; this file targets the lenient extensions.
"""

from __future__ import annotations

import json

from coderouter.translation.tool_repair import (
    deduplicate_tool_calls,
    repair_tool_calls_in_text,
)


def _names(calls: list[dict]) -> list[str]:
    return [c["function"]["name"] for c in calls]


def _args(call: dict) -> dict:
    return json.loads(call["function"]["arguments"])


# ======================================================================
# R1 — lenient JSON parsing
# ======================================================================


def test_r1_trailing_comma_is_repaired() -> None:
    text = '{"name": "Bash", "arguments": {"command": "ls",}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]
    assert _args(calls[0]) == {"command": "ls"}


def test_r1_trailing_comma_top_level() -> None:
    text = '{"name": "Bash", "arguments": {"command": "ls"},}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]


def test_r1_single_quotes_are_repaired() -> None:
    text = "{'name': 'Bash', 'arguments': {'command': 'ls'}}"
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]
    assert _args(calls[0]) == {"command": "ls"}


def test_r1_single_quotes_with_cjk() -> None:
    text = "了解しました。{'name': 'Bash', 'arguments': {'command': 'ls 日本語ディレクトリ'}}"
    cleaned, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]
    assert _args(calls[0]) == {"command": "ls 日本語ディレクトリ"}
    assert "了解しました。" in cleaned


def test_r1_unquoted_keys_are_repaired() -> None:
    text = '{name: "Bash", arguments: {command: "ls"}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]
    assert _args(calls[0]) == {"command": "ls"}


def test_r1_double_brace_in_json_fence() -> None:
    text = '```json\n{{"name": "echo", "arguments": {"message": "probe"}}}\n```'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}


def test_r1_double_brace_in_xml_fence() -> None:
    text = '```xml\n{{"name": "echo", "arguments": {"message": "demo"}}}\n```'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]


def test_r1_double_brace_bare() -> None:
    text = '{{"name": "Bash", "arguments": {"command": "pwd"}}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]


def test_r1_single_quote_apostrophe_in_value_preserved() -> None:
    """An escaped apostrophe inside a single-quoted value must survive."""
    text = "{'name': 'Bash', 'arguments': {'command': 'echo it\\'s ok'}}"
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]
    assert _args(calls[0]) == {"command": "echo it's ok"}


def test_r1_strict_json_still_wins() -> None:
    """Well-formed JSON must not be perturbed by the lenient path."""
    text = '{"name": "Bash", "arguments": {"a": 1, "b": [1, 2, 3]}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _args(calls[0]) == {"a": 1, "b": [1, 2, 3]}


# ======================================================================
# R2 — shape dictionary aliases
# ======================================================================


def test_r2_key_tool() -> None:
    text = '{"tool": "Bash", "arguments": {"command": "ls"}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]


def test_r2_key_tool_name_and_input() -> None:
    text = '{"tool_name": "Bash", "input": {"command": "ls"}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]
    assert _args(calls[0]) == {"command": "ls"}


def test_r2_key_parameters() -> None:
    text = '{"name": "Bash", "parameters": {"command": "ls"}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]


def test_r2_key_args() -> None:
    text = '{"name": "Bash", "args": {"command": "ls"}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]


def test_r2_function_wrapper_with_aliases() -> None:
    text = '{"function": {"tool_name": "Grep", "parameters": {"pattern": "x"}}}'
    _, calls = repair_tool_calls_in_text(text, ["Grep"])
    assert _names(calls) == ["Grep"]


def test_r2_ambiguous_name_keys_not_repaired() -> None:
    """When both 'name' and 'tool' are present, decline (ambiguous -> safe)."""
    text = '{"name": "Bash", "tool": "Grep", "arguments": {"x": 1}}'
    cleaned, calls = repair_tool_calls_in_text(text, ["Bash", "Grep"])
    assert calls == []
    assert cleaned == text


def test_r2_ambiguous_arg_keys_not_repaired() -> None:
    text = '{"name": "Bash", "arguments": {"a": 1}, "parameters": {"b": 2}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert calls == []


def test_r2_alias_still_respects_allowlist() -> None:
    text = '{"tool": "DeleteEverything", "arguments": {"path": "/"}}'
    cleaned, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert calls == []
    assert cleaned == text


def test_r2_alias_no_args_key_is_not_a_call() -> None:
    """A name alias without any args container is not a tool call."""
    text = '{"tool": "Bash"}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert calls == []


# ======================================================================
# R3 — XML-flavoured forms
# ======================================================================


def test_r3_self_closing_attribute_form() -> None:
    text = '<echo message="probe"/>'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}
    assert cleaned == ""


def test_r3_attribute_form_in_prose() -> None:
    text = '了解しました。<echo message="こんにちは"/> を実行します。'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "こんにちは"}
    assert "了解しました。" in cleaned
    assert "を実行します。" in cleaned


def test_r3_multiple_attributes() -> None:
    text = '<Write file_path="a.txt" content="hi"/>'
    _, calls = repair_tool_calls_in_text(text, ["Write"])
    assert _names(calls) == ["Write"]
    assert _args(calls[0]) == {"file_path": "a.txt", "content": "hi"}


def test_r3_tool_wrapper_form() -> None:
    text = '<tool>{"name": "echo", "arguments": {"message": "probe"}}</tool>'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}
    assert cleaned == ""


def test_r3_named_wrapper_form() -> None:
    text = '<echo>{"message": "probe"}</echo>'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "probe"}


def test_r3_named_wrapper_with_whitespace() -> None:
    text = '<echo>   {"message": "demo"} </echo>'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]
    assert _args(calls[0]) == {"message": "demo"}


def test_r3_wrapper_delegates_to_lenient() -> None:
    """Wrapper body may itself be malformed; R1 lenient parse rescues it."""
    text = "<tool>{'name': 'echo', 'arguments': {'message': 'probe'}}</tool>"
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert _names(calls) == ["echo"]


def test_r3_tag_not_in_allowlist_is_ignored() -> None:
    text = '<danger message="x"/>'
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert cleaned == text


def test_r3_no_allowlist_skips_xml() -> None:
    """Without an allow-list, XML extraction is skipped entirely (safe)."""
    text = '<echo message="probe"/>'
    cleaned, calls = repair_tool_calls_in_text(text)
    assert calls == []
    assert cleaned == text


def test_r3_think_tag_excluded() -> None:
    text = (
        '<think>I could emit <echo message="secret"/> but the user only '
        "asked a question.</think>\nNo action is required."
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "<echo" in cleaned  # left untouched inside the think block


def test_r3_html_tags_not_treated_as_tools() -> None:
    text = (
        "<p>Here is the plan:</p>\n<ul><li>echo the result</li></ul>\n"
        '<div class="note">No tool needed.</div>'
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo", "read_file"])
    assert calls == []
    assert cleaned == text


# ======================================================================
# False-positive guards (the safety-critical class)
# ======================================================================


def test_guard_python_fence_with_allowed_name() -> None:
    text = (
        "You can build the payload in Python:\n```python\n"
        'payload = {"name": "echo", "arguments": {"message": "probe"}}\n'
        "send(payload)\n```\nThen post it."
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert "```python" in cleaned


def test_guard_python_fence_single_quote_dict() -> None:
    text = "```python\nresult = {'name': 'echo', 'level': 'info'}\nprint(result)\n```"
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert cleaned == text


def test_guard_prose_example_bare_json() -> None:
    text = (
        'For example, you would write {"name": "echo", "arguments": '
        '{"message": "probe"}} to invoke it, but here we don\'t need any tool.'
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []
    assert cleaned == text


def test_guard_prose_example_japanese() -> None:
    text = '例えば {"name": "echo", "arguments": {"message": "x"}} のように書きます。'
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []


def test_guard_xml_doc_example_placeholder() -> None:
    text = (
        "The XML tool syntax is documented like this: write "
        '<echo message="..."/> where the message attribute holds your text.'
    )
    _, calls = repair_tool_calls_in_text(text, ["echo"])
    assert calls == []


def test_guard_out_of_allowlist_always_wins() -> None:
    """Even a perfectly-shaped call with a disallowed name is never repaired."""
    for text in (
        '{"name": "Nuke", "arguments": {}}',
        "{'tool': 'Nuke', 'parameters': {}}",
        '<Nuke path="/"/>',
        '{{"tool_name": "Nuke", "input": {}}}',
    ):
        _, calls = repair_tool_calls_in_text(text, ["echo", "Bash"])
        assert calls == [], text


def test_guard_prose_does_not_block_genuine_call_cue() -> None:
    """A normal narration lead-in ('Let me run that.') must NOT be suppressed."""
    text = 'Let me run that. {"name": "Bash", "arguments": {"command": "date"}}'
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]


# ======================================================================
# Residue cleanup (trap #2)
# ======================================================================


def test_residue_empty_fence_removed() -> None:
    text = '```json\n{"name": "Bash", "arguments": {"command": "ls"}}\n```'
    cleaned, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert _names(calls) == ["Bash"]
    assert "```" not in cleaned
    assert cleaned == ""


def test_residue_empty_array_removed() -> None:
    text = '[{"name": "ls", "arguments": {"p": "."}}, {"name": "ls", "arguments": {"p": "."}}]'
    cleaned, calls = repair_tool_calls_in_text(text, ["ls"])
    assert _names(calls) == ["ls"]  # deduped
    assert "[," not in cleaned
    assert "[]" not in cleaned


def test_residue_nested_fence_arg_leaves_no_stray_fence() -> None:
    text = (
        '```json\n{"name": "Write", "arguments": '
        '{"file_path": "s.md", "content": "```py\\nprint(1)\\n```"}}\n```'
    )
    cleaned, calls = repair_tool_calls_in_text(text, ["Write"])
    assert _names(calls) == ["Write"]
    assert cleaned.strip() == ""


# ======================================================================
# Integration / regression sanity
# ======================================================================


def test_mixed_forms_in_one_response() -> None:
    """A fenced call, an XML attribute call, and a bare malformed call."""
    text = (
        "First:\n```json\n{\"name\": \"Bash\", \"arguments\": {\"command\": \"pwd\"}}\n```\n"
        'Then: <echo message="hi"/>\n'
        "Finally: {'name': 'Read', 'arguments': {'path': '/x'}}"
    )
    _, calls = repair_tool_calls_in_text(text, ["Bash", "echo", "Read"])
    assert sorted(_names(calls)) == ["Bash", "Read", "echo"]


def test_malformed_and_valid_dedup() -> None:
    """A strict and a lenient rendering of the same call dedup to one."""
    text = (
        '{"name": "Bash", "arguments": {"command": "ls"}}\n'
        "{'name': 'Bash', 'arguments': {'command': 'ls'}}"
    )
    _, calls = repair_tool_calls_in_text(text, ["Bash"])
    assert len(calls) == 1


def test_deduplicate_standalone_still_works() -> None:
    calls = [
        {"function": {"name": "Bash", "arguments": '{"c": "ls"}'}},
        {"function": {"name": "Bash", "arguments": '{"c": "ls"}'}},
    ]
    assert len(deduplicate_tool_calls(calls)) == 1
