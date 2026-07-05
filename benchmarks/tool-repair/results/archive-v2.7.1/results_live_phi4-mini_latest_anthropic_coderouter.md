# L2 Live Tool-Call Benchmark — phi4-mini:latest

- Generated: 2026-07-05T04:27:10.325489+00:00
- Endpoint: http://localhost:8088  (wire=anthropic)
- Reps per prompt: 20   Prompts: simple_echo, complex_args, multi_tool_select, japanese_instruction, no_tool_temptation
- Total responses: 100

## Totals

| verdict | count | rate |
|---|---|---|
| native | 40 | 40.0% |
| repair | 0 | 0.0% |
| fail | 60 | 60.0% |
| error | 0 | — |

## Per-prompt

| prompt | native | repair | fail | error |
|---|---|---|---|---|
| simple_echo | 20 | 0 | 0 | 0 |
| complex_args | 0 | 0 | 20 | 0 |
| multi_tool_select | 0 | 0 | 20 | 0 |
| japanese_instruction | 20 | 0 | 0 | 0 |
| no_tool_temptation | 0 | 0 | 20 | 0 |

## Sample non-native / failed responses

- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
- `complex_args` [fail]: write_note({"path":"notes/日本語.txt","text":"line1\ndefine("quotes", 1234) and a comma, plus {braces}\n"})
