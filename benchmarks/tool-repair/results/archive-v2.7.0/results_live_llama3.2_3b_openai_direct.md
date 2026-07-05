# L2 Live Tool-Call Benchmark — llama3.2:3b

- Generated: 2026-07-05T03:46:21.678432+00:00
- Endpoint: http://localhost:11434/v1  (wire=openai)
- Reps per prompt: 20   Prompts: simple_echo, complex_args, multi_tool_select, japanese_instruction, no_tool_temptation
- Total responses: 100

## Totals

| verdict | count | rate |
|---|---|---|
| native | 100 | 100.0% |
| repair | 0 | 0.0% |
| fail | 0 | 0.0% |
| error | 0 | — |

## Per-prompt

| prompt | native | repair | fail | error |
|---|---|---|---|---|
| simple_echo | 20 | 0 | 0 | 0 |
| complex_args | 20 | 0 | 0 | 0 |
| multi_tool_select | 20 | 0 | 0 | 0 |
| japanese_instruction | 20 | 0 | 0 | 0 |
| no_tool_temptation | 20 | 0 | 0 | 0 |

## Sample non-native / failed responses

_all responses were native tool_calls_
