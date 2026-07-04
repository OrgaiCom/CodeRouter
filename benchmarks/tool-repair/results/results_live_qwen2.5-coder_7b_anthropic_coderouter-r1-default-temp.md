# L2 Live Tool-Call Benchmark — qwen2.5-coder:7b

- Generated: 2026-07-04T13:13:46.931028+00:00
- Endpoint: http://localhost:8088  (wire=anthropic)
- Reps per prompt: 20   Prompts: simple_echo, complex_args, multi_tool_select, japanese_instruction, no_tool_temptation
- Total responses: 100

## Totals

| verdict | count | rate |
|---|---|---|
| native | 99 | 99.0% |
| repair | 0 | 0.0% |
| fail | 1 | 1.0% |
| error | 0 | — |

## Per-prompt

| prompt | native | repair | fail | error |
|---|---|---|---|---|
| simple_echo | 20 | 0 | 0 | 0 |
| complex_args | 20 | 0 | 0 | 0 |
| multi_tool_select | 20 | 0 | 0 | 0 |
| japanese_instruction | 20 | 0 | 0 | 0 |
| no_tool_temptation | 19 | 0 | 1 | 0 |

## Sample non-native / failed responses

- `no_tool_temptation` [fail]: ```xml <tools>   <function name="echo" arguments='{"message": "demo"}' /> </tools> ```
