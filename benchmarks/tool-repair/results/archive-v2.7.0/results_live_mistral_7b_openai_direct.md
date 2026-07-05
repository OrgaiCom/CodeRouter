# L2 Live Tool-Call Benchmark — mistral:7b

- Generated: 2026-07-05T03:48:05.462578+00:00
- Endpoint: http://localhost:11434/v1  (wire=openai)
- Reps per prompt: 20   Prompts: simple_echo, complex_args, multi_tool_select, japanese_instruction, no_tool_temptation
- Total responses: 100

## Totals

| verdict | count | rate |
|---|---|---|
| native | 0 | 0.0% |
| repair | 80 | 80.0% |
| fail | 20 | 20.0% |
| error | 0 | — |

## Per-prompt

| prompt | native | repair | fail | error |
|---|---|---|---|---|
| simple_echo | 0 | 20 | 0 | 0 |
| complex_args | 0 | 20 | 0 | 0 |
| multi_tool_select | 0 | 20 | 0 | 0 |
| japanese_instruction | 0 | 20 | 0 | 0 |
| no_tool_temptation | 0 | 0 | 20 | 0 |

## Sample non-native / failed responses

- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
- `simple_echo` [repair]:  [{"name":"echo","arguments":{"message":"probe"}}]
