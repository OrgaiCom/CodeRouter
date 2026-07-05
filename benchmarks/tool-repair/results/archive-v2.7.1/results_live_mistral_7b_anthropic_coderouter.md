# L2 Live Tool-Call Benchmark — mistral:7b

- Generated: 2026-07-05T04:21:21.437773+00:00
- Endpoint: http://localhost:8088  (wire=anthropic)
- Reps per prompt: 20   Prompts: simple_echo, complex_args, multi_tool_select, japanese_instruction, no_tool_temptation
- Total responses: 100

## Totals

| verdict | count | rate |
|---|---|---|
| native | 80 | 80.0% |
| repair | 0 | 0.0% |
| fail | 20 | 20.0% |
| error | 0 | — |

## Per-prompt

| prompt | native | repair | fail | error |
|---|---|---|---|---|
| simple_echo | 20 | 0 | 0 | 0 |
| complex_args | 20 | 0 | 0 | 0 |
| multi_tool_select | 20 | 0 | 0 | 0 |
| japanese_instruction | 20 | 0 | 0 | 0 |
| no_tool_temptation | 0 | 0 | 20 | 0 |

## Sample non-native / failed responses

- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
- `no_tool_temptation` [fail]:  The `echo` function echoes back the provided message.  ``` echo(message: 'demo') ```
