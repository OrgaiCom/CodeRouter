# L2 Live Tool-Call Benchmark — gemma3:27b

- Generated: 2026-07-05T03:53:54.373594+00:00
- Endpoint: http://localhost:11434/v1  (wire=openai)
- Reps per prompt: 20   Prompts: simple_echo, complex_args, multi_tool_select, japanese_instruction, no_tool_temptation
- Total responses: 100

## Totals

| verdict | count | rate |
|---|---|---|
| native | 0 | 0.0% |
| repair | 0 | 0.0% |
| fail | 0 | 0.0% |
| error | 100 | — |

## Per-prompt

| prompt | native | repair | fail | error |
|---|---|---|---|---|
| simple_echo | 0 | 0 | 0 | 20 |
| complex_args | 0 | 0 | 0 | 20 |
| multi_tool_select | 0 | 0 | 0 | 20 |
| japanese_instruction | 0 | 0 | 0 | 20 |
| no_tool_temptation | 0 | 0 | 0 | 20 |

## Sample non-native / failed responses

- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
- `simple_echo` [error]: Client error '400 Bad Request' for url 'http://localhost:11434/v1/chat/completions' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTT
