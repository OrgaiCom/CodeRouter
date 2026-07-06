# Context Budget Management (v2.0.0)

日本語版: [`context-budget.md`](./context-budget.md)

CodeRouter proactively prevents the problem where a long agent session exhausts the context window and dies.

## Why it's needed

When you run coding sessions over 8 hours with Claude Code / Cline / OpenClaw etc., messages asymptote toward the backend's context window (32K–200K tokens). The moment the limit is crossed the backend returns a 400 error and the agent session dies instantly.

Conventional workarounds:

- Manually "start a new session" → interrupts work + loses context
- Monitor token count with an external tool → tedious + complex configuration

CodeRouter v2.0.0's Context Budget Management solves it **automatically**:

1. **warn** — notifies via a response header once usage exceeds 80%
2. **auto trim** — automatically removes old messages once usage exceeds 90% and continues the session

## Benefits

- **Zero session deaths**: no matter how long the session, it never dies from context overflow
- **Tool-pair preservation**: preserves `tool_use` / `tool_result` pairs atomically. The agent loop is not broken after a trim
- **Safe with no configuration**: default `off`. Enable with a one-line opt-in; no impact on existing environments
- **Zero external dependencies**: estimates tokens with the char/4 heuristic. No extra deps like tiktoken (the 5-deps invariant is unchanged)
- **Full observability**: state is visible via 4 paths — response header / structured log / Prometheus metrics / stats TUI

## How to configure

Add the following to a profile in `providers.yaml`:

```yaml
profiles:
  - name: default
    providers:
      - ollama-qwen3
    # Context Budget Guard (v2.0.0)
    context_budget_action: trim          # off | warn | trim
    context_budget_warn_threshold: 0.80  # warning at 80% usage
    context_budget_trim_threshold: 0.90  # auto trim at 90% usage
    context_budget_trim_target: 0.75     # target usage after trim
    context_budget_preserve_last_n: 4    # always keep the last N messages
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `context_budget_action` | `off` | `off`: disabled / `warn`: warn only / `trim`: warn + auto trim |
| `context_budget_warn_threshold` | `0.80` | context usage at which a warning is issued |
| `context_budget_trim_threshold` | `0.90` | context usage at which an auto trim fires |
| `context_budget_trim_target` | `0.75` | target usage after trim (removes messages down to here) |
| `context_budget_preserve_last_n` | `4` | the last N messages are always kept even during a trim |

### Choosing an action

| Use case | Recommended action |
|---|---|
| Want to observe first | `warn` — notifies via log and header, does not touch messages |
| Want stable long-session operation | `trim` — automatically prevents overflow |
| Managing tokens yourself | `off` — guard disabled |

## How it works

```
request arrives
    │
    ▼
estimate_context_usage()
    │  estimate token count with char/4
    │  usage_ratio = estimated / max_context_tokens
    │
    ├─ ratio < warn_threshold → passes through unchanged
    │
    ├─ warn_threshold ≤ ratio < trim_threshold
    │   → emit WARNING log
    │   → add X-CodeRouter-Context-Budget: warning header
    │   → (if action=warn, ends here)
    │
    └─ ratio ≥ trim_threshold (when action=trim)
        → run trim_to_budget()
        → remove old messages from the front
        → preserve tool_use/tool_result pairs atomically
        → add X-CodeRouter-Context-Budget: trimmed header
        → send the trimmed request to the backend
```

## How token estimation works

The char/4 heuristic with no external dependency:

```
estimated_tokens ≈ len(request_json) // 4
```

- Within ±10% of the measured value for English text
- For CJK (Japanese / Chinese) it tends to underestimate, so setting the threshold 5–10% lower is safer

`max_context_tokens` for major models is already bundled in `model-capabilities.yaml`:

| Model | max_context_tokens |
|---|---|
| Claude (Sonnet/Opus/Haiku) | 200,000 |
| Qwen3 | 32,768 |
| Qwen3-Coder / Qwen3.5 / Qwen3.6 | 131,072 |
| Gemma 4 | 131,072 |
| DeepSeek V3 / R1 | 131,072 |
| GPT-OSS | 131,072 |

For models not in model-capabilities.yaml, specify it explicitly in the provider config:

```yaml
providers:
  - name: my-custom-model
    kind: openai_compat
    base_url: http://localhost:11434/v1
    model: custom-model
    max_context_tokens: 65536   # explicit
```

## Trim algorithm

1. The system prompt is never removed
2. The last `preserve_last_n` messages are always kept
3. Old messages are removed from the front
4. **Tool-pair preservation**: identifies the corresponding `tool_use` / `tool_result` by `tool_use_id` and prevents a state where only one half remains (fixpoint algorithm)
5. Re-estimate after removal → if still over `trim_target`, decrement `preserve_last_n` by 1 and retry (minimum floor: 2)

## Observability

### Response Header

```
X-CodeRouter-Context-Budget: warning   # warning state
X-CodeRouter-Context-Budget: trimmed   # trim performed
```

For streaming responses too, the header is added before SSE begins.

### Prometheus Metrics

```
coderouter_context_budget_warnings_total{profile="default"}  # warning count
coderouter_context_budget_trims_total{profile="default"}     # trim count
coderouter_context_budget_usage_ratio{profile="default"}     # latest usage ratio (gauge)
```

```bash
curl http://localhost:8088/metrics | grep context_budget
```

### Stats TUI

```bash
coderouter stats --port 8088
```

`ctx_budget_warnings` / `ctx_budget_trims` / `latest_ratio` are shown in the Gates section.

### Structured Log

```json
{"msg": "context-budget-warning", "profile": "default", "usage_ratio": 0.83, ...}
{"msg": "context-budget-trimmed", "profile": "default", "messages_removed": 5, ...}
```

## Verification config

A providers.yaml for verification is bundled:

```bash
cp examples/providers.v2-context-budget.yaml ~/.coderouter/providers.yaml
coderouter serve --port 8088 --log-level debug
```

It has lowered thresholds so you can trigger warn / trim with a small amount of data. See `docs/inside/verification-v2.0-F.md` for details.
