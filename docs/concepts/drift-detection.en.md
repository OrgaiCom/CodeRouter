# Drift Detection (v2.0-G)

日本語版: [`drift-detection.md`](./drift-detection.md)

A guard that automatically detects the gradual degradation (drift) of model
response quality in long-running agent sessions and executes corrective
action.

## Background

Local LLM backends such as Ollama can exhibit the following quality-degradation patterns when run for a long time:

- responses become empty (output_tokens=0)
- response length gradually shortens (length collapse)
- stops returning tool_use in situations where it should (tool silence)
- abnormal stop_reason increases
- error rate rises
- repeats the same response and stops making progress (goal progress stall — only when response_fingerprint is set)

Individually these are not fatal, but they accumulate until the agent session is effectively inoperable.
Drift Detection monitors these as 6 quality signals over a rolling window and,
once a threshold is crossed, automatically executes corrective action.

## Configuration

Add the following fields to a profile in `providers.yaml`:

```yaml
profiles:
  - name: long-session
    providers: [ollama-qwen3, ollama-gemma4]
    drift_detection_action: reload      # off | warn | promote | reload
    drift_detection_sensitivity: normal # low | normal | high
    drift_detection_window_size: 20     # per-provider rolling window size (4-200)
    drift_detection_cooldown_s: 300     # recovery wait seconds after promote/reload (10-3600)
```

### Action

| Action | Behavior |
|--------|------|
| `off` | detection disabled (default) |
| `warn` | detection + log only (adds X-CodeRouter-Drift header) |
| `promote` | warn + demote the provider's rank in the chain (divert traffic to the next provider) |
| `reload` | promote + Ollama KV cache flush (unload model with keep_alive=0 → fresh reload on next request) |

### Sensitivity

| Preset | empty_response_rate | length_collapse_ratio | tool_silence_rate | stop_anomaly_rate | error_rate | min_window_fill |
|--------|--------------------:|----------------------:|------------------:|------------------:|-----------:|----------------:|
| `low` | 0.5 | 0.3 | 0.8 | 0.6 | 0.4 | 10 |
| `normal` | 0.3 | 0.5 | 0.7 | 0.4 | 0.25 | 6 |
| `high` | 0.2 | 0.7 | 0.5 | 0.3 | 0.15 | 4 |

(Each preset also includes a `repetition_rate_threshold` for `goal_progress_stall`: low=0.6 / normal=0.4 / high=0.25)

> **goal_mode**: setting `goal_mode: true` on a profile applies a dedicated `goal` preset (empty=0.2 / collapse=0.6 / tool_silence=0.5 / stop=0.3 / error=0.15 / repetition=0.2 / min_window_fill=4) regardless of `drift_detection_sensitivity`. goal_mode itself is not a signal but a bool flag that switches the threshold preset. It picks up progress stalls and length collapse earlier, for `/goal` sessions.

## The 6 quality signals

1. **empty_response_rate** — rate of responses with output_tokens=0 (excluding errors)
2. **length_collapse_ratio** — median output_tokens of the second half of the window / median of the first half (collapse when the ratio is below the threshold)
3. **tool_silence_rate** — rate of not returning tool_use for requests that include tools[]
4. **stop_anomaly_rate** — rate of stop_reason other than end_turn / tool_use / max_tokens
5. **error_rate** — rate of provider errors
6. **goal_progress_stall** — rate at which a previously seen fingerprint reappears within the window (only for observations where response_fingerprint is set, fires at 3 or more). A sign that the model is not making progress and repeating the same response

## Severity determination

- severe signal ×1 (empty_response_rate, length_collapse) → **severe**
- mild signal ×2 or more (tool_silence, stop_anomaly, error_rate, goal_progress_stall) → **severe**
- mild signal ×1 → **mild**

## Response Header

`X-CodeRouter-Drift: mild` or `X-CodeRouter-Drift: severe` is added to the response header.
The verdict is managed per request (request-scoped), so detection results from other concurrent requests are never mixed into the header.
For streaming, detection runs after the header has been sent, so in principle no header is added — check detection results in logs / metrics.

## Cooldown & Recovery

When a `promote` / `reload` action fires:

1. While the target provider is in cooldown, it is demoted to the tail of the chain (reflected in the actual attempt order regardless of the profile's `adaptive` setting)
2. For `reload`, additionally attempt an Ollama KV cache flush (best-effort)
3. Skip re-detection for `drift_detection_cooldown_s` seconds
4. After cooldown expires, restore rank + clear the window on the next observation record
5. Emit a `drift-recovered` log

## Observability

### Log events

| Event | Level | Description |
|-------|-------|------|
| `drift-detected` | WARNING | drift detected (includes severity, signals, action) |
| `drift-promoted` | INFO | provider rank demoted |
| `drift-reload-attempted` | INFO | Ollama KV flush attempt result |
| `drift-recovered` | INFO | cooldown expired → rank restored |

### Prometheus Metrics

```
coderouter_drift_detected_total{provider="..."} — detection count
coderouter_drift_promoted_total                  — demotion count
coderouter_drift_reload_total                    — reload attempt count
coderouter_drift_reload_success_total            — reload success count
```

### /metrics.json

```json
{
  "counters": {
    "drift_detected_total": 3,
    "drift_detected_by_provider": {"ollama-qwen3": 3},
    "drift_promoted_total": 2,
    "drift_reload_total": 2,
    "drift_reload_success_total": 1
  }
}
```

## Recommended configuration patterns

### Claude Code + Ollama (long coding sessions)

```yaml
profiles:
  - name: default
    providers: [ollama-qwen3, ollama-gemma4]
    drift_detection_action: reload
    drift_detection_sensitivity: normal
    drift_detection_cooldown_s: 300
```

### Monitoring only (assess the situation first)

```yaml
profiles:
  - name: default
    providers: [ollama-qwen3]
    drift_detection_action: warn
    drift_detection_sensitivity: high
```

### Failover-focused with multiple providers

```yaml
profiles:
  - name: default
    providers: [ollama-qwen3, ollama-gemma4, openrouter-llama4]
    drift_detection_action: promote
    drift_detection_sensitivity: normal
    drift_detection_cooldown_s: 600
```
