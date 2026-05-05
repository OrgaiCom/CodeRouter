# Continuous Probing (v2.0-I)

## Overview

CodeRouter's **continuous probing** sends periodic 1-token health-check
requests to each configured provider, detecting backend failures during
idle periods when no user traffic flows. Without this, the L5 backend
health state machine only transitions on real request outcomes — meaning
a crashed Ollama instance stays marked HEALTHY until the next user
request hits it and fails.

## Configuration

```yaml
# providers.yaml (global section)
continuous_probe: active    # off (default) | active
probe_interval_s: 60        # seconds between probe rounds
probe_paid: false           # also probe paid providers?
probe_timeout_s: 10         # per-provider timeout in seconds
```

| Field | Default | Description |
|-------|---------|-------------|
| `continuous_probe` | `off` | Set to `active` to enable background probing |
| `probe_interval_s` | `60` | Seconds between full probe sweeps |
| `probe_paid` | `false` | When false, providers with `paid: true` are skipped |
| `probe_timeout_s` | `10` | Per-provider request timeout |

## How It Works

1. **Startup**: When `continuous_probe: active`, an asyncio background
   task starts during the FastAPI lifespan. It waits one full interval
   before the first probe round (letting the server finish startup).

2. **Probe round**: Each eligible provider receives a minimal 1-token
   completion request:
   - `openai_compat`: `POST {base_url}/chat/completions` with
     `max_tokens: 1`
   - `anthropic`: `POST {base_url}/v1/messages` with `max_tokens: 1`

3. **Health integration**: Probe outcomes feed into the L5 backend
   health state machine via `record_attempt()`. A provider that fails
   3 consecutive probes (the default threshold) transitions from
   HEALTHY → DEGRADED → UNHEALTHY, causing it to be demoted to the
   back of the chain on subsequent real requests.

4. **Model drift detection**: On successful probes, the response's
   `model` field is compared against the configured `provider.model`.
   A mismatch emits a `probe-capabilities-drift` warning — useful for
   detecting silent Ollama model updates or misconfiguration.

5. **Graceful shutdown**: The lifespan exit path signals the probe loop
   to stop, and awaits its completion for clean resource teardown.

## Log Events

| Event | Level | Description |
|-------|-------|-------------|
| `continuous-probe-started` | INFO | Emitted once at startup |
| `probe-completed` | INFO | Per-provider probe result |
| `probe-round-completed` | INFO | Summary after one full sweep |
| `probe-capabilities-drift` | WARN | Model name mismatch detected |

## Metrics

### Prometheus (`/metrics`)

| Metric | Type | Labels |
|--------|------|--------|
| `coderouter_probe_total` | counter | `provider` |
| `coderouter_probe_outcomes_total` | counter | `provider`, `outcome` |
| `coderouter_probe_rounds_total` | counter | — |
| `coderouter_probe_latency_ms` | gauge | `provider` |
| `coderouter_probe_drift_detected_total` | counter | `provider` |

### JSON (`/metrics.json`)

The `counters` object includes:

```json
{
  "probe_total": {"ollama-local": 120},
  "probe_success": {"ollama-local": 118},
  "probe_failure": {"ollama-local": 2},
  "probe_rounds_total": 60,
  "probe_latency_ms": {"ollama-local": 23.4},
  "probe_drift_detected": {}
}
```

## Design Decisions

- **1-token completion** rather than `/api/version` or `/api/tags`
  because version endpoints are Ollama-only; a generate confirms the
  entire model-serving pipeline is operational.
- **Sequential probing** (not parallel) to avoid hammering backends
  and to keep the implementation trivially correct.
- **No new dependency** — uses httpx (already a runtime dep) + asyncio
  (stdlib). The 5-deps invariant is preserved.
- **Paid provider protection** — `probe_paid: false` (default) ensures
  operators don't accidentally spend money on probe requests to
  OpenRouter / Anthropic API.
