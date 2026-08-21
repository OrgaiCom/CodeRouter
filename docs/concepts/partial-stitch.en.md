# Mid-stream Partial Stitching (v2.0-H)

日本語版: [`partial-stitch.md`](./partial-stitch.md)

## Overview

When a provider crashes during a streaming response, normally all the text
generated up to that point is discarded and the client receives only an
`event: error`. The Partial Stitch feature in v2.0-H adds a "surface mode"
that gracefully returns the accumulated text to the client.

## Configuration

```yaml
profiles:
  - name: long-session
    partial_stitch_action: surface   # off | surface
```

| Value | Behavior |
|------|------|
| `off` (default) | returns `event: error` as before |
| `surface` | returns the accumulated text and ends the stream cleanly |

## Flow (surface mode)

1. During a streaming response, `_StreamUsageAccumulator` accumulates the text content_block
2. The provider crashes mid-stream → raises `MidStreamError(partial_content=[...])`
3. Ingress checks `partial_stitch_action`
4. When it is `surface` and partial_content exists:
   - `event: message_delta` (stop_reason: null, usage: {output_tokens: 0})
   - `event: message_stop`
   - `event: coderouter_partial` (metadata + accumulated text)
5. The client can process this as a normal stream termination

## coderouter_partial event

```json
{
  "type": "coderouter_partial",
  "partial_content": [
    {"type": "text", "text": "accumulated text..."}
  ],
  "provider": "ollama-local",
  "reason": "mid_stream_failure",
  "original_error": "connection reset by peer"
}
```

`reason` defaults to `mid_stream_failure`. Since v2.15.0 it becomes
`stream_truncated` when the failure came from truncation detection
(`stream_truncation_action: error`) →
[`stream-truncation.en.md`](./stream-truncation.en.md)

**Compatibility**: the Anthropic SDK automatically ignores unknown event
types, so there is no impact on existing clients. CodeRouter-aware clients
can read the `coderouter_partial` event to display the partial response.

## Limitations

- **Only text blocks are accumulated**: partial JSON of `tool_use` blocks is not surfaced
- **Memory**: request-lifecycle scoped, so no problem even with long responses
- **Phase 2 (future)**: retry mode — inject the partial into context and resend to the next provider

## Logs / metrics

- Log event: `partial-stitch-surfaced` (info)
  - fields: `provider`, `profile`, `reason`, `text_blocks`, `text_length`
- MetricsCollector: `partial_stitch_surfaced_total` counter
- Prometheus: `coderouter_partial_stitch_surfaced_total`

## Use cases

| Scenario | Recommended setting |
|----------|----------|
| Short Q&A (< 5s) | `off` — resending is faster |
| Long generation (code, documents, 30s+) | `surface` — don't waste 30s of generation |
| Claude Code agent session | `surface` — the user can review the partial result |
