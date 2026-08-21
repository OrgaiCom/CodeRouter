# Stream Truncation Detection (v2.15.0)

日本語版: [`stream-truncation.md`](./stream-truncation.md)

## Overview

This detects the case where an upstream **closed its HTTP response cleanly
while the LLM protocol carried inside it was still mid-message**:

- Anthropic wire: no `message_stop`
- OpenAI wire: no `data: [DONE]` and no `finish_reason`

llama.cpp slot preemption, an `--n-predict` cut-off, a front proxy closing an
EOF-delimited body, a local server that OOM'd — all of them produce exactly
this shape. **Transport-level** breakage (timeouts, `httpx.RemoteProtocolError`)
was already caught as an `AdapterError`. What this closes is the layer the
transport cannot see.

Up to v2.14.0 such a stream was indistinguishable from a complete one: the
adapters never recorded whether a terminator arrived, and the translation
layer's terminator-synthesis guards (H6 / M9 in
`coderouter/translation/convert.py`) then fabricated `stop_reason: end_turn` /
`finish_reason: "stop"`.

**The synthesis itself is correct and stays** — removing it hangs the client
(Claude Code). All v2.15.0 changes is that the engine is now *told* it
happened.

### Important: the synthesis does not cover every route

H6 / M9 are guards in `translation/convert.py`. **A route that does not go
through the translation layer therefore has no synthesis.** There are four:

| Client wire | Backend `kind` | Translation | Terminator guaranteed |
|-------------|----------------|-------------|-----------------------|
| `/v1/messages` | `openai_compat` | `stream_chat_to_anthropic_events` (H6) | yes |
| `/v1/chat/completions` | `anthropic` | `stream_anthropic_to_chat_chunks` (M9) | yes |
| `/v1/chat/completions` | `openai_compat` | passthrough (but the ingress always appends `data: [DONE]`) | yes |
| **`/v1/messages`** | **`anthropic`** | **passthrough (no translation)** | **no** |

On the last row — **native Anthropic passthrough** — the adapter hands the
events it received straight to the client and nothing anywhere adds a
`message_stop`. If the upstream goes quiet, **an unterminated SSE stream
reaches the client even under the default `off`.** That is v2.14.0 behavior,
not something v2.15.0 introduced, but it means the assumption "`off` is safe
because the synthesis will close the stream" does not hold on this route.

Claude Code talking to a local llama.cpp declared as `kind: anthropic` *is*
this route. It is the single strongest reason to move from `warn` to
`error`: with `error` + `empty_response_action: fallback`, a truncation
before real content is swapped to the next provider, and an exhausted chain
**always terminates** with an `overloaded_error` `event: error` frame (see
"When the whole chain truncates" below).

## Configuration

```yaml
profiles:
  - name: local-first
    stream_truncation_action: error   # off | warn | error
    empty_response_action: fallback   # required for pre-content fallback
    partial_stitch_action: surface    # catches the already-forwarded case
```

| Value | Behavior |
|------|------|
| `off` (default) | No detection, no log, no metric. Byte-for-byte identical to v2.14.0 |
| `warn` | Emits a `stream-truncation-detected` log and metric only; the stream still ends through the legacy terminator synthesis |
| `error` | The adapter raises `StreamTruncatedError` (a retryable `AdapterError`) and the engine's existing branches take over |

**Recommended rollout**: run `warn` first to measure your backends' real
truncation rate and check for false positives, then move to `error`.

## What counts as a terminator

Deliberately generous, to keep the false-positive rate down.

| Wire | Accepted as a terminator |
|------|--------------------------|
| Anthropic | `message_stop`, a `message_delta` carrying a `stop_reason`, or a top-level `error` event |
| OpenAI | `data: [DONE]`, or a `finish_reason` on any choice |

Anthropic-compatible servers that omit `message_stop` and OpenAI-compatible
servers that omit `[DONE]` do exist — the M9 guard's comment at
`convert.py:1379` explicitly anticipates "a provider that omits the
terminator". Accepting `message_delta` / `finish_reason` keeps those from
being misread as truncation.

## Flow (`error`)

```
truncation detected (adapter raises StreamTruncatedError)
  ├─ no real content forwarded to the client yet
  │    → fall back to the next provider
  │      reason = "stream-truncated"
  │
  └─ already forwarded
       → MidStreamError
         → ingress consults partial_stitch_action
            ├─ off     : close with event: error (api_error)
            └─ surface : accumulated text + message_delta + message_stop +
                         coderouter_partial (reason: "stream_truncated")
```

**No new exit was created.** The truncation joins the existing
empty-response branch and the existing mid-stream branch, with the reason
swapped. As a side effect every self-healing guard —
`memory_pressure` (L2), `drift_detection` (L4), `backend_health` (L5),
`self_healing` (L6) — learns the truncation as a failure automatically. A
backend that keeps going quiet is demoted by adaptive routing and becomes a
self-healing restart candidate.

### When the whole chain truncates

`empty_response_action: fallback` withholds the opening events until real
content appears and, when the chain runs out, **flushes the last buffer it
held** so the SSE terminates normally. For a genuinely empty response that
buffer is a complete `message_start … message_stop` sequence, which is what
turns an all-empty chain into a 200 blank.

But **a truncated stream's buffer has no `message_stop`, by definition.**
Flushing it emits an SSE sequence that never ends: no exception, no error
frame, and a client that waits forever. So in v2.15.0 the flush is gated on
**whether the buffer being held actually terminates**.

| All providers exhausted… | Result |
|--------------------------|--------|
| every one genuinely empty (clean end) | 200 blank, as before |
| the buffer being held came from a truncation | `NoProvidersAvailableError` → the ingress emits `event: error` (`overloaded_error`) |
| a truncation, then a provider that ended cleanly empty | that buffer terminates → 200 blank |

An operator who set `error` asked not to have truncation swallowed;
synthesizing an `end_turn` over a half-written answer and presenting it as a
complete reply would be exactly that. The branch is only reachable when a
`StreamTruncatedError` exists, so the default `off` is identical to v2.14.0.
A truncation no longer inflates `empty-response-detected` /
`empty_responses_total` either — it was never an *empty* response.

## Interaction with `empty_response_action` (important)

On the Anthropic streaming path the opening `message_start` is **forwarded to
the client the moment it arrives**. By the time a truncation is detected,
bytes are therefore already out — which makes it mid-stream by definition.

`empty_response_action: fallback` is the knob that **withholds** the preamble
(`message_start` / an empty `content_block_start` / `ping`) until real
content appears. Only with it can a truncation before real content be swapped
to the next provider without the client noticing.

| Configuration | Truncation before real content | After real content |
|---------------|-------------------------------|--------------------|
| `stream_truncation_action: error` alone | `MidStreamError` | `MidStreamError` |
| `+ empty_response_action: fallback` | **falls back to the next provider** | `MidStreamError` |

## Cost note

Falling back on truncation throws away the tokens and time the cut attempt
already spent and regenerates the answer on the next provider. In a chain
with a paid cloud in the third tier that is real money, and it is counted
against the monthly budget in `budget.py`.

## Logs / metrics

- Log event: `stream-truncation-detected` (warning)
  - fields: `provider`, `action`, `wire` (`anthropic` | `openai`),
    `events_forwarded`, `saw_stream_start`, `tool_call_in_flight`
- MetricsCollector: `stream_truncated_total` / `stream_truncated_by_provider`
  / `stream_truncated_by_action`
- Prometheus: `coderouter_stream_truncated_total{provider="..."}` /
  `coderouter_stream_truncated_by_action_total{action="warn|error"}`
- Fallback reason: `stream-truncated` (surfaces in the
  `X-CodeRouter-Fallback-Reason` header and the `coderouter_fallback` SSE
  metadata event)

The `action` label exists so one dashboard can separate the `warn`-phase
measurement from the `error`-phase interventions.

## About truncated tool calls

`tool_call_in_flight: true` means a `tool_use` / `tool_calls` block was still
open when the stream was cut. `_close_current_block` in
`translation/convert.py` only emits `content_block_stop` — it neither repairs
nor validates the partial argument JSON. So in that case a structurally valid
`tool_use` block reaches the client carrying an `input` that cannot be parsed.

The flag is **observational, not a trigger**. The missing-terminator signal
already covers this case (a stream cut mid-arguments never sends a
terminator). A rule like "incomplete JSON is unconditionally a truncation"
would be a new mechanism that misfires whenever the arguments legitimately
completed, so it is not implemented.

## Limitations

- **`off` by default.** Doing nothing is the default; enabling is an
  explicit opt-in.
- **Non-streaming paths are out of scope** — `empty_response_action` and
  drift detection already cover completed responses.
- **No retry to the same provider.** CodeRouter is designed around fallback,
  not retry.
- **Residual false positives**: an upstream that sends neither
  `message_stop`, nor a `message_delta.stop_reason`, nor a `finish_reason`
  is flagged even when it is healthy. That is why measuring with `warn`
  first is recommended.
- **Native Anthropic passthrough has no terminator synthesis.** The
  `/v1/messages` × `kind: anthropic` route does not pass through
  `translation/convert.py`, so under `off` a truncated stream reaches the
  client unterminated (see "the synthesis does not cover every route"
  above). On this one route, doing nothing is not the safe option.
- **Multi-choice (`n>1`) detection is lenient.** On the OpenAI wire a
  `finish_reason` on *any* choice counts as a terminator, so a stream cut
  while only some of the choices had finished is not flagged. Claude Code,
  CodeRouter's primary client, does not use `n>1`.

## Use cases

| Scenario | Recommended setting |
|----------|---------------------|
| You just want to know the real rate | `warn` |
| Unstable local tier 1 with a cloud behind it | `error` + `empty_response_action: fallback` |
| A paid cloud is tier 1 | `warn` (avoid paying twice) |
| Long generations where partial output still helps | `error` + `partial_stitch_action: surface` |
