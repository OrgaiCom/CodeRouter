# Routing Sub-Agents to Different Models

日本語版: [`docs/guides/subagent-routing.md`](./subagent-routing.md)

A practical guide to routing Claude Code sub-agents (`.claude/agents/*.md`) to different backends — local LLM, cloud, or an external agent CLI (`agent_cli`) — through CodeRouter, one per sub-agent role.

Table of contents:

1. [Overview](#1-overview)
2. [How it works — the three channels and their precedence](#2-how-it-works--the-three-channels-and-their-precedence)
3. [Client-side setup (Claude Code)](#3-client-side-setup-claude-code)
4. [CodeRouter-side setup](#4-coderouter-side-setup)
5. [Real-world patterns](#5-real-world-patterns)
6. [Verifying it works](#6-verifying-it-works)
7. [Limitations and known gaps](#7-limitations-and-known-gaps)
8. [Related documents](#8-related-documents)

---

## 1. Overview

CodeRouter can assign a different model/backend to each Claude Code sub-agent. A reviewer role can go to a cheap local model, an architect role to a high-capability cloud model, an audit role to an external `claude` CLI — and so on.

Start with the honest core of the mechanism. **There is no dedicated identifier for "this is a sub-agent" on the wire (the HTTP request).** Claude Code simply resolves a model name per sub-agent and sends it in the same `model` field as any other request. From CodeRouter's side, that `model` name is effectively the only practical signal for distinguishing sub-agents. So the routing described in this guide is fundamentally a two-step combination: **(1) set a distinct `model` in each sub-agent's frontmatter, then (2) let CodeRouter route on that `model` name** — it is not a dedicated feature of CodeRouter by itself.

## 2. How it works — the three channels and their precedence

CodeRouter has three channels for assigning a request to a profile (a bundle of backends), evaluated in this order (same for both ingress paths — Anthropic Messages API and OpenAI Chat Completions):

```
body.profile  >  X-CodeRouter-Profile header  >  X-CodeRouter-Mode header  >  auto_router  >  default_profile
```

Source: `coderouter/routing/auto_router.py`, `coderouter/ingress/anthropic_routes.py`, `coderouter/ingress/openai_routes.py`.

| # | Channel | Mechanism | Where it fits sub-agent routing |
|---|---|---|---|
| ① **Model-name match** | auto_router's `model_pattern` matcher (`re.fullmatch` against body's `model`) | **The primary lever.** Give each sub-agent a distinct `model` in frontmatter and route on the resolved name. Works even with a stock Claude Code sub-agent launch, which cannot inject custom headers. |
| ② **Explicit header** | `X-CodeRouter-Profile` (names a profile directly) / `X-CodeRouter-Mode` (resolved via `mode_alias`) | The right tool when your own orchestrator drives each sub-call deterministically, e.g. attaching `X-CodeRouter-Profile: planner\|coder\|reviewer-audit`. It's an override, not a condition — you cannot write a rule that branches on "if this header equals X". |
| ③ **Content-based** | auto_router's remaining 6 matchers (CJK ratio, code-fence ratio, token count, etc.) | A fallback for when the client declares no role at all. Heuristics such as: long content → planner, CJK-heavy → local, code-dense → coder. |

All three channels ultimately resolve to a profile name, and a profile is bound to a bundle of backends (a fallback chain). Sub-agent routing is built as a three-stage pipeline: (model name or explicit header) → profile → backend.

## 3. Client-side setup (Claude Code)

### The `model` frontmatter field

A sub-agent definition lives at `.claude/agents/*.md` (project) or `~/.claude/agents/*.md` (user), as YAML frontmatter plus a Markdown body; identity comes only from the `name` frontmatter field. The `model` field accepts (as of the official docs, 2026-07):

- A model alias: `sonnet` / `opus` / `haiku` / `fable`
- A full model ID: e.g. `claude-opus-4-8` / `claude-sonnet-5` (the same values accepted by `--model`)
- `inherit`: use the same model as the main conversation
- Default when unset: `inherit`

Source: <https://code.claude.com/docs/en/sub-agents> (the frontmatter table and the "Choose a model" section).

Example:

```markdown
---
name: reviewer
description: Code review specialist. Use proactively after code changes.
model: haiku        # ← pin to a cheap model → CodeRouter routes it to local
---
(system prompt body)
```

```markdown
---
name: architect
description: Architecture design and planning specialist.
model: opus         # ← pin to a high-capability model → CodeRouter routes it to the planner profile
---
```

If every sub-agent stays on `inherit` (the same model as the main conversation), the model name can no longer distinguish them. This pattern only works if you give each sub-agent role a distinct `model` in its frontmatter.

### Sub-agent model resolution order

Inside Claude Code, the resolution order is (highest priority first):

1. Environment variable `CLAUDE_CODE_SUBAGENT_MODEL` (when set to an alias or model ID)
2. The per-invocation `model` parameter passed by the Agent/Task tool at launch time
3. The sub-agent definition's `model` frontmatter
4. The main conversation's model (`inherit`)

Source: <https://code.claude.com/docs/en/sub-agents> ("Choose a model" section, numbered list). `CLAUDE_CODE_SUBAGENT_MODEL=inherit` is treated as equivalent to unset, so resolution continues to per-invocation → frontmatter (since v2.1.196).

Frontmatter has no field for injecting an HTTP header (`tools` / `disallowedTools` / `permissionMode` / `effort` / `isolation` / `color`, etc. control behavior, not headers). In other words, channel ② (the explicit header) is not reachable from a stock Claude Code sub-agent launch — it only applies when your own orchestrator adds the header.

### Pointing Claude Code at CodeRouter with ANTHROPIC_BASE_URL

```bash
export ANTHROPIC_BASE_URL="http://localhost:8088"
export ANTHROPIC_AUTH_TOKEN="dummy"   # CodeRouter ignores auth; any non-empty value works
claude
```

`ANTHROPIC_BASE_URL` only changes *where* the request goes — it has no effect on *which model answers* (model selection is the job of the frontmatter/env settings above). With a custom `ANTHROPIC_BASE_URL`, Claude Code passes the model-name string through without allowlist validation, so CodeRouter can target arbitrary model-name strings with `model_pattern`. Source: <https://code.claude.com/docs/en/model-config>.

## 4. CodeRouter-side setup

### auto_router's `model_pattern` rules

auto_router only runs when `default_profile: auto`. Writing your own `rules` **completely replaces** the bundled defaults (image → multi / code-dense → coding / fallthrough → writing) — it does not merge with them.

```yaml
default_profile: auto
auto_router:
  default_rule_profile: coder
  rules:
    - id: user:opus-to-planner
      profile: planner
      match: { model_pattern: "(claude-)?opus.*" }     # fullmatch — routes opus-family models to planner
    - id: user:haiku-to-local
      profile: reviewer-light
      match: { model_pattern: "(claude-.*)?haiku.*" }   # fullmatch — routes haiku-family models to local reviewer
```

**Two things to watch for:**

- `model_pattern` uses **`re.fullmatch`**, not `re.search`. Your regex must match the *entire* string of whatever model name Claude Code actually sends (whether it stays `opus` or expands to `claude-opus-4-8` is environment-dependent). Forgetting the trailing `.*` is a common way for a rule to silently miss.
- The exact model-name string that arrives is environment- and version-dependent, and cannot be assumed. **Before relying on a rule in production, confirm the real value from the auto-router log** (the `signals.model` field of the `auto-router-resolved` event). See [§6](#6-verifying-it-works) for the procedure.

### The full list of auto_router matchers (8 total)

A `RuleMatcher` allows **exactly one matcher per rule** — a load-time validator enforces this, so AND-composing multiple conditions is not possible (see gap G1 in [§7](#7-limitations-and-known-gaps)). Rules are evaluated top-to-bottom, first match wins.

| Field | Type | Meaning | Evaluated against |
|---|---|---|---|
| `has_image` | `bool` (only `true` is valid) | Matches if the latest user message has an image block | Latest user message |
| `code_fence_ratio_min` | `float` (0.0–1.0) | Matches if the fraction of characters inside ` ``` ` fences is ≥ threshold | Latest user message |
| `cjk_ratio_min` | `float` (0.0–1.0) | Matches if the CJK character ratio is ≥ threshold | Latest user message |
| `content_contains` | `str` | Case-sensitive substring match | Latest user message |
| `content_regex` | `str` | `re.search` (compiled and validated at load time) | Latest user message |
| `model_pattern` | `str` | `re.fullmatch` against the body's `model` field (compiled and validated at load time) | Body's `model` |
| `content_token_count_min` | `int` (≥1) | Matches if the estimated token count (char/4 heuristic) over system + all messages is ≥ threshold | Entire request |
| `has_tools` | `bool` (only `true` is valid) | Matches if the body declares 1+ entries in `tools[]` (OpenAI/Anthropic common) or legacy OpenAI `functions[]` | Entire body |

`model_pattern`, `content_token_count_min`, and `has_tools` can fire even without a user message (e.g. a system-only request, or a body carrying just `model` + `tools`). Every other matcher requires a user message to be present. Boolean matchers (`has_image` / `has_tools`) reject an explicit `false` at load time (dead-rule prevention — omit the field instead if unused).

### The X-CodeRouter-Profile header

Use this when there's no `profile` field in the body and your own orchestrator wants to drive each sub-call deterministically via header.

```bash
curl http://localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-CodeRouter-Profile: reviewer-audit' \
  -d '{"model":"opus","messages":[{"role":"user","content":"Say hi in one line"}]}'
```

Resolution order: body.profile → `X-CodeRouter-Profile` → `X-CodeRouter-Mode` (resolved to a profile via `mode_aliases`) → auto_router → `default_profile`. Naming a profile that doesn't exist returns 400.

## 5. Real-world patterns

### (a) opusplan style — Opus for planning, local/mid-tier for execution

Planning goes to Claude Opus (via `agent_cli`, read-only); execution goes to a mid-tier local model with a cloud fallback; review splits audit work (Opus) from light review (local).

```yaml
allow_paid: true            # needed for agent_cli (claude = Opus subscription)
default_profile: coder      # falls back to the day-to-day coding role when no header/rule matches

providers:
  - name: agent-claude-opus     # Planning role: Claude Opus (agent_cli, read-only)
    kind: agent_cli
    model: opus
    paid: true
    capabilities: { streaming: false, tools: false }   # agent_cli default
    agent_cli: { agent: claude, sandbox_mode: read_only, exec_timeout_s: 600 }
  - name: local-coder            # Execution role: mid-tier local (zero tax)
    kind: anthropic               # Ollama v0.23.1+ passthrough
    base_url: http://localhost:11434
    model: qwen3-coder:30b
  - name: cloud-mid               # Execution role fallback: cloud mid-tier
    kind: openai_compat
    base_url: https://openrouter.ai/api/v1
    model: qwen/qwen3-coder:free
    paid: true
  - name: agent-claude-review    # Review role: audits go to Opus
    kind: agent_cli
    model: opus
    paid: true
    capabilities: { streaming: false, tools: false }
    agent_cli: { agent: claude, sandbox_mode: read_only }
  - name: local-reviewer         # Review role: light review to a separate local model
    kind: anthropic
    base_url: http://localhost:11434
    model: qwen2.5-coder:7b

profiles:
  - name: planner
    providers: [agent-claude-opus]        # agent_cli is meant to be solo
  - name: coder
    providers: [local-coder, cloud-mid]   # local first, cloud on failure
  - name: reviewer-audit
    providers: [agent-claude-review]      # security audits go to Opus
  - name: reviewer-light
    providers: [local-reviewer]           # light review runs locally

auto_router:   # fallback for when the client declares no profile
  default_rule_profile: coder
  rules:
    - id: user:image-to-multi
      profile: planner
      match: { has_image: true }
    - id: user:dense-code-to-coder
      profile: coder
      match: { code_fence_ratio_min: 0.3 }
    - id: user:long-context-to-planner
      profile: planner
      match: { content_token_count_min: 32000 }
    - id: user:cjk-to-local
      profile: coder
      match: { cjk_ratio_min: 0.5 }
    - id: user:review-keyword
      profile: reviewer-audit
      match: { content_contains: "review" }

plugins:
  enabled: [agents]   # required since v2.9.0 for any kind: agent_cli provider
```

**A note on how this is actually driven**: the auto_router above is just the fallback for when the client declares no role. The intended way to drive opusplan is for an upper-layer orchestrator to attach `X-CodeRouter-Profile: planner|coder|reviewer-audit` explicitly to each sub-call (precedence as in [§2](#2-how-it-works--the-three-channels-and-their-precedence)). Note this is a different thing from **Claude Code's native `opusplan` alias** (which auto-switches from `opus` during plan mode to `sonnet` for execution) — don't conflate the two.

### (b) Minimal per-sub-agent model configuration

The same idea as the frontmatter example in [§3](#3-client-side-setup-claude-code): pin `reviewer` to `model: haiku` and `architect` to `model: opus`, then route on `model_pattern` in CodeRouter.

```yaml
default_profile: auto

providers:
  - name: local-reviewer
    kind: openai_compat
    base_url: http://localhost:11434/v1
    model: qwen2.5-coder:7b
  - name: cloud-planner
    kind: openai_compat
    base_url: https://openrouter.ai/api/v1
    model: anthropic/claude-opus-4-8
    paid: true

profiles:
  - name: reviewer-light
    providers: [local-reviewer]
  - name: planner
    providers: [cloud-planner]

auto_router:
  default_rule_profile: reviewer-light
  rules:
    - id: user:opus-to-planner
      profile: planner
      match: { model_pattern: "(claude-)?opus.*" }
    - id: user:haiku-to-local
      profile: reviewer-light
      match: { model_pattern: "(claude-.*)?haiku.*" }
```

### (c) Mixing in agent_cli — an audit role via an external claude CLI

`kind: agent_cli` registers an external CLI — claude / codex / grok / antigravity — as a single CodeRouter provider. **As of v2.9.0, this requires installing `coderouter-plugin-agents` and adding `plugins.enabled: [agents]` to `providers.yaml`** (without it, `coderouter serve` fails at startup as soon as any `kind: agent_cli` provider is present). See [`docs/backends/external-agents.md`](../backends/external-agents.en.md) for full details.

```bash
uv pip install "coderouter-plugin-agents @ git+https://github.com/zephel01/coderouter-plugin-agents"
```

```yaml
allow_paid: true
default_profile: reviewer-light

plugins:
  enabled: [agents]        # required since v2.9.0 for any kind: agent_cli provider

providers:
  - name: agent-claude-review
    kind: agent_cli
    model: opus
    paid: true
    capabilities: { streaming: false, tools: false }
    agent_cli: { agent: claude, sandbox_mode: read_only }
  - name: local-reviewer
    kind: openai_compat
    base_url: http://localhost:11434/v1
    model: qwen2.5-coder:7b

profiles:
  - name: reviewer-audit      # audit role = external claude CLI (Opus subscription)
    providers: [agent-claude-review]     # agent_cli is solo
  - name: reviewer-light      # light review = local
    providers: [local-reviewer]
```

`agent_cli` defaults to `capabilities: { streaming: false, tools: false }`, so it cannot sit mid-chain — it's limited to being the sole provider of a dedicated profile, or the terminus of a chain (gap G6 in [§7](#7-limitations-and-known-gaps)).

### (d) Content-based supplementary rules

You can also add a standalone content-based matcher as a fallback for when the client declares no role.

```yaml
auto_router:
  default_rule_profile: coder
  rules:
    - id: user:cjk-to-local
      profile: coder
      match: { cjk_ratio_min: 0.5 }        # send CJK-heavy turns to local (zero tax)
    - id: user:long-context-to-planner
      profile: planner
      match: { content_token_count_min: 32000 }   # send long-context turns to planner
```

## 6. Verifying it works

1. **At startup**: check the `coderouter serve` startup log for `plugin-loaded` (if using agent_cli) and for the absence of config-load errors.
2. **Send one real request**: from Claude Code, or a curl request shaped like a sub-agent call.
3. **Read the auto-router log**: a matched rule is recorded as an `auto-router-resolved` event, and `signals.model` carries the actual `model` string that arrived — this is your primary source for confirming whether the alias stayed as-is or expanded to a full ID. The shape of the log line is roughly:

   ```json
   {"ts":"2026-07-11T10:03:21","level":"INFO","logger":"coderouter.routing.auto_router",
    "msg":"auto-router-resolved","rule_id":"user:opus-to-planner","resolved_profile":"planner",
    "signals":{"has_image":false,"code_fence_ratio":0.0,"content_len":42,
               "model":"opus","estimated_tokens":15,"has_tools":true}}
   ```

4. **Confirm the intended profile**: check that `resolved_profile` is the profile name you intended (`planner`, `reviewer-audit`, etc.). If it isn't, suspect a `fullmatch` miss in `model_pattern` (a missing trailing `.*` is the usual culprit).

## 7. Limitations and known gaps

- **One matcher per rule (no AND)**: a `RuleMatcher` cannot express compound conditions. "CJK-heavy AND long" isn't expressible as a single rule today (gap G1). Work around it by ordering multiple single-condition rules and letting first-match-wins do the job.
- **No dedicated sub-agent-declaration channel (under consideration)**: there is currently no channel that carries "this is a sub-agent, and its role is X." In practice, the answer is what this guide describes — route on model name. A future proposal exists (not yet started) to add a top-priority rule ahead of auto_router that detects a tag embedded in the system prompt or first message, but this is not implemented.
- **How this differs from ccr's (claude-code-router) tag approach**: CodeRouter actively reads wire-level metadata (model name, headers) to route. ccr instead passively detects and extracts a tag embedded in the prompt body. Both are different answers to the same underlying constraint — there is no native sub-agent identifier on the Anthropic wire.
- **UNCONFIRMED — the shape of the model name that arrives**: whether the model-name string a Claude Code sub-agent sends stays an alias (e.g. `opus`) or expands to a full ID (e.g. `claude-opus-4-8`) is environment- and version-dependent, and was not verified as of this guide's writing. Whenever you write a `model_pattern` rule, confirm the real value via `signals.model` in the log as described in [§6](#6-verifying-it-works).

## 8. Related documents

- [`docs/backends/external-agents.md`](../backends/external-agents.en.md) — full configuration reference, authentication, and troubleshooting for `agent_cli` (claude/codex/grok/antigravity)
- [`docs/guides/usage-guide.md`](./usage-guide.en.md) — general CodeRouter usage guide
- Claude Code official docs: <https://code.claude.com/docs/en/sub-agents>, <https://code.claude.com/docs/en/model-config>
