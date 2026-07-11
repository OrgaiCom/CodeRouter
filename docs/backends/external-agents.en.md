# External Coding Agent CLI (agent_cli)

> 日本語版: [`external-agents.md`](./external-agents.md)

`kind: "agent_cli"` is an adapter that registers an external coding-agent CLI, such as the Claude Code CLI, as a single CodeRouter provider. It was newly added in v2.7.7 (Phase 1a: claude), v2.7.8 added grok (Phase 1d), and v2.7.9 added codex (Phase 1b). See [`docs/designs/external-agents-adapter.md`](../designs/external-agents-adapter.md) for the full design.

---

## Overview

A coding-agent CLI is normally a stateful control loop that runs many turns autonomously while editing files — a poor fit for CodeRouter's "one request = one transformation" ethos. `agent_cli` reconciles the two by collapsing the CLI into a **one-shot `exec`** (prompt in → final answer text out). Orchestration (multi-turn control, tool execution) stays entirely inside the agent CLI process; from CodeRouter's side it looks like just another provider that answers a single exchange.

- **Supported CLIs**: the `agent` field can declare `claude` / `codex` / `gemini` / `grok`.
- **Implementation status (as of v2.7.9)**: **`claude` (Claude Code CLI, Phase 1a), `codex` (OpenAI Codex CLI, Phase 1b), and `grok` (Grok CLI, Phase 1d) are implemented**. Only `gemini` can be written into `providers.yaml` at the schema level, but constructing the adapter always rejects it (Phase 1c is not yet implemented). The error message looks roughly like the following (the exact wording may differ between versions):

  ```
  AdapterError: agent 'gemini' is not implemented yet (implemented: claude, codex, grok).
  Wait for the agent's phase (1c).
  ```

  This rejection is `retryable=False` — even if other providers exist in the fallback chain, it stops immediately as a configuration error.

- The shared parts of this document (authentication design, configuration reference, limitations) are written against the `claude` target; codex- and grok-specific behavior are collected in the [codex (OpenAI Codex CLI)](#codex-openai-codex-cli) and [grok (Grok CLI)](#grok-grok-cli) sections, respectively. Example configs for `gemini` exist only as a commented-out preview at the bottom of `examples/providers-agent-cli.yaml` and do not work.

---

## Quickstart

1. **Install the Claude Code CLI** (verify with `claude --version`).
2. **Log in** — either run `claude` interactively and complete `/login`, or, on a headless machine, follow the [platform-specific authentication](#platform-specific-authentication) steps using `claude setup-token`.
3. **Start with the example config**:

   ```bash
   uv run coderouter serve --config examples/providers-agent-cli.yaml --port 8088
   ```

4. **Verify it works**:

   ```bash
   curl http://localhost:8088/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -H 'X-CodeRouter-Profile: claude-agent' \
     -d '{"model":"opus","messages":[{"role":"user","content":"1行でこんにちはと言って"}]}'
   ```

### The first call is slow

`agent_cli` launches a fresh CLI process on every call (nothing is kept resident). The first call pays for process startup plus one round trip with the real Claude backend, so it is noticeably slower than a typical HTTP-backed provider.

You may also notice `usage.prompt_tokens` in the tens of thousands. This is because Claude Code's own system prompt (hooks/CLAUDE.md discovery, the full tool-definition set) rides along on every call regardless of how much text CodeRouter actually sent. When running on subscription OAuth, this does **not** cost anything in dollars (it does consume the 5-hour window / weekly quota — see [Limitations](#limitations)).

From the second call onward, Anthropic's prompt cache kicks in and `usage.prompt_tokens_details.cached_tokens` grows. In one measured run against the same `workdir`, the first call's `coderouter_cost_usd` (a dollar-equivalent figure based on API pricing, not an actual charge — see [Configuration reference](#configuration-reference)) was **about $0.22-equivalent**, and once the cache took effect on subsequent calls it dropped to **about $0.05-equivalent**.

---

## Platform-specific authentication

`AgentCliAdapter` does **not** let the child process inherit the parent process's environment as-is. It explicitly injects a fixed, safe `PATH` / `NO_COLOR=1` / `TERM=dumb`, plus `HOME` / `USER` / `LOGNAME` (only when set), plus whatever names are listed in `passthrough_env`. As a result, `ANTHROPIC_API_KEY` is not forwarded by default, and subscription authentication (OAuth) takes priority.

| Platform | Where credentials live | Env vars that must be inherited | v2.7.7 status |
|---|---|---|---|
| **macOS** | Keychain | `USER` (required for Keychain entry resolution) | Fixed in v2.7.7, which now inherits `USER` / `LOGNAME`. Works as-is once `claude /login` has been done (field-verified) |
| **Linux** | `~/.claude/.credentials.json` (mode `0600`) | `HOME` | Works with `HOME` inheritance alone. Works as-is once `claude /login` has been done (field-verified) |
| **Headless server / container** (no browser) | Either of the above, or a long-lived token | `CLAUDE_CODE_OAUTH_TOKEN` (via explicit `passthrough_env`) | See the procedure below |
| **Windows** | Not natively supported | — | Run CodeRouter itself inside WSL2 (this is then treated the same as Linux) |

### macOS

The Claude Code CLI reads the `USER` environment variable when resolving its Keychain entry. Before v2.7.7, the env allowlist only inherited `HOME`, so `USER` was missing and headless/server runs on macOS failed Keychain resolution with `Not logged in`. v2.7.7 fixed `_build_child_env()` to also inherit `USER` / `LOGNAME`, resolving this. As long as `claude` has already been logged in interactively via `/login`, it works with no extra configuration — verified on real hardware.

### Linux

Credentials live in `~/.claude/.credentials.json` (mode `0600`). The child process only needs `HOME` inherited to read this file. As on macOS, once `claude /login` has been completed it works as-is — verified on real hardware.

### Headless server / container (no browser)

For environments where an interactive browser login isn't possible, there's a path via a long-lived token forwarded through an environment variable.

1. On a **machine with a browser**, run `claude setup-token` to issue an OAuth token valid for one year.
2. Put the issued token in the target server's `.env` as `CLAUDE_CODE_OAUTH_TOKEN=...`. Make sure the file is mode `0600` and excluded via `.gitignore` (both are checked by the `env_security` check in `coderouter doctor --check-env`).
3. In `providers.yaml`, set `agent_cli.passthrough_env: [CLAUDE_CODE_OAUTH_TOKEN]` on that provider to explicitly forward it into the child process.

### Windows

`AgentCliAdapter` is implemented assuming POSIX (`os.killpg`-based process-group kill, a fixed `PATH` in `/usr/local/bin`-style form, etc.), so it does not run natively on Windows. Running the whole CodeRouter stack inside WSL2 and calling `claude` from within WSL2 effectively makes this the same as the Linux case.

### Important note — API keys are not forwarded automatically

Because the child process does not inherit the parent environment, exporting `ANTHROPIC_API_KEY` in your shell will **not** reach the claude CLI. This is intentional: it prioritizes subscription authentication and prevents a stray API key left in the environment from silently overriding subscription auth. Only if you want to run on API-key metered billing should you explicitly list it, e.g. `passthrough_env: [ANTHROPIC_API_KEY]`.

---

## Configuration reference

All fields of the `agent_cli:` sub-config (`AgentCliConfig`) in `providers.yaml`. `extra: forbid` applies, so an unknown key fails immediately at config load.

| Field | Type | Default | Description |
|---|---|---|---|
| `agent` | `"claude" \| "codex" \| "gemini" \| "grok"` | (required) | Which CLI to invoke. **As of v2.7.9, `claude`, `codex`, and `grok` are implemented; `gemini` is rejected when the adapter is constructed** |
| `command` | `str \| null` | `null` (defaults to the same name as `agent`) | CLI executable name or absolute path, resolved via `PATH` |
| `workdir` | `str \| null` | `null` (defaults to `~/.coderouter/agents/<provider name>`) | Working directory for the one-shot exec. `~` / env-var expansion is applied; a path containing `..` is rejected |
| `exec_timeout_s` | `float` | `600.0` (range `1.0`–`1800.0`) | Forced timeout (seconds) for the whole exec. **Separate** from `ProviderConfig.timeout_s` (the latter is not used by agent_cli) |
| `allow_file_writes` | `bool` | `false` | Whether to allow file writes. When `false`, the effective mode is clamped to read-only regardless of `sandbox_mode` |
| `sandbox_mode` | `"read_only" \| "edit" \| "full_auto"` | `"read_only"` | Maps to each CLI's sandbox/approval flags (claude: [table below](#sandbox_mode--permission-mode-mapping-claude); codex: [codex section](#sandbox_mode--codex-flag-mapping); grok: [grok section](#sandbox_mode--grok-flag-mapping)) |
| `model` | `str \| null` | `null` (defaults to `ProviderConfig.model`) | Model name passed to the CLI's `--model` / `-m` (claude: `opus` / `sonnet` / `haiku` / `fable` etc.; codex: `gpt-5.5` etc.; grok: `grok-4.5` etc.) |
| `max_turns` | `int \| null` | `8` (range `1`–`50`) | Turn cap inside the CLI. Passed as `--max-turns`. **codex has no corresponding CLI flag, so this is always ignored** (for codex, `exec_timeout_s` + process-group kill is the only time bound) |
| `passthrough_env` | `list[str]` | `[]` | Allowlist of environment variable names forwarded from the parent process into the child. `ANTHROPIC_API_KEY` is not forwarded unless listed here |
| `agent_depth_limit` | `int` | `2` (range `1`–`4`) | Recursion nesting cap. When `CODEROUTER_AGENT_DEPTH` reaches or exceeds this, the call stops immediately with `AdapterError(retryable=False)` |

When `command` is unset it defaults to the same name as `agent`. Also, specifying `allow_file_writes: true` together with `sandbox_mode: read_only` is treated as a contradictory configuration and raises a **`ValueError` at config-load time** (set `sandbox_mode` to `edit` or `full_auto` if you want to permit writes).

### `sandbox_mode` → `--permission-mode` mapping (claude)

| `sandbox_mode` | claude `--permission-mode` | Notes |
|---|---|---|
| `read_only` (default) | `plan` | No file changes. Always clamped to this mode when `allow_file_writes=false` |
| `edit` | `acceptEdits` | Auto-approves file edits |
| `full_auto` | `acceptEdits` | For claude this maps the same as `edit` (claude has no separate full_auto-equivalent mode in use yet). grok distinguishes it via `--always-approve` (see the [grok section](#sandbox_mode--grok-flag-mapping)) |

### Why `paid: false`

The `agent-claude` provider in the example config `examples/providers-agent-cli.yaml` is set to `paid: false`. That's because running on subscription OAuth incurs **zero metered cost** (only the 5-hour window / weekly quota described below is consumed). If you want to run it on metered API-key billing instead, change it to `paid: true` and pass `ALLOW_PAID=true` as an environment variable when starting CodeRouter. Note that the `ALLOW_PAID` environment variable **overrides** whatever `allow_paid` value is written in `providers.yaml` at startup — a `paid: true` provider is excluded from routing whenever `ALLOW_PAID` is unset.

---

## codex (OpenAI Codex CLI)

v2.7.9 (Phase 1b) implements `agent: codex`. Like claude, it delivers the prompt via **stdin** (unlike grok's file-based delivery). It has its own behavior around JSONL output, `--ephemeral`, and running outside a git repository. Everything below is based on codex CLI **0.144.1** (field-verified on the author's Mac, 2026-07-11).

### Example configuration

```yaml
providers:
  - name: agent-codex
    kind: agent_cli
    model: gpt-5.5                # a current frontier-model example; the default depends on the environment/plan, so set it explicitly
    paid: false                   # ChatGPT-plan subscription OAuth = zero metered cost
    capabilities:
      streaming: false
      tools: false
    agent_cli:
      agent: codex
      command: codex
      workdir: ~/.coderouter/agents/codex
      exec_timeout_s: 600
      allow_file_writes: false
      sandbox_mode: read_only
      max_turns: 8                 # ignored by codex — no corresponding flag (see below)
      passthrough_env: []          # OAuth reads ~/.codex/auth.json via the inherited HOME, so empty is fine.
                                    # Only list CODEX_API_KEY (exec-only) or OPENAI_API_KEY
                                    # when using an API key in CI
```

### The argv the adapter builds

With `sandbox_mode: read_only` (the default), the adapter builds the following argv.

```
codex exec --json --skip-git-repo-check --ephemeral -m <model> -C <workdir> -s read-only -
```

### The prompt is delivered via stdin (same as claude, unlike grok)

codex's `exec` subcommand reads the prompt from stdin when the PROMPT argument is omitted (or explicitly given as `-`). The adapter places an explicit trailing `-` on the argv to force this path. Unlike grok, there's no need for a temp file (`--prompt-file`) inside the isolated workdir — this is the same stdin scheme as claude.

### Why `--skip-git-repo-check` is always passed

CodeRouter's isolated workdir is not a git repository. By default, codex runs a "trusted directory" check and, outside a git repo in an unrecognized directory, fails immediately with exit 1 and stderr `Not inside a trusted directory and --skip-git-repo-check was not specified.` (field-verified). The adapter always passes this flag, so this error message should never appear in normal operation.

### Why `--ephemeral` is always passed

`--ephemeral` prevents the session from being persisted to disk. For the same reason as grok's `--no-memory` — keeping with CodeRouter's "one request = one stateless transformation" ethos — the adapter always passes this flag (it cannot be turned off in config).

### `sandbox_mode` → codex flag mapping

As with claude/grok, when `allow_file_writes=false` the effective mode is clamped to `read_only` regardless of `sandbox_mode`.

| `sandbox_mode` | codex flags | Notes |
|---|---|---|
| `read_only` (default) | `-s read-only` | No file changes. Always clamped to this mode when `allow_file_writes=false` |
| `edit` | `-s workspace-write` | Permits file edits inside the workspace-write sandbox |
| `full_auto` | `-s workspace-write` | **`codex exec` has no approval flag (`-a` / `--ask-for-approval`)** — it's absent from `exec --help` in 0.144.1, since non-interactive execution has no approval prompt to control in the first place. So this maps identically to `edit`. `--dangerously-bypass-approvals-and-sandbox` is never used |

### No `--max-turns` / `--timeout` exist

codex exec has neither `--max-turns` nor `--timeout`. Consequently `AgentCliConfig.max_turns` is **ignored for codex**. The only time bound is the existing `exec_timeout_s` + process-group `SIGKILL`.

### JSONL output and usage normalization

`--json` output is JSONL (one event per line); progress goes to stderr, and the event stream (including the final answer) goes to stdout (confirmed by both the official docs and the real CLI). A verified one-shot run:

```
$ codex exec --json --skip-git-repo-check "What's 1+1? Answer with just the digit"
{"type":"thread.started","thread_id":"019f4e74-08fd-77b2-9cc6-9afa744df130"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"2"}}
{"type":"turn.completed","usage":{"input_tokens":13810,"cached_input_tokens":9984,"output_tokens":5,"reasoning_output_tokens":0}}
```

- Final answer: the **last** `item.completed` whose `item.type=="agent_message"`, taking `item.text`.
- Usage normalization: `turn.completed.usage`'s `input_tokens` → `prompt_tokens`, `output_tokens` → `completion_tokens`, and their sum → `total_tokens`. `cached_input_tokens` is a **subset** of `input_tokens` (measured: input 13810 ⊃ cached 9984), so it is not added on top — it's kept as `prompt_tokens_details.cached_tokens`. When `reasoning_output_tokens` is greater than 0, it's kept as `completion_tokens_details.reasoning_tokens` (defensive). Multiple `turn.completed` events, if they occur, are summed.
- `thread_id` (from `thread.started`) is surfaced as the `coderouter_session_id` response metadata.
- If an `error` event or `turn.failed` is seen, or no agent_message is ever produced / stdout is empty / every line is non-JSON, this raises a retryable `AdapterError` and the fallback chain advances to the next provider. Individual non-JSON lines in the JSONL stream don't stop processing of the remaining lines (defense against stray stderr-like output mixed in).

### Authentication (ChatGPT-plan subscription OAuth / API key)

The codex CLI supports ChatGPT-plan OAuth login. Credentials are stored at `~/.codex/auth.json` (or the OS keyring), and the adapter's `HOME` inheritance makes it work with `passthrough_env: []`.

1. Run `codex login` to complete login. `codex login status` exiting 0 confirms you're logged in.
2. The OAuth token goes **stale after about 8 days**. It auto-refreshes on use, but a setup that doesn't call codex for a long stretch can fail while stale — running codex occasionally, or re-logging in, is recommended.

For CI or metered API-key billing, list `CODEX_API_KEY` (**exec-only**) or `OPENAI_API_KEY` (general) in `passthrough_env`. `CODEX_HOME` can also override the config/credentials directory itself.

### Error reporting

The codex CLI exits 0 on success, and on failure exits non-zero with the error text on stderr (e.g. the git-repo-check message). JSONL may also carry an `error` event or `turn.failed`; both of those also become a retryable `AdapterError` and the fallback chain advances to the next provider.

### Pre-1.0 caveat

The codex CLI is pre-1.0 and releases nearly daily. `--json`'s alias is still `--experimental-json`, and the JSONL schema is not frozen. **Version pinning is recommended** (you can point `command` at a pinned binary's full path). If the schema does change, defensive parsing turns it into a retryable `AdapterError` and the fallback chain demotes to the next provider.

---

## grok (Grok CLI)

v2.7.8 (Phase 1d) implements `agent: grok`. It uses the same one-shot exec scheme as claude, but grok has its own behavior around prompt delivery, cross-session memory disabling, and usage reporting. Everything below is based on grok CLI **v0.2.93** ([stable] channel, field-verified on 2026-07-10).

### Example configuration

```yaml
providers:
  - name: agent-grok
    kind: agent_cli
    model: grok-4.5              # the default model on a current install; list with `grok models`
    paid: false                  # subscription OAuth = zero metered cost
    capabilities:
      streaming: false
      tools: false
    agent_cli:
      agent: grok
      command: grok
      workdir: ~/.coderouter/agents/grok
      exec_timeout_s: 600
      allow_file_writes: false
      sandbox_mode: read_only
      max_turns: 8
      passthrough_env: []        # OAuth reads ~/.grok/auth.json via the inherited HOME, so empty is fine.
                                 # Only list GROK_CODE_XAI_API_KEY when using an API key in CI
```

### The argv the adapter builds

With `sandbox_mode: read_only` (the default), the adapter builds the following argv.

```
grok --prompt-file <workdir>/.coderouter-prompt-<uuid>.txt \
     --output-format json -m <model> --cwd <workdir> \
     --max-turns <N> --no-memory \
     --sandbox read-only --permission-mode plan
```

### The prompt is delivered via a file (`--prompt-file`)

grok's `-p` / `--single` accepts the prompt **only as an argv value** (verified on the real CLI: stdin is not accepted as the prompt). Putting a huge prompt on argv runs into Linux's `MAX_ARG_STRLEN` limit (~128KiB) and exposes the full prompt text to `ps`. The adapter therefore writes the prompt to a mode-`0600` temp file inside the isolated workdir (`.coderouter-prompt-<uuid>.txt`) and passes it via `--prompt-file`. That temp file is **always deleted** after the exec finishes, including the timeout and error paths.

### `sandbox_mode` → grok flag mapping

As with claude, when `allow_file_writes=false` the effective mode is clamped to `read_only` regardless of `sandbox_mode`.

| `sandbox_mode` | grok flags | Notes |
|---|---|---|
| `read_only` (default) | `--sandbox read-only --permission-mode plan` | No file changes. Always clamped to this mode when `allow_file_writes=false` |
| `edit` | `--sandbox workspace --permission-mode acceptEdits` | Auto-approves file edits inside the workspace sandbox |
| `full_auto` | `--sandbox workspace --always-approve` | Unlike claude, grok maps this distinctly from `edit` |

### `--no-memory` is always passed

The grok CLI has a cross-session memory feature. Letting a previous call's memory leak into the next response conflicts with CodeRouter's "one request = one stateless transformation" ethos, so the adapter **always** passes `--no-memory` to disable it (this cannot be turned off in config).

### JSON output and usage / cost

The `--output-format json` output is a single JSON object `{"text", "stopReason", "sessionId", "requestId", "thought"?}` (verified on grok v0.2.93). `text` becomes the final answer, and `sessionId` is surfaced as the `coderouter_session_id` response metadata. **There are no token-usage or cost fields**, so usage is reported as all zeros, and `coderouter_cost_usd` stays 0 unless the operator sets unit prices in `ProviderConfig.cost` (in contrast to claude, which emits `total_cost_usd` directly). The JSON is parsed defensively: anything malformed becomes an `AdapterError(retryable=True)` and the fallback chain advances to the next provider.

### Authentication (subscription OAuth / API key)

The grok CLI supports OAuth subscription login (SuperGrok / X Premium+). Credentials are stored at `~/.grok/auth.json` (7-day expiry with auto-refresh; `GROK_HOME` overrides the location), and the adapter's `HOME` inheritance makes OAuth work with `passthrough_env: []`. Setup steps:

1. Run `grok login` to complete the subscription login.
2. Smoke-check by running `grok models` and confirming the model list comes back. On a current install it lists `grok-4.5` (default) and `grok-composer-2.5-fast`.

Only when running on API-key metered billing (e.g. CI) should you list `passthrough_env: [GROK_CODE_XAI_API_KEY]`. Note that the environment variable name is **`GROK_CODE_XAI_API_KEY`, not `XAI_API_KEY`**. When the API key is forwarded, it takes precedence over OAuth.

### Error reporting

The grok CLI exits 0 on success, and exits 1 on auth/network/runtime errors with the error text on **stderr**. The adapter includes a tail of stderr in the `AdapterError` message, so you can use the message shown as your lead.

### Early-beta caveat

The grok CLI is early beta (v0.2.93 [stable] channel as of 2026-07-10). Its JSON schema may still churn, so **version pinning is recommended** (you can point `command` at a pinned binary's full path). If the schema does change, defensive parsing turns it into a retryable `AdapterError` and the fallback chain demotes to the next provider.

---

## Limitations

| Limitation | Details |
|---|---|
| **One-shot only** | No session continuation (resume). Every call launches a fresh CLI process; nothing from a previous call carries over (this is an intentional non-goal, not a bug) |
| **Pseudo-streaming** | No implemented CLI exposes a stable token-level stream, so `stream()` just splits the final text from `generate()` into fixed-size chunks and yields them in order. The example config explicitly sets `capabilities.streaming: false` (the default `true` must be overridden) |
| **May carry plan-mode framing** | The default `sandbox_mode: read_only` maps to `--permission-mode plan`. Plan mode is designed as a response format for an interactive human-review UI, so in one-shot execution the returned text can lean toward "here's my plan" phrasing rather than a direct answer, since no actual change is made |
| **Consumes subscription quota** | Uses up the Claude Code subscription's 5-hour window / weekly quota. Zero API billing doesn't mean unlimited calls |
| **Recursion cap** | Nested calls beyond `agent_depth_limit` (default 2, max 4) are rejected. Be careful if you build a setup where the agent CLI calls back into CodeRouter internally |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Fails with `Not logged in · Please run /login` | The claude CLI isn't logged in for the user/environment it's running as | First run `claude` interactively and check the `/login` state. On macOS headless runs, versions before v2.7.7 failed to forward the `USER` environment variable to the child process, causing this error; v2.7.7 fixes it by also inheriting `USER` / `LOGNAME` |
| Requests are rejected / not routed, effectively "paid gate blocked" | The `agent_cli` provider has `paid: true` but `ALLOW_PAID` isn't set | For subscription usage, set `paid: false`. For metered API-key billing, set `ALLOW_PAID=true` when starting CodeRouter |
| `claude exited 1: ...` shows a specific reason | The claude CLI sometimes reports auth/API errors as an `is_error: true` JSON document on **stdout** (with stderr left empty) and exits with code 1 | v2.7.7's `_error_detail()` now prefers the `result` field of that stdout `is_error` JSON (the actual error text, e.g. `Not logged in · Please run /login`) even when stderr is empty, and includes it in the raised error message — use the message shown as your lead |
| Fails with `grok exited 1: ...` | The grok CLI exits with code 1 on auth/network/runtime errors, with the error text on stderr | The adapter includes a tail of stderr in the `AdapterError`, so use the message shown as your lead. For auth errors, re-run `grok login` and smoke-check that `grok models` works |
| Fails with `codex exited 1: ...` (suspect stale OAuth) | codex's OAuth token goes stale after about 8 days. It auto-refreshes on use, but a long gap between calls can leave it stale and failing | Check login state with `codex login status` and re-run `codex login` if needed. Running codex occasionally helps avoid staleness |
| `Not inside a trusted directory and --skip-git-repo-check was not specified.` appears | This should never happen — the adapter always passes `--skip-git-repo-check` | If you see this, it likely indicates a bug in CodeRouter's argv construction. Check your version and file an issue if it reproduces |
| CLI fails to launch (`failed to launch ...`) | `command` (defaults to the same name as `agent`) isn't on `PATH` | Confirm `claude --version` / `codex --version` / `grok --version` works. You can also point `command` at a full path |

---

## Related docs

- [External Agents Adapter design doc](../designs/external-agents-adapter.md) — authentication design, argv construction, and security requirements in detail
- [`examples/providers-agent-cli.yaml`](../../examples/providers-agent-cli.yaml) — a working example configuration
- [Secrets handling & security posture](../guides/security.md)
