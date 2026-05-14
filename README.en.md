<h1 align="center">CodeRouter</h1>

<p align="center">
  <strong>Tool calling breaks when you run Claude Code on local LLMs.<br>One router fixes it.</strong>
</p>

<p align="center">
  <a href="https://github.com/zephel01/CodeRouter/actions/workflows/ci.yml"><img src="https://github.com/zephel01/CodeRouter/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/coderouter-cli/"><img src="https://img.shields.io/pypi/v/coderouter-cli?include_prereleases&color=blue&label=pypi" alt="pypi"></a>
  <a href=""><img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="python"></a>
  <a href=""><img src="https://img.shields.io/badge/deps-5-brightgreen" alt="deps"></a>
  <a href=""><img src="https://img.shields.io/badge/license-MIT-yellow" alt="license"></a>
</p>

<p align="center">
  <strong>English</strong> · <a href="./README.md">日本語</a> · <a href="./docs/quickstart.en.md">Get started in 10 min</a> · <a href="./docs/architecture.md">Architecture</a>
</p>

---

## What it does — in 30 seconds

```
Your agent (Claude Code / gemini-cli / codex)
        │
        ▼
  ┌─ CodeRouter ──┐
  │  translate     │──→  ① Local (Ollama — free, fastest)
  │  repair        │──→  ② Free cloud (OpenRouter / NIM)
  │  guard + heal  │──→  ③ Paid (Claude — opt-in only)
  └────────────────┘
```

**What it does for you:**

- Repairs broken tool calling from local models before it reaches Claude Code
- Automatically falls back to the next provider when one goes down
- Only uses paid APIs when you explicitly allow it (free-only by default)
- Keeps your agent running for 8+ hours with 6 types of guards
- Diagnoses what's wrong with one command: `coderouter doctor`

---

## Install (3 lines)

```bash
# 1. Drop a sample config
mkdir -p ~/.coderouter
curl -fsSL https://raw.githubusercontent.com/zephel01/CodeRouter/main/examples/providers.yaml \
  > ~/.coderouter/providers.yaml

# 2. Run (Python 3.12+)
uvx --from coderouter-cli coderouter serve --port 8088
```

For a permanent install: `uv tool install coderouter-cli`

---

## Use with Claude Code

```bash
# Terminal 1
coderouter serve --port 8088

# Terminal 2
ANTHROPIC_BASE_URL=http://localhost:8088 ANTHROPIC_AUTH_TOKEN=dummy claude
```

That's it. Claude Code works as usual, but your local Ollama is answering behind the scenes.

---

## Do you need it?

| Your situation | CodeRouter? |
|---|---|
| Claude Code + local Ollama, tool calling breaks | **Yes** — wire translation + tool repair |
| Claude Code + local, dies after long sessions | **Helpful** — 6 guards + self-healing |
| codex / gemini-cli + Ollama works fine | Optional — if you want fallback |
| Using Claude API directly, no issues | Not needed |

Full decision matrix → [Do I need CodeRouter?](./docs/when-do-i-need-coderouter.en.md)

---

## Key Features

### Connection & Repair

| Feature | What it does |
|---|---|
| **Wire translation** | Claude Code (Anthropic format) ↔ Ollama (OpenAI format) auto-converted |
| **Tool-call repair** | JSON that local models emit as plain text → valid tool_use blocks |
| **3-tier fallback** | Local → free cloud → paid, automatic switching |
| **Output filters** | Strips leaked `<think>` tags, stop markers, XML tool tags |

### Long-running Session Guards

| Guard | What it protects against |
|---|---|
| **Context Budget** | Messages piling up → context window overflow. Auto-trim at 90% |
| **Drift Detection** | Model quality degrading over time → switch provider or flush KV cache (6 signals incl. `goal_progress_stall`; `goal_mode` for tighter thresholds) |
| **Self-healing** | Backend crashes → auto-exclude + restart + recovery probe → auto-restore |
| **Tool Loop Guard** | Agent calling the same tool forever → detect and break |
| **Memory Pressure** | GPU running out of VRAM → switch to lighter model |
| **Mid-stream Guard** | Response dies mid-stream → safely return accumulated text |

### Diagnostics & Visibility

| Feature | What you learn |
|---|---|
| **`coderouter doctor`** | 6-probe diagnosis of provider issues + copy-paste YAML patches |
| **`/dashboard`** | Real-time browser view of what's happening |
| **`coderouter audit`** | Search guard activation history |
| **`coderouter replay`** | Compare providers statistically (A/B analysis) / `--suggest-rules` for automated rule suggestions |
| **Continuous Probe** | Background health monitoring even during idle |

---

## Minimal Config

```yaml
# ~/.coderouter/providers.yaml
default_profile: claude-code

profiles:
  - name: claude-code
    providers: [ollama-local, openrouter-free]

providers:
  - name: ollama-local
    kind: openai_compat
    base_url: http://localhost:11434/v1
    model: qwen3-coder:7b

  - name: openrouter-free
    kind: openai_compat
    base_url: https://openrouter.ai/api/v1
    model: qwen/qwen3-coder:free
    api_key_env: OPENROUTER_API_KEY
```

More detail → [Usage guide](./docs/usage-guide.en.md) · [Architecture](./docs/architecture.md)

---

## Documentation

| Goal | Document |
|---|---|
| Get running fast | [Quickstart](./docs/quickstart.en.md) |
| Use it well | [Usage guide](./docs/usage-guide.en.md) |
| Run for free | [Free-tier guide](./docs/free-tier-guide.en.md) |
| Stuck? | [Troubleshooting](./docs/troubleshooting.en.md) |
| Understand the design | [Architecture](./docs/architecture.md) |
| Full release history | [CHANGELOG](./CHANGELOG.md) |

日本語: [Quickstart](./docs/quickstart.md) · [利用ガイド](./docs/usage-guide.md) · [無料枠ガイド](./docs/free-tier-guide.md) · [トラブルシューティング](./docs/troubleshooting.md)

---

## Troubleshooting (cheat sheet)

**First move**: run `coderouter doctor --check-model <provider>`. It usually finds the problem.

| Symptom | Cause | Details |
|---|---|---|
| 401 error | API key not set / missing `export` in `.env` | [§1](./docs/troubleshooting.en.md#1-five-startup--config-gotchas-added-in-v162) |
| Empty / garbage replies | Ollama `num_ctx` truncated to 2048 | [§3](./docs/troubleshooting.en.md#3-ollama-beginner--5-silent-fail-symptoms-v07-c) |
| `<think>` tags leaking | Add `output_filters: [strip_thinking]` | [§3](./docs/troubleshooting.en.md#3-ollama-beginner--5-silent-fail-symptoms-v07-c) |
| Tool calls misbehaving in Claude Code | Tool-call repair not kicking in | [§4](./docs/troubleshooting.en.md#4-claude-code-integration-gotchas-added-in-v162) |

Open `http://localhost:8088/dashboard` while debugging — most issues become visible in 10 seconds.

---

## Tech Specs

- **Runtime deps**: `fastapi` / `uvicorn` / `httpx` / `pydantic` / `pyyaml` — only 5
- **Tests**: 964 (41 consecutive sub-releases without adding a dep)
- **OS**: macOS (Apple Silicon recommended) / Linux / Windows WSL2
- **Backends**: Ollama / llama.cpp / LM Studio / vLLM / MLX-LM / OpenRouter / NVIDIA NIM / Anthropic API
- **License**: MIT

---

## Ecosystem

CodeRouter runs as an independent backend router layer. Point any project's `OPENAI_BASE_URL` at CodeRouter and it gets fallback + observability for free:

- **[Voice Bridge](https://github.com/zephel01/voice-bridge)** — Real-time voice translation + AI voice chat. Route through CodeRouter so your voice assistant doesn't go silent when the local LLM hiccups.

---

## Security

Secrets go in env vars, not config files. See [`docs/security.en.md`](./docs/security.en.md) for the full policy and reporting instructions.

## License

MIT
