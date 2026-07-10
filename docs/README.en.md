# CodeRouter Documentation

日本語版: [`README.md`](./README.md)

Index of CodeRouter's public documentation — find the right page by what you want to do.

> Developer-internal notes and article drafts live in `inside/` and `articles/` (local-only, not shipped in the public repo).

---

## Quick start by goal

| Goal | Read |
|---|---|
| Get running now | [start/quickstart](start/quickstart.en.md) |
| Is it for me? | [start/when-do-i-need-coderouter](start/when-do-i-need-coderouter.en.md) |
| Run for free | [guides/free-tier-guide](guides/free-tier-guide.en.md) |
| Learn the features | [guides/usage-guide](guides/usage-guide.en.md) |
| Measure & avoid the language tax | [guides/language-tax](guides/language-tax.en.md) |
| Something broke | [guides/troubleshooting](guides/troubleshooting.en.md) |
| Launch a local LLM | [backends/launcher-quickstart](backends/launcher-quickstart.md) |
| Secrets & security | [guides/security](guides/security.en.md) |
| Understand the design | [concepts/architecture](concepts/architecture.en.md) |
| Extend with plugins | [Plugins](#plugins) |

---

## Layout

```
docs/
├── start/             Getting started
├── guides/            How-to guides
├── backends/          Local LLM backends
├── concepts/          Architecture & internals
├── designs/           Design docs
├── retrospectives/    Release retrospectives
├── evidence/          Verification logs
├── openrouter-roster/ OpenRouter model roster
└── assets/            Images
```

Many documents have a Japanese version (`.md`) and an English version (`.en.md`).

---

## 1. Getting started — `start/`

For first-time users.

- **quickstart** — Get running in one sitting · [日本語](start/quickstart.md) · [English](start/quickstart.en.md)
- **when-do-i-need-coderouter** — Decide whether you need it · [日本語](start/when-do-i-need-coderouter.md) · [English](start/when-do-i-need-coderouter.en.md)

## 2. How-to guides — `guides/`

Day-to-day usage.

- **usage-guide** — Full feature guide · [日本語](guides/usage-guide.md) · [English](guides/usage-guide.en.md)
- **language-tax** — Measure, route around, and visualize the CJK language tax · [日本語](guides/language-tax.md) · [English](guides/language-tax.en.md)
- **free-tier-guide** — Zero-cost operation with NVIDIA NIM × OpenRouter Free · [日本語](guides/free-tier-guide.md) · [English](guides/free-tier-guide.en.md)
- **troubleshooting** — Fixing problems · [日本語](guides/troubleshooting.md) · [English](guides/troubleshooting.en.md)
- **security** — Secrets handling & security posture · [日本語](guides/security.md) · [English](guides/security.en.md)

## 3. Local LLM backends — `backends/`

Installing, launching, and connecting local inference backends.

- **install-backends** — Installing the three backends (llama.cpp / vLLM / MLX) · [日本語](backends/install-backends.md) · [English](backends/install-backends.en.md)
- **launcher-quickstart** — Install a backend and launch, the shortest path · [日本語](backends/launcher-quickstart.md)
- **launcher** — Launcher guide (Web & Desktop GUI) · [日本語](backends/launcher.md)
- **external-agents** — External coding-agent CLI (agent_cli, v2.7.7, claude only) · [日本語](backends/external-agents.md) · [English](backends/external-agents.en.md)
- **llamacpp-direct** — Connect llama.cpp directly · [日本語](backends/llamacpp-direct.md) · [English](backends/llamacpp-direct.en.md)
- **lmstudio-direct** — Connect LM Studio directly · [日本語](backends/lmstudio-direct.md) · [English](backends/lmstudio-direct.en.md)
- **hf-ollama-models** — Use HF models via Ollama · [日本語](backends/hf-ollama-models.md)
- **gguf_dl** — GGUF download helper · [日本語](backends/gguf_dl.md)
- **verify-ollama-0.23.1** — Ollama v0.23.1 verification checklist · [日本語](backends/verify-ollama-0.23.1.md)

## 4. Architecture & internals — `concepts/`

How CodeRouter works and its reliability mechanisms.

- **architecture** — Architecture overview · [日本語](concepts/architecture.md) · [English](concepts/architecture.en.md)
- **context-budget** — Context budget management (v2.0.0) · [日本語](concepts/context-budget.md) · [English](concepts/context-budget.en.md)
- **drift-detection** — Drift detection (v2.0-G) · [日本語](concepts/drift-detection.md) · [English](concepts/drift-detection.en.md)
- **partial-stitch** — Mid-stream partial stitching (v2.0-H) · [日本語](concepts/partial-stitch.md) · [English](concepts/partial-stitch.en.md)
- **continuous-probing** — Continuous probing (v2.0-I) · [日本語](concepts/continuous-probing.md) · [English](concepts/continuous-probing.en.md)

## 5. Design docs & records

- **designs/** — Feature design docs ([v1.6 auto-router](designs/v1.6-auto-router.md) and others)
- **retrospectives/** — Release retrospectives ([v0.4](retrospectives/v0.4.md) – [v1.0](retrospectives/v1.0.md))
- **evidence/** — Verification run logs
- **openrouter-roster/** — OpenRouter model roster — [README](openrouter-roster/README.md)

---

## Plugins

CodeRouter's **Plugin SDK** (since v2.3.0) loads out-of-tree plugins *opt-in*: a plugin runs only when its name is listed in `plugins.enabled` (supply-chain defense), so installing one does nothing by itself. Each plugin ships as a separate PyPI package, so **the core's dependencies never grow**.

| Plugin | What it does | Install | Repo |
|---|---|---|---|
| **compress** | Compresses tool output (JSON / logs) before it reaches the LLM to cut tokens; originals kept locally and reversible (CCR). `cache-align` also aligns Anthropic prompt caching. | `pip install coderouter-plugin-compress` | [coderouter-plugin-compress](https://github.com/zephel01/coderouter-plugin-compress) |
| **memory** | Extracts key facts from responses into `facts.jsonl` and auto-injects them into the next session's system prompt — solving "explain it every time" at the wire layer. | `pip install coderouter-plugin-memory` | [coderouter-plugin-memory](https://github.com/zephel01/coderouter-plugin-memory) |

Enable by adding to `providers.yaml`; a `plugin-loaded` line in the startup log confirms activation.

```yaml
plugins:
  enabled:
    - compress          # compress tool output
    - compress-stats    # report compression ratio in coderouter stats
    - cache-align       # align prompt-cache breakpoints
    - memory            # cross-session memory
  config:
    compress:
      mode: safe        # off | safe | aggressive
      ccr: true         # reversible re-expansion (default on)
    memory:
      consolidate_model: qwen3:1.7b
```

See each plugin's repo README for full configuration.

---

Last updated: 2026-06-24
