# Using HuggingFace-distributed models via Ollama

> 日本語版: [`hf-ollama-models.md`](./hf-ollama-models.md)

> CodeRouter's `examples/providers.yaml` includes **commented-out provider
> stanzas** for trending models not yet registered in the official Ollama
> registry (Gemma 4 26B-A4B, GLM-4.5-Air, Opus-distilled Qwen3 fine-tunes,
> etc.). This document walks through the steps to actually run them.

---

## Prerequisites

- Ollama 0.3.13 or later (direct HF execution support)
- A local GPU/Mac with sufficient VRAM/unified memory
- A HuggingFace account (only needed to pull gated repos.
  Qwen3-Coder / Gemma 4 / GLM families are mostly free public)

Check your Ollama version:

```bash
ollama --version
# OK if ollama version is 0.3.13 or higher
```

---

## Basic steps

### 1. Pull the HF GGUF

```bash
# Example: pull Qwen3-Coder 30B-A3B (Q4_K_M quantization)
ollama pull hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
```

Key points:
- The `:<quant>` suffix is **required**. Omitting it makes Ollama return
  `404 model not found` (CodeRouter v0.7-B doctor's `auth+basic-chat`
  probe detects this as `UNSUPPORTED`).
- Choosing a quantization variant:
  - `Q4_K_M`: the standard size/quality balance. **Try this first.**
  - `Q5_K_M`, `Q6_K_M`: better quality if you have memory to spare.
  - `Q8_0`: near-lossless relative to the original model. Needs roughly 2x the VRAM.
  - `IQ3_XS` etc.: extreme size reduction (for lightweight machines). Clear quality degradation.

### 2. (Optional) give it a short alias

Writing a long name like
`hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M` in
`providers.yaml` is hard to read, so it's recommended to cut a local
alias with `ollama cp`:

```bash
ollama cp hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M qwen3-coder:30b-a3b
```

This lets you refer to it as `qwen3-coder:30b-a3b`.

### 3. Enable the corresponding stanza in providers.yaml

Uncomment the corresponding stanza in the HF-on-Ollama section of
`examples/providers.yaml`, and rewrite the `model:` field to the name
you pulled in step 1 (or the alias from step 2):

```yaml
# Before (commented out):
# - name: ollama-qwen3-coder-480b-hf
#   kind: openai_compat
#   ...

# After:
- name: ollama-qwen3-coder-30b-hf
  kind: openai_compat
  base_url: http://localhost:11434/v1
  # If you cut an alias in step 2:
  model: qwen3-coder:30b-a3b
  # If you didn't cut an alias:
  # model: hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
  paid: false
  timeout_s: 240
  output_filters: [strip_thinking, strip_stop_markers]
  capabilities:
    chat: true
    streaming: true
    tools: true
```

### 4. Also append it to the target profile's `providers:` list

For example, to place it as the local primary in the `coding` profile:

```yaml
profiles:
  - name: coding
    append_system_prompt: |
      ...
    providers:
      - ollama-qwen3-coder-30b-hf  # ← added
      - ollama-qwen-coder-14b
      - ...
```

### 5. Verify with `coderouter doctor --check-model`

```bash
coderouter doctor --check-model ollama-qwen3-coder-30b-hf
```

Expected output:
- `auth+basic-chat`: OK
- `num_ctx`: NEEDS_TUNING (Ollama's default of 2048 fails the canary probe)
- `tool_calls`: OK
- `streaming`: OK or NEEDS_TUNING

If NEEDS_TUNING appears, you can use the auto-patch feature implemented in v1.8.0:

```bash
coderouter doctor --check-model ollama-qwen3-coder-30b-hf --apply
# → non-destructively writes back extra_body.options.num_ctx: 32768 etc. to providers.yaml
```

---

## Registration examples for recommended models

### For coding

```bash
# Qwen3-Coder 30B-A3B (24GB+ VRAM recommended, coding profile primary)
ollama pull hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
ollama cp  hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M qwen3-coder:30b-a3b

# Qwen3-Coder 480B-A35B (needs Mac M3 Ultra 192GB / NVIDIA H100x2 class hardware)
ollama pull hf.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:Q4_K_M
ollama cp  hf.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:Q4_K_M qwen3-coder:480b-a35b
```

### For general/miscellaneous use (recommended in the note article)

> **2026-04 update**: Gemma 4 / Qwen3.6 have been registered in the
> official Ollama registry. Going through HF is no longer needed. You can
> use `ollama pull gemma4:26b` directly. The `ollama-gemma4-*` stanza in
> providers.yaml is already enabled.

> **⚠️ Qwen3.6 family (qwen3.6:35b / qwen3.6:27b) tends to get stuck via
> Ollama (confirmed via real-machine verification in v1.8.1–v1.8.3 plus
> community reports on X/Reddit)**:
>
> - `tool_calls [NEEDS TUNING]` (Ollama's chat template/tool spec is immature)
> - Broad reports of hard crashes / reboots / memory-calculation bugs (mainly on Mac Metal)
> - Variants like `qwen3.6:35b-a3b-coding-nvfp4` 500-error on the MLX backend
>
> **If you're aiming for Qwen3.6 as a Sonnet-class model, direct
> llama.cpp is recommended over Ollama**: `Unsloth/Qwen3.6-35B-A3B-GGUF`
> (UD-Q4_K_M) + `llama-server` gives clean, fully-working native
> `tool_calls`. See
> [`docs/llamacpp-direct.en.md`](./llamacpp-direct.en.md) for steps
> (real-machine verified in CodeRouter v1.8.3; a
> `llamacpp-qwen3-6-35b-a3b` provider example is also bundled in
> `examples/providers.yaml`).

### For reasoning (GLM / Opus distillation)

```bash
# GLM-4.5-Air ("intent understanding at Claude Opus level")
# Recommended: use Z.AI's cloud API (zai-coding-glm-4-5-air) — no
# registration needed. The following is only if you want to run it locally:
ollama pull hf.co/unsloth/GLM-4.5-Air-Instruct-GGUF:Q4_K_M
ollama cp  hf.co/unsloth/GLM-4.5-Air-Instruct-GGUF:Q4_K_M glm-4.5-air

# For Opus-distilled Qwen3 fine-tunes, search HF for "qwen3 opus distill"
# or "claude-distill qwen3". Several community fine-tunes come up.
# Example (replace with an actual repo):
# ollama pull hf.co/<author>/Qwen3-Opus-Distill-30B-GGUF:Q4_K_M
```

### Note: Z.AI Coding Plan's "unauthorized tool" warning

If you want to seriously use the GLM family, Z.AI's Coding Plan
(from $18/month) is the most cost-effective option. However, the
official docs (docs.z.ai/devpack/overview) explicitly state that
"access via unauthorized third-party tools may result in benefit
restrictions." CodeRouter provides Anthropic-API-compatible ingress,
so it should look like an authorized tool, but this depends on their
detection logic.

If you want to be certain, do one of the following:

1. **Connect Claude Code directly to Z.AI** (bypassing CodeRouter)
2. **Use the Z.AI General API (`/api/paas/v4`) on a pay-as-you-go basis** —
   enable `zai-paas-glm-4-7` (commented) in `examples/providers.yaml`

---

## Known pitfalls

### 1. Forgetting the `:<quant>` suffix

```
$ ollama pull hf.co/unsloth/Qwen3-Coder-30B-A3B-GGUF
Error: 404 page not found
```

→ A quantization suffix is required. Check the repo on HF and append e.g. `Q4_K_M`.

### 2. Ollama's default `num_ctx: 2048` is too small

Claude Code sends a 15-20K token system prompt every turn. With
Ollama's default context window of 2048, **the beginning of the prompt
is silently truncated**, tool declarations disappear, and Claude Code
ends up in a state where it "somehow can't use tools."

Fix: CodeRouter emits `extra_body.options.num_ctx: 32768` as a patch
in v1.0-B. Apply it automatically with
`coderouter doctor --check-model <name> --apply`.

### 3. Quantization size vs. VRAM mismatch

| Quantization | Approx. size (30B model) | VRAM needed |
|---|---|---|
| Q4_K_M | ~18 GB | 20 GB+ |
| Q5_K_M | ~22 GB | 24 GB+ |
| Q6_K_M | ~26 GB | 28 GB+ |
| Q8_0 | ~32 GB | 36 GB+ |

Insufficient VRAM triggers CPU offload, causing a severe slowdown. You
can check whether the running model is on the GPU with `ollama ps`.

### 4. Mismatch with CodeRouter's capability registry

Model names in the `hf.co/...` format don't match the globs (e.g.
`qwen3-coder:*`) in CodeRouter's bundled `model-capabilities.yaml`, so
automatic capability resolution doesn't kick in. Either **explicitly
declare** things like `capabilities.tools: true` on the
`providers.yaml` side, or cut a short alias with `ollama cp` so it
matches the glob.

```yaml
# Example: if using the raw HF name, declare capabilities explicitly
- name: ollama-qwen3-coder-30b-hf
  model: hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
  capabilities:
    tools: true       # ← explicit, since it doesn't match the registry's glob
```

Or, recommended:

```bash
ollama cp hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M qwen3-coder:30b-a3b
```

→ This matches the bundled registry's `qwen3-coder:*` glob, and
`tools: true` / `claude_code_suitability: ok` are resolved automatically.

---

## Reference links

- Ollama HF integration: <https://huggingface.co/docs/hub/en/ollama>
- Unsloth (a leading uploader of fast quantized versions): <https://huggingface.co/unsloth>
- **Unsloth: Tool calling guide for local LLMs (Japanese)**: <https://unsloth.ai/docs/jp/ji-ben/tool-calling-guide-for-local-llms>
  — organizes, model by model, why tool-calling fails or breaks for local LLMs like Qwen / Llama / Gemma and how to fix it. A good background read for when CodeRouter reports `tool_calls: NEEDS_TUNING`.
- bartowski (quality-focused quantized versions): <https://huggingface.co/bartowski>
- Qwen3-Coder (official Alibaba): <https://huggingface.co/collections/Qwen/qwen3-coder>
- Gemma 4 (official Google): <https://huggingface.co/collections/google/gemma-4>
- CodeRouter doctor details: [`docs/troubleshooting.md`](../guides/troubleshooting.md)
- Full `providers.yaml` structure: [`examples/providers.yaml`](../../examples/providers.yaml)
