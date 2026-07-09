# Launcher Guide — starting llama.cpp / vllm / mlx from a GUI

> 日本語版: [`launcher.md`](./launcher.md)

The CodeRouter Launcher is a tool that **starts and manages** local inference backends (llama.cpp / vllm / mlx) **through on-screen operation**. Instead of typing a long startup command every time, you pick a model and press a button.

The Launcher comes in two forms:

- **Desktop GUI edition** (`launcher_gui.py`) — a tkinter desktop app. No browser required. CodeRouter itself can also be started from here.
- **Web edition** (`/launcher`) — a browser page served by CodeRouter.

Configuration (the `launcher:` block in `providers.yaml`), the screen layout, and troubleshooting are shared between both editions. This guide describes the shared parts once each.

> For backend installation steps, see the [Backend Installation Guide](./install-backends.en.md); for the full walkthrough from installation to startup, see the [Launcher Quickstart](./launcher-quickstart.en.md).

---

## Overview — what you can do with the Launcher

- Recursively scan `.gguf` / `.safetensors` etc. under `model_dirs` and display a model list
- Pick an option profile (preset) from a dropdown and launch
- Manage multiple processes at once (e.g. llama.cpp + vllm running side by side)
- View each process's log in real time
- See [memory recommendations](#memory-recommendations) for each model, checked against installed memory

---

## The two launchers — which one to use

| | Desktop GUI edition (`launcher_gui.py`) | Web edition (`/launcher`) |
|---|---|---|
| Form | tkinter desktop app | Browser page |
| Starting CodeRouter | **Can start it from this app** | Cannot (it runs inside CodeRouter) |
| Main use | First bootstrap — bring up backend and CodeRouter together | Operational UI for managing backends while CodeRouter is running |
| Configuration | `launcher:` block in `providers.yaml` (shared) | Same as left |

The two are not competitors but complements. The natural split is: **the desktop edition for the first bootstrap, and the Web edition for day-to-day operation once CodeRouter is up and running**.

---

## Desktop GUI edition — how to start

`launcher_gui.py` is a tkinter app that starts and manages the backend and CodeRouter **without a browser**. CodeRouter itself can be launched directly from this GUI, letting you go all the way to connecting a local LLM to Claude Code in a single window.

### Requirements

- Python 3.10 or later
- tkinter — part of Python's standard library (no extra install needed; some Linux distros require a separate `python3-tk` package)
- PyYAML — an existing CodeRouter dependency; running from CodeRouter's venv pulls it in automatically

### Starting it

```bash
# Normal startup
python3 launcher_gui.py

# Via CodeRouter's venv (guarantees PyYAML is available)
uv run python launcher_gui.py

# Explicitly specify a config file
python3 launcher_gui.py --config ~/.coderouter/providers.yaml
```

Config file lookup order: ① `--config` if given → ② `providers.yaml` in the current directory → ③ `~/.coderouter/providers.yaml`. If none exist, it starts with an empty configuration (you can still start things by entering values manually in the UI).

### CodeRouter bar (desktop edition only)

At the top of the desktop edition there is a **CodeRouter bar** not present in the Web edition.

- Status dot — shows `stopped` / `starting…` / `running` / `error` in color
- Port — CodeRouter's listen port (default `8088`). Editable only while stopped or in error state
- ▶ Start CodeRouter / ■ Stop
- Claude Code connection string — `ANTHROPIC_BASE_URL=http://localhost:<port> ANTHROPIC_AUTH_TOKEN=dummy claude`. Click or use "Copy" to send it to the clipboard

When CodeRouter starts, if `~/.coderouter/providers.yaml` doesn't exist yet, a minimal config is auto-generated (this auto-generated file does not include a `launcher:` block — more on this below). Closing the window automatically stops the CodeRouter instance and all backend processes it started.

---

## Web edition — how to start

The operational UI you use in a browser while CodeRouter is running.

1. Add a `launcher:` section to `providers.yaml` (see [Configuration Reference](#configuration-reference))
2. Start CodeRouter — `coderouter serve --port 8088`
3. Open `http://localhost:8088/launcher` in a browser

---

## Using the screen

The Launcher screen is made up of a "MODELS panel," a "LAUNCH form," a "PROCESSES table," and "logs." The look differs between the desktop edition (tkinter) and the Web edition (browser), but **the structure and operations are shared**.

### MODELS panel

- The scan button re-scans `model_dirs` and refreshes the model list
- Clicking a model name auto-fills the "Model path" field (the desktop edition also auto-fills "Name"; a manually typed name is preserved)
- File size (GB) is shown alongside, making it easy to weigh against VRAM/memory
- Each model shows a **memory recommendation badge** (`✓ Recommended` / `⚠ Memory tight`) → [Memory recommendations](#memory-recommendations)
- The header shows the detected hardware (e.g. `Metal · RAM 64GB`)
- Target extensions: `.gguf` `.safetensors` `.bin` `.pt` `.pth` `.ggml` (subfolders are searched recursively too)

### LAUNCH form

| Field | Description |
|---|---|
| **Name** | Any identifier for management purposes (e.g. `qwen-coder-8080`) |
| **Port** | The port the server will run on (default `8080`) |
| **Backend** | Choose from `llama.cpp` / `vllm` / `mlx`. The resolved binary path and availability are shown below |
| **Model path** | Selected from the MODELS panel or entered directly |
| **Option profile** | Choose a preset defined in `providers.yaml` |
| **MTP/draft gguf** | Explicit companion draft/MTP gguf path (llama.cpp only). Leave blank for auto-detection → [MTP / speculative decoding](#mtp--speculative-decoding-llamacpp) |
| **MTP** | `auto` (default, auto-detect) / `off` (disable speculative decoding) |
| **Extra options** | Enter flags not in the profile on the spot. Parsed with `shlex` and appended to the end of the command |

`▶ Launch` starts the process and it appears in the PROCESSES table. If the binary isn't found, **the launch button is automatically disabled** and the reason is displayed. See [Memory recommendations](#memory-recommendations) for the **⚙ Recommended values** button next to the "Extra options" field.

### Automatic provider sync (v2.7.4, Web edition only)

When you start a backend in the Web edition, that backend is **automatically registered as a provider** (no need to edit providers.yaml).

- The provider name is `launcher-<backend>-<port>` (e.g. `launcher-llamacpp-8085`). Restarting under the same name **replaces** the entry — no duplicates
- It's registered under the `launcher` profile (auto-created if absent). **The most recently started backend comes first**
- Routing is explicit opt-in: via the `X-CodeRouter-Profile: launcher` header, or `"profile": "launcher"` in the body. **`default_profile` is not changed**
- Registration is **in-memory only** (nothing is written to providers.yaml — to avoid breaking hand-written comments). It disappears on a serve restart, but since the Launcher's own processes share that lifetime, this stays consistent. If you want it to persist, transcribe it into providers.yaml by hand
- Because the provider is registered with `model: ""`, `/v1/models` returns the **actual model ID (gguf name) the upstream is currently loading** (model-name pass-through, also v2.7.4). Swapping the gguf requires no config edit, and external benchmarks can identify the model (a 30-second TTL cache applies)

Verification:

```bash
# After starting
curl http://localhost:8088/v1/models
#   → "id": "<the loaded gguf name>", "owned_by": "coderouter/launcher-llamacpp-8085"

curl http://localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-CodeRouter-Profile: launcher' \
  -d '{"model":"x","messages":[{"role":"user","content":"say hi"}]}'
#   → connectivity is OK if coderouter_provider is launcher-llamacpp-<port>
```

The desktop GUI edition (launcher_gui.py) runs as a separate process, so it's excluded from automatic sync. As before, adjust the `base_url` of the entry in providers.yaml (e.g. the auto-generated `llama-cpp-local`) to match the launched port.

### PROCESSES table

A list of launched backend processes. Shows NAME / BACKEND (llama.cpp / vllm / mlx) / MODEL / PORT / PID / STATUS (color-coded `starting` / `running` / `stopped` / `error`), and lets you select a process to **stop** (SIGTERM), **remove** (from the registry), or **view logs**.

### Logs

Real-time display of the selected process's stdout/stderr. In the Web edition, the log panel auto-refreshes every 3 seconds while running. There are caps on retained lines and displayed lines so long-running sessions don't eat memory.

### Typical workflow (desktop edition)

1. **Pick a model** — click the model you want to use from MODELS
2. **Start the backend** — choose an option profile and press the launch button. It shows as `running` in PROCESSES
3. **Start CodeRouter** — "▶ Start CodeRouter" in the top bar
4. **Connect Claude Code** — copy the connection string and run it in a terminal

---

## MTP / speculative decoding (llama.cpp)

llama.cpp's `llama-server` supports Multi-Token Prediction (MTP) / speculative decoding via `--spec-type`-family flags. The Launcher assembles these flags automatically from the LAUNCH form's **MTP/draft gguf** field and **MTP** field (`auto` / `off`). **llama.cpp only** — specifying `draft_model_path` or `mtp_mode` for vllm/mlx makes the launch request fail with a 400.

### Auto-detection order (`mtp_mode: auto`, the default)

1. **Embedded nextn** — if the selected main gguf's metadata has `{arch}.nextn_predict_layers > 0`, `--spec-type draft-mtp` is added with no separate draft model needed.
2. **Same-folder companion gguf** — if there's no embedded nextn, the Launcher scans the **same directory** as the main gguf for a companion that satisfies all of:
   - the filename contains `mtp` or `draft`, or shares the main file's name prefix (with shard/quant suffixes stripped), and
   - its file size is under 50% of the main gguf, and
   - if its gguf architecture is readable, it matches the main model's (a mismatch is rejected — to avoid a tokenizer/vocabulary mismatch).

   When a candidate is selected, filenames containing `mtp` get `--spec-type draft-mtp`; otherwise `--spec-type draft-simple` — both paired with `--model-draft <path>`.
3. **Nothing found** — the process starts normally without speculative decoding. The process log records `[launcher] MTP/draft gguf not found next to <main>.gguf; starting without speculative decoding`.

### Specifying an explicit draft/MTP gguf

You can point the **MTP/draft gguf** field directly at a companion gguf. If the given path doesn't exist, the launch request is rejected with a 400. Filenames containing `mtp` get `--spec-type draft-mtp`; otherwise `--spec-type draft-simple`.

### `mtp_mode: off`

Choosing `off` in the **MTP** field never emits speculative-decoding flags (reproduces the historical launch command exactly). Combining `off` with an explicit **MTP/draft gguf** is a conflict and is rejected with a 400.

### When `--spec-type` is already supplied via extra options

If "Extra options" or the option profile already contains `--spec-type`, the Launcher's auto-detection is skipped entirely (no flags are added) — an explicit operator choice always wins.

### `-md` / `--model-draft` cannot be used in extra options

Just like `-m` / `--model`, the draft model path can only be set via the **MTP/draft gguf** field. Writing `-md` / `--model-draft` / `--spec-draft-model` into "Extra options" or an option profile causes the launch request to be rejected with a 400. The remaining speculative knobs (`--spec-type` / `--spec-draft-n-max` / `--spec-draft-n-min` / `--spec-draft-p-min` / `-ngld` / `-devd`) stay free-form.

### Known issue: `--split-mode tensor` combination (llama.cpp issue #24309)

Combining a nextn-embedded model / active speculative decoding with `--split-mode tensor` is known to crash llama.cpp ([issue #24309](https://github.com/ggml-org/llama.cpp/issues/24309)). The Launcher detects this combination but does not block the launch — it records a warning in the process log recommending `--split-mode layer` instead.

### API

`POST /api/launcher/start` (Web edition) accepts these additional fields (llama.cpp backend only; other backends get a 400):

| Field | Type | Default | Description |
|---|---|---|---|
| `draft_model_path` | `string \| null` | `null` | Explicit companion draft/MTP gguf path |
| `mtp_mode` | `"auto" \| "off"` | `"auto"` | `auto` = auto-detect, `off` = disable speculative decoding |

On a successful start, the response JSON includes the resolved speculative flags under the `"speculative"` key (a token array, e.g. `["--spec-type", "draft-mtp"]`; an empty array when nothing was added).

---

## Memory recommendations

Each model in the MODELS list shows a verdict checked against the memory installed on the machine running CodeRouter (unified memory for Apple Silicon, VRAM for NVIDIA GPUs, RAM otherwise).

- **✓ Recommended** — expected to run with margin (`model size × 1.2 + 2GB` fits within available memory)
- **⚠ Memory tight** — doesn't fit, or margin is thin. May swap and become significantly slower

The **⚙ Recommended values** button next to the "Extra options" field fills that field with suggested launch flags based on the selected model, hardware, and **backend**. The output differs per backend.

- **llama.cpp** — `-ngl` (`99` if it fits the GPU, `0` for CPU-only) / `--ctx-size` (`4096`–`32768` depending on available memory) / `--threads` (CPU core count − 2)
- **vllm** — empty. `--max-model-len` and similar depend on the model's actual context length, so this is left to the engine's own auto-derivation
- **mlx** — empty. Since it assumes unified memory, no launch-time tuning flags are needed

All of these are **estimates** — they don't account for other processes' memory usage or quantization scheme. Adjust on the actual machine.

---

## Configuration reference

The MODELS list, option profiles, and binary paths are all loaded from the `launcher:` block in `~/.coderouter/providers.yaml`. **Shared between the desktop and Web editions**.

### The full `launcher:` block

```yaml
# ~/.coderouter/providers.yaml
launcher:
  model_dirs:           # list[str]  required
    - ~/llm/models
  backends:             # dict  optional
    llama.cpp:
      binary: null      # null = llama-server from PATH
    vllm:
      binary: null      # null = python from PATH
    mlx:
      binary: null      # null = python from PATH
  option_profiles:      # dict  optional
    llama.cpp: [...]
    vllm: [...]
```

> The `providers.yaml` auto-generated by the "Start CodeRouter" button does not include a `launcher:` block. To use the model list or profiles, you'll need to add the `launcher:` block yourself. You can start from a copy of the `launcher_profiles.yaml.example` template.

### `backends` — binary path configuration

Specify a full path when the binary isn't in PATH (source builds, venv environments, etc.).

```yaml
launcher:
  backends:
    llama.cpp:
      binary: ~/llama.cpp/build/bin/llama-server         # source build example
    vllm:
      binary: ~/.coderouter/backends/vllm/bin/python     # venv example
    mlx:
      binary: ~/.coderouter/backends/mlx/bin/python      # venv example
```

If `binary` is omitted or `null`, the default name (`llama-server` / `python`) is looked up in PATH. Tilde (`~`) expansion is supported. For vLLM/MLX, it's recommended to keep separate venvs per backend under `~/.coderouter/backends/<backend-name>/` (see the [Installation Guide](./install-backends.en.md) for details). The resolved path is shown below the "Backend" select in the UI.

### `model_dirs`

- Tilde (`~`) expansion supported
- Non-existent paths are silently skipped during scanning (no startup error)
- Extensions searched: `.gguf` `.safetensors` `.bin` `.pt` `.pth` `.ggml`
- Subfolders are searched recursively

### `option_profiles`

```yaml
option_profiles:
  llama.cpp:            # backend name (key)
    - name: "A readable name"   # shown in the UI dropdown
      args:
        "-ngl": 99              # int → "-ngl 99"
        "--ctx-size": 4096
        "--dtype": "float16"    # str → "--dtype float16"
        "--mlock": true         # bool true → "--mlock" (no value)
        "--no-mmap": false      # bool false → omitted
```

**Type rules for `args`:**

| YAML type | CLI conversion |
|---|---|
| `int` / `float` / `str` | 2 tokens: `--flag value` |
| `bool: true` | `--flag` only (no value) |
| `bool: false` | this flag is omitted |

### Extra options (free-form input)

The string in the UI's "Extra options" field is parsed with `shlex.split()` and appended to the end of the command. Use this to try experimental flags not in a profile.

```
-ngl 40 --rope-scale 2.0 --rope-freq-base 10000
```

> **Note**: re-specifying the model via `-m` / `--model` (or the `--model=...` form) is not accepted in either extra options or option profiles — doing so causes the launch request to be rejected with a 400. Specify the model only via the "Model path" field. Likewise, re-specifying the draft model via `-md` / `--model-draft` / `--spec-draft-model` (llama.cpp-only flags) is not accepted in extra options or option profiles either. Specify the draft model only via the "MTP/draft gguf" field — see [MTP / speculative decoding](#mtp--speculative-decoding-llamacpp).

---

## Option quick reference

### llama.cpp

Only the commonly used flags are listed. See `llama-server --help` for the full list.

| Flag | Description | Suggested value |
|---|---|---|
| `-ngl` | Number of layers offloaded to GPU | `99` (all) / `0` (CPU only) |
| `--ctx-size` | Context length (tokens) | `4096` / `8192` / `131072` |
| `--threads` | Number of CPU threads | CPU core count − 2 |
| `--batch-size` | Batch size | `512` |
| `--mlock` | Lock into memory (prevent swap) | `true` |
| `--embedding` | Start in embedding mode | `true` |

### vllm

See `python -m vllm.entrypoints.openai.api_server --help` for the full list.

| Flag | Description | Suggested value |
|---|---|---|
| `--dtype` | Tensor data type | `"auto"` / `"float16"` / `"bfloat16"` |
| `--max-model-len` | Maximum context length | `4096` / `32768` |
| `--gpu-memory-utilization` | GPU memory utilization (0–1) | `0.85` |
| `--quantization` | Quantization scheme | `"awq"` / `"gptq"` |
| `--tensor-parallel-size` | Tensor parallelism degree (number of GPUs) | `2` |

### mlx

MLX (`mlx_lm.server`) assumes unified memory and has no concept of layer offloading like `-ngl`. It runs as soon as the Launcher sets `--model` and `--port`; startup performance-tuning flags are generally unnecessary.

---

## Using it after startup — connecting to CodeRouter

The backend started by the Launcher provides an OpenAI-compatible API. Register it as a CodeRouter provider in `providers.yaml` to gain routing, guards, and fallback.

```yaml
providers:
  - name: local-qwen-launcher
    kind: openai_compat
    base_url: http://localhost:8080/v1   # the port specified in the Launcher
    model: Qwen2.5-Coder-7B-Instruct

profiles:
  - name: default
    providers: [local-qwen-launcher]
```

Start Claude Code pointed at CodeRouter:

```bash
ANTHROPIC_BASE_URL=http://localhost:8088 ANTHROPIC_AUTH_TOKEN=dummy claude
```

---

## Adding and sharing profiles

You can add a new preset just by appending to `option_profiles`. No code changes needed.

```yaml
launcher:
  option_profiles:
    llama.cpp:
      - name: "My custom setup"
        args:
          "-ngl": 40
          "--ctx-size": 8192
```

Restarting CodeRouter reflects it in the UI. `launcher_profiles.yaml.example` is bundled in the repository, so you can add a new profile to it and share it via a PR.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Launch button is disabled (grayed out) | The backend's binary can't be found | Check the display below the backend field, and set a full path in `launcher.backends.<name>.binary` |
| Model list is empty | `launcher.model_dirs` isn't set, or the config file wasn't found | Set `model_dirs` in `providers.yaml` (the desktop edition can also specify it explicitly via `--config`) |
| Option profile can't be selected | `launcher.option_profiles` is missing | Add `option_profiles` to `providers.yaml` |
| Goes to `error` right after starting | Wrong model path / insufficient VRAM | Check the error details in the log |
| Port conflict | Another process is already using that port | Change the port number |
| `PyYAML not found` (desktop edition) | Ran from a plain Python install | Run from CodeRouter's venv with `uv run python launcher_gui.py` |
| Process disappears after a restart | By design — the registry is in-memory | Use OS-level launchd/systemd if you need it to persist |

---

## Related docs

- [Backend Installation Guide](./install-backends.en.md) — installing llama.cpp / vLLM / MLX
- [Launcher Quickstart](./launcher-quickstart.en.md) — the full walkthrough from install to startup
- [Architecture details — Launcher section](../concepts/architecture.en.md#launcher--llamacpp--vllm-プロセス管理-v250)
- [Usage Guide](../guides/usage-guide.md)
- [llama.cpp direct connection guide](./llamacpp-direct.en.md)
