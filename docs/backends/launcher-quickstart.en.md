# Launcher Quickstart — from backend install to startup

> 日本語版: [`launcher-quickstart.md`](./launcher-quickstart.md)

A guide for using the CodeRouter Launcher for the first time. It walks through installing the backend (llama.cpp / vLLM / mlx) that the Launcher starts and manages, all the way to starting the Launcher and connecting Claude Code.

Supported platforms: macOS / Linux / Windows

---

## Overall flow

1. Install a backend (one of **llama.cpp** / **vLLM** / **mlx**)
2. Prepare a model
3. Write a `launcher:` block in `providers.yaml`
4. Start the Launcher (desktop GUI edition or Web edition)
5. From the Launcher, start the backend + CodeRouter → connect Claude Code

You only need one backend to get started. **When in doubt, llama.cpp is recommended** — it runs on all OSes, has a huge selection of `.gguf` models, and is lightweight to set up.

---

## 1. Install a backend

Install one inference backend that the Launcher will start. **When in doubt, llama.cpp** — it runs on all OSes and has a huge selection of `.gguf` models.

| Backend | Fastest install | Support |
|---|---|---|
| **llama.cpp** | `brew install llama.cpp` (macOS/Linux) / `winget install ggml.llamacpp` (Windows) | All OSes |
| **MLX** | `pip install mlx-lm` (venv recommended) | macOS / Apple Silicon |
| **vLLM** | `uv pip install vllm` (venv recommended) | Linux + NVIDIA GPU |

Multiple install methods, OS-specific details, verification steps, and common pitfalls are collected in the **[Backend Installation Guide](./install-backends.en.md)**.

> vLLM / MLX need a dedicated Python virtual environment (venv). The policy is to keep a separate venv per backend under `~/.coderouter/backends/<backend-name>/` (e.g. `~/.coderouter/backends/vllm/`). See that guide for details.

---

## 2. Prepare a model

- **llama.cpp** — `.gguf` format. Get one from Hugging Face or elsewhere
- **MLX** — MLX format (distributed under `mlx-community`). Cannot read `.gguf`
- **vLLM** — a Hugging Face model ID, or a local path

Keep local files like `.gguf` in one directory (e.g. `~/llm/models/`). Subfolders are also scanned recursively.

---

## 3. Write the launcher block in providers.yaml

The Launcher loads the model list, option profiles, and binary paths from the `launcher:` block in `~/.coderouter/providers.yaml`.

```yaml
# ~/.coderouter/providers.yaml
launcher:
  model_dirs:
    - ~/llm/models                      # recursively searches for .gguf etc.
  backends:
    llama.cpp:
      # Specify the full path if built from source.
      # For Homebrew / winget installs, `backends` can be omitted entirely
      # (auto-resolved from PATH).
      binary: ~/llama.cpp/build/bin/llama-server
    vllm:
      binary: ~/.coderouter/backends/vllm/bin/python   # the venv where vLLM is installed
  option_profiles:
    llama.cpp:
      - name: "Full GPU utilization"
        args:
          "-ngl": 99
          "--ctx-size": 32768
```

You can start from a copy of the `launcher_profiles.yaml.example` template. For configuration details, see the [Launcher Guide's configuration reference](./launcher.en.md#configuration-reference).

---

## 4. Start the Launcher

There are two kinds of Launcher. The **desktop GUI edition** is easiest for the first run (CodeRouter itself can also be started from there).

### Desktop GUI edition — no browser needed

From the root of the CodeRouter repository:

```bash
python3 launcher_gui.py
# or via CodeRouter's venv (guarantees PyYAML is available)
uv run python launcher_gui.py
```

Once the window opens:

1. Click the model you want to use from MODELS (a `✓ Recommended` one is a safe bet memory-wise)
2. Choose an option profile and press "▶ Launch llama.cpp / vllm / mlx"
3. Press "▶ Start CodeRouter" in the top bar
4. Copy the connection string that appears

See the [Launcher Guide](./launcher.en.md) for details.

### Web edition — operational UI while CodeRouter is running

The Web edition runs inside CodeRouter, so start CodeRouter first:

```bash
coderouter serve --port 8088
```

Open `http://localhost:8088/launcher` in a browser, pick a model, and press "▶ Launch".

A backend started via the Web edition is automatically registered as a provider without editing `providers.yaml` (v2.7.4). Specifying `X-CodeRouter-Profile: launcher` makes it immediately eligible for routing. See the [Launcher Guide](./launcher.en.md) for details.

> **Using llama.cpp with an MTP-capable gguf?** Leave the LAUNCH form's "MTP/draft gguf" field blank and "MTP" set to `auto` to get speculative decoding auto-detected. See [MTP / speculative decoding](./launcher.en.md#mtp--speculative-decoding-llamacpp) for details.

---

## 5. Use it from Claude Code

Once CodeRouter is running, start Claude Code pointed at it:

```bash
ANTHROPIC_BASE_URL=http://localhost:8088 ANTHROPIC_AUTH_TOKEN=dummy claude
```

In the desktop GUI edition, this connection string is displayed at the top of the screen and can be copied from there.

---

## If you get stuck

| Symptom | Fix |
|---|---|
| Launch button is grayed out | The backend's binary can't be found. Set a full path in `backends.<name>.binary` |
| Model list is empty | Set `launcher.model_dirs` and check that it contains `.gguf` etc. |
| `PyYAML not found` (desktop edition) | Run from CodeRouter's venv with `uv run python launcher_gui.py` |
| vLLM is slow/won't run on macOS | vLLM targets Linux/CUDA. Use llama.cpp on macOS |
| A model shows `⚠ Memory tight` | The model is large relative to installed memory. Choose a smaller quantization |

See the [Launcher Guide](./launcher.en.md) for more detailed troubleshooting.

---

## Related docs

- [Backend Installation Guide (llama.cpp / vLLM / MLX)](./install-backends.en.md)
- [Launcher Guide (Web edition / desktop GUI edition)](./launcher.en.md)
- [CodeRouter Quickstart](../start/quickstart.md)
- [llama.cpp direct connection guide](./llamacpp-direct.en.md)
