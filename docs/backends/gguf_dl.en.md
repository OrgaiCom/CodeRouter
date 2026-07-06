# gguf_dl.py

> 日本語版: [`gguf_dl.md`](./gguf_dl.md)

A helper script for downloading GGUF (and other) files from Hugging Face into a local folder. Just paste a `huggingface.co` URL and it auto-parses the `repo_id` / revision / filename; it also supports an interactive mode and bulk downloads.

It's a thin wrapper around the new CLI `hf` (formerly `huggingface-cli`), but saves you the trouble of building a command from scratch every time, thanks to three things:

- **Works from a pasted URL** (`hf download` requires the `repo_id` form)
- **Drops into interactive mode with no arguments**
- **A fallback chain for the destination** (`--dest` → `$GGUF_DL_DIR` → `./models/`)

---

## Requirements

- Python 3.8 or later
- `huggingface_hub` (the new `hf` CLI ships in the same package)

Installing `hf_transfer` optionally enables faster parallel downloads.

```bash
pip install --upgrade "huggingface_hub[hf_transfer]"
```

For private repos, set a token with one of the following:

```bash
hf auth login            # recommended. Enter the token interactively
export HF_TOKEN=hf_xxxxx # or as an environment variable
python gguf_dl.py --token hf_xxxxx ...   # or pass it as a script argument
```

> The old `huggingface-cli login` has been replaced by `hf auth login`.

---

## Quickstart

```bash
# 1. A single file (just paste the URL)
python gguf_dl.py https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf

# 2. Specify repo_id and filename separately
python gguf_dl.py TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf

# 3. Specify a destination
python gguf_dl.py <URL> -d ~/models/gguf

# 4. Bulk-download multiple files by pattern (split GGUFs supported too)
python gguf_dl.py bartowski/Qwen3-30B-A3B-GGUF -p "*Q4_K_M*.gguf"

# 5. Just list the files in a repo
python gguf_dl.py bartowski/Qwen3-30B-A3B-GGUF --list

# 6. No arguments → interactive mode
python gguf_dl.py
```

---

## Accepted input formats

You can pass any of the following as the first argument (or at the interactive prompt):

| Format | Example |
| --- | --- |
| File blob URL | `https://huggingface.co/<owner>/<repo>/blob/<rev>/<path>` |
| File resolve URL | `https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path>` |
| Tree URL | `https://huggingface.co/<owner>/<repo>/tree/<rev>` |
| Repo root URL | `https://huggingface.co/<owner>/<repo>` |
| Bare repo_id | `<owner>/<repo>` |

If the URL includes a file path or revision, it's auto-parsed and you can omit the second argument or `--revision`.

---

## Options

| Option | Description |
| --- | --- |
| `target` | A URL or `<owner>/<repo>` (omit to enter interactive mode) |
| `filename` | The file to download (omit if included in the URL) |
| `-d, --dest <DIR>` | Destination folder |
| `-r, --revision <REV>` | Branch/tag/commit (default: from URL, or `main`) |
| `-p, --pattern <GLOB>` | Wildcard spec (repeatable, e.g. `-p '*Q4_K_M*.gguf'`) |
| `--list` | Only list the files in the repo |
| `--nested` | Save in HF cache format (snapshots layout). Default is flat layout |
| `--token <TOKEN>` | Token for private repos (the `HF_TOKEN` env var also works) |
| `-y, --yes` | Skip confirmation prompts |

### Destination resolution order

1. Location explicitly given via `--dest`
2. The `GGUF_DL_DIR` environment variable
3. `./models/` under the current directory

Created automatically if it doesn't exist.

---

## Interactive mode

Running with no arguments asks the following questions in order:

1. URL or `<owner>/<repo>`
2. Revision (default: `main`)
3. How to specify files (`single` / `pattern` / `all-gguf`)
4. Destination folder

Step 3 is skipped if the URL already includes a filename.

---

## Resuming and speeding up downloads

- **Resume**: `huggingface_hub`'s own mechanism auto-detects via ETag/size. Re-running the same command continues from where it left off.
- **Speedup**: if `hf_transfer` is installed, parallel transfer is enabled automatically (it sets the `HF_HUB_ENABLE_HF_TRANSFER=1` environment variable internally). Faster networks benefit more.
- **Timeout**: on slow connections, if you see `httpx.TimeoutException`, extend it with e.g. `export HF_HUB_DOWNLOAD_TIMEOUT=60` (default 10 seconds).

---

## File placement

Under the default **flat layout** (using `local_dir`), paths inside the repo are expanded as-is directly under `<dest>/`. For example, fetching `text_encoder/model.safetensors` produces `<dest>/text_encoder/model.safetensors`.

With `--nested`, files are saved in HF cache format under `<dest>/.hf_cache/`. This is better suited for workflows that switch between multiple revisions, or want to share blobs across multiple models.

---

## Usage examples

### Fetching a split GGUF (multi-part file) as a set

```bash
python gguf_dl.py bartowski/Some-Big-Model-GGUF \
    -p "*Q4_K_M*.gguf-*-of-*" \
    -d ~/models/gguf
```

### A private repo

```bash
hf auth login          # log in once
python gguf_dl.py myorg/private-llama -p "*.gguf"
```

### Just browsing the list to pick what you want

```bash
python gguf_dl.py TheBloke/Llama-2-7B-GGUF --list
# Look at the output, then re-run for the quantization you want
python gguf_dl.py TheBloke/Llama-2-7B-GGUF llama-2-7b.Q5_K_M.gguf
```

---

## Troubleshooting

**`huggingface_hub is not installed`**
→ Run `pip install --upgrade "huggingface_hub[hf_transfer]"`.

**Fails with 401 / 403**
→ It's a private or gated repo. Either log in with `hf auth login`, or agree to the model card's terms on the web and then pass `--token` or `HF_TOKEN`.

**It stopped partway / I want to run it again**
→ Just re-run the same command and it resumes from where it stopped. To start completely fresh, delete the destination files (and anything under `.hf_cache/`).

**Nothing matches the pattern**
→ Check the filenames with `--list`. `-p` patterns match against both the full path and the filename (e.g. `sub/file.gguf` can be matched by either `sub/*.gguf` or `*.gguf`).

**`hf download` alone seems like it would be enough**
→ For simple cases, the new CLI's `hf download <repo> --include "*.gguf" --local-dir <dir>` does the equivalent. What this script adds is "URL paste support," "interactive mode," and "a default-destination fallback chain."

---

## What it does internally

Roughly, it does the following:

1. Parses the input string with a regex into `(repo_id, revision, filename)`
2. Fetches the file list via `HfApi.repo_info()` if needed
3. Filters with `fnmatch` using `--pattern`
4. Fetches each file with `hf_hub_download(local_dir=...)` (resuming and progress bars are built in)

---

## License

This script is a personal helper tool — feel free to modify and redistribute it freely.
