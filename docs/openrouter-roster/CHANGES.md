# OpenRouter free-tier roster — change log

Appended by `scripts/openrouter_roster_diff.py` on each run. Newest
entries appear at the top. Each section records a delta between two
consecutive snapshots — not a cumulative list of free-tier models.
For the current list, see `latest.json` in this directory.

## 2026-07-13T00:07:11Z

### Removed (1) ⚠️

- `poolside/laguna-xs.2:free` (was: ctx=262144, prompt=0, completion=0)

### Added (1)

- `tencent/hy3:free` (ctx=262144)

## 2026-07-06T00:07:28Z

### Removed (6) ⚠️

- `arcee-ai/trinity-large-thinking:free` (was: ctx=262144, prompt=0, completion=0)
- `baidu/cobuddy:free` (was: ctx=131072, prompt=0, completion=0)
- `deepseek/deepseek-v4-flash:free` (was: ctx=1048576, prompt=0, completion=0)
- `minimax/minimax-m2.5:free` (was: ctx=204800, prompt=0, completion=0)
- `openrouter/owl-alpha` (was: ctx=1048756, prompt=0, completion=0)
- `z-ai/glm-4.5-air:free` (was: ctx=131072, prompt=0, completion=0)

### Added (4)

- `cohere/north-mini-code:free` (ctx=256000)
- `nvidia/nemotron-3-ultra-550b-a55b:free` (ctx=1000000)
- `nvidia/nemotron-3.5-content-safety:free` (ctx=128000)
- `poolside/laguna-xs-2.1:free` (ctx=262144)

### Context changed (2)

- `poolside/laguna-m.1:free`: 131072 → 262144
- `poolside/laguna-xs.2:free`: 131072 → 262144

## 2026-05-18T01:21:41Z

### Removed (8) ⚠️

- `baidu/qianfan-ocr-fast:free` (was: ctx=65536, prompt=0, completion=0)
- `google/gemma-3-12b-it:free` (was: ctx=32768, prompt=0, completion=0)
- `google/gemma-3-27b-it:free` (was: ctx=131072, prompt=0, completion=0)
- `google/gemma-3-4b-it:free` (was: ctx=32768, prompt=0, completion=0)
- `google/gemma-3n-e2b-it:free` (was: ctx=8192, prompt=0, completion=0)
- `google/gemma-3n-e4b-it:free` (was: ctx=8192, prompt=0, completion=0)
- `inclusionai/ling-2.6-1t:free` (was: ctx=262144, prompt=0, completion=0)
- `tencent/hy3-preview:free` (was: ctx=262144, prompt=0, completion=0)

### Added (3)

- `arcee-ai/trinity-large-thinking:free` (ctx=262144)
- `baidu/cobuddy:free` (ctx=131072)
- `deepseek/deepseek-v4-flash:free` (ctx=1048576)

### Context changed (4)

- `meta-llama/llama-3.3-70b-instruct:free`: 65536 → 131072
- `minimax/minimax-m2.5:free`: 196608 → 204800
- `nvidia/nemotron-3-super-120b-a12b:free`: 262144 → 1000000
- `qwen/qwen3-coder:free`: 262000 → 1048576

## 2026-05-04T03:31:58Z

### Removed (2) ⚠️

- `arcee-ai/trinity-large-preview:free` (was: ctx=131000, prompt=0, completion=0)
- `openrouter/elephant-alpha` (was: ctx=262144, prompt=0, completion=0)

### Added (7)

- `baidu/qianfan-ocr-fast:free` (ctx=65536)
- `inclusionai/ling-2.6-1t:free` (ctx=262144)
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` (ctx=256000)
- `openrouter/owl-alpha` (ctx=1048756)
- `poolside/laguna-m.1:free` (ctx=131072)
- `poolside/laguna-xs.2:free` (ctx=131072)
- `tencent/hy3-preview:free` (ctx=262144)

