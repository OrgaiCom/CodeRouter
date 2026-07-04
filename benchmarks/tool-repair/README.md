# Tool-call repair benchmark

Measures how well CodeRouter's `repair_tool_calls_in_text`
(`coderouter/translation/tool_repair.py`) rescues tool calls that local
models write into assistant *text* instead of the structured `tool_calls`
field — and how often a live model needs that rescue at all.

Two layers:

- **L1 — `run_offline.py`**: deterministic, no network, stdlib only. Runs the
  repairer over a fixed corpus of broken outputs (`corpus.jsonl`, 55 cases in
  8 categories + 12 negatives) and scores **recovery** and **false
  positives**. This is the regression gate for the repairer itself; it also
  runs in CI via `tests/test_toolrepair_bench.py`.
- **L2 — `run_live.py`**: talks to a real endpoint (a backend directly, or
  CodeRouter in front of it), applies the same 3-value verdict CodeRouter's
  `doctor` uses (**native** / **repair** / **fail**), and reports per-model
  rates.

Both load `tool_repair.py` by file path (`--tool-repair` to override), so you
can point the benchmark at any branch's repairer and diff before/after on the
same corpus.

## L1 — offline

```bash
python benchmarks/tool-repair/run_offline.py
```

Writes `results_offline.json` / `results_offline.md` next to the script.

Outcome classes: `recovered` (expected repair, got the right call),
`correct_pass` (expected NO repair, repairer stayed quiet), `missed`
(expected repair, got nothing/wrong), `false_positive` (repairer fabricated
a call — the dangerous direction; the `negative` category guards it and must
stay at zero).

Corpus policy: cases the current repairer *cannot* fix are still marked
`expect.repaired: true` so they show up as `missed` — the benchmark surfaces
gaps instead of rubber-stamping today's behaviour.

## L2 — live

```bash
# Self-test (no server needed) — run this first:
python benchmarks/tool-repair/run_live.py --dry-run

# Backend directly (Ollama's OpenAI-compatible endpoint):
python benchmarks/tool-repair/run_live.py \
  --base-url http://localhost:11434/v1 --wire openai \
  --model qwen2.5-coder:7b --reps 20 --tag direct

# Through CodeRouter (start it with the bundled bench config first):
#   coderouter serve --port 8088 --config benchmarks/tool-repair/providers.bench.yaml
python benchmarks/tool-repair/run_live.py \
  --base-url http://localhost:8088 --wire anthropic \
  --model qwen2.5-coder:7b --profile bench-qwen7b --reps 20 --tag coderouter
```

Notes:

- CodeRouter treats the client-sent `model` as a routing placeholder (the
  provider's configured model wins), so on the router path you select the
  backend model with `--profile` (see `providers.bench.yaml`).
- `--temperature 0` is the default so a direct-vs-router comparison measures
  the path, not the sampler. Use `--temperature none` for backend-default
  sampling.
- Requires `httpx` (already a CodeRouter runtime dep).

## Measured results (2026-07-04, M3 Max, 100 requests per cell)

Kept under `results/` as evidence for the write-ups.

| model | direct | via CodeRouter | reading |
|---|:-:|:-:|---|
| qwen2.5-coder:7b | 0% usable calls | **100%** | weak models: repair does all the work |
| qwen3-coder:30b | 100% | **100%** | strong models: zero degradation |
| gemma4:26b | 80% | 80% | empty responses — repair can't fix absent text (fallback territory) |

Offline (post-v2.7.1 repairer): recall **100%** (43/43), false positives
**0/12**. The pre-v2.7.1 repairer scored 80.6% overall and 14.3% on the
`malformed` category — that gap is what drove the lenient-repair upgrade
shipped in v2.7.1.

## Growing the corpus

Real-world broken tool-call outputs are welcome — add a line to
`corpus.jsonl` with an honest `expect` and a `note` explaining the failure
mode, and keep the `negative` cases passing (false positives stay at zero,
no exceptions).
