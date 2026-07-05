#!/usr/bin/env bash
# L3 P1 一括計測スクリプト(直結 vs CodeRouter、temperature=0)
# 実行前提:
#   - リポジトリルートで実行
#   - Ollama 稼働中(localhost:11434)
#   - CodeRouter 稼働中: coderouter serve --port 8088 --config \
#       benchmarks/tool-repair/providers.bench.yaml
# 使い方:
#   bash benchmarks/tool-repair/bench_l3_p1.sh          # P1 のみ
#   P2=1 bash benchmarks/tool-repair/bench_l3_p1.sh    # P2 も実行
set -euo pipefail

BENCH="benchmarks/tool-repair"
REPS="${REPS:-20}"
TEMP="${TEMP:-0}"
ROUTER_URL="${ROUTER_URL:-http://localhost:8088}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"

# model:profile ペア
P1_TARGETS=(
  "llama3.2:3b|bench-llama32-3b"
  "mistral:7b|bench-mistral7b"
  "qwen2.5-coder:1.5b|bench-qwen25-coder-15b"
)
P2_TARGETS=(
  "llama3.1:8b|bench-llama31-8b"
  # gemma3:27b は tools capability 無し(400/502)のため除外。gemma系は gemma4:26b を使用
  "phi4-mini:latest|bench-phi4mini"
)

TARGETS=("${P1_TARGETS[@]}")
[ "${P2:-0}" = "1" ] && TARGETS+=("${P2_TARGETS[@]}")

run_pair() {
  local model="$1" profile="$2"
  echo "=== ${model} / 直結(openai wire) ==="
  python3 "${BENCH}/run_live.py" \
    --base-url "${OLLAMA_URL}" --wire openai \
    --model "${model}" --reps "${REPS}" --temperature "${TEMP}" \
    --tag direct --out-dir "${BENCH}"

  echo "=== ${model} / CodeRouter(anthropic wire, profile=${profile}) ==="
  python3 "${BENCH}/run_live.py" \
    --base-url "${ROUTER_URL}" --wire anthropic \
    --model "${model}" --profile "${profile}" \
    --reps "${REPS}" --temperature "${TEMP}" \
    --tag coderouter --out-dir "${BENCH}"
}

for t in "${TARGETS[@]}"; do
  model="${t%%|*}"; profile="${t##*|}"
  run_pair "${model}" "${profile}"
done

echo "完了。結果: ${BENCH}/results_live_*_{direct,coderouter}.{json,md}"
