#!/usr/bin/env bash
# ============================================================
# CodeRouter E2E — サブエージェント/オーケストレータ × agent_cli 4種
#
#   Phase 0: preflight (CLI / ログイン / Ollama / ポート確認)
#   Phase A: X-CodeRouter-Profile ヘッダ疎通 (OpenAI ingress)   — チャネル②
#   Phase B: model_pattern 振り分け (Anthropic ingress)          — チャネル①の配管
#   Phase C: Claude Code サブエージェント E2E (Task ツール経由)  — 本命
#   Phase D: レポート生成 (results-<ts>/report.md)
#
# 実行:  cd _run/e2e-agents && bash run_e2e.sh
# 環境変数:
#   PORT (既定 8189) / SKIP_C=1 で Phase C をスキップ
#   MAIN_MODEL_ARG (既定 sonnet — Phase C の claude -p --model 値)
#   CODEROUTER_CMD (既定 coderouter — uv run 等に差し替え可)
# ============================================================
set -uo pipefail

PORT="${PORT:-8189}"
MAIN_MODEL_ARG="${MAIN_MODEL_ARG:-sonnet}"
CODEROUTER_CMD="${CODEROUTER_CMD:-coderouter}"
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d-%H%M%S)"
RES="$KIT_DIR/results-$TS"
mkdir -p "$RES"
SERVE_LOG="$RES/serve.log"
REPORT="$RES/report.md"
BASE="http://localhost:$PORT"

PASS_LINES=()
note()  { printf '%s\n' "$*"; }
record() { PASS_LINES+=("$1"); printf '  -> %s\n' "$1"; }

# ---------- JSON 抽出ヘルパ (jq 非依存 / python3 使用) ----------
extract_openai()    { python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"] or "")
except Exception as e: print(f"<<PARSE-ERROR: {e}>>")' ; }
extract_anthropic() { python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print("".join(b.get("text","") for b in d.get("content",[]) if isinstance(b,dict)))
except Exception as e: print(f"<<PARSE-ERROR: {e}>>")' ; }
extract_claude_p()  { python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d.get("result",""))
except Exception as e: print(f"<<PARSE-ERROR: {e}>>")' ; }

# ---------- サブエージェント定義を .claude/agents/ へ展開 ----------
# (リモートツールから .claude/ へ直接書けないため claude-agents/ を staging にしている)
if [ -d "$KIT_DIR/claude-agents" ]; then
  mkdir -p "$KIT_DIR/.claude/agents"
  cp "$KIT_DIR"/claude-agents/*.md "$KIT_DIR/.claude/agents/"
fi

# ============================================================
note "=========================================================="
note " Phase 0: preflight"
note "=========================================================="
FATAL=0
for c in curl python3 "$CODEROUTER_CMD" claude codex grok agy; do
  if command -v "$c" >/dev/null 2>&1; then
    note "[ok]   $c => $(command -v "$c")"
  else
    note "[FATAL] $c が PATH にありません"; FATAL=1
  fi
done

if [ -n "${CLAUDE_CODE_SUBAGENT_MODEL:-}" ]; then
  note "[warn] CLAUDE_CODE_SUBAGENT_MODEL=$CLAUDE_CODE_SUBAGENT_MODEL が設定済み — frontmatter model を上書きするため Phase C では unset して実行します"
fi

# ログイン状態 (軽いものだけ。claude はサブスク前提でスキップ)
if codex login status >/dev/null 2>&1; then note "[ok]   codex login status: ログイン済み"
else note "[warn] codex login status が非0 — 'codex login' を実行してから再試行を推奨"; fi
if grok models >/dev/null 2>&1; then note "[ok]   grok models: 応答あり (認証OK)"
else note "[warn] grok models が失敗 — 'grok login' を確認"; fi
if agy models </dev/null >/dev/null 2>&1; then note "[ok]   agy models: 応答あり (認証OK)"
else note "[warn] agy models が失敗 — agy の初回ログインを確認 (stdin は必ず空で)"; fi

# main 用 Ollama モデル (providers.yaml の main-ollama-9b / main-ollama の model を読む)
# 9B化対応: プロバイダ名を厳密一致 (^  - name: <name>$) で拾ってから、その直後に
# 現れる最初の model: 行を読む。"main-ollama" だけの部分一致だと "main-ollama-9b"
# にもヒットしてしまうため、両方を個別に解決する。
_provider_model() { # $1=provider name (exact)
  awk -v want="$1" '
    /^  - name: / { cur=$3 }
    cur==want && /^    model:/ { print $2; exit }
  ' "$KIT_DIR/providers.yaml"
}
MAIN_OLLAMA_MODEL="$(_provider_model main-ollama)"
MAIN_OLLAMA_MODEL="${MAIN_OLLAMA_MODEL:-qwen3-coder:30b}"
MAIN_OLLAMA_9B_MODEL="$(_provider_model main-ollama-9b)"
if curl -sf --max-time 5 http://localhost:11434/api/tags | grep -q "$(printf '%s' "$MAIN_OLLAMA_MODEL" | cut -d: -f1)"; then
  note "[ok]   Ollama 稼働中 + main モデル系列あり ($MAIN_OLLAMA_MODEL)"
else
  note "[warn] Ollama の $MAIN_OLLAMA_MODEL が見つからない — Phase C の main ループが失敗する可能性。providers.yaml の main-ollama.model を手元のモデルに変更するか 'ollama pull $MAIN_OLLAMA_MODEL'"
fi
if [ -n "$MAIN_OLLAMA_9B_MODEL" ]; then
  if curl -sf --max-time 5 http://localhost:11434/api/tags | grep -q "$(printf '%s' "$MAIN_OLLAMA_9B_MODEL" | cut -d: -f1)"; then
    note "[ok]   Ollama に main-ollama-9b 系列あり ($MAIN_OLLAMA_9B_MODEL) — main chain 先頭候補が有効"
  else
    note "[info] main-ollama-9b の $MAIN_OLLAMA_9B_MODEL が未pull — main chain は自動的に main-ollama ($MAIN_OLLAMA_MODEL) へフォールバックします (致命的ではありません。'ollama pull $MAIN_OLLAMA_9B_MODEL' で有効化可)"
  fi
fi

if lsof -i ":$PORT" >/dev/null 2>&1; then
  note "[FATAL] port $PORT は使用中 — PORT=別番号 で再実行してください"; FATAL=1
fi
[ "$FATAL" -eq 1 ] && { note "preflight FATAL により中断"; exit 1; }

# ============================================================
note ""
note "=========================================================="
note " coderouter serve 起動 (port $PORT)"
note "=========================================================="
( cd "$KIT_DIR" && "$CODEROUTER_CMD" serve --config "$KIT_DIR/providers.yaml" --port "$PORT" --log-level info ) >"$SERVE_LOG" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null' EXIT

UP=0
for _ in $(seq 1 30); do
  if curl -sf --max-time 2 "$BASE/healthz" >/dev/null 2>&1; then UP=1; break; fi
  kill -0 "$SERVE_PID" 2>/dev/null || break
  sleep 1
done
if [ "$UP" -ne 1 ]; then
  note "[FATAL] serve が起動しませんでした。ログ末尾:"
  tail -n 40 "$SERVE_LOG"
  note "(kind: agent_cli 起因なら coderouter-plugin-agents 未導入の可能性 — README 参照)"
  exit 1
fi
note "[ok]   serve 起動確認 (/healthz)"

# ============================================================
note ""
note "=========================================================="
note " Phase A: X-CodeRouter-Profile ヘッダ疎通 (OpenAI ingress)"
note "=========================================================="
phase_a() { # $1=profile $2=marker $3=outfile-suffix
  local profile="$1" marker="$2" out="$RES/phaseA-$3.json" body content
  body=$(printf '{"model":"x","messages":[{"role":"user","content":"Reply with exactly this single token and nothing else: %s"}]}' "$marker")
  curl -s --max-time 330 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "X-CodeRouter-Profile: $profile" \
    -d "$body" >"$out" 2>&1
  content="$(extract_openai <"$out")"
  printf '  [%s] 応答: %.120s\n' "$profile" "$content"
  if printf '%s' "$content" | grep -q "$marker"; then
    record "PASS PhaseA/$profile — マーカー $marker を確認"
  else
    record "FAIL PhaseA/$profile — マーカー $marker 不在 (raw: $out)"
  fi
}
phase_a claude-agent PONG-CLAUDE a-claude
phase_a codex        PONG-CODEX  a-codex
phase_a grok         PONG-GROK   a-grok
phase_a antigravity  PONG-AGY    a-agy

# ============================================================
note ""
note "=========================================================="
note " Phase B: model_pattern 振り分け (Anthropic ingress /v1/messages)"
note "=========================================================="
phase_b() { # $1=model $2=rule_id $3=suffix
  local model="$1" rule="$2" out="$RES/phaseB-$3.json" body content routed
  body=$(printf '{"model":"%s","max_tokens":128,"messages":[{"role":"user","content":"What is 6*7? Reply with the number only."}]}' "$model")
  curl -s --max-time 330 "$BASE/v1/messages" \
    -H 'Content-Type: application/json' \
    -H 'anthropic-version: 2023-06-01' \
    -d "$body" >"$out" 2>&1
  content="$(extract_anthropic <"$out")"
  printf '  [model=%s] 応答: %.120s\n' "$model" "$content"
  routed=$(grep -c "\"rule_id\": *\"$rule\"\|\"rule_id\":\"$rule\"" "$SERVE_LOG" || true)
  if printf '%s' "$content" | grep -q '42' && [ "$routed" -ge 1 ]; then
    record "PASS PhaseB/$model — 応答42 + ルール $rule 発火 (auto-router-resolved x$routed)"
  elif [ "$routed" -ge 1 ]; then
    record "FAIL PhaseB/$model — ルール $rule は発火したが応答が不正 (raw: $out)"
  else
    record "FAIL PhaseB/$model — ルール $rule が serve ログに見当たらない"
  fi
}
phase_b e2e-claude "e2e:claude" b-claude
phase_b e2e-codex  "e2e:codex"  b-codex
phase_b e2e-grok   "e2e:grok"   b-grok
phase_b e2e-agy    "e2e:agy"    b-agy

# ============================================================
note ""
note "=========================================================="
note " Phase C: Claude Code サブエージェント E2E (Task ツール)"
note "=========================================================="
if [ "${SKIP_C:-0}" = "1" ]; then
  note "  SKIP_C=1 のためスキップ"
else
  # v2 (リトライ化): 最大3回まで再試行し、1回でも "144" が出力に含まれれば PASS
  # とする (クライアント側 flaky 対策)。試行2回目以降の out/err は -try2/-try3
  # サフィックス付きの別ファイルに残す (1回目は従来どおり phaseC-<suffix>.json/.err)。
  # 各試行後、当該試行中に該当 auto_router ルール (rule_id) が serve ログへ発火
  # したかを確認し、3回とも失敗した場合は「ルーティング障害(ルール未発火)」か
  # 「クライアント側flaky(発火したが応答未着/未発行)」かを record に明示する。
  phase_c() { # $1=subagent $2=marker $3=suffix $4=auto_router rule_id
    local name="$1" marker="$2" suffix="$3" rule="$4"
    local attempt out err result found=0 routed_any=0 serve_lines_before
    for attempt in 1 2 3; do
      if [ "$attempt" -eq 1 ]; then
        out="$RES/phaseC-$suffix.json"; err="$RES/phaseC-$suffix.err"
      else
        out="$RES/phaseC-$suffix-try$attempt.json"; err="$RES/phaseC-$suffix-try$attempt.err"
      fi
      note "  [$name] claude -p 実行中 (試行 $attempt/3, main=$MAIN_MODEL_ARG, 数分かかることあり)..."
      serve_lines_before=$(wc -l <"$SERVE_LOG" 2>/dev/null || echo 0)
      ( cd "$KIT_DIR" && env -u CLAUDE_CODE_SUBAGENT_MODEL -u ANTHROPIC_API_KEY \
          ANTHROPIC_BASE_URL="$BASE" ANTHROPIC_AUTH_TOKEN=dummy \
          claude -p "Use the Task tool to launch the '$name' subagent with exactly this prompt: \"What is 12*12? Reply with the number only.\" After the subagent returns, output its full answer verbatim. Do not answer the question yourself. After calling the Task tool once, WAIT for its completion notification. Do NOT poll with SendMessage or TaskOutput. Do NOT claim the agent is unreachable - it exists in your agent list." \
          --model "$MAIN_MODEL_ARG" --allowedTools Task --output-format json \
        ) >"$out" 2>"$err"
      result="$(extract_claude_p <"$out")"
      printf '  [%s] (試行 %d/3) orchestrator 出力: %.200s\n' "$name" "$attempt" "$result"
      if tail -n "+$((serve_lines_before + 1))" "$SERVE_LOG" 2>/dev/null | grep -q "\"rule_id\": *\"$rule\"\|\"rule_id\":\"$rule\""; then
        routed_any=1
      fi
      if printf '%s' "$result" | grep -q '144'; then
        found=1
        break
      fi
    done
    if [ "$found" -eq 1 ]; then
      if printf '%s' "$result" | grep -q "$marker"; then
        record "PASS PhaseC/$name — 144 + マーカー $marker (system prompt 貫通も確認, try $attempt/3)"
      else
        record "PASS PhaseC/$name — 144 を確認 (マーカー $marker は非表示 — 情報のみ, try $attempt/3)"
      fi
    else
      if [ "$routed_any" -eq 1 ]; then
        record "FAIL PhaseC/$name — 144 が出力に無い (3試行とも失敗; ルール $rule は発火済み — クライアント側flaky疑い; out: $out / err: $err)"
      else
        record "FAIL PhaseC/$name — 144 が出力に無い (3試行とも失敗; ルール $rule が serve ログに未発火 — ルーティング障害疑い; out: $out / err: $err)"
      fi
    fi
  }
  phase_c ext-claude CLAUDE-OK c-claude e2e:claude
  phase_c ext-codex  CODEX-OK  c-codex  e2e:codex
  phase_c ext-grok   GROK-OK   c-grok   e2e:grok
  phase_c ext-agy    AGY-OK    c-agy    e2e:agy
fi

# ============================================================
note ""
note "=========================================================="
note " Phase D: レポート生成"
note "=========================================================="
{
  echo "# CodeRouter E2E レポート — $TS"
  echo
  echo "- kit: $KIT_DIR / port: $PORT / main(--model): $MAIN_MODEL_ARG"
  echo "- coderouter: $("$CODEROUTER_CMD" --version 2>/dev/null || echo '?')"
  echo "- claude: $(claude --version 2>/dev/null | head -1 || echo '?') / codex: $(codex --version 2>/dev/null | head -1 || echo '?')"
  echo "- grok: $(grok --version 2>/dev/null | head -1 || echo '?') / agy: $(agy --version </dev/null 2>/dev/null | head -1 || echo '?')"
  echo
  echo "## 結果"
  echo
  for l in "${PASS_LINES[@]}"; do echo "- $l"; done
  echo
  echo "## auto-router-resolved イベント (signals.model の実測値 — UNCONFIRMED 事項の確定用)"
  echo
  echo '```'
  grep 'auto-router-resolved' "$SERVE_LOG" | tail -n 40 || echo "(なし)"
  echo '```'
  echo
  echo "## capability / fallback / agent_cli 関連イベント"
  echo
  echo '```'
  grep -E 'capability-degraded|fallback|agent-cli|adapter-error|AdapterError' "$SERVE_LOG" | tail -n 60 || echo "(なし)"
  echo '```'
} >"$REPORT"

note ""
note "レポート: $REPORT"
note "serve ログ: $SERVE_LOG"
FAILS=$(printf '%s\n' "${PASS_LINES[@]}" | grep -c '^FAIL' || true)
note ""
if [ "$FAILS" -eq 0 ]; then
  note "★ ALL PASS — report.md と serve.log をそのまま Claude に共有してください"
else
  note "★ FAIL x$FAILS — results-$TS/ フォルダごと Claude に共有してください (判定します)"
fi
