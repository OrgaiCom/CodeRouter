#!/usr/bin/env bash
# ============================================================
# CodeRouter swap-test — launcher.swap (Phase 1) 実機テスト
#
#   Phase 0: preflight (coderouter修正版 / llama-server / GGUF / ポート)
#   Phase A: コールドスタート — 未起動モデルへのリクエストで自動spawn+応答
#   Phase B: ウォーム — 2回目は再spawnせず高速応答
#   Phase C: カタログ外model名 — フォールスルー先(swapプロファイル)で応答
#            (=カタログ非一致modelでもリース保護される経路の確認)
#   Phase D: TTLアンロード — アイドル @TTL@ 秒で自動停止(swap-unload)
#   Phase E: 再spawn — アンロード後のリクエストで再び自動起動
#   Phase F: レポート生成 (results-<ts>/report.md)
#
# 実行:  cd _run/swap-test && bash run_swap_test.sh
# 環境変数:
#   GGUF_PATH    テストに使うGGUF(既定: ~/models と repo models/ から自動探索。
#                小さいモデルほどテストが速い)
#   LLAMA_SERVER llama-serverバイナリ(既定: PATHから探索)
#   PORT         CodeRouterポート(既定 8288)
#   MODEL_PORT   swapモデルのポート(既定 18081)
#   TTL          アイドルアンロード秒(既定 25)
#   CODEROUTER_CMD (既定: "uv run coderouter" — 作業ツリーの修正版を使う)
# ============================================================
set -uo pipefail

PORT="${PORT:-8288}"
MODEL_PORT="${MODEL_PORT:-18081}"
TTL="${TTL:-25}"
CODEROUTER_CMD="${CODEROUTER_CMD:-uv run coderouter}"
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$KIT_DIR/../.." && pwd)"
TS="$(date +%Y%m%d-%H%M%S)"
RES="$KIT_DIR/results-$TS"
mkdir -p "$RES"
SERVE_LOG="$RES/serve.log"
REPORT="$RES/report.md"
BASE="http://localhost:$PORT"
MODEL_NAME="swap-test-model"
SWAP_PROVIDER="launcher-swap-$MODEL_NAME"

PASS_LINES=()
note()   { printf '%s\n' "$*"; }
record() { PASS_LINES+=("$1"); printf '  -> %s\n' "$1"; }

extract_anthropic() { python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print("".join(b.get("text","") for b in d.get("content",[]) if isinstance(b,dict)))
except Exception as e: print(f"<<PARSE-ERROR: {e}>>")' ; }
extract_provider() { python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d.get("coderouter_provider",""))
except Exception: print("")' ; }

model_port_listening() { lsof -i ":$MODEL_PORT" -sTCP:LISTEN >/dev/null 2>&1; }

ask() { # $1=model $2=outfile — /v1/messages に投げて応答秒数をSECS変数に入れる
  local model="$1" out="$2" t0 t1
  t0=$(date +%s)
  curl -s --max-time 330 "$BASE/v1/messages" \
    -H 'Content-Type: application/json' \
    -H 'anthropic-version: 2023-06-01' \
    -d "$(printf '{"model":"%s","max_tokens":32,"messages":[{"role":"user","content":"Reply with the single word: pong"}]}' "$model")" >"$out" 2>&1
  t1=$(date +%s); SECS=$((t1 - t0))
}

# ============================================================
note "=========================================================="
note " Phase 0: preflight"
note "=========================================================="
FATAL=0

# 修正版が入っているか(coderouter.launcher_swap の存在で判定)
if (cd "$REPO_DIR" && $CODEROUTER_CMD --version >/dev/null 2>&1); then
  VER="$(cd "$REPO_DIR" && $CODEROUTER_CMD --version 2>/dev/null | head -1)"
  note "[ok]   coderouter: $VER"
else
  note "[FATAL] '$CODEROUTER_CMD' が実行できません — リポジトリ直下で 'uv sync' 済みか確認"; FATAL=1
fi
if (cd "$REPO_DIR" && uv run python -c 'import coderouter.launcher_swap' 2>/dev/null); then
  note "[ok]   coderouter.launcher_swap (swap修正版) を確認"
else
  note "[FATAL] coderouter.launcher_swap が import できません — 修正版の作業ツリーで実行していますか?"; FATAL=1
fi

# llama-server (探索順: 環境変数 → PATH → 既知のビルド場所)
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
if [ -z "$LLAMA_SERVER" ] && [ -x "$HOME/llm/apps/llama.cpp/build/bin/llama-server" ]; then
  LLAMA_SERVER="$HOME/llm/apps/llama.cpp/build/bin/llama-server"
fi
if [ -n "$LLAMA_SERVER" ] && [ -x "$LLAMA_SERVER" ]; then
  note "[ok]   llama-server: $LLAMA_SERVER"
else
  note "[FATAL] llama-server が見つかりません — LLAMA_SERVER=/path/to/llama-server を指定"; FATAL=1
fi

# GGUF 自動探索(未指定時): ~/models → repo models/ の順で最小サイズのものを選ぶ
if [ -z "${GGUF_PATH:-}" ]; then
  GGUF_PATH="$( { ls -S -r "$HOME"/models/*.gguf 2>/dev/null; ls -S -r "$REPO_DIR"/models/*.gguf 2>/dev/null; } | head -1 || true)"
fi
if [ -n "${GGUF_PATH:-}" ] && [ -f "$GGUF_PATH" ]; then
  note "[ok]   GGUF: $GGUF_PATH ($(du -h "$GGUF_PATH" | cut -f1))"
else
  note "[FATAL] GGUF が見つかりません — GGUF_PATH=/path/to/model.gguf を指定 (小さいモデル推奨)"; FATAL=1
fi

for p in "$PORT" "$MODEL_PORT"; do
  if lsof -i ":$p" >/dev/null 2>&1; then
    note "[FATAL] port $p は使用中 — PORT/MODEL_PORT で変更してください"; FATAL=1
  fi
done
[ "$FATAL" -eq 1 ] && { note "preflight FATAL により中断"; exit 1; }

MODEL_DIR="$(cd "$(dirname "$GGUF_PATH")" && pwd)"
GGUF_ABS="$MODEL_DIR/$(basename "$GGUF_PATH")"

# providers.yaml 生成
GEN="$RES/providers.generated.yaml"
sed -e "s|@GGUF_PATH@|$GGUF_ABS|" \
    -e "s|@MODEL_DIR@|$MODEL_DIR|" \
    -e "s|@LLAMA_SERVER@|$LLAMA_SERVER|" \
    -e "s|@MODEL_PORT@|$MODEL_PORT|" \
    -e "s|@TTL@|$TTL|" \
    "$KIT_DIR/providers.tpl.yaml" >"$GEN"
note "[ok]   設定生成: $GEN"

# ============================================================
note ""
note "=========================================================="
note " coderouter serve 起動 (port $PORT)"
note "=========================================================="
( cd "$REPO_DIR" && $CODEROUTER_CMD serve --config "$GEN" --port "$PORT" --log-level info ) >"$SERVE_LOG" 2>&1 &
SERVE_PID=$!
# 後始末: $! はサブシェルのPID。coderouter 本体と、swapが起動した
# llama-server(MODEL_PORT)まで、ポート基準で確実に始末する。
cleanup_serve() {
  kill "$SERVE_PID" 2>/dev/null
  wait "$SERVE_PID" 2>/dev/null
  local p pids
  for p in "$PORT" "$MODEL_PORT"; do
    pids="$(lsof -ti ":$p" 2>/dev/null)"
    if [ -n "$pids" ]; then
      kill $pids 2>/dev/null
      sleep 2
      pids="$(lsof -ti ":$p" 2>/dev/null)"
      [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    fi
  done
  return 0
}
trap cleanup_serve EXIT

UP=0
for _ in $(seq 1 30); do
  curl -sf --max-time 2 "$BASE/healthz" >/dev/null 2>&1 && { UP=1; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || break
  sleep 1
done
if [ "$UP" -ne 1 ]; then
  note "[FATAL] serve が起動しませんでした。ログ末尾:"; tail -n 30 "$SERVE_LOG"; exit 1
fi
note "[ok]   serve 起動確認 (/healthz)"
if model_port_listening; then
  note "[FATAL] serve直後に $MODEL_PORT が既にLISTEN — オンデマンドのはずが先行起動している"; exit 1
fi
note "[ok]   swapモデルは未起動 (port $MODEL_PORT 閉、オンデマンド待機)"

# ============================================================
note ""
note "=========================================================="
note " Phase A: コールドスタート (自動spawn + 応答)"
note "=========================================================="
ask "$MODEL_NAME" "$RES/phaseA.json"; COLD_SECS=$SECS
A_TEXT="$(extract_anthropic <"$RES/phaseA.json")"
A_PROV="$(extract_provider <"$RES/phaseA.json")"
printf '  応答(%.60s...) provider=%s %ss\n' "$A_TEXT" "$A_PROV" "$COLD_SECS"
if [ -n "$A_TEXT" ] && [ "$A_PROV" = "$SWAP_PROVIDER" ] && model_port_listening \
   && grep -q '"rule_id": *"swap:'"$MODEL_NAME"'"' "$SERVE_LOG"; then
  record "PASS PhaseA/cold-spawn — 自動起動+応答 (provider=$SWAP_PROVIDER, ${COLD_SECS}s, ルールswap:${MODEL_NAME} 発火)"
else
  record "FAIL PhaseA/cold-spawn — text='${A_TEXT:0:40}' provider='$A_PROV' port_listen=$(model_port_listening && echo yes || echo no) (raw: $RES/phaseA.json)"
fi

# ============================================================
note ""
note "=========================================================="
note " Phase B: ウォーム (再spawnなし)"
note "=========================================================="
ask "$MODEL_NAME" "$RES/phaseB.json"; WARM_SECS=$SECS
B_TEXT="$(extract_anthropic <"$RES/phaseB.json")"
NPROC=$(lsof -t -i ":$MODEL_PORT" -sTCP:LISTEN 2>/dev/null | sort -u | wc -l | tr -d ' ')
printf '  応答(%.40s...) %ss (cold %ss) listenプロセス数=%s\n' "$B_TEXT" "$WARM_SECS" "$COLD_SECS" "$NPROC"
if [ -n "$B_TEXT" ] && [ "$NPROC" = "1" ]; then
  record "PASS PhaseB/warm — 応答あり・プロセス1個のまま (warm ${WARM_SECS}s vs cold ${COLD_SECS}s)"
else
  record "FAIL PhaseB/warm — text空 or プロセス数=$NPROC (raw: $RES/phaseB.json)"
fi

# ============================================================
note ""
note "=========================================================="
note " Phase C: カタログ外model名 (フォールスルー+リース保護)"
note "=========================================================="
ask "totally-unknown-model" "$RES/phaseC.json"
C_TEXT="$(extract_anthropic <"$RES/phaseC.json")"
C_PROV="$(extract_provider <"$RES/phaseC.json")"
printf '  応答(%.40s...) provider=%s\n' "$C_TEXT" "$C_PROV"
if [ -n "$C_TEXT" ] && [ "$C_PROV" = "$SWAP_PROVIDER" ]; then
  record "PASS PhaseC/alien-model — カタログ外model名がフォールスルー先(swap)で応答"
else
  record "FAIL PhaseC/alien-model — text='${C_TEXT:0:40}' provider='$C_PROV' (raw: $RES/phaseC.json)"
fi

# ============================================================
note ""
note "=========================================================="
note " Phase D: TTLアンロード (アイドル ${TTL}s + sweep 5s)"
note "=========================================================="
DEADLINE=$((TTL + 40)); UNLOADED=0; WAITED=0
note "  アイドル待機中 (最大 ${DEADLINE}s)..."
while [ "$WAITED" -lt "$DEADLINE" ]; do
  sleep 5; WAITED=$((WAITED + 5))
  if grep -q 'swap-unload' "$SERVE_LOG" && ! model_port_listening; then UNLOADED=1; break; fi
done
if [ "$UNLOADED" -eq 1 ]; then
  record "PASS PhaseD/ttl-unload — ${WAITED}s で swap-unload 発火 + port $MODEL_PORT 解放"
else
  record "FAIL PhaseD/ttl-unload — ${DEADLINE}s 待っても unload されない (swap-unload: $(grep -c 'swap-unload' "$SERVE_LOG" || true)件, port_listen=$(model_port_listening && echo yes || echo no))"
fi

# ============================================================
note ""
note "=========================================================="
note " Phase E: 再spawn (アンロード後の復帰)"
note "=========================================================="
ask "$MODEL_NAME" "$RES/phaseE.json"; RESPAWN_SECS=$SECS
E_TEXT="$(extract_anthropic <"$RES/phaseE.json")"
printf '  応答(%.40s...) %ss\n' "$E_TEXT" "$RESPAWN_SECS"
if [ -n "$E_TEXT" ] && model_port_listening; then
  record "PASS PhaseE/respawn — アンロード後に再び自動起動して応答 (${RESPAWN_SECS}s)"
else
  record "FAIL PhaseE/respawn — text='${E_TEXT:0:40}' (raw: $RES/phaseE.json)"
fi

# ============================================================
note ""
note "=========================================================="
note " Phase F: レポート生成"
note "=========================================================="
{
  echo "# CodeRouter swap-test レポート — $TS"
  echo
  echo "- kit: $KIT_DIR / port: $PORT / model_port: $MODEL_PORT / ttl: ${TTL}s"
  echo "- coderouter: ${VER:-?}"
  echo "- llama-server: $LLAMA_SERVER"
  echo "- GGUF: $GGUF_ABS"
  echo "- 所要: cold=${COLD_SECS:-?}s / warm=${WARM_SECS:-?}s / respawn=${RESPAWN_SECS:-?}s"
  echo
  echo "## 結果"
  echo
  for l in "${PASS_LINES[@]}"; do echo "- $l"; done
  echo
  echo "## swap / auto-router 関連イベント"
  echo
  echo '```'
  grep -E 'swap-unload|"rule_id": *"swap:|launcher-swap|auto-router-resolved' "$SERVE_LOG" | tail -n 40 || echo "(なし)"
  echo '```'
} >"$REPORT"

note ""
note "レポート: $REPORT"
note "serve ログ: $SERVE_LOG"
FAILS=$(printf '%s\n' "${PASS_LINES[@]}" | grep -c '^FAIL' || true)
note ""
if [ "$FAILS" -eq 0 ]; then
  note "★ ALL PASS (5/5) — swap Phase 1 は実機で機能しています"
else
  note "★ FAIL x$FAILS — results-$TS/ フォルダごと Claude に共有してください (判定します)"
fi
