#!/usr/bin/env bash
# ============================================================
# CodeRouter E2E flaky統計ランナー — run_e2e.sh を N 回実行して集計
#
# 実行:  cd _run/e2e-agents && bash run_e2e_stats.sh
# 環境変数:
#   RUNS (既定 5)  / PORT (既定 8189、run_e2e.sh に伝播)
#   その他 run_e2e.sh の環境変数はそのまま伝播 (MAIN_MODEL_ARG 等)
#
# 出力: stats-<ts>/summary.md (項目別PASS率、PhaseCの試行回数分布、所要時間)
#       各runの results-* はそのまま残る
# 所要目安: 1 run 15〜20分 × RUNS (5なら約1.5時間)
# ============================================================
set -uo pipefail

RUNS="${RUNS:-5}"
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d-%H%M%S)"
STATS_DIR="$KIT_DIR/stats-$TS"
mkdir -p "$STATS_DIR"
SUMMARY="$STATS_DIR/summary.md"

note() { printf '%s\n' "$*"; }

RUN_DIRS=()
RUN_SECS=()
PORT="${PORT:-8189}"
FAST_FAILS=0

ensure_port_free() {
  local pids
  pids="$(lsof -ti ":$PORT" 2>/dev/null)"
  if [ -n "$pids" ]; then
    note "[warn] port $PORT に残留プロセス($pids) — 前runの残骸を掃除します"
    kill $pids 2>/dev/null; sleep 2
    pids="$(lsof -ti ":$PORT" 2>/dev/null)"
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    sleep 1
  fi
}

for i in $(seq 1 "$RUNS"); do
  ensure_port_free
  note "=========================================================="
  note " run $i/$RUNS 開始 ($(date +%H:%M:%S))"
  note "=========================================================="
  before="$(ls -d "$KIT_DIR"/results-* 2>/dev/null | sort)"
  t0=$(date +%s)
  bash "$KIT_DIR/run_e2e.sh" || note "[warn] run $i が非0終了 (レポートがあれば集計は続行)"
  t1=$(date +%s)
  after="$(ls -d "$KIT_DIR"/results-* 2>/dev/null | sort)"
  newdir="$(comm -13 <(printf '%s\n' $before) <(printf '%s\n' $after) | tail -1)"
  if [ -z "$newdir" ] || [ ! -f "$newdir/report.md" ]; then
    note "[warn] run $i の results ディレクトリが見つからない — スキップ"
    if [ $((t1 - t0)) -lt 60 ]; then
      FAST_FAILS=$((FAST_FAILS + 1))
      if [ "$FAST_FAILS" -ge 2 ]; then
        note "[FATAL] 短時間失敗が連続 — 環境エラーと判断して中断します (preflightのFATALメッセージを確認)"
        break
      fi
    fi
    continue
  fi
  FAST_FAILS=0
  RUN_DIRS+=("$newdir")
  RUN_SECS+=($((t1 - t0)))
  note " run $i 完了: $(basename "$newdir") ($((t1 - t0))s)"
  sleep 5   # ポート解放の猶予
done

note ""
note "集計中..."

python3 - "$SUMMARY" "${RUN_DIRS[@]}" <<'PYEOF'
import re, sys, statistics
summary_path = sys.argv[1]
run_dirs = sys.argv[2:]

items = {}       # item名 -> [True/False per run]
tries = {}       # PhaseC item名 -> [try番号 per run(PASS時)]
lines_by_run = []

for d in run_dirs:
    txt = open(f"{d}/report.md", encoding="utf-8").read()
    run_items = {}
    for m in re.finditer(r'^- (PASS|FAIL) (\S+)(.*)$', txt, re.M):
        ok, name, rest = m.group(1) == "PASS", m.group(2), m.group(3)
        run_items[name] = (ok, rest)
        items.setdefault(name, []).append(ok)
        t = re.search(r'try (\d)/3', rest)
        if ok and t:
            tries.setdefault(name, []).append(int(t.group(1)))
    lines_by_run.append((d.rsplit('/',1)[-1], run_items))

with open(summary_path, "w", encoding="utf-8") as f:
    w = f.write
    w(f"# E2E flaky統計 — {len(run_dirs)} runs\n\n")
    w("## 項目別PASS率\n\n")
    total_pass = 0; total_cells = 0
    for name, results in items.items():
        n = len(results); p = sum(results)
        total_pass += p; total_cells += n
        mark = "" if p == n else "  ← **flaky/FAIL**"
        w(f"- {name}: **{p}/{n}**{mark}\n")
    w(f"\n全体: {total_pass}/{total_cells} ")
    w(f"({100.0*total_pass/total_cells:.1f}%)\n" if total_cells else "\n")
    if tries:
        w("\n## PhaseC 試行回数の分布 (PASSしたrunのみ)\n\n")
        for name, ts_ in tries.items():
            dist = {k: ts_.count(k) for k in sorted(set(ts_))}
            diststr = ", ".join(f"try{k}: {v}回" for k, v in dist.items())
            w(f"- {name}: {diststr}\n")
    w("\n## run一覧\n\n")
    for rn, ri in lines_by_run:
        fails = [k for k, (ok, _) in ri.items() if not ok]
        w(f"- {rn}: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}\n")
print("summary written:", summary_path)
PYEOF

# 所要時間の追記
{
  echo ""
  echo "## 所要時間"
  echo ""
  i=1
  for s in "${RUN_SECS[@]}"; do echo "- run $i: ${s}s"; i=$((i+1)); done
} >> "$SUMMARY"

note ""
note "統計レポート: $SUMMARY"
cat "$SUMMARY"
