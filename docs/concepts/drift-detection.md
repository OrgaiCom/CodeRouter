# Drift Detection (v2.0-G)

English: [`drift-detection.en.md`](./drift-detection.en.md)

長時間稼働する agent session でモデル応答品質の漸進的な劣化 (drift) を自動検知し、
corrective action を実行するガードです。

## 背景

Ollama 等のローカル LLM バックエンドは長時間稼働すると以下のような品質低下パターンを示すことがあります:

- 応答が空 (output_tokens=0) になる
- 応答長が徐々に短くなる (length collapse)
- tool_use を返すべき場面で返さなくなる (tool silence)
- 異常な stop_reason が増加する
- エラー率が上昇する
- 同じ内容の応答を繰り返し前進しなくなる (goal progress stall — response_fingerprint が設定された場合のみ)

これらは個別には致命的ではないものの、蓄積すると agent session が事実上動作不能に陥ります。
Drift Detection はこれらを 6 つの品質シグナルとして rolling window で監視し、
閾値を超えた時点で corrective action を自動実行します。

## 設定

`providers.yaml` の profile に以下のフィールドを追加します:

```yaml
profiles:
  - name: long-session
    providers: [ollama-qwen3, ollama-gemma4]
    drift_detection_action: reload      # off | warn | promote | reload
    drift_detection_sensitivity: normal # low | normal | high
    drift_detection_window_size: 20     # per-provider rolling window サイズ (4-200)
    drift_detection_cooldown_s: 300     # promote/reload 後の復帰待機秒数 (10-3600)
```

### Action

| Action | 動作 |
|--------|------|
| `off` | 検知無効 (default) |
| `warn` | 検知 + ログ出力のみ (X-CodeRouter-Drift header 付与) |
| `promote` | warn + chain 内で provider rank を降格 (traffic を次の provider へ迂回) |
| `reload` | promote + Ollama KV cache flush (keep_alive=0 で model unload → 次 request で fresh reload) |

### Sensitivity

| Preset | empty_response_rate | length_collapse_ratio | tool_silence_rate | stop_anomaly_rate | error_rate | min_window_fill |
|--------|--------------------:|----------------------:|------------------:|------------------:|-----------:|----------------:|
| `low` | 0.5 | 0.3 | 0.8 | 0.6 | 0.4 | 10 |
| `normal` | 0.3 | 0.5 | 0.7 | 0.4 | 0.25 | 6 |
| `high` | 0.2 | 0.7 | 0.5 | 0.3 | 0.15 | 4 |

（各プリセットには `goal_progress_stall` 用の `repetition_rate_threshold` も含まれます: low=0.6 / normal=0.4 / high=0.25）

> **goal_mode**: profile に `goal_mode: true` を設定すると、`drift_detection_sensitivity` に関わらず専用の `goal` プリセット (empty=0.2 / collapse=0.6 / tool_silence=0.5 / stop=0.3 / error=0.15 / repetition=0.2 / min_window_fill=4) が適用されます。goal_mode 自体はシグナルではなく、閾値プリセットを切り替える bool フラグです。`/goal` セッション向けに前進停滞・length collapse を早く拾います。

## 6 品質シグナル

1. **empty_response_rate** — output_tokens=0 の応答率 (error を除外)
2. **length_collapse_ratio** — window 後半の median output_tokens / 前半の median (比率が閾値未満で collapse)
3. **tool_silence_rate** — tools[] を含む request に対して tool_use を返さない率
4. **stop_anomaly_rate** — end_turn / tool_use / max_tokens 以外の stop_reason の率
5. **error_rate** — provider error の率
6. **goal_progress_stall** — window 内で以前と同じ fingerprint が再出現した率 (response_fingerprint が設定された observation のみ対象、3 件以上で発火)。model が前進せず同じ応答を繰り返している兆候

## Severity 判定

- severe signal ×1 (empty_response_rate, length_collapse) → **severe**
- mild signal ×2 以上 (tool_silence, stop_anomaly, error_rate, goal_progress_stall) → **severe**
- mild signal ×1 → **mild**

## Response Header

`X-CodeRouter-Drift: mild` または `X-CodeRouter-Drift: severe` が response header に付与されます。
verdict はリクエスト単位 (request-scoped) で管理され、他の並行リクエストの検知結果がヘッダに混ざることはありません。
Streaming ではヘッダ送出後に検知が走るため、原則ヘッダは付与されません — 検知結果はログ / metrics で確認してください。

## Cooldown & Recovery

`promote` / `reload` action が発火すると:

1. 対象 provider が cooldown 中はチェーン末尾へ降格される (profile の `adaptive` 設定に関係なく実際の試行順に反映)
2. `reload` の場合は追加で Ollama KV cache flush を試行 (best-effort)
3. `drift_detection_cooldown_s` 秒間は再検知をスキップ
4. cooldown 満了後、次の observation 記録時に rank 復帰 + window クリア
5. `drift-recovered` ログを出力

## Observability

### ログイベント

| Event | Level | 説明 |
|-------|-------|------|
| `drift-detected` | WARNING | drift 検知 (severity, signals, action を含む) |
| `drift-promoted` | INFO | provider rank 降格 |
| `drift-reload-attempted` | INFO | Ollama KV flush 試行結果 |
| `drift-recovered` | INFO | cooldown 満了 → rank 復帰 |

### Prometheus Metrics

```
coderouter_drift_detected_total{provider="..."} — 検知回数
coderouter_drift_promoted_total                  — 降格回数
coderouter_drift_reload_total                    — reload 試行回数
coderouter_drift_reload_success_total            — reload 成功回数
```

### /metrics.json

```json
{
  "counters": {
    "drift_detected_total": 3,
    "drift_detected_by_provider": {"ollama-qwen3": 3},
    "drift_promoted_total": 2,
    "drift_reload_total": 2,
    "drift_reload_success_total": 1
  }
}
```

## 推奨設定パターン

### Claude Code + Ollama (長時間コーディング)

```yaml
profiles:
  - name: default
    providers: [ollama-qwen3, ollama-gemma4]
    drift_detection_action: reload
    drift_detection_sensitivity: normal
    drift_detection_cooldown_s: 300
```

### 監視のみ (まず状況把握)

```yaml
profiles:
  - name: default
    providers: [ollama-qwen3]
    drift_detection_action: warn
    drift_detection_sensitivity: high
```

### 複数 provider でのフェイルオーバー重視

```yaml
profiles:
  - name: default
    providers: [ollama-qwen3, ollama-gemma4, openrouter-llama4]
    drift_detection_action: promote
    drift_detection_sensitivity: normal
    drift_detection_cooldown_s: 600
```
