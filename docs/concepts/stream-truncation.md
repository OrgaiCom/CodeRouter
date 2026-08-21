# ストリーム断絶検知 (Stream Truncation Detection, v2.15.0)

English: [`stream-truncation.en.md`](./stream-truncation.en.md)

## 概要

上流が **HTTP としては綺麗にレスポンスを閉じたのに、LLM プロトコルとしては
メッセージの途中だった** — このケースを検知します。

- Anthropic ワイヤ: `message_stop` が来ない
- OpenAI ワイヤ: `data: [DONE]` も `finish_reason` も来ない

llama.cpp のスロット横取り、`--n-predict` 打ち切り、前段プロキシによる
EOF 終端ボディの切断、ローカルサーバの OOM — いずれもこの形になります。
タイムアウトや `httpx.RemoteProtocolError` のような**トランスポート層の**
故障は以前から `AdapterError` として検知していました。ここで塞ぐのは、
トランスポート層からは見えない層です。

v2.14.0 まで、この断絶は正常完了と区別が付きませんでした。アダプタは終端
イベントを観測したかどうかを記録しておらず、翻訳層の終端合成ガード
(`coderouter/translation/convert.py` の H6 / M9) が
`stop_reason: end_turn` / `finish_reason: "stop"` を合成していたためです。

**合成そのものは正しい判断で、v2.15.0 でも残します** — 消せばクライアント
(Claude Code) がハングします。v2.15.0 が変えるのは「**合成した事実を
エンジンに伝える**」ことだけです。

## 設定

```yaml
profiles:
  - name: local-first
    stream_truncation_action: error   # off | warn | error
    empty_response_action: fallback   # 断絶前フォールバックに必要 (後述)
    partial_stitch_action: surface    # 送出済みだった場合の受け皿
```

| 値 | 動作 |
|------|------|
| `off` (デフォルト) | 検知なし・ログなし・メトリクスなし。v2.14.0 とバイト単位で同一 |
| `warn` | `stream-truncation-detected` ログ + メトリクスのみ。ストリームは従来どおり合成終端で正常終了する |
| `error` | アダプタが `StreamTruncatedError` (retryable な `AdapterError`) を送出し、エンジンの既存分岐に合流する |

**推奨の導入手順**: まず `warn` で実測し、自分のバックエンドの断絶率と
偽陽性の有無を確認してから `error` に上げてください。

## 検知の判定基準

偽陽性を避けるため、終端イベントの判定は意図的に緩めにしてあります。

| ワイヤ | 終端とみなすもの |
|--------|------------------|
| Anthropic | `message_stop` / `stop_reason` を伴う `message_delta` / トップレベルの `error` イベント |
| OpenAI | `data: [DONE]` / いずれかの choice の `finish_reason` |

`message_stop` を送らない Anthropic 互換サーバ、`[DONE]` を送らない
OpenAI 互換サーバは実在します (`convert.py:1379` の M9 ガードのコメントが
"provider that omits the terminator" を明示的に想定しています)。
`message_delta` / `finish_reason` を終端として受け入れることで、
それらを断絶と誤認しません。

## 動作フロー (`error`)

```
断絶検知 (アダプタが StreamTruncatedError を送出)
  ├─ クライアントに実コンテンツ未送出
  │    → 次プロバイダへフォールバック
  │      reason = "stream-truncated"
  │
  └─ 送出済み
       → MidStreamError
         → partial_stitch_action を参照
            ├─ off     : event: error (api_error) で終了
            └─ surface : 蓄積テキスト + message_delta + message_stop +
                         coderouter_partial (reason: "stream_truncated")
```

**新しい出口は作っていません。** 既存の empty-response フォールバック枝と
mid-stream 枝に reason を差し替えて合流させているだけです。副作用として、
`memory_pressure` (L2) / `drift_detection` (L4) / `backend_health` (L5) /
`self_healing` (L6) の全ガードが断絶を「失敗」として自動的に学習します。
断絶を繰り返すバックエンドは適応ルーティングで降格し、self-healing の
再起動対象になります。

## `empty_response_action` との関係 (重要)

Anthropic streaming 経路では、**先頭の `message_start` は届いた瞬間に
クライアントへ転送されます**。したがって断絶が検知された時点では既に
バイトが出ており、定義上 mid-stream です。

`empty_response_action: fallback` は、実コンテンツが現れるまで前置イベント
(`message_start` / 空の `content_block_start` / `ping`) を**保留する**
機能です。これを併用したときに限り、実コンテンツ送出前の断絶を
クライアントに気付かれずに次プロバイダへ回せます。

| 構成 | 実コンテンツ前の断絶 | 実コンテンツ後の断絶 |
|------|----------------------|----------------------|
| `stream_truncation_action: error` のみ | `MidStreamError` | `MidStreamError` |
| `+ empty_response_action: fallback` | **次プロバイダへフォールバック** | `MidStreamError` |

## コスト上の注意

`error` でフォールバックすると、断絶したプロバイダが消費したトークンと
時間は捨てられ、次のプロバイダで生成をやり直します。第3層に有料クラウドを
置く構成では実費が二重にかかります。`budget.py` の月次予算にも計上されます。

## ログ / メトリクス

- ログイベント: `stream-truncation-detected` (warning)
  - fields: `provider`, `action`, `wire` (`anthropic` | `openai`),
    `events_forwarded`, `saw_stream_start`, `tool_call_in_flight`
- MetricsCollector: `stream_truncated_total` / `stream_truncated_by_provider`
  / `stream_truncated_by_action`
- Prometheus: `coderouter_stream_truncated_total{provider="..."}` /
  `coderouter_stream_truncated_by_action_total{action="warn|error"}`
- フォールバック reason: `stream-truncated`
  (`X-CodeRouter-Fallback-Reason` ヘッダ / `coderouter_fallback` SSE
  メタイベントに載ります)

`action` ラベルで分けているのは、`warn` 期の実測値と `error` 期の
実際の介入回数を 1 枚のダッシュボードで区別するためです。

## tool_use 断絶について

`tool_call_in_flight: true` は、`tool_use` / `tool_calls` ブロックが開いた
まま切れたことを意味します。`translation/convert.py` の
`_close_current_block` は `content_block_stop` を発行するだけで、
不完全な引数 JSON の修復も検証も行いません。つまりこのとき、構造上は正しい
`tool_use` ブロックが、パースできない `input` を抱えたままクライアントに
渡ります。

このフラグは**観測用であり、判定には使いません**。終端欠落の検知が既に
このケースを含んでいるためです (引数の途中で切れたストリームが終端イベントを
送ることはありません)。「JSON が不完全なら無条件に断絶扱い」という規則は、
引数が正当に完結している場合に誤判定する新機構になるため採用していません。

## 制限事項

- **`off` 既定**: 何もしないのが既定です。有効化は明示的なオプトインです。
- **非ストリーミング経路は対象外**: 完成したレスポンスに対しては
  `empty_response_action` と drift 検知が既にあります。
- **同一プロバイダへの再送は行いません**: CodeRouter はリトライではなく
  フォールバックで設計されています。
- **偽陽性の残存**: `message_stop` も `message_delta.stop_reason` も
  `finish_reason` も一切送らない上流は、正常でも断絶と判定されます。
  `warn` での実測を挟む理由がこれです。

## ユースケース

| シナリオ | 推奨設定 |
|----------|----------|
| まず実態を知りたい | `warn` |
| ローカル第1層が不安定 / クラウドが控えている | `error` + `empty_response_action: fallback` |
| 有料クラウドが第1層 | `warn` (二重課金を避ける) |
| 長い生成でとにかく途中結果が欲しい | `error` + `partial_stitch_action: surface` |
