# Mid-stream Partial Stitching (v2.0-H)

## 概要

streaming 応答中にプロバイダがクラッシュすると、通常はそれまでに生成された
テキストが全て破棄され、クライアントには `event: error` のみが届きます。
v2.0-H の Partial Stitch 機能は、蓄積されたテキストをクライアントに
グレースフルに返却する「surface mode」を追加します。

## 設定

```yaml
profiles:
  - name: long-session
    partial_stitch_action: surface   # off | surface
```

| 値 | 動作 |
|------|------|
| `off` (デフォルト) | 従来通り `event: error` を返す |
| `surface` | 蓄積テキストを返却してストリームを正常終了する |

## 動作フロー (surface mode)

1. streaming 応答中、`_StreamUsageAccumulator` がテキスト content_block を蓄積
2. プロバイダが mid-stream でクラッシュ → `MidStreamError(partial_content=[...])` raise
3. ingress が `partial_stitch_action` を確認
4. `surface` かつ partial_content が存在する場合:
   - `event: message_delta` (stop_reason: null, usage: {output_tokens: 0})
   - `event: message_stop`
   - `event: coderouter_partial` (メタデータ + 蓄積テキスト)
5. クライアントは正常なストリーム終了として処理可能

## coderouter_partial イベント

```json
{
  "type": "coderouter_partial",
  "partial_content": [
    {"type": "text", "text": "蓄積されたテキスト..."}
  ],
  "provider": "ollama-local",
  "reason": "mid_stream_failure",
  "original_error": "connection reset by peer"
}
```

**互換性**: Anthropic SDK は未知の event type を自動的に無視するため、
既存クライアントに影響はありません。CodeRouter 対応クライアントは
`coderouter_partial` イベントを読み取って部分応答を表示できます。

## 制限事項

- **テキストブロックのみ蓄積**: `tool_use` ブロックの部分 JSON は surface しません
- **メモリ**: request ライフサイクルなので長大応答でも問題なし
- **Phase 2 (将来)**: retry mode — partial を context injection して次プロバイダに再送

## ログ / メトリクス

- ログイベント: `partial-stitch-surfaced` (info)
  - fields: `provider`, `profile`, `text_blocks`, `text_length`
- MetricsCollector: `partial_stitch_surfaced_total` counter
- Prometheus: `coderouter_partial_stitch_surfaced_total`

## ユースケース

| シナリオ | 推奨設定 |
|----------|----------|
| 短い Q&A (< 5秒) | `off` — 再送の方が早い |
| 長い生成 (コード、文書、30秒+) | `surface` — 30秒の生成を無駄にしない |
| Claude Code エージェントセッション | `surface` — ユーザが部分結果を確認可能 |
