# E2E テストキット — サブエージェント/オーケストレータ × agent_cli 4種

Claude Code (orchestrator) → CodeRouter → 外部エージェント CLI (claude / codex / grok / antigravity) の経路が動くかを検証するキット。Mac のターミナルで 1 本実行するだけで、3 フェーズの検証とレポート生成まで行う。

## 前提条件

1. `coderouter` v2.9.0+ と **coderouter-plugin-agents** がインストール済みであること:

   ```bash
   uv pip install "coderouter-plugin-agents @ git+https://github.com/zephel01/coderouter-plugin-agents"
   # uv tool install 構成なら:
   #   uv tool install coderouter-cli --with "coderouter-plugin-agents @ git+https://github.com/zephel01/coderouter-plugin-agents"
   ```

2. 4 つの CLI がインストール + ログイン済み: `claude`(/login 済み)、`codex`(`codex login status` が 0)、`grok`(`grok models` が通る)、`agy`(`agy models </dev/null` が通る)。
3. Ollama 稼働中 + `qwen3-coder:30b`(Phase C の orchestrator 本体ループ用。無ければ `providers.yaml` の `main-ollama.model` を手元の tool 対応モデルに変更)。
4. ポート 8189 が空いていること(既存の 8088 serve とは共存可)。

## 実行

```bash
cd ~/works/project/CodeRouter/_run/e2e-agents
bash run_e2e.sh
```

所要 5〜15 分(agent_cli はコールドスタートが遅い)。終了時に `results-<timestamp>/report.md` が出るので、**そのフォルダごと(最低でも report.md と serve.log)を Claude に共有**してください。判定と次のアクションを返します。

部分実行: `SKIP_C=1 bash run_e2e.sh`(Phase A/B のみ)、`PORT=8200 bash run_e2e.sh`。

## 何を検証しているか

| Phase | 経路 | 検証内容 |
|---|---|---|
| A | curl → OpenAI ingress + `X-CodeRouter-Profile` | チャネル②(明示ヘッダ)。4 つの agent_cli プロバイダそれぞれへの疎通・応答マーカー確認 |
| B | curl → Anthropic ingress `/v1/messages`、model=`e2e-*` | チャネル①の配管。auto_router `model_pattern` ルール(fullmatch)の発火を serve ログの `auto-router-resolved` で確認 |
| C | `claude -p` → Task ツール → `.claude/agents/ext-*` → CodeRouter → 各 CLI | 本命の E2E。サブエージェント frontmatter `model: e2e-*` が wire に載り、CodeRouter が外部 CLI へ振り分け、応答が orchestrator まで還流するか |

サブエージェント定義の原本は `claude-agents/*.md` にあり、`run_e2e.sh` 実行時に `.claude/agents/` へ自動コピーされる(手作業不要)。frontmatter を編集する場合は `claude-agents/` 側を編集すること。

Phase C は同時に、docs/guides/subagent-routing.md §7 で **UNCONFIRMED** の 2 点を実測で確定させる:

- サブエージェントの model 名がエイリアスのまま届くか / フル ID 展開されるか → report.md の `auto-router-resolved` の `signals.model`
- frontmatter にカスタム model 名(`e2e-*`)が使えるか → 拒否されたら下記プラン B

## 構成のポイント

- **main プロファイル(Ollama qwen3-coder:30b)が必須な理由**: orchestrator 本体は Task ツールを呼ぶため tool 対応 backend が必要。agent_cli は `tools: false`(テキスト in/out)なので main には使えない。`--model sonnet` で起動した本体ループは `default_rule_profile: main` に落ちる。
- **サブエージェント側に tools 往復は期待しない**: `ext-*` サブエージェントのリクエストに tools[] が載っても、CodeRouter の capability 層で処理される(report.md に `capability-degraded` が出るのは想定内)。サブエージェントは一問一答のリレー役として設計してある。
- **課金**: 4 プロバイダとも `paid: false`(サブスク OAuth)。従量課金は発生しないが、各サブスクのクォータは消費する(claude は 5 時間窓、テスト全体で 3 呼び出し程度)。
- **grok/agy は usage 常時ゼロ**が仕様。レポートのトークン数 0 は異常ではない。

## プラン B — frontmatter が `e2e-*` を拒否した場合

Phase C で claude が「invalid model」系エラーを出したら、`.claude/agents/*.md` の `model:` をエイリアスに書き換え、`providers.yaml` 末尾のコメントアウト済み auto_router ルール(opus/haiku/fable ベース)に差し替えて再実行:

| ファイル | model |
|---|---|
| ext-claude.md | `opus` |
| ext-codex.md | `haiku` |
| ext-grok.md | `fable` |
| ext-agy.md | `claude-agy-e2e` |

## トラブルシューティング早見

| 症状 | 対処 |
|---|---|
| serve が即死(agent_cli エラー) | plugin 未導入。前提条件 1 を実行 + `plugins.enabled: [agents]`(本キットの yaml には設定済み) |
| PhaseA/claude-agent で `Not logged in` | `claude` を対話起動して /login |
| PhaseA/agy がタイムアウト | `agy models </dev/null` で認証確認。agy は stdin パイプ厳禁 |
| PhaseC で main が Task を呼ばない | ローカルモデルの限界の可能性。`MAIN_MODEL_ARG` はそのままに、providers.yaml の main-ollama.model をより強いモデルへ。または providers.yaml 内コメントの main-anthropic-api(要 API キー + ALLOW_PAID=true)へ切替 |
| 詳細 | `docs/backends/external-agents.md` / `docs/guides/subagent-routing.md` |
