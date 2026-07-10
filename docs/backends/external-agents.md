# 外部コーディングエージェント CLI (agent_cli)

> English: [`external-agents.en.md`](./external-agents.en.md)

`kind: "agent_cli"` は、Claude Code CLI のような外部コーディングエージェントを CodeRouter の 1 プロバイダとして登録するアダプタである。v2.7.7 で新規追加された(Phase 1a)。詳細設計は [`docs/designs/external-agents-adapter.md`](../designs/external-agents-adapter.md) を参照。

---

## 概要

コーディングエージェント CLI は本来、ファイルを書き換えながら何ターンも自律的に動くステートフルな制御ループであり、CodeRouter の「1リクエスト = 1変換」という思想とは相性が悪い。`agent_cli` はこれを **ワンショット `exec`**(プロンプト in → 最終回答テキスト out)に押し込めることで両立させている。オーケストレーション(マルチターン制御・ツール実行)はエージェント CLI 内部で完結し、CodeRouter 側からは「1回の対話で答えを返すだけの1つのプロバイダ」として見える。

- **対象 CLI**: `agent` フィールドで `claude` / `codex` / `gemini` / `grok` の4種を宣言できる。
- **実装状況(v2.7.7 時点)**: **`claude`(Claude Code CLI)のみ実装済み**。`codex` / `gemini` / `grok` は `providers.yaml` のスキーマ上は書けるが、アダプタ構築時に必ず次のエラーで拒否される(Phase 1b〜1d は未実装)。

  ```
  AdapterError: agent 'codex' is not implemented in Phase 1a (claude only).
  Configure agent='claude' or wait for the agent's phase.
  ```

  この拒否は `retryable=False` — フォールバックチェーンに他プロバイダがあっても、設定ミスとして即座に停止する。

- **新機能・claude のみ**という段階であり、本ドキュメントも claude ターゲットを前提に記述する。他ターゲットの設定例は `examples/providers-agent-cli.yaml` 末尾にコメントアウトされたプレビューとして置かれているが、動作しない。

---

## クイックスタート

1. **Claude Code CLI をインストールする**(`claude --version` で確認できること)。
2. **ログインを済ませる** — 対話起動して `claude` を実行し `/login` を通すか、ヘッドレス環境なら [プラットフォーム別の認証](#プラットフォーム別の認証) の手順で `claude setup-token` を使う。
3. **サンプル設定で起動する**:

   ```bash
   uv run coderouter serve --config examples/providers-agent-cli.yaml --port 8088
   ```

4. **動作確認**:

   ```bash
   curl http://localhost:8088/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -H 'X-CodeRouter-Profile: claude-agent' \
     -d '{"model":"opus","messages":[{"role":"user","content":"1行でこんにちはと言って"}]}'
   ```

### 初回呼び出しは遅い

`agent_cli` は毎回 CLI プロセスを新規起動する(常駐なし)。初回はプロセス起動 + Claude 本体との1往復が乗るため、通常の HTTP バックエンドより明らかに遅い。

さらに `usage.prompt_tokens` が **2万トークン台**になることがある。これは Claude Code 自身のシステムプロンプト(hooks/CLAUDE.md 探索・ツール定義一式)が毎回プロンプトに乗るためで、CodeRouter 側から渡した実際のメッセージ量とは無関係である。サブスクリプション OAuth で動かしている場合、この分の**課金は発生しない**(5時間窓/週次クォータの消費にはなる — [制限事項](#制限事項) 参照)。

2回目以降は Anthropic 側のプロンプトキャッシュが効き、レスポンスの `usage.prompt_tokens_details.cached_tokens` が増える。実測では、同一 `workdir` への初回呼び出しで `coderouter_cost_usd`(API従量換算のドル相当額。課金額そのものではない — [設定リファレンス](#設定リファレンス) 参照)が **約 $0.22 相当**だったのに対し、キャッシュが効いた2回目以降は **約 $0.05 相当**まで下がった。

---

## プラットフォーム別の認証

`AgentCliAdapter` は子プロセスに**親プロセスの環境をそのまま継承させない**。固定の安全な `PATH` / `NO_COLOR=1` / `TERM=dumb`、そして `HOME` / `USER` / `LOGNAME`(値が設定されている場合のみ)、`passthrough_env` に列挙した変数だけを明示的に注入する。この設計により `ANTHROPIC_API_KEY` は既定では子プロセスに渡らず、サブスクリプション認証(OAuth)が優先される。

| プラットフォーム | 資格情報の保管場所 | 必要な環境変数の継承 | v2.7.7 での対応状況 |
|---|---|---|---|
| **macOS** | Keychain | `USER`(Keychain エントリ解決に必須) | v2.7.7 で `USER` / `LOGNAME` を継承するようになり対応済み。`claude /login` 済みならそのまま動作(実機確認済み) |
| **Linux** | `~/.claude/.credentials.json`(パーミッション `0600`) | `HOME` | `HOME` 継承のみで動作。`claude /login` 済みならそのまま動作(実機確認済み) |
| **ヘッドレスサーバー / コンテナ**(ブラウザ無し) | 上記いずれか、または長期トークン | `CLAUDE_CODE_OAUTH_TOKEN`(`passthrough_env` で明示) | 下記手順を参照 |
| **Windows** | ネイティブ未対応 | — | WSL2 上で CodeRouter ごと動かす(Linux と同じ扱いになる) |

### macOS

Claude Code CLI は Keychain からトークンを解決する際に `USER` 環境変数を参照する。v2.7.7 以前の env allowlist は `HOME` しか継承しておらず、`USER` が欠けていたため macOS のヘッドレス/サーバー実行で Keychain 解決に失敗し `Not logged in` になっていた。v2.7.7 で `_build_child_env()` が `USER` / `LOGNAME` も継承するよう修正され、この問題は解消している。事前に `claude` を対話起動して `/login` を完了させておけば、追加設定なしでそのまま動作することを実機で確認済み。

### Linux

資格情報は `~/.claude/.credentials.json`(パーミッション `0600`)に保存される。子プロセスは `HOME` を継承するだけでこのファイルを読める。macOS と同様、事前に `claude /login` を済ませておけばそのまま動作することを実機で確認済み。

### ヘッドレスサーバー / コンテナ(ブラウザ無し)

ブラウザ付きの対話ログインができない環境向けに、長期トークンを発行して環境変数経由で渡す経路がある。

1. **ブラウザのあるマシン**で `claude setup-token` を実行し、1年間有効な OAuth トークンを発行する。
2. 発行されたトークンを対象サーバーの `.env` に `CLAUDE_CODE_OAUTH_TOKEN=...` として置く。パーミッションは `0600`、`.gitignore` での除外を必ず確認する(`coderouter doctor --check-env` の `env_security` チェックがこの2点を検査する)。
3. `providers.yaml` の該当プロバイダで `agent_cli.passthrough_env: [CLAUDE_CODE_OAUTH_TOKEN]` を指定し、子プロセスへ明示的に転送する。

### Windows

`AgentCliAdapter` は POSIX 前提で実装されている(`os.killpg` によるプロセスグループ kill、`/usr/local/bin` 形式の固定 `PATH` など)ため、Windows ネイティブでは動作しない。WSL2 内に CodeRouter 一式を立て、WSL2 上の `claude` を呼ぶ構成にすれば、実質的に Linux と同じ扱いになる。

### 重要な注意 — API キーは自動では渡らない

子プロセスは親環境を継承しないため、シェルで `ANTHROPIC_API_KEY` をエクスポートしていても **claude CLI には渡らない**。これはサブスクリプション認証を優先し、環境に残った API キーがうっかりサブスク認証を上書きする事故を防ぐための意図的な設計である。API キー従量課金で動かしたい場合のみ、`passthrough_env: [ANTHROPIC_API_KEY]` のように明示的に列挙すること。

---

## 設定リファレンス

`providers.yaml` の `agent_cli:` サブ設定(`AgentCliConfig`)の全フィールド。`extra: forbid` なので未知のキーは設定読み込み時に即座にエラーになる。

| フィールド | 型 | 既定値 | 説明 |
|---|---|---|---|
| `agent` | `"claude" \| "codex" \| "gemini" \| "grok"` | (必須) | 呼び出す CLI。**v2.7.7 では `claude` 以外はアダプタ構築時に拒否される** |
| `command` | `str \| null` | `null`(未設定時は `agent` と同名) | CLI 実行ファイル名 or 絶対パス。`PATH` から解決 |
| `workdir` | `str \| null` | `null`(未設定時は `~/.coderouter/agents/<プロバイダ名>`) | ワンショット exec の作業ディレクトリ。`~` / 環境変数展開あり。`..` を含むパスは拒否される |
| `exec_timeout_s` | `float` | `600.0`(範囲 `1.0`–`1800.0`) | exec 全体の強制タイムアウト(秒)。`ProviderConfig.timeout_s` とは**別系統**(後者は agent_cli では使われない) |
| `allow_file_writes` | `bool` | `false` | ファイル書き込みを許可するか。`false` のときは `sandbox_mode` の値に関わらず read-only にクランプされる |
| `sandbox_mode` | `"read_only" \| "edit" \| "full_auto"` | `"read_only"` | claude の `--permission-mode` へマッピングされる([下表](#sandbox_mode--permission-mode-マッピング)参照) |
| `model` | `str \| null` | `null`(未設定時は `ProviderConfig.model` を使用) | CLI の `--model` に渡すモデル名(`opus` / `sonnet` / `haiku` / `fable` 等) |
| `max_turns` | `int \| null` | `8`(範囲 `1`–`50`) | CLI 内部のターン上限。`--max-turns` として渡る |
| `passthrough_env` | `list[str]` | `[]` | 親環境から子プロセスへ転送する環境変数名のallowlist。`ANTHROPIC_API_KEY` はここに書かない限り渡らない |
| `agent_depth_limit` | `int` | `2`(範囲 `1`–`4`) | 再帰ネストの上限。`CODEROUTER_AGENT_DEPTH` が上限以上なら `AdapterError(retryable=False)` で即停止 |

`command` が未設定の場合は `agent` と同名がデフォルトになる。また、`allow_file_writes: true` と `sandbox_mode: read_only` を同時に指定すると、矛盾した設定として**設定読み込み時に `ValueError`** で弾かれる(書き込みを許可したいなら `sandbox_mode` を `edit` か `full_auto` にすること)。

### `sandbox_mode` → `--permission-mode` マッピング

| `sandbox_mode` | claude `--permission-mode` | 備考 |
|---|---|---|
| `read_only`(既定) | `plan` | ファイル変更なし。`allow_file_writes=false` のとき常にこのモードにクランプされる |
| `edit` | `acceptEdits` | ファイル編集を自動承認 |
| `full_auto` | `acceptEdits` | Phase 1a では `edit` と同じマッピング(claude 側に full_auto 相当の別モードは未使用) |

### `paid: false` の理由

サンプル設定 `examples/providers-agent-cli.yaml` の `agent-claude` プロバイダは `paid: false` になっている。これはサブスクリプション OAuth で運用する限り**従量課金が一切発生しない**(消費するのは後述の5時間窓/週次クォータのみ)ためである。API キー従量課金で運用したい場合は `paid: true` に変更し、CodeRouter 起動時に `ALLOW_PAID=true` を環境変数として渡す必要がある。`ALLOW_PAID` 環境変数は `providers.yaml` に書いた値(`allow_paid`)を**起動時に上書きする**ため、`paid: true` のプロバイダは `ALLOW_PAID` 未設定時にはルーティングから除外される点に注意する。

---

## 制限事項

| 制限 | 内容 |
|---|---|
| **one-shot のみ** | セッション継続(resume)は非対応。呼び出しごとに新しい CLI プロセスが起動し、前回のやり取りは引き継がれない(設計方針として意図的に非スコープ) |
| **擬似ストリーミング** | CLI 側にトークン単位の安定したストリーム出力面が無いため、`generate()` の最終テキストを固定サイズのチャンクに分割して順に yield するだけの擬似ストリームになる。サンプル設定でも `capabilities.streaming: false` を明示している(既定 `true` の上書きが必須) |
| **plan モードの色が付くことがある** | 既定の `sandbox_mode: read_only` は `--permission-mode plan` にマップされる。plan モードは本来インタラクティブな人間によるレビュー UI 向けの応答形式であり、one-shot 実行では実際の変更を伴わない「計画の説明」寄りの文面が返ってくることがある |
| **サブスクのクォータを消費する** | Claude Code サブスクリプションの5時間窓/週次クォータを消費する。API 課金がゼロでも無制限に呼べるわけではない |
| **再帰上限あり** | `agent_depth_limit`(既定2、最大4)を超えるネスト呼び出しは拒否される。エージェント CLI が内部で CodeRouter を呼び返すような構成を組む場合は特に注意 |

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `Not logged in · Please run /login` で失敗する | claude CLI がその実行ユーザー/環境でログイン状態にない | まず `claude` を対話起動して `/login` の状態を確認する。macOS でヘッドレス実行している場合、v2.7.7 以前は `USER` 環境変数が子プロセスに渡らずこのエラーになっていたが、v2.7.7 で修正済み(`USER` / `LOGNAME` を継承するようになった) |
| リクエストが `paid gate blocked` 相当で弾かれる/ルーティングされない | `agent_cli` プロバイダが `paid: true` なのに `ALLOW_PAID` が立っていない | サブスク運用なら `paid: false` にする。API キー従量課金で使うなら CodeRouter 起動時に `ALLOW_PAID=true` を設定する |
| `claude exited 1: ...` のエラーメッセージに具体的な理由が出る | claude CLI は認証エラー等を **stdout** に `is_error: true` の JSON として出力し(stderr は空のまま)終了コード1で終わることがある | v2.7.7 の `_error_detail()` は stderr が空でも stdout の `is_error` JSON から `result` フィールド(実際のエラー文言、例: `Not logged in · Please run /login`)を優先的に拾ってエラーメッセージに含めるようになっている。表示されたメッセージをそのまま手がかりにできる |
| CLI 起動に失敗する(`failed to launch ...`) | `command`(既定は `agent` と同名)が `PATH` 上に無い | `claude --version` が通ることを確認する。フルパスを `command` に指定してもよい |

---

## 関連ドキュメント

- [外部エージェントアダプタ 設計ドキュメント](../designs/external-agents-adapter.md) — 認証設計・argv構築・セキュリティ要件の詳細
- [`examples/providers-agent-cli.yaml`](../../examples/providers-agent-cli.yaml) — 実際に動く設定例
- [シークレット管理とセキュリティ方針](../guides/security.md)
