# 外部コーディングエージェント CLI (agent_cli)

> English: [`external-agents.en.md`](./external-agents.en.md)

`kind: "agent_cli"` は、Claude Code CLI のような外部コーディングエージェントを CodeRouter の 1 プロバイダとして登録するアダプタである。v2.7.7 で新規追加され(Phase 1a: claude)、v2.7.8 で grok が追加された(Phase 1d)。詳細設計は [`docs/designs/external-agents-adapter.md`](../designs/external-agents-adapter.md) を参照。

---

## 概要

コーディングエージェント CLI は本来、ファイルを書き換えながら何ターンも自律的に動くステートフルな制御ループであり、CodeRouter の「1リクエスト = 1変換」という思想とは相性が悪い。`agent_cli` はこれを **ワンショット `exec`**(プロンプト in → 最終回答テキスト out)に押し込めることで両立させている。オーケストレーション(マルチターン制御・ツール実行)はエージェント CLI 内部で完結し、CodeRouter 側からは「1回の対話で答えを返すだけの1つのプロバイダ」として見える。

- **対象 CLI**: `agent` フィールドで `claude` / `codex` / `gemini` / `grok` の4種を宣言できる。
- **実装状況(v2.7.8 時点)**: **`claude`(Claude Code CLI・Phase 1a)と `grok`(Grok CLI・Phase 1d)が実装済み**。`codex` / `gemini` は `providers.yaml` のスキーマ上は書けるが、アダプタ構築時に必ず AdapterError で拒否される(Phase 1b/1c は未実装)。エラーメッセージは概ね次の形である(正確な文言はバージョンにより変わりうる):

  ```
  AdapterError: agent 'codex' is not implemented yet (implemented: claude, grok).
  Wait for the agent's phase (1b/1c).
  ```

  この拒否は `retryable=False` — フォールバックチェーンに他プロバイダがあっても、設定ミスとして即座に停止する。

- 本ドキュメントの共通部分(認証設計・設定リファレンス・制限事項)は claude ターゲットを軸に記述し、grok 固有の挙動は [grok(Grok CLI)](#grokgrok-cli) セクションにまとめる。codex / gemini の設定例は `examples/providers-agent-cli.yaml` 末尾にコメントアウトされたプレビューとして置かれているが、動作しない。

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
| `agent` | `"claude" \| "codex" \| "gemini" \| "grok"` | (必須) | 呼び出す CLI。**v2.7.8 で実装済みなのは `claude` と `grok`。`codex` / `gemini` はアダプタ構築時に拒否される** |
| `command` | `str \| null` | `null`(未設定時は `agent` と同名) | CLI 実行ファイル名 or 絶対パス。`PATH` から解決 |
| `workdir` | `str \| null` | `null`(未設定時は `~/.coderouter/agents/<プロバイダ名>`) | ワンショット exec の作業ディレクトリ。`~` / 環境変数展開あり。`..` を含むパスは拒否される |
| `exec_timeout_s` | `float` | `600.0`(範囲 `1.0`–`1800.0`) | exec 全体の強制タイムアウト(秒)。`ProviderConfig.timeout_s` とは**別系統**(後者は agent_cli では使われない) |
| `allow_file_writes` | `bool` | `false` | ファイル書き込みを許可するか。`false` のときは `sandbox_mode` の値に関わらず read-only にクランプされる |
| `sandbox_mode` | `"read_only" \| "edit" \| "full_auto"` | `"read_only"` | 各 CLI のサンドボックス/承認フラグへマッピングされる(claude は[下表](#sandbox_mode--permission-mode-マッピングclaude)、grok は [grok セクション](#sandbox_mode--grok-フラグのマッピング)参照) |
| `model` | `str \| null` | `null`(未設定時は `ProviderConfig.model` を使用) | CLI の `--model` / `-m` に渡すモデル名(claude: `opus` / `sonnet` / `haiku` / `fable` 等、grok: `grok-4.5` 等) |
| `max_turns` | `int \| null` | `8`(範囲 `1`–`50`) | CLI 内部のターン上限。`--max-turns` として渡る |
| `passthrough_env` | `list[str]` | `[]` | 親環境から子プロセスへ転送する環境変数名のallowlist。`ANTHROPIC_API_KEY` はここに書かない限り渡らない |
| `agent_depth_limit` | `int` | `2`(範囲 `1`–`4`) | 再帰ネストの上限。`CODEROUTER_AGENT_DEPTH` が上限以上なら `AdapterError(retryable=False)` で即停止 |

`command` が未設定の場合は `agent` と同名がデフォルトになる。また、`allow_file_writes: true` と `sandbox_mode: read_only` を同時に指定すると、矛盾した設定として**設定読み込み時に `ValueError`** で弾かれる(書き込みを許可したいなら `sandbox_mode` を `edit` か `full_auto` にすること)。

### `sandbox_mode` → `--permission-mode` マッピング(claude)

| `sandbox_mode` | claude `--permission-mode` | 備考 |
|---|---|---|
| `read_only`(既定) | `plan` | ファイル変更なし。`allow_file_writes=false` のとき常にこのモードにクランプされる |
| `edit` | `acceptEdits` | ファイル編集を自動承認 |
| `full_auto` | `acceptEdits` | claude では `edit` と同じマッピング(claude 側に full_auto 相当の別モードは未使用)。grok は `--always-approve` で区別される([grok セクション](#sandbox_mode--grok-フラグのマッピング)参照) |

### `paid: false` の理由

サンプル設定 `examples/providers-agent-cli.yaml` の `agent-claude` プロバイダは `paid: false` になっている。これはサブスクリプション OAuth で運用する限り**従量課金が一切発生しない**(消費するのは後述の5時間窓/週次クォータのみ)ためである。API キー従量課金で運用したい場合は `paid: true` に変更し、CodeRouter 起動時に `ALLOW_PAID=true` を環境変数として渡す必要がある。`ALLOW_PAID` 環境変数は `providers.yaml` に書いた値(`allow_paid`)を**起動時に上書きする**ため、`paid: true` のプロバイダは `ALLOW_PAID` 未設定時にはルーティングから除外される点に注意する。

---

## grok(Grok CLI)

v2.7.8(Phase 1d)で `agent: grok` が実装された。claude と同じワンショット exec 方式だが、プロンプトの渡し方・クロスセッションメモリの無効化・usage 報告の各点で grok 固有の挙動がある。以下は grok CLI **v0.2.93**([stable] チャネル、2026-07-10 実機検証)を基準とする。

### 設定例

```yaml
providers:
  - name: agent-grok
    kind: agent_cli
    model: grok-4.5              # 現行インストールの既定モデル。`grok models` で一覧確認
    paid: false                  # サブスクリプション OAuth 運用 = 従量課金ゼロ
    capabilities:
      streaming: false
      tools: false
    agent_cli:
      agent: grok
      command: grok
      workdir: ~/.coderouter/agents/grok
      exec_timeout_s: 600
      allow_file_writes: false
      sandbox_mode: read_only
      max_turns: 8
      passthrough_env: []        # OAuth は ~/.grok/auth.json を HOME 継承で読むため空でよい。
                                 # CI で API キーを使う場合のみ GROK_CODE_XAI_API_KEY を列挙
```

### アダプタが構築する argv

`sandbox_mode: read_only`(既定)の場合、アダプタは次の argv を構築する。

```
grok --prompt-file <workdir>/.coderouter-prompt-<uuid>.txt \
     --output-format json -m <model> --cwd <workdir> \
     --max-turns <N> --no-memory \
     --sandbox read-only --permission-mode plan
```

### プロンプトはファイル経由で渡す(`--prompt-file`)

grok の `-p` / `--single` はプロンプトを **argv の値としてしか受け取らない**(stdin をプロンプトとして受け付けないことを実 CLI で確認済み)。argv に巨大なプロンプトを載せると Linux の `MAX_ARG_STRLEN`(約 128KiB)の上限に当たるうえ、`ps` からプロンプト全文が見えてしまう。そのためアダプタは隔離 workdir 内にパーミッション `0600` の一時ファイル(`.coderouter-prompt-<uuid>.txt`)としてプロンプトを書き出し、`--prompt-file` で渡す。この一時ファイルは exec 終了後に**必ず削除される**(タイムアウト・エラー経路を含む)。

### `sandbox_mode` → grok フラグのマッピング

claude と同様、`allow_file_writes=false` のときは `sandbox_mode` の値に関わらず `read_only` にクランプされる。

| `sandbox_mode` | grok フラグ | 備考 |
|---|---|---|
| `read_only`(既定) | `--sandbox read-only --permission-mode plan` | ファイル変更なし。`allow_file_writes=false` のとき常にこのモードにクランプされる |
| `edit` | `--sandbox workspace --permission-mode acceptEdits` | workspace サンドボックス内でのファイル編集を自動承認 |
| `full_auto` | `--sandbox workspace --always-approve` | claude と異なり grok では `edit` と区別されたマッピングになる |

### `--no-memory` を常に付与する

grok CLI はセッションをまたぐメモリ機能を持つ。前回呼び出しの記憶が次の応答へ漏れることは CodeRouter の「1リクエスト = 1ステートレス変換」思想と衝突するため、アダプタは**常に** `--no-memory` を付与してこれを無効化する(設定で外すことはできない)。

### JSON 出力と usage / cost

`--output-format json` の出力は単一 JSON オブジェクト `{"text", "stopReason", "sessionId", "requestId", "thought"?}` である(grok v0.2.93 で確認)。`text` が最終回答として、`sessionId` がレスポンスメタデータ `coderouter_session_id` として返る。**トークン usage・コストのフィールドは存在しない**ため、usage はすべてゼロで報告され、`coderouter_cost_usd` も運用者が `ProviderConfig.cost` に単価を設定しない限り 0 のままである(claude が `total_cost_usd` を直接出力するのとは対照的)。JSON は防御的にパースされ、想定外の形は `AdapterError(retryable=True)` としてフォールバックチェーンを次のプロバイダへ進める。

### 認証(サブスクリプション OAuth / API キー)

grok CLI は OAuth によるサブスクリプションログインに対応している(SuperGrok / X Premium+)。資格情報は `~/.grok/auth.json` に保存され(7日で失効・自動リフレッシュあり。`GROK_HOME` で保管場所を上書き可能)、アダプタの `HOME` 継承によって `passthrough_env: []` のままで OAuth が機能する。セットアップ手順:

1. `grok login` を実行してサブスクリプションログインを済ませる。
2. `grok models` でモデル一覧が返ることをスモーク確認する。現行インストールでは `grok-4.5`(既定)と `grok-composer-2.5-fast` が返る。

CI などで API キー従量課金を使う場合のみ、`passthrough_env: [GROK_CODE_XAI_API_KEY]` を列挙する。環境変数名は **`GROK_CODE_XAI_API_KEY` であり `XAI_API_KEY` ではない**点に注意。API キーが渡っている場合は OAuth より優先される。

### エラー報告

grok CLI は成功時に終了コード 0、認証・ネットワーク・実行時エラーでは終了コード 1 でエラーテキストを **stderr** に出力する。アダプタは stderr の末尾を `AdapterError` メッセージに含めるため、表示されたメッセージをそのまま手がかりにできる。

### early beta であることの注意

grok CLI は early beta である(v0.2.93 [stable] チャネル、2026-07-10 時点)。JSON スキーマが今後変わる可能性があるため、**バージョンの pin を推奨する**(`command` に固定版バイナリのフルパスを指定できる)。スキーマ変化が起きた場合も防御的パースにより retryable な `AdapterError` となり、フォールバックチェーンの次のプロバイダへ降格する。

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
| `grok exited 1: ...` で失敗する | grok CLI は認証・ネットワーク・実行時エラーを終了コード1で終わり、エラーテキストを stderr に出す | アダプタが stderr の末尾を `AdapterError` に含めるので、そのメッセージを手がかりにする。認証エラーなら `grok login` を再実行し、`grok models` が通ることをスモーク確認する |
| CLI 起動に失敗する(`failed to launch ...`) | `command`(既定は `agent` と同名)が `PATH` 上に無い | `claude --version` / `grok --version` が通ることを確認する。フルパスを `command` に指定してもよい |

---

## 関連ドキュメント

- [外部エージェントアダプタ 設計ドキュメント](../designs/external-agents-adapter.md) — 認証設計・argv構築・セキュリティ要件の詳細
- [`examples/providers-agent-cli.yaml`](../../examples/providers-agent-cli.yaml) — 実際に動く設定例
- [シークレット管理とセキュリティ方針](../guides/security.md)
