# サブエージェントのモデル振り分け

English version: [`docs/guides/subagent-routing.en.md`](./subagent-routing.en.md)

Claude Code のサブエージェント(`.claude/agents/*.md`)を、種類ごとに別のバックエンド — ローカル LLM、クラウド、外部エージェント CLI(`agent_cli`) — へ CodeRouter で振り分けるための実践ガイドである。

目次:

1. [概要](#1-概要)
2. [仕組み — 3チャネルと優先順位](#2-仕組み--3チャネルと優先順位)
3. [クライアント側の設定(Claude Code)](#3-クライアント側の設定claude-code)
4. [CodeRouter 側の設定](#4-coderouter-側の設定)
5. [実利用パターン集](#5-実利用パターン集)
6. [動作確認](#6-動作確認)
7. [制限と既知の課題](#7-制限と既知の課題)
8. [関連ドキュメント](#8-関連ドキュメント)

---

## 1. 概要

CodeRouter で、Claude Code のサブエージェントごとに異なるモデル/バックエンドを割り当てられる。レビュー役はローカルの安価なモデルへ、設計役はクラウドの高能力モデルへ、監査役は外部の `claude` CLI へ、といった運用が可能である。

先に仕組みの核心を正直に書く。**ワイヤ(HTTP リクエスト)上には「これはサブエージェントである」という専用の識別子は無い。** Claude Code はサブエージェントごとに解決したモデル名を、通常のリクエストと同じ `model` フィールドに載せて送るだけである。CodeRouter からは、この `model` 名がサブエージェントを見分ける実質的に唯一の手がかりになる。したがって本ガイドの振り分けは「① frontmatter でサブエージェントごとに `model` を変える → ② CodeRouter がその `model` 名で振り分ける」という 2 段の組み合わせが基本線であり、CodeRouter 単体の専用機能ではない。

## 2. 仕組み — 3チャネルと優先順位

CodeRouter がリクエストを profile(=バックエンドの束)に割り当てる経路は 3 つあり、優先順位は次の通り(両 ingress[Anthropic Messages API / OpenAI Chat Completions] 共通)。

```
body.profile  >  X-CodeRouter-Profile ヘッダ  >  X-CodeRouter-Mode ヘッダ  >  auto_router  >  default_profile
```

出典: `coderouter/routing/auto_router.py`、`coderouter/ingress/anthropic_routes.py`、`coderouter/ingress/openai_routes.py`。

| # | チャネル | 実体 | サブエージェント振り分けでの使いどころ |
|---|---|---|---|
| ① **model 名マッチ** | auto_router の `model_pattern` matcher(body の `model` に `re.fullmatch`) | **主力**。frontmatter の `model` を種別ごとに変えておけば、その解決済みモデル名で振り分けられる。専用ヘッダを注入できない標準の Claude Code サブエージェント起動でも使える。 |
| ② **明示ヘッダ** | `X-CodeRouter-Profile`(profile 名を直指定)/ `X-CodeRouter-Mode`(mode_alias 経由) | 自前の orchestrator を挟み、各サブ呼び出しに `X-CodeRouter-Profile: planner\|coder\|reviewer-audit` のようなヘッダを付与して確定的に駆動する場合の本命。override であって「条件」ではないため、ルールの中に「このヘッダの値なら」という分岐は書けない。 |
| ③ **内容ベース** | auto_router の残り 6 matcher(CJK 比率・コードフェンス比率・トークン数など) | クライアントが役割を宣言しない場合の保険。長文なら planner、CJK が濃ければローカル、コード濃度が高ければ coder、といった推測分岐。 |

いずれのチャネルも最終的には「profile 名」に解決され、profile が backend の束(フォールバックチェーン)に紐づく。サブエージェント振り分けは「(model 名 or 明示ヘッダ) → profile → backend」の 3 段で組む。

## 3. クライアント側の設定(Claude Code)

### frontmatter の `model` フィールド

サブエージェント定義は `.claude/agents/*.md`(プロジェクト)または `~/.claude/agents/*.md`(ユーザー)に置く YAML frontmatter + Markdown 本文で、識別子は `name` frontmatter のみに由来する。`model` フィールドに取れる値は次の通り(2026-07 時点、公式仕様)。

- モデルエイリアス: `sonnet` / `opus` / `haiku` / `fable`
- フルモデル ID: 例 `claude-opus-4-8` / `claude-sonnet-5`(`--model` フラグと同じ値)
- `inherit`: メイン会話と同じモデル
- 未指定時の既定は `inherit`

出典: <https://code.claude.com/docs/en/sub-agents>(frontmatter 表・"Choose a model" 節)。

例:

```markdown
---
name: reviewer
description: コードレビュー担当。use proactively after code changes.
model: haiku        # ← 安価モデルに固定 → CodeRouter で local へ寄せる
---
(システムプロンプト本文)
```

```markdown
---
name: architect
description: アーキテクチャ設計・計画担当。
model: opus         # ← 高能力に固定 → CodeRouter で planner profile へ
---
```

全サブエージェントが `inherit`(=メイン会話と同一モデル)のままだと、`model` 名では区別できなくなる点に注意。この方式を使うなら frontmatter の `model` を種別ごとに変えておくのが前提になる。

### サブエージェントのモデル解決順

Claude Code 内部でのモデル解決順は次の通り(上が優先)。

1. 環境変数 `CLAUDE_CODE_SUBAGENT_MODEL`(エイリアス or モデル ID の場合)
2. 起動時の per-invocation `model` パラメータ(Agent/Task ツールが渡す)
3. サブエージェント定義の `model` frontmatter
4. メイン会話のモデル(`inherit`)

出典: <https://code.claude.com/docs/en/sub-agents>("Choose a model" 節)。`CLAUDE_CODE_SUBAGENT_MODEL=inherit` は「未設定」と同義で、解決は per-invocation → frontmatter へ続く(v2.1.196 以降)。

frontmatter には HTTP ヘッダを注入するフィールドは無い(`tools` / `disallowedTools` / `permissionMode` / `effort` / `isolation` / `color` 等は挙動制御用で、ヘッダ注入用ではない)。つまり②の明示ヘッダ経路は、標準の Claude Code サブエージェント起動だけでは使えず、自前 orchestrator が付与する場合の経路になる。

### ANTHROPIC_BASE_URL で CodeRouter に向ける

```bash
export ANTHROPIC_BASE_URL="http://localhost:8088"
export ANTHROPIC_AUTH_TOKEN="dummy"   # CodeRouter は認証を見ない。非空であればOK
claude
```

`ANTHROPIC_BASE_URL` は「送り先」を変えるだけで「どのモデルが答えるか」には関与しない(モデル選択は上記 frontmatter/環境変数の役目)。カスタム `ANTHROPIC_BASE_URL` 経由の場合、Claude Code はモデル名文字列を allowlist 検証せずそのまま通すため、CodeRouter 側では任意のモデル名を `model_pattern` の対象にできる。出典: <https://code.claude.com/docs/en/model-config>。

## 4. CodeRouter 側の設定

### auto_router の `model_pattern` ルール

`default_profile: auto` のときだけ auto_router が働く。ルールを自分で書くと、バンドル済みの既定ルール(画像→multi / コード濃度→coding / フォールスルー→writing)は**完全に置き換わる**(マージされない)。

```yaml
default_profile: auto
auto_router:
  default_rule_profile: coder
  rules:
    - id: user:opus-to-planner
      profile: planner
      match: { model_pattern: "(claude-)?opus.*" }     # フルmatch。opus 系を planner へ
    - id: user:haiku-to-local
      profile: reviewer-light
      match: { model_pattern: "(claude-.*)?haiku.*" }   # haiku 系を local reviewer へ
```

**注意すべき2点**:

- `model_pattern` は `re.search` ではなく **`re.fullmatch`** である。Claude Code が実際に送るモデル名文字列(`opus` のままか `claude-opus-4-8` に展開されるかは環境依存)の**全体**にマッチする正規表現を書く必要がある。末尾の `.*` を忘れると一致しない。
- 実際に届くモデル名は環境・バージョン依存で断定できない。**運用前に必ず auto-router のログ(`auto-router-resolved` イベントの `signals.model`)で実値を確認する**こと。確認手順は [§6 動作確認](#6-動作確認) を参照。

### auto_router matcher 一覧(全 8 種)

`RuleMatcher` は **1 ルール = ちょうど 1 matcher**。読み込み時のバリデータが強制するため、複数条件の AND 合成はできない(→ [§7](#7-制限と既知の課題) の G1)。ルール順は「先着一致」で、上から順に評価される。

| フィールド | 型 | 意味 | 評価対象 |
|---|---|---|---|
| `has_image` | `bool`(`true` のみ有効) | 最新 user メッセージに画像ブロックがあれば一致 | 最新 user メッセージ |
| `code_fence_ratio_min` | `float`(0.0–1.0) | ` ``` ` フェンス内文字数の比率が閾値以上 | 最新 user メッセージ |
| `cjk_ratio_min` | `float`(0.0–1.0) | CJK 文字比率が閾値以上 | 最新 user メッセージ |
| `content_contains` | `str` | 部分文字列(大小区別あり) | 最新 user メッセージ |
| `content_regex` | `str` | `re.search`(読み込み時にコンパイル検証) | 最新 user メッセージ |
| `model_pattern` | `str` | body の `model` に対する `re.fullmatch`(読み込み時にコンパイル検証) | body の `model` |
| `content_token_count_min` | `int`(≥1) | 全メッセージ(system + messages)の推定トークン数(char/4)が閾値以上 | リクエスト全体 |
| `has_tools` | `bool`(`true` のみ有効) | `tools[]`(OpenAI/Anthropic 共通)または legacy `functions[]` を1個以上宣言 | body 全体 |

`model_pattern` / `content_token_count_min` / `has_tools` は user メッセージが無くても発火できる(system-only なリクエストや model+tools のみの body)。それ以外の matcher は user メッセージが必須。ブール matcher(`has_image` / `has_tools`)に `false` を明示すると読み込みエラーになる(死にルール防止のため、未使用は省略する)。

### X-CodeRouter-Profile ヘッダ

body に `profile` フィールドが無く、自前 orchestrator がサブ呼び出しごとにヘッダで確定的に駆動したい場合に使う。

```bash
curl http://localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-CodeRouter-Profile: reviewer-audit' \
  -d '{"model":"opus","messages":[{"role":"user","content":"1行でこんにちはと言って"}]}'
```

処理順は「body.profile → `X-CodeRouter-Profile` → `X-CodeRouter-Mode`(`mode_aliases` で profile へ解決)→ auto_router → `default_profile`」。存在しない profile 名を指定すると 400 になる。

## 5. 実利用パターン集

### (a) opusplan 型 — Plan は Opus、実行はローカル/中位

Plan 段階を Claude Opus(`agent_cli`、read-only)、実行を中位ローカル→クラウドの順、レビューは監査を Opus・軽微をローカルに振り分ける構成。

```yaml
allow_paid: true            # agent_cli(claude=Opus サブスク) 利用のため
default_profile: coder      # 明示ヘッダ/ルール不一致時は日常コーディング役へ

providers:
  - name: agent-claude-opus     # Plan役: Claude Opus (agent_cli, read-only)
    kind: agent_cli
    model: opus
    paid: true
    capabilities: { streaming: false, tools: false }   # agent_cli 既定
    agent_cli: { agent: claude, sandbox_mode: read_only, exec_timeout_s: 600 }
  - name: local-coder            # 実行役: 中位ローカル (税ゼロ)
    kind: anthropic               # Ollama v0.23.1+ passthrough
    base_url: http://localhost:11434
    model: qwen3-coder:30b
  - name: cloud-mid               # 実行役の保険: クラウド中位
    kind: openai_compat
    base_url: https://openrouter.ai/api/v1
    model: qwen/qwen3-coder:free
    paid: true
  - name: agent-claude-review    # レビュー役: 監査は Opus
    kind: agent_cli
    model: opus
    paid: true
    capabilities: { streaming: false, tools: false }
    agent_cli: { agent: claude, sandbox_mode: read_only }
  - name: local-reviewer         # レビュー役: 軽微は別ローカル
    kind: anthropic
    base_url: http://localhost:11434
    model: qwen2.5-coder:7b

profiles:
  - name: planner
    providers: [agent-claude-opus]        # agent_cli は単独が原則
  - name: coder
    providers: [local-coder, cloud-mid]   # ローカル優先→落ちたらクラウド
  - name: reviewer-audit
    providers: [agent-claude-review]      # セキュリティ監査は Opus
  - name: reviewer-light
    providers: [local-reviewer]           # 軽微レビューはローカル

auto_router:   # クライアントが profile を明示しない場合の保険
  default_rule_profile: coder
  rules:
    - id: user:image-to-multi
      profile: planner
      match: { has_image: true }
    - id: user:dense-code-to-coder
      profile: coder
      match: { code_fence_ratio_min: 0.3 }
    - id: user:long-context-to-planner
      profile: planner
      match: { content_token_count_min: 32000 }
    - id: user:cjk-to-local
      profile: coder
      match: { cjk_ratio_min: 0.5 }
    - id: user:review-keyword
      profile: reviewer-audit
      match: { content_contains: "レビュー" }

plugins:
  enabled: [agents]   # kind: agent_cli を使うため v2.9.0 以降は必須
```

**駆動方法の注記**: 上記 auto_router はあくまで「クライアントが役割を宣言しない時の保険」である。本来の opusplan 駆動は、上位層 orchestrator が各サブ呼び出しに `X-CodeRouter-Profile: planner|coder|reviewer-audit` を明示付与する形になる(precedence は [§2](#2-仕組み--3チャネルと優先順位) の通り)。なお「Claude Code 純正の `opusplan` エイリアス(Plan モード中は opus、実行に移ると sonnet へ自動切替)」とは別物であり、混同しないこと。

### (b) サブエージェント別モデルの最小構成

考え方は [§3](#3-クライアント側の設定claude-code) の frontmatter 例の通り: `reviewer` は `model: haiku`、`architect` は `model: opus` に固定し、CodeRouter 側は `model_pattern` で振り分ける。

```yaml
default_profile: auto

providers:
  - name: local-reviewer
    kind: openai_compat
    base_url: http://localhost:11434/v1
    model: qwen2.5-coder:7b
  - name: cloud-planner
    kind: openai_compat
    base_url: https://openrouter.ai/api/v1
    model: anthropic/claude-opus-4-8
    paid: true

profiles:
  - name: reviewer-light
    providers: [local-reviewer]
  - name: planner
    providers: [cloud-planner]

auto_router:
  default_rule_profile: reviewer-light
  rules:
    - id: user:opus-to-planner
      profile: planner
      match: { model_pattern: "(claude-)?opus.*" }
    - id: user:haiku-to-local
      profile: reviewer-light
      match: { model_pattern: "(claude-.*)?haiku.*" }
```

### (c) agent_cli を役として混ぜる — 監査役に外部 claude CLI

`kind: agent_cli` は claude / codex / grok / antigravity の外部 CLI を「1 プロバイダ」として登録するアダプタである。**v2.9.0 以降は `coderouter-plugin-agents` のインストールと `providers.yaml` への `plugins.enabled: [agents]` が必須**(未導入だと `kind: agent_cli` の provider がある状態で `coderouter serve` が起動時エラーになる)。詳細は [`docs/backends/external-agents.md`](../backends/external-agents.md) を参照。

```bash
uv pip install "coderouter-plugin-agents @ git+https://github.com/zephel01/coderouter-plugin-agents"
```

```yaml
allow_paid: true
default_profile: reviewer-light

plugins:
  enabled: [agents]        # v2.9.0 以降、kind: agent_cli を使うなら必須

providers:
  - name: agent-claude-review
    kind: agent_cli
    model: opus
    paid: true
    capabilities: { streaming: false, tools: false }
    agent_cli: { agent: claude, sandbox_mode: read_only }
  - name: local-reviewer
    kind: openai_compat
    base_url: http://localhost:11434/v1
    model: qwen2.5-coder:7b

profiles:
  - name: reviewer-audit      # 監査役 = 外部 claude CLI（Opus サブスク）
    providers: [agent-claude-review]     # agent_cli は単独
  - name: reviewer-light      # 軽微レビュー = ローカル
    providers: [local-reviewer]
```

`agent_cli` の既定 `capabilities` は `streaming: false` / `tools: false` のため、fallback chain の中間には置けず、専用 profile の単独 provider か chain 終端に限定される([§7](#7-制限と既知の課題) の G6)。

### (d) 内容ベースの補助ルール

クライアントが役割を宣言しない時の保険として、内容ベースの matcher を単独で足すこともできる。

```yaml
auto_router:
  default_rule_profile: coder
  rules:
    - id: user:cjk-to-local
      profile: coder
      match: { cjk_ratio_min: 0.5 }        # CJK 濃い turn を local(税ゼロ)へ
    - id: user:long-context-to-planner
      profile: planner
      match: { content_token_count_min: 32000 }   # 長文脈を planner へ
```

## 6. 動作確認

1. **起動時**: `coderouter serve` の起動ログで `plugin-loaded`(agent_cli を使う場合)や設定読み込みエラーが無いことを確認する。
2. **実リクエストを1本流す**: Claude Code またはサブエージェント相当の curl リクエストを送る。
3. **auto-router のログを見る**: マッチしたルールは `auto-router-resolved` イベントとして記録され、`signals.model` に実際に届いた `model` 文字列が入る。これが「エイリアスのまま届くか、フルIDに展開されるか」を確認する一次情報になる。ログの形は概ね次の通り。

   ```json
   {"ts":"2026-07-11T10:03:21","level":"INFO","logger":"coderouter.routing.auto_router",
    "msg":"auto-router-resolved","rule_id":"user:opus-to-planner","resolved_profile":"planner",
    "signals":{"has_image":false,"code_fence_ratio":0.0,"content_len":42,
               "model":"opus","estimated_tokens":15,"has_tools":true}}
   ```

4. **意図した profile に着地しているか**: `resolved_profile` が狙った profile 名(`planner` / `reviewer-audit` 等)になっているかを確認する。想定と違えば `model_pattern` の `fullmatch` 漏れ(末尾 `.*` 忘れ等)を疑う。

## 7. 制限と既知の課題

- **1 ルール = 1 matcher(AND 不可)**: `RuleMatcher` は複合条件を書けない。「CJK かつ長文なら…」のような AND 合成は現状できない(G1)。回避するには、優先させたい条件を先に並べたルールを複数用意し、先着一致に任せる。
- **明示的サブエージェント指定チャネルは無い(検討中)**: 「これはサブエージェントで種別は X」を運ぶ専用チャネルは現時点で存在しない。実用上の答えは本ガイドの通り「model 名を振り分けキーに使う」こと。将来案として、system prompt や先頭メッセージに埋め込んだタグを検出する最優先ルールを auto_router の前段に足す構想はあるが、未着手である。
- **ccr(claude-code-router)のタグ方式との違い**: CodeRouter は wire 上のメタデータ(model 名・ヘッダ)を能動的に読んで振り分けるのに対し、ccr はプロンプト本文に埋め込まれたタグを検出・抽出して受動的に振り分ける方式を採る。どちらも「Anthropic wire にネイティブなサブエージェント識別子が無い」という同じ制約への異なる解法である。
- **UNCONFIRMED — 届く model 名の形**: Claude Code のサブエージェントが送るモデル名文字列が、エイリアス(`opus` 等)のまま届くかフル ID(`claude-opus-4-8` 等)に展開されるかは環境・バージョン依存であり、本ガイド執筆時点では未検証。`model_pattern` ルールを書いたら、必ず [§6 動作確認](#6-動作確認) の手順でログの `signals.model` を見て実値を確認すること。

## 8. 関連ドキュメント

- [`docs/backends/external-agents.md`](../backends/external-agents.md) — `agent_cli`(claude/codex/grok/antigravity)の詳細な設定リファレンス・認証・トラブルシューティング
- [`docs/guides/usage-guide.md`](./usage-guide.md) — CodeRouter 全般の使い方
- Claude Code 公式ドキュメント: <https://code.claude.com/docs/en/sub-agents>、<https://code.claude.com/docs/en/model-config>
