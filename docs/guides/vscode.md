# VSCode 連携ガイド — CodeRouter を VSCode 上のエージェント拡張から使う

> English: (未整備)

VSCode（および Cursor / Windsurf / VSCodium）上で動く AI 拡張から CodeRouter に繋ぐためのガイドです。結論を先に: **Claude Code は `coderouter vscode-init` で一発、他拡張は下のスニペットをコピペ**。

`ANTHROPIC_BASE_URL` を毎回シェルに export し忘れる、`.envrc` を書いたはいいがワークスペース外に漏れる、といった罠を CLI で構造的に避けます。

---

## 前提 — CodeRouter の 2 つの入口

`coderouter serve --port 8088` を起動すると、以下の 2 つの入口が同じプロセスで待ち受けます。

| 入口 | パス | 使う拡張 |
|---|---|---|
| **Anthropic 互換** | `http://localhost:8088`（`/v1/messages`） | Claude Code |
| **OpenAI 互換** | `http://localhost:8088/v1`（`/v1/chat/completions`） | Cline / Roo Code / Kilo Code / Continue.dev |

`coderouter serve` の `--port` を指定していない場合、既定は **4000** です。README・本ガイド・`docs/backends/*.md` はすべて 8088 を前提に書いているので、迷ったら `--port 8088` で揃えるのが楽です。

> 別 PC から繋ぐ場合は [remote-access.md](./remote-access.md) を先に。

---

## Claude Code — `coderouter vscode-init` で自動化

VSCode の統合ターミナルから `claude` を叩くとき、`ANTHROPIC_BASE_URL` と `ANTHROPIC_AUTH_TOKEN` が環境変数として渡っている必要があります。手作業だと **シェル起動時のみ有効・別プロジェクトに漏れる・claude.ai コネクタと競合** といった罠を踏みがちなので、`vscode-init` に任せます。

### 使い方

プロジェクトルートで一発:

```bash
cd /path/to/your/project
coderouter vscode-init
```

これで `.vscode/settings.json` に以下が **マージ書き込み** されます（既存キーは触りません）:

```json
{
  "terminal.integrated.env.osx":     { "ANTHROPIC_BASE_URL": "http://localhost:8088", "ANTHROPIC_AUTH_TOKEN": "dummy" },
  "terminal.integrated.env.linux":   { "ANTHROPIC_BASE_URL": "http://localhost:8088", "ANTHROPIC_AUTH_TOKEN": "dummy" },
  "terminal.integrated.env.windows": { "ANTHROPIC_BASE_URL": "http://localhost:8088", "ANTHROPIC_AUTH_TOKEN": "dummy" }
}
```

以後、そのプロジェクトを VSCode で開き、統合ターミナルで `claude` と打つだけで CodeRouter 経由になります。**そのワークスペースにいる間だけ**環境変数が効くので、他プロジェクトや claude.ai には影響しません。

### 主なオプション

```bash
coderouter vscode-init [--target PATH]
                       [--port PORT]        # デフォルト 8088
                       [--profile NAME]     # CODEROUTER_MODE を追加
                       [--with-envrc]       # direnv 用 .envrc も生成
                       [--dry-run]          # 差分だけ表示、書き込まない
                       [--force]            # 既存の異なる値を上書き
```

- `--port 4000`: `coderouter serve` を素で起動して 4000 で動かしている場合
- `--profile local-first`: 常に `local-first` プロファイルへルーティング（`CODEROUTER_MODE=local-first` が terminal env に載る）
- `--dry-run`: `.vscode/settings.json` に何を書くかの unified diff だけ表示
- `--force`: 既存の `ANTHROPIC_BASE_URL` などが違う値だったとき上書き（既定はコンフリクト報告のみで書かない）

### 再実行しても壊れない

`vscode-init` は冪等です。同じ引数で再実行すれば `unchanged` を報告して終了。異なる値と衝突した場合は `conflict` を出して**ファイルに触りません**（`--force` 必要）。オンボーディングスクリプトに含めて安全です。

### direnv 派の場合

```bash
coderouter vscode-init --with-envrc
```

これで `.envrc` も生成されます。生成後に **`direnv allow`** を 1 回実行してください。シークレットは `.envrc.local` に分けて `source_env_if_exists .envrc.local` の形が安全です（`.envrc` はプロジェクトに commit することが多いので）。

### 注意点

- `.vscode/` は CodeRouter リポジトリの `.gitignore` に既に入っているため、`.vscode/settings.json` は既定で **git に含まれません**。あなたのプロジェクトの `.gitignore` は個別に確認してください
- `ANTHROPIC_AUTH_TOKEN` はダミー値です（CodeRouter は検証しない）。**本物の API キーは絶対に置かない**でください
- **claude.ai コネクタとの競合**: `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` がグローバルに export されていると、`claude.ai connectors are disabled…` のエラーが出ます。`vscode-init` はワークスペース内のターミナルにだけ export するので構造的に安全ですが、既存の `.zshrc` / `.bashrc` に手書きで残っていたら消してください

---

## Cline / Roo Code / Kilo Code — 手動設定

これらは自前の設定 UI を持つので、`vscode-init` は触りません。拡張の設定画面で以下を入れます。

| 項目 | 値 |
|---|---|
| API Provider | **OpenAI Compatible** |
| Base URL | `http://localhost:8088/v1` |
| API Key | `dummy`（任意の非空文字列） |
| Model ID | 任意（CodeRouter が `default_profile` / `auto_router` で解決） |

プロファイルを明示指定したい場合は、拡張がカスタムヘッダを送れるなら `X-CodeRouter-Profile: <プロファイル名>` を追加してください。送れない拡張なら、`providers.yaml` の `default_profile` を切り替えるか、`auto_router.rules` の `model_pattern` で拡張が送る Model ID にマッチさせます。

---

## Continue.dev — `config.json` にスニペット追記

`~/.continue/config.json` の `models` 配列に以下を追加:

```json
{
  "title": "CodeRouter",
  "provider": "openai",
  "model": "any-model-id",
  "apiBase": "http://localhost:8088/v1",
  "apiKey": "dummy"
}
```

Continue はモデル ID をそのままサーバに渡すので、CodeRouter 側で `auto_router.rules` に `model_pattern: any-model-id` のようなマッチを書けば、Continue から届いたリクエストを狙いのプロファイルへ回せます。

Anthropic 互換入口（`/v1/messages`）を叩きたい場合は `"provider": "anthropic"` + `"apiBase": "http://localhost:8088"` にしてください。

---

## プロファイルの指定順（precedence）

どの入口・どの拡張から届いたリクエストも、CodeRouter が最終的にどのプロファイルを使うかは次の順で決まります:

```
body.profile > X-CodeRouter-Profile ヘッダ > X-CodeRouter-Mode ヘッダ > auto_router > default_profile
```

拡張がカスタムヘッダ / body に何も足せない場合は、`default_profile` か `auto_router` で受けます。

---

## トラブルシューティング

### `claude.ai connectors are disabled...` が出る

`ANTHROPIC_AUTH_TOKEN` または `ANTHROPIC_API_KEY` がグローバル環境に残っています。`.zshrc` / `.bashrc` を確認して手書きの `export` を消し、ワークスペーススコープ（`vscode-init` が書く `terminal.integrated.env.*` か direnv `.envrc`）だけに絞ってください。

### `vscode-init` が `conflict` を出す

既存 `.vscode/settings.json` の `ANTHROPIC_BASE_URL` が違う値のときの安全側動作です。`--dry-run` で差分を確認し、上書きしていいなら `--force` で再実行。

### 統合ターミナルの `claude` が繋がらない

VSCode を**開き直してください**。`terminal.integrated.env.*` は新規ターミナル起動時にしか反映されません。既存のターミナルは古い env を持ち続けます。

### Cline 等から 403 `Host '...' is not allowed`

CodeRouter 側の Host 検証（DNS リバインディング対策）に引っかかっています。localhost で完結する構成なら発生しないはずですが、`--host 0.0.0.0` にしていたり、別ホスト名で叩いている場合は `CODEROUTER_ALLOWED_HOSTS` を設定してください（詳細は [remote-access.md](./remote-access.md)）。

### 別 PC の VSCode から繋ぎたい

[remote-access.md](./remote-access.md) の SSH トンネル or Tailscale が推奨。トンネル側で `localhost:8088` として現れるようにすれば、上記のスニペット・`vscode-init` はそのまま動きます。

---

## 関連

- [Quickstart](../start/quickstart.md) — CodeRouter の 10 分導入
- [利用ガイド](./usage-guide.md) — `providers.yaml` の書き方
- [リモートアクセス](./remote-access.md) — 別 PC から繋ぐ
- [セキュリティ](./security.md) — 信頼境界と脅威モデル
