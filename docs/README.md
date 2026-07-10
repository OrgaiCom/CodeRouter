# CodeRouter ドキュメント / Documentation

English: [`README.en.md`](./README.en.md)

CodeRouter の公開ドキュメント索引です。「やりたいこと」から読むべきページを引けるよう整理しています。
Index of CodeRouter's public documentation — find the right page by what you want to do.

> 開発者向けの内部メモ・記事原稿は `inside/` と `articles/` にあります(ローカル専用 / gitignored、公開リポジトリには含まれません)。
> Developer-internal notes and article drafts live in `inside/` and `articles/` (local-only, not shipped in the public repo).

---

## こういう時はこれを読む / Quick start by goal

| やりたいこと / Goal | 読むもの / Read |
|---|---|
| 今すぐ動かしたい / Get running now | [start/quickstart](start/quickstart.md) |
| 自分に必要か知りたい / Is it for me? | [start/when-do-i-need-coderouter](start/when-do-i-need-coderouter.md) |
| 無料で運用したい / Run for free | [guides/free-tier-guide](guides/free-tier-guide.md) |
| 機能を一通り知りたい / Learn the features | [guides/usage-guide](guides/usage-guide.md) |
| 言語税を計測・回避したい / Measure & avoid the language tax | [guides/language-tax](guides/language-tax.md) |
| エラーで詰まった / Something broke | [guides/troubleshooting](guides/troubleshooting.md) |
| ローカル LLM を起動したい / Launch a local LLM | [backends/launcher-quickstart](backends/launcher-quickstart.md) |
| APIキー・機密の扱い / Secrets & security | [guides/security](guides/security.md) |
| 仕組みを理解したい / Understand the design | [concepts/architecture](concepts/architecture.md) |
| プラグインで拡張したい / Extend with plugins | [対応プラグイン / Plugins](#対応プラグイン--plugins) |

---

## 構成 / Layout

```
docs/
├── start/             はじめに / Getting started
├── guides/            使い方ガイド / How-to guides
├── backends/          ローカルLLMバックエンド / Local LLM backends
├── concepts/          設計・内部動作 / Architecture & internals
├── designs/           設計ドキュメント / Design docs
├── retrospectives/    リリース振り返り / Release retrospectives
├── evidence/          検証ログ / Verification logs
├── openrouter-roster/ OpenRouter モデル一覧 / OpenRouter model roster
└── assets/            画像など / Images
```

各ドキュメントは日本語版 (`.md`) と英語版 (`.en.md`) が揃っているものがあります。
Many documents have a Japanese version (`.md`) and an English version (`.en.md`).

---

## 1. はじめに / Getting started — `start/`

初めて CodeRouter に触れる人向け。 / For first-time users.

- **quickstart** — 最短セットアップで動かす / Get running in one sitting · [日本語](start/quickstart.md) · [English](start/quickstart.en.md)
- **when-do-i-need-coderouter** — 自分に必要かを判断する / Decide whether you need it · [日本語](start/when-do-i-need-coderouter.md) · [English](start/when-do-i-need-coderouter.en.md)

## 2. 使い方ガイド / How-to guides — `guides/`

日常的に使いこなすためのガイド。 / Day-to-day usage.

- **usage-guide** — 機能を一通り使いこなす / Full feature guide · [日本語](guides/usage-guide.md) · [English](guides/usage-guide.en.md)
- **language-tax** — 日本語の言語税を計測・ルーティング回避・可視化 / Measure, route around, and visualize the CJK language tax · [日本語](guides/language-tax.md) · [English](guides/language-tax.en.md)
- **free-tier-guide** — NVIDIA NIM × OpenRouter Free でコストゼロ運用 / Zero-cost operation · [日本語](guides/free-tier-guide.md) · [English](guides/free-tier-guide.en.md)
- **troubleshooting** — つまずいたときの解決集 / Fixing problems · [日本語](guides/troubleshooting.md) · [English](guides/troubleshooting.en.md)
- **security** — シークレット管理とセキュリティ方針 / Secrets handling & security posture · [日本語](guides/security.md) · [English](guides/security.en.md)

## 3. ローカル LLM バックエンド / Local LLM backends — `backends/`

ローカル推論バックエンドの導入・起動・接続。 / Installing, launching, and connecting local inference backends.

- **install-backends** — llama.cpp / vLLM / MLX のインストール手順 / Installing the three backends · [日本語](backends/install-backends.md) · [English](backends/install-backends.en.md)
- **launcher-quickstart** — バックエンド導入から起動までの最短手順 / Install a backend and launch · [日本語](backends/launcher-quickstart.md)
- **launcher** — Launcher ガイド(Web版・デスクトップGUI版) / Launcher guide (Web & Desktop GUI) · [日本語](backends/launcher.md)
- **external-agents** — 外部コーディングエージェント CLI (agent_cli、v2.7.7・claude のみ) / External coding-agent CLI (agent_cli, v2.7.7, claude only) · [日本語](backends/external-agents.md) · [English](backends/external-agents.en.md)
- **llamacpp-direct** — llama.cpp に直結する / Connect llama.cpp directly · [日本語](backends/llamacpp-direct.md) · [English](backends/llamacpp-direct.en.md)
- **lmstudio-direct** — LM Studio に直結する / Connect LM Studio directly · [日本語](backends/lmstudio-direct.md) · [English](backends/lmstudio-direct.en.md)
- **hf-ollama-models** — HuggingFace 配布モデルを Ollama で使う / Use HF models via Ollama · [日本語](backends/hf-ollama-models.md)
- **gguf_dl** — GGUF モデルのダウンロードツール / GGUF download helper · [日本語](backends/gguf_dl.md)
- **verify-ollama-0.23.1** — Ollama v0.23.1 実機検証チェックリスト / Ollama verification checklist · [日本語](backends/verify-ollama-0.23.1.md)

## 4. 設計・内部動作 / Architecture & internals — `concepts/`

CodeRouter の仕組みと信頼性機構。 / How CodeRouter works and its reliability mechanisms.

- **architecture** — アーキテクチャ全体像 / Architecture overview · [日本語](concepts/architecture.md)
- **context-budget** — コンテキスト予算管理 (v2.0.0) / Context budget management · [日本語](concepts/context-budget.md)
- **drift-detection** — ドリフト検出 (v2.0-G) / Drift detection · [日本語](concepts/drift-detection.md)
- **partial-stitch** — ストリーム途中の部分ステッチ (v2.0-H) / Mid-stream partial stitching · [日本語](concepts/partial-stitch.md)
- **continuous-probing** — 継続プロービング (v2.0-I) / Continuous probing · [日本語](concepts/continuous-probing.md)

## 5. 設計資料・記録 / Design docs & records

- **designs/** — 機能の設計ドキュメント / Feature design docs ([v1.6 auto-router](designs/v1.6-auto-router.md) ほか)
- **retrospectives/** — リリース振り返り / Release retrospectives ([v0.4](retrospectives/v0.4.md) 〜 [v1.0](retrospectives/v1.0.md))
- **evidence/** — 実機検証ログ / Verification run logs
- **openrouter-roster/** — OpenRouter 利用可能モデル一覧 / OpenRouter model roster — [README](openrouter-roster/README.md)

---

## 対応プラグイン / Plugins

CodeRouter は v2.3.0 で入った **Plugin SDK** により、別パッケージのプラグインを *opt-in* で読み込めます。`plugins.enabled` に名前を明示したときだけ作動する（サプライチェーン防御）ため、インストールしただけでは何も起きません。各プラグインは独立した PyPI パッケージなので、**コアの依存は一切増えません**。

CodeRouter's **Plugin SDK** (since v2.3.0) loads out-of-tree plugins *opt-in*: a plugin runs only when its name is listed in `plugins.enabled` (supply-chain defense), so installing one does nothing by itself. Each plugin ships as a separate PyPI package, so **the core's dependencies never grow**.

| プラグイン / Plugin | 何をするか / What it does | インストール / Install | リポジトリ / Repo |
|---|---|---|---|
| **compress** | ツール出力（JSON / ログ）を LLM に届く前に圧縮してトークンを削減。原文はローカル保持で可逆（CCR）。`cache-align` で Anthropic プロンプトキャッシュも整列。<br>Compresses tool output (JSON / logs) before it reaches the LLM to cut tokens; originals kept locally and reversible (CCR). `cache-align` also aligns Anthropic prompt caching. | `pip install coderouter-plugin-compress` | [coderouter-plugin-compress](https://github.com/zephel01/coderouter-plugin-compress) |
| **memory** | 応答から key facts を抽出して `facts.jsonl` に蓄積し、次セッションの system prompt へ自動注入。「毎回同じ説明」を wire 層で解消。<br>Extracts key facts from responses into `facts.jsonl` and auto-injects them into the next session's system prompt — solving "explain it every time" at the wire layer. | `pip install coderouter-plugin-memory` | [coderouter-plugin-memory](https://github.com/zephel01/coderouter-plugin-memory) |

有効化は `providers.yaml` に追記するだけ。起動ログに `plugin-loaded` が出れば有効です。
Enable by adding to `providers.yaml`; a `plugin-loaded` line in the startup log confirms activation.

```yaml
plugins:
  enabled:
    - compress          # ツール出力を圧縮 / compress tool output
    - compress-stats    # 圧縮率を coderouter stats に出力 / report compression ratio
    - cache-align       # プロンプトキャッシュのブレークポイント整列 / align prompt-cache breakpoints
    - memory            # セッション横断メモリ / cross-session memory
  config:
    compress:
      mode: safe        # off | safe | aggressive
      ccr: true         # 圧縮の可逆復元（既定 on）/ reversible re-expansion (default on)
    memory:
      consolidate_model: qwen3:1.7b
```

各プラグインの詳細・設定は上記リポジトリの README を参照してください。
See each plugin's repo README for full configuration.

---

最終更新 / Last updated: 2026-06-24
