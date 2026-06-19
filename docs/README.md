# CodeRouter ドキュメント / Documentation

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

最終更新 / Last updated: 2026-05-22
