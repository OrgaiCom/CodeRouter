<h1 align="center">CodeRouter</h1>

<p align="center">
  <strong>ローカル LLM で Claude Code を動かすと壊れる問題、<br>ルーター 1 つで直します。</strong>
</p>

<p align="center">
  <a href="https://github.com/zephel01/CodeRouter/actions/workflows/ci.yml"><img src="https://github.com/zephel01/CodeRouter/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/coderouter-cli/"><img src="https://img.shields.io/pypi/v/coderouter-cli?include_prereleases&color=blue&label=pypi" alt="pypi"></a>
  <a href=""><img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="python"></a>
  <a href=""><img src="https://img.shields.io/badge/deps-5-brightgreen" alt="deps"></a>
  <a href=""><img src="https://img.shields.io/badge/license-MIT-yellow" alt="license"></a>
</p>

<p align="center">
  <a href="./README.en.md">English</a> · <strong>日本語</strong> · <a href="./docs/quickstart.md">10 分で動かす</a> · <a href="./docs/architecture.md">設計詳細</a>
</p>

---

## 何ができるか — 30 秒で

```
あなたのエージェント (Claude Code / gemini-cli / codex)
        │
        ▼
  ┌─ CodeRouter ─┐
  │  翻訳 + 修復  │──→  ① ローカル (Ollama — 無料・最速)
  │  ガード + 監視 │──→  ② 無料クラウド (OpenRouter / NIM)
  │  自動フォールバック │──→  ③ 有料 (Claude — opt-in 時のみ)
  └──────────────┘
```

**やってくれること:**

- ローカルモデルが壊した tool calling を Claude Code に届く前に修復する
- 1 つ目が落ちたら自動で次のプロバイダに切り替える
- 有料 API は明示的に許可したときだけ使う (デフォルトは無料のみ)
- 8 時間回しても止まらないように 6 種類のガードで守る
- 何がおかしいか `coderouter doctor` コマンド一発で診断する

---

## インストール (3 行)

```bash
# 1. サンプル設定を置く
mkdir -p ~/.coderouter
curl -fsSL https://raw.githubusercontent.com/zephel01/CodeRouter/main/examples/providers.yaml \
  > ~/.coderouter/providers.yaml

# 2. 起動 (Python 3.12+)
uvx --from coderouter-cli coderouter serve --port 8088
```

恒久インストールしたい場合: `uv tool install coderouter-cli`

---

## Claude Code で使う

```bash
# ターミナル 1
coderouter serve --port 8088

# ターミナル 2
ANTHROPIC_BASE_URL=http://localhost:8088 ANTHROPIC_AUTH_TOKEN=dummy claude
```

これだけ。Claude Code はいつも通り動きますが、裏ではローカルの Ollama が答えています。

---

## 自分に必要？

| あなたの状況 | CodeRouter は？ |
|---|---|
| Claude Code + ローカル Ollama で tool calling が壊れる | **必須** — wire 変換 + tool 修復 |
| Claude Code + ローカルで長時間回すと止まる | **便利** — 6 系統ガード + self-healing |
| codex / gemini-cli + Ollama 直繋ぎで動いてる | オプション — フォールバックが欲しいなら |
| Claude API を直接叩いてて問題ない | 不要 |

詳細は → [要否判定ガイド](./docs/when-do-i-need-coderouter.md)

---

## 主な機能

### 接続と修復

| 機能 | 何をしてくれるか |
|---|---|
| **Wire 翻訳** | Claude Code (Anthropic形式) ↔ Ollama (OpenAI形式) を自動変換 |
| **Tool-call 修復** | ローカルモデルがテキストで吐いた JSON を正しい tool_use ブロックに復元 |
| **3 層フォールバック** | ローカル → 無料クラウド → 有料の順に自動切替 |
| **出力フィルタ** | `<think>` タグ漏れ、stop marker 漏れを自動除去 |

### 長時間運用ガード

| ガード | 何から守るか |
|---|---|
| **Context Budget** | メッセージが溜まりすぎて context window 溢れ → 自動 trim |
| **Drift Detection** | モデルの応答品質が徐々に劣化 → 別 provider に切替 or KV cache flush (6 シグナル、`goal_mode` で目標達成停滞も検知) |
| **Self-healing** | backend が落ちた → 自動除外 + restart + 回復 probe で自動復帰 |
| **Tool Loop Guard** | 同じツールを無限に呼び続ける → 検知して停止 |
| **Memory Pressure** | GPU メモリ不足を検知 → 軽量モデルに切替 |
| **Mid-stream Guard** | 応答途中で落ちた → 溜まったテキストを安全に返却 |

### 診断と可視化

| 機能 | 何がわかるか |
|---|---|
| **`coderouter doctor`** | プロバイダの問題を 6 プローブで即診断 + 修正パッチ出力 |
| **`/dashboard`** | ブラウザで今何が起きてるかリアルタイム確認 |
| **`coderouter audit`** | guard 発火履歴を検索 |
| **`coderouter replay`** | provider 切替の効果を統計比較 (A/B 分析) / `--suggest-rules` でルール最適化提案 |
| **Continuous Probe** | idle 時も定期的に backend を監視 |

### Launcher — llama.cpp / vllm 起動 UI

`http://localhost:8088/launcher` で開けるブラウザ UI。llama.cpp や vllm を GUI で起動・管理できます。

| 機能 | 詳細 |
|---|---|
| **モデルスキャン** | `model_dirs` に指定したフォルダを再帰スキャンして `.gguf` / `.safetensors` をリスト化 |
| **オプションプロファイル** | `providers.yaml` に名前付きプリセットを定義 → ドロップダウンで選択するだけ |
| **複数プロセス管理** | llama.cpp と vllm を同時に起動し、ポートごとに独立管理 |
| **ログビューア** | 各プロセスの stdout/stderr をブラウザ内でリアルタイム確認 |

```yaml
# providers.yaml に追記するだけで有効になる
launcher:
  model_dirs:
    - ~/models
  option_profiles:
    llama.cpp:
      - name: "GPU フル活用"
        args:
          "-ngl": 99
          "--ctx-size": 4096
    vllm:
      - name: "標準"
        args:
          "--dtype": "auto"
          "--max-model-len": 4096
```

詳細 → [Launcher ガイド](./docs/launcher.md)

---

## 設定例 (最小)

```yaml
# ~/.coderouter/providers.yaml
default_profile: claude-code

profiles:
  - name: claude-code
    providers: [ollama-local, openrouter-free]

providers:
  - name: ollama-local
    kind: openai_compat
    base_url: http://localhost:11434/v1
    model: qwen3-coder:7b

  - name: openrouter-free
    kind: openai_compat
    base_url: https://openrouter.ai/api/v1
    model: qwen/qwen3-coder:free
    api_key_env: OPENROUTER_API_KEY
```

もっと詳しい設定 → [利用ガイド](./docs/usage-guide.md) · [設計詳細](./docs/architecture.md)

---

## ドキュメント

| やりたいこと | ドキュメント |
|---|---|
| すぐ動かす | [Quickstart](./docs/quickstart.md) |
| 使いこなす | [利用ガイド](./docs/usage-guide.md) |
| 無料で回す | [無料枠ガイド](./docs/free-tier-guide.md) |
| llama.cpp / vllm を GUI で起動 | [Launcher ガイド](./docs/launcher.md) |
| 詰まった | [トラブルシューティング](./docs/troubleshooting.md) |
| 設計を知りたい | [アーキテクチャ詳細](./docs/architecture.md) |
| 全リリース履歴 | [CHANGELOG](./CHANGELOG.md) |

English: [Quickstart](./docs/quickstart.en.md) · [Usage guide](./docs/usage-guide.en.md) · [Free-tier](./docs/free-tier-guide.en.md) · [Troubleshooting](./docs/troubleshooting.en.md)

---

## トラブルシューティング (早見表)

**まず**: `coderouter doctor --check-model <provider名>` を走らせてください。大体これで原因がわかります。

| 症状 | 原因 | 詳細 |
|---|---|---|
| 401 エラー | API キー未設定 / `.env` に `export` 忘れ | [§1](./docs/troubleshooting.md#1-起動設定で踏みやすい-5-つの罠-v162-追加) |
| 返信が空 / 意味不明 | Ollama の `num_ctx` が 2048 に切り詰め | [§3](./docs/troubleshooting.md#3-ollama-初心者--サイレント失敗-5-症状-v07-c) |
| `<think>` タグが漏れる | `output_filters: [strip_thinking]` を付ける | [§3](./docs/troubleshooting.md#3-ollama-初心者--サイレント失敗-5-症状-v07-c) |
| Claude Code でツール呼び出しがおかしい | tool-call 修復が効いてない | [§4](./docs/troubleshooting.md#4-claude-code-連携で踏みやすい罠-v162-追加) |

`http://localhost:8088/dashboard` を開いておくと、ほとんどの問題が見て 10 秒でわかります。

---

## 技術スペック

- **ランタイム依存**: `fastapi` / `uvicorn` / `httpx` / `pydantic` / `pyyaml` の 5 個のみ
- **テスト**: 964 本 (41 sub-release 連続で依存追加なし)
- **対応 OS**: macOS (Apple Silicon 推奨) / Linux / Windows WSL2
- **対応 backend**: Ollama / llama.cpp / LM Studio / vLLM / MLX-LM / OpenRouter / NVIDIA NIM / Anthropic API
- **ライセンス**: MIT

---

## エコシステム

CodeRouter は backend ルーター層として独立して動きます。`OPENAI_BASE_URL` を CodeRouter に向けるだけで、他プロジェクトを無改造で吸収:

- **[Voice Bridge](https://github.com/zephel01/voice-bridge)** — リアルタイム音声翻訳 + AI 音声チャット。CodeRouter 経由でローカル LLM のフォールバックを効かせると、ずんだもんが沈黙しなくなる

---

## Security

シークレットは環境変数に置きます。[`docs/security.md`](./docs/security.md) に完全な方針と報告手順があります。

## License

MIT
