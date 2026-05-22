# Launcher クイックスタート — バックエンド導入から起動まで

CodeRouter Launcher を初めて使うための手引きです。Launcher が起動・管理するバックエンド（llama.cpp / vLLM）の導入から、Launcher の起動・Claude Code 接続までを通しで説明します。

対象プラットフォーム: macOS (Apple Silicon) / Linux

---

## 全体の流れ

1. バックエンド（**llama.cpp** か **vLLM**）をインストール
2. モデル（`.gguf` 等）を用意
3. `providers.yaml` に `launcher:` ブロックを書く
4. Launcher を起動（デスクトップGUI版 または Web版）
5. Launcher からバックエンド＋CodeRouter を起動 → Claude Code を接続

llama.cpp と vLLM はどちらか一方があれば始められます。**迷ったら llama.cpp を推奨** — macOS でも Linux でも動き、`.gguf` モデルが豊富で、セットアップが軽量です。vLLM は Linux + NVIDIA GPU 向けです。

---

## 1. llama.cpp をインストール

OpenAI 互換 API を提供する `llama-server` を用意します。

### 方法 A — Homebrew（macOS / Linux、最も簡単）

```bash
brew install llama.cpp
```

`llama-server` が PATH に入ります。これで完了です。

### 方法 B — ソースからビルド（最新版・GPU 最適化したい場合）

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

**macOS (Apple Silicon)** — Metal は既定で有効:

```bash
cmake -B build
cmake --build build --config Release -j
```

**Linux (NVIDIA CUDA)** — CUDA Toolkit が必要:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

ビルド後、サーバーバイナリは `build/bin/llama-server` に生成されます。このフルパスを後で Launcher に設定します。

> CUDA と Metal は同一バイナリに同梱できません。実行マシンに合わせてそれぞれビルドしてください。

---

## 2. vLLM をインストール（任意）

vLLM は **Linux + NVIDIA GPU (CUDA)** 向けの高速推論サーバーです。macOS では CPU バックエンドのみで実用的ではないため、**macOS なら llama.cpp を使ってください**。

`uv`（高速な Python 環境管理ツール）での導入が推奨されています:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

`pip` でも可:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install vllm
```

Launcher は vLLM を `<python> -m vllm.entrypoints.openai.api_server` の形で起動します。後の設定では、この **vLLM を入れた venv の python** のパスを指定します（例: `~/.venv/bin/python`）。

---

## 3. モデルを用意

llama.cpp は `.gguf` 形式のモデルを使います。Hugging Face などから入手し、1 つのディレクトリにまとめておきます（例: `~/llm/models/`）。サブフォルダも再帰的にスキャンされます。

vLLM は Hugging Face のモデル ID、またはローカルパスを使います。

---

## 4. providers.yaml に launcher ブロックを書く

Launcher はモデル一覧・オプションプロファイル・バイナリパスを `~/.coderouter/providers.yaml` の `launcher:` ブロックから読み込みます。

```yaml
# ~/.coderouter/providers.yaml
launcher:
  model_dirs:
    - ~/llm/models                      # .gguf 等を再帰検索
  backends:
    llama.cpp:
      # 方法B でビルドした場合はフルパスを指定。
      # Homebrew (方法A) なら backends ごと省略可（PATH から自動解決）。
      binary: ~/llama.cpp/build/bin/llama-server
    vllm:
      binary: ~/.venv/bin/python        # vLLM を入れた venv の python
  option_profiles:
    llama.cpp:
      - name: "GPU フル活用"
        args:
          "-ngl": 99
          "--ctx-size": 32768
```

テンプレートは `launcher_profiles.yaml.example` をコピーして始められます。設定項目の詳細は [Launcher ガイド（Web版）の設定リファレンス](./launcher.md#設定リファレンス) を参照してください。

---

## 5. Launcher を起動

Launcher には 2 種類あります。初回は **デスクトップGUI版** が簡単です（CodeRouter 自体もそこから起動できます）。

### デスクトップGUI版 — ブラウザ不要

CodeRouter のリポジトリ直下で:

```bash
python3 launcher_gui.py
# または CodeRouter の venv 経由（PyYAML を確実に使う）
uv run python launcher_gui.py
```

ウィンドウが開いたら:

1. MODELS から使うモデルをクリック（メモリ的に `✓ 推奨` のものが安心）
2. オプションプロファイルを選び「▶ llama.cpp / vllm 起動」
3. 上部バーの「▶ CodeRouter 起動」
4. 表示される接続文字列をコピー

詳細は [Launcher ガイド（デスクトップGUI版）](./launcher-gui.md)。

### Web版 — CodeRouter 稼働中の運用 UI

Web版は CodeRouter の中で動くため、先に CodeRouter を起動します:

```bash
coderouter serve --port 8088
```

ブラウザで `http://localhost:8088/launcher` を開き、モデルを選んで「▶ 起動」します。

詳細は [Launcher ガイド（Web版）](./launcher.md)。

---

## 6. Claude Code から使う

CodeRouter が稼働したら、Claude Code を CodeRouter に向けて起動します:

```bash
ANTHROPIC_BASE_URL=http://localhost:8088 ANTHROPIC_AUTH_TOKEN=dummy claude
```

デスクトップGUI版では、この接続文字列が画面上部に表示されコピーできます。

---

## つまずいたら

| 症状 | 対処 |
|---|---|
| 起動ボタンがグレーアウト | バックエンドのバイナリが見つからない。`backends.<name>.binary` にフルパスを設定 |
| モデル一覧が空 | `launcher.model_dirs` を設定し、`.gguf` 等が入っているか確認 |
| `PyYAML が見つかりません`（デスクトップ版） | `uv run python launcher_gui.py` で CodeRouter の venv から実行 |
| vLLM が macOS で遅い／動かない | vLLM は Linux/CUDA 向け。macOS では llama.cpp を使う |
| モデルに `⚠ メモリ厳しい` と出る | 搭載メモリに対しモデルが大きい。より小さい量子化版を選ぶ |

さらに詳しいトラブルシューティングは各 Launcher ガイドを参照してください。

---

## 関連ドキュメント

- [Launcher ガイド（デスクトップGUI版）](./launcher-gui.md)
- [Launcher ガイド（Web版）](./launcher.md)
- [CodeRouter クイックスタート](./quickstart.md)
- [llama.cpp 直接接続ガイド](./llamacpp-direct.md)
