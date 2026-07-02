# Launcher ガイド — llama.cpp / vllm / mlx を GUI で起動する

CodeRouter Launcher は、ローカル推論バックエンド(llama.cpp / vllm / mlx)を**画面の操作で起動・管理**するツールです。長い起動コマンドを毎回打つ代わりに、モデルを選んでボタンを押すだけで起動できます。

Launcher には 2 つの形態があります。

- **デスクトップGUI版**(`launcher_gui.py`)— tkinter 製のデスクトップアプリ。ブラウザ不要。CodeRouter 自体もここから起動できる。
- **Web版**(`/launcher`)— CodeRouter が配信するブラウザページ。

設定(`providers.yaml` の `launcher:` ブロック)・画面構成・トラブルシューティングは両版で共通です。本ガイドは共通部を 1 回ずつ記載しています。

> バックエンドの導入手順は [バックエンド インストール手順書](./install-backends.md)、導入から起動までの通し手順は [Launcher クイックスタート](./launcher-quickstart.md) を参照してください。

---

## 概要 — Launcher でできること

- `model_dirs` 配下の `.gguf` / `.safetensors` 等を再帰スキャンしてモデル一覧を表示
- オプションプロファイル(プリセット)をドロップダウンで選んで起動
- 複数プロセスの同時管理(llama.cpp + vllm 並走など)
- 各プロセスのログをリアルタイム確認
- 搭載メモリと照らしたモデルの[メモリ推奨](#メモリ推奨)表示

---

## 2 つのランチャー — どちらを使うか

| | デスクトップGUI版(`launcher_gui.py`) | Web版(`/launcher`) |
|---|---|---|
| 形態 | tkinter デスクトップアプリ | ブラウザページ |
| CodeRouter の起動 | **このアプリから起動できる** | できない(CodeRouter の中で動くため) |
| 主な用途 | 最初の一発 — backend と CodeRouter をまとめて立ち上げる | CodeRouter 稼働中に backend を管理する運用 UI |
| 設定 | `providers.yaml` の `launcher:` ブロック(共通) | 同左 |

両者は競合ではなく補完関係です。**最初の一発(ブートストラップ)はデスクトップ版、CodeRouter が回り始めた後の日常運用は Web版**、という住み分けになります。

---

## デスクトップGUI版 — 起動方法

`launcher_gui.py` は backend と CodeRouter を**ブラウザなし**で起動・管理する tkinter 製アプリです。CodeRouter 自体もこの GUI から直接起動でき、ローカル LLM を Claude Code に繋ぐまでを 1 ウィンドウで完結できます。

### 必要なもの

- Python 3.10 以上
- tkinter — Python 標準ライブラリ(追加インストール不要。一部の Linux では `python3-tk` パッケージが別途必要)
- PyYAML — CodeRouter の既存依存。CodeRouter の venv から実行すれば自動的に揃う

### 起動

```bash
# 通常起動
python3 launcher_gui.py

# CodeRouter の venv 経由(PyYAML を確実に使う)
uv run python launcher_gui.py

# 設定ファイルを明示指定
python3 launcher_gui.py --config ~/.coderouter/providers.yaml
```

設定ファイルの探索順: ① `--config` 指定 → ② カレントの `providers.yaml` → ③ `~/.coderouter/providers.yaml`。どれも無ければ空の設定で起動します(UI から手動入力すれば起動自体は可能)。

### CodeRouter バー(デスクトップ版のみ)

デスクトップ版の最上部には、Web版に無い **CodeRouter バー**があります。

- ステータスドット — `停止中` / `起動中…` / `稼働中` / `エラー` を色付き表示
- ポート — CodeRouter のリッスンポート(既定 `8088`)。停止中・エラー時のみ編集可
- ▶ CodeRouter 起動 / ■ 停止
- Claude Code 接続文字列 — `ANTHROPIC_BASE_URL=http://localhost:<ポート> ANTHROPIC_AUTH_TOKEN=dummy claude`。クリックまたは「コピー」でクリップボードへ

CodeRouter 起動時、`~/.coderouter/providers.yaml` が無ければ最小構成を自動生成します(この自動生成ファイルには `launcher:` ブロックは含まれません — 後述)。ウィンドウを閉じると、起動した CodeRouter と全 backend プロセスは自動的に停止します。

---

## Web版 — 起動方法

CodeRouter が稼働しているとき、ブラウザで使う運用 UI です。

1. `providers.yaml` に `launcher:` セクションを追加([設定リファレンス](#設定リファレンス)参照)
2. CodeRouter を起動 — `coderouter serve --port 8088`
3. ブラウザで `http://localhost:8088/launcher` を開く

---

## 画面の使い方

Launcher の画面は「MODELS パネル」「LAUNCH フォーム」「PROCESSES テーブル」「ログ」で構成されます。見た目はデスクトップ版(tkinter)と Web版(ブラウザ)で異なりますが、**構成と操作は共通**です。

### MODELS パネル

- スキャンボタンで `model_dirs` を再スキャンしてモデル一覧を更新
- モデル名をクリックすると「モデルパス」欄に自動入力(デスクトップ版は「名前」も自動入力。手入力した名前は保持される)
- ファイルサイズ (GB) を併記。VRAM / メモリと相談しやすい
- 各モデルに**メモリ推奨バッジ**(`✓ 推奨` / `⚠ メモリ厳しい`)を表示 → [メモリ推奨](#メモリ推奨)
- ヘッダに検出ハード(例: `Metal · RAM 64GB`)を表示
- 対象拡張子: `.gguf` `.safetensors` `.bin` `.pt` `.pth` `.ggml`(サブフォルダも再帰検索)

### LAUNCH フォーム

| 項目 | 説明 |
|---|---|
| **名前** | 管理用の任意の識別子(例: `qwen-coder-8080`) |
| **ポート** | 起動するサーバーのポート(既定 `8080`) |
| **バックエンド** | `llama.cpp` / `vllm` / `mlx` から選択。解決されたバイナリパスと利用可否が下に表示される |
| **モデルパス** | MODELS パネルから選択するか直接入力 |
| **オプションプロファイル** | `providers.yaml` で定義したプリセットを選択 |
| **追加オプション** | プロファイルにないフラグをその場で入力。`shlex` でパースされコマンド末尾に追加される |

`▶ 起動` でプロセスが起動し、PROCESSES テーブルに表示されます。バイナリが見つからない場合は**起動ボタンが自動的に無効化**され、理由が表示されます。「追加オプション」欄の横の **⚙ 推奨値** ボタンについては [メモリ推奨](#メモリ推奨) を参照してください。

### PROCESSES テーブル

起動した backend プロセスの一覧です。NAME / BACKEND(llama.cpp / vllm / mlx)/ MODEL / PORT / PID / STATUS(`starting` / `running` / `stopped` / `error` を色分け)を表示し、プロセスを選んで **停止**(SIGTERM)・**削除**(レジストリから除去)・**ログ表示**ができます。

### ログ

選択中プロセスの標準出力 / 標準エラーをリアルタイム表示します。Web版はログパネルが running 中に 3 秒ごと自動更新されます。長時間稼働でもメモリを圧迫しないよう、保持行数・表示行数に上限が設けられています。

### 典型的な使い方(デスクトップ版)

1. **モデルを選ぶ** — MODELS から使うモデルをクリック
2. **backend を起動** — オプションプロファイルを選び起動ボタンを押す。PROCESSES に `running` で表示される
3. **CodeRouter を起動** — 上部バーの「▶ CodeRouter 起動」
4. **Claude Code を繋ぐ** — 接続文字列をコピーしてターミナルで実行

---

## メモリ推奨

MODELS 一覧の各モデルには、CodeRouter を動かしているマシンの搭載メモリ(Apple Silicon は統合メモリ、NVIDIA GPU は VRAM、それ以外は RAM)と照らした判定が表示されます。

- **✓ 推奨** — 余裕を持って動く目安(`モデルサイズ × 1.2 + 2GB` が利用可能メモリ以内)
- **⚠ メモリ厳しい** — 収まらない／余裕が乏しい。スワップして大幅に遅くなる可能性

「追加オプション」欄の横の **⚙ 推奨値** ボタンは、選択中モデル・ハード・**バックエンド**に応じた起動フラグの目安を同欄に入れます。出力はバックエンドで異なります。

- **llama.cpp** — `-ngl`(GPU に載るなら `99`・CPU のみ `0`)/ `--ctx-size`(空きメモリに応じ `4096`〜`32768`)/ `--threads`(CPU コア数 − 2)
- **vllm** — 空。`--max-model-len` 等はモデルの実コンテキスト長に依存するため、エンジンの自動導出に任せます
- **mlx** — 空。統合メモリ前提で、起動時の調整フラグは不要です

いずれも**目安**で、他プロセスのメモリ使用や量子化方式までは考慮しません。実機で調整してください。

---

## 設定リファレンス

MODELS 一覧・オプションプロファイル・バイナリパスは `~/.coderouter/providers.yaml` の `launcher:` ブロックから読み込まれます。**デスクトップ版・Web版で共通**です。

### `launcher:` ブロック全体

```yaml
# ~/.coderouter/providers.yaml
launcher:
  model_dirs:           # list[str]  必須
    - ~/llm/models
  backends:             # dict  省略可
    llama.cpp:
      binary: null      # null = PATH の llama-server
    vllm:
      binary: null      # null = PATH の python
    mlx:
      binary: null      # null = PATH の python
  option_profiles:      # dict  省略可
    llama.cpp: [...]
    vllm: [...]
```

> CodeRouter 起動ボタンが自動生成する `providers.yaml` には `launcher:` ブロックは含まれません。モデル一覧やプロファイルを使うには `launcher:` ブロックを自分で用意してください。テンプレートは `launcher_profiles.yaml.example` をコピーして始められます。

### `backends` — バイナリパス設定

バイナリが PATH に無い場合(ソースビルド、venv 環境など)にフルパスを指定します。

```yaml
launcher:
  backends:
    llama.cpp:
      binary: ~/llama.cpp/build/bin/llama-server         # ソースビルド例
    vllm:
      binary: ~/.coderouter/backends/vllm/bin/python     # venv 例
    mlx:
      binary: ~/.coderouter/backends/mlx/bin/python      # venv 例
```

`binary` を省略または `null` にすると、PATH からデフォルト名(`llama-server` / `python`)を探します。チルダ (`~`) 展開に対応。vLLM / MLX 用の venv は `~/.coderouter/backends/<バックエンド名>/` 配下にバックエンドごとに分けて作るのが推奨です(詳細は [インストール手順書](./install-backends.md))。UI の「バックエンド」セレクト下に解決されたパスが表示されます。

### `model_dirs`

- チルダ (`~`) 展開あり
- 存在しないパスはスキャン時に無視(起動エラーなし)
- 検索対象拡張子: `.gguf` `.safetensors` `.bin` `.pt` `.pth` `.ggml`
- サブフォルダを再帰検索

### `option_profiles`

```yaml
option_profiles:
  llama.cpp:            # バックエンド名(キー)
    - name: "わかりやすい名前"   # UI ドロップダウンに表示
      args:
        "-ngl": 99              # int → "-ngl 99"
        "--ctx-size": 4096
        "--dtype": "float16"    # str → "--dtype float16"
        "--mlock": true         # bool true → "--mlock"(値なし)
        "--no-mmap": false      # bool false → 省略
```

**`args` の型ルール:**

| YAML 型 | CLI 変換 |
|---|---|
| `int` / `float` / `str` | `--flag value` の 2 引数 |
| `bool: true` | `--flag` のみ(値なし) |
| `bool: false` | このフラグを省略 |

### 追加オプション(自由入力)

UI の「追加オプション」欄の文字列は `shlex.split()` でパースされ、コマンド末尾に追加されます。プロファイルに無い実験的なフラグを試すときに使います。

```
-ngl 40 --rope-scale 2.0 --rope-freq-base 10000
```

> **注意**: `-m` / `--model` (および `--model=...` 形式) によるモデルの再指定は、追加オプション・オプションプロファイルのどちらでも受け付けません — 指定すると起動リクエストは 400 で拒否されます。モデルは「モデルパス」欄でのみ指定してください。

---

## オプション早見表

### llama.cpp

よく使うフラグのみ抜粋。完全リストは `llama-server --help`。

| フラグ | 説明 | 推奨値例 |
|---|---|---|
| `-ngl` | GPU にオフロードするレイヤー数 | `99`(全部)/ `0`(CPU のみ) |
| `--ctx-size` | コンテキスト長(トークン) | `4096` / `8192` / `131072` |
| `--threads` | CPU スレッド数 | CPU コア数 − 2 |
| `--batch-size` | バッチサイズ | `512` |
| `--mlock` | メモリにロック(スワップ防止) | `true` |
| `--embedding` | Embedding モードで起動 | `true` |

### vllm

完全リストは `python -m vllm.entrypoints.openai.api_server --help`。

| フラグ | 説明 | 推奨値例 |
|---|---|---|
| `--dtype` | テンソルデータ型 | `"auto"` / `"float16"` / `"bfloat16"` |
| `--max-model-len` | 最大コンテキスト長 | `4096` / `32768` |
| `--gpu-memory-utilization` | GPU メモリ使用率(0–1) | `0.85` |
| `--quantization` | 量子化方式 | `"awq"` / `"gptq"` |
| `--tensor-parallel-size` | テンソル並列数(GPU 台数) | `2` |

### mlx

MLX(`mlx_lm.server`)は統合メモリ前提で、`-ngl` のようなレイヤーオフロードの概念がありません。Launcher が `--model` と `--port` を設定すれば動き、起動時の性能チューニングフラグは基本的に不要です。

---

## 起動後の使い方 — CodeRouter への接続

Launcher で起動した backend は OpenAI 互換 API を提供します。これを CodeRouter のプロバイダとして `providers.yaml` に登録すれば、ルーティング・ガード・フォールバックが使えます。

```yaml
providers:
  - name: local-qwen-launcher
    kind: openai_compat
    base_url: http://localhost:8080/v1   # Launcher で指定したポート
    model: Qwen2.5-Coder-7B-Instruct

profiles:
  - name: default
    providers: [local-qwen-launcher]
```

Claude Code は接続先を CodeRouter に向けて起動します:

```bash
ANTHROPIC_BASE_URL=http://localhost:8088 ANTHROPIC_AUTH_TOKEN=dummy claude
```

---

## プロファイルの追加・共有

`option_profiles` に追記するだけで新しいプリセットを足せます。コード変更は不要です。

```yaml
launcher:
  option_profiles:
    llama.cpp:
      - name: "私のカスタム設定"
        args:
          "-ngl": 40
          "--ctx-size": 8192
```

CodeRouter を再起動すると UI に反映されます。`launcher_profiles.yaml.example` をリポジトリに含めてあるので、新プロファイルを追記して PR を送れば共有できます。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| 起動ボタンが押せない(グレーアウト) | バックエンドのバイナリが見つからない | バックエンド欄下の表示を確認し、`launcher.backends.<name>.binary` にフルパスを設定 |
| モデル一覧が空 | `launcher.model_dirs` 未設定、または設定ファイル未検出 | `providers.yaml` に `model_dirs` を設定(デスクトップ版は `--config` で明示指定も可) |
| オプションプロファイルが選べない | `launcher.option_profiles` が無い | `providers.yaml` に `option_profiles` を追加 |
| 起動後すぐ `error` になる | モデルパスの誤り / VRAM 不足 | ログでエラー内容を確認 |
| ポートが衝突する | 同じポートで別プロセスが動いている | ポート番号を変える |
| `PyYAML が見つかりません`(デスクトップ版) | 素の Python から実行した | `uv run python launcher_gui.py` で CodeRouter の venv から実行 |
| 再起動後にプロセスが消える | 仕様 — レジストリは in-memory | 常駐させたい場合は OS の launchd / systemd で管理 |

---

## 関連ドキュメント

- [バックエンド インストール手順書](./install-backends.md) — llama.cpp / vLLM / MLX の導入
- [Launcher クイックスタート](./launcher-quickstart.md) — 導入から起動までの通し手順
- [アーキテクチャ詳細 — Launcher セクション](../concepts/architecture.md#launcher--llamacpp--vllm-プロセス管理-v250)
- [利用ガイド](../guides/usage-guide.md)
- [llama.cpp 直接接続ガイド](./llamacpp-direct.md)
