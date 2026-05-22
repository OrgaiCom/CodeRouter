# Launcher ガイド — デスクトップGUI版 (`launcher_gui.py`)

`launcher_gui.py` は llama.cpp / vllm と CodeRouter を **ブラウザなし**で起動・管理する tkinter 製デスクトップアプリです。CodeRouter 自体もこの GUI から直接起動でき、ローカル LLM を Claude Code に繋ぐまでを 1 つのウィンドウで完結できます。

> Web版ランチャー（`/launcher` ページ）もあります。違いは [Web版との違い](#web版との違い) を参照してください。

---

## 必要なもの

> llama.cpp / vLLM をまだ導入していない場合は、先に [Launcher クイックスタート](./launcher-quickstart.md)（バックエンド導入〜起動の通し手順）を参照してください。

- Python 3.10 以上
- tkinter — Python 標準ライブラリ（追加インストール不要。一部の Linux ディストリビューションでは `python3-tk` パッケージが別途必要）
- PyYAML — CodeRouter の既存依存。CodeRouter の venv から実行すれば自動的に揃う

追加パッケージのインストールは不要です。

---

## 起動方法

```bash
# 通常起動
python3 launcher_gui.py

# CodeRouter の venv 経由（PyYAML を確実に使う）
uv run python launcher_gui.py

# 設定ファイルを明示指定
python3 launcher_gui.py --config ~/.coderouter/providers.yaml
```

設定ファイル（`providers.yaml`）の探索順:

1. `--config` で指定したパス
2. カレントディレクトリの `providers.yaml`
3. `~/.coderouter/providers.yaml`

どれも見つからない場合は空の設定で起動します（UI から手動でオプションを入力すれば起動自体は可能です）。

---

## Web版との違い

| | デスクトップGUI版（`launcher_gui.py`） | Web版（`/launcher`） |
|---|---|---|
| 形態 | tkinter デスクトップアプリ | ブラウザページ |
| CodeRouter の起動 | **このアプリから起動できる** | できない（CodeRouter の中で動くため） |
| 主な用途 | 最初の一発 — llama.cpp と CodeRouter をまとめて立ち上げる | CodeRouter 稼働中に backend を管理する運用 UI |
| 設定 | `providers.yaml` の `launcher:` ブロック（共通） | 同左 |

両者は競合ではなく補完関係です。設定はどちらも同じ `launcher:` ブロックを読みます。

---

## 画面構成

ウィンドウは上から「CodeRouter バー」「MODELS / LAUNCH」「PROCESSES」「ログ」で構成されます。

### CodeRouter バー（最上部）

CodeRouter の起動・停止と接続情報を扱います。

- **ステータスドット + ラベル** — `停止中` / `起動中…` / `稼働中` / `エラー` を色付きで表示
- **ポート** — CodeRouter のリッスンポート（既定 `8088`）。停止中・エラー時のみ編集可能で、起動中・稼働中はロックされる
- **▶ CodeRouter 起動 / ■ 停止** — CodeRouter プロセスの起動・停止
- **Claude Code 接続文字列** — `ANTHROPIC_BASE_URL=http://localhost:<ポート> ANTHROPIC_AUTH_TOKEN=dummy claude`。ポート欄を変更すると即座に追従する。文字列クリックまたは「コピー」ボタンでクリップボードにコピーできる

CodeRouter 起動時、`~/.coderouter/providers.yaml` が存在しなければ最小構成のファイルを自動生成します。

### MODELS パネル（左）

- **↻ スキャン** — `model_dirs` を再スキャンしてモデル一覧を更新
- モデル名をクリックすると LAUNCH フォームの「モデルパス」に自動入力される
- ファイルサイズ（GB）を併記。VRAM と相談しやすい
- 各モデルに **メモリ推奨バッジ**（`✓ 推奨` / `⚠ メモリ厳しい`）を表示。判定基準は [メモリ推奨](#メモリ推奨) を参照
- ヘッダに検出ハード（例: `Metal · RAM 64GB`）を表示
- 対象拡張子: `.gguf` `.ggml` `.safetensors` `.bin` `.pt` `.pth`（サブフォルダも再帰検索）

### LAUNCH フォーム（右）

| 項目 | 説明 |
|---|---|
| **名前** | 管理用の任意の識別子。モデルを選ぶと自動入力されるが、手入力した名前は保持される |
| **ポート** | 起動する llama.cpp / vllm サーバーのポート（既定 `8080`） |
| **バックエンド** | `llama.cpp` か `vllm`。選択するとバイナリの解決パスと利用可否が下に表示される |
| **モデルパス** | 左のリストから選択するか直接入力 |
| **オプションプロファイル** | `providers.yaml` で定義したプリセットを選択。選ぶと展開後の引数が表示される |
| **追加オプション** | プロファイルにないフラグをその場で入力（例: `--threads 8`）。`shlex` でパースされコマンド末尾に追加される |

バイナリが見つからない場合、**▶ llama.cpp / vllm 起動ボタンは自動的に無効化**され、理由が赤字で表示されます。

「追加オプション」欄の横の **⚙ 推奨値** ボタンを押すと、選択中モデルと検出ハードに応じた起動フラグの目安（`-ngl` / `--ctx-size` / `--threads`）が同欄に入ります。詳細は [メモリ推奨](#メモリ推奨) を参照。

### PROCESSES テーブル

起動した llama.cpp / vllm プロセスの一覧です。

| 列 | 意味 |
|---|---|
| NAME / BACKEND / MODEL / PORT / PID | 起動時の設定と OS プロセス情報 |
| STATUS | `starting` / `running` / `stopped` / `error` を色分け表示 |

行を選択して **■ 停止**（SIGTERM）/ **✕ 削除**（レジストリから除去）を実行できます。

### ログ

選択中プロセスの標準出力 / 標準エラーをリアルタイム表示します。「クリア」で表示を消去できます。長時間稼働でもメモリを圧迫しないよう、保持行数と表示行数に上限が設けられています。

---

## メモリ推奨

MODELS 一覧の各モデルには、検出した搭載メモリ（Apple Silicon はユニファイドメモリ、NVIDIA GPU は VRAM、それ以外は RAM）と照らした判定が表示されます。

- **✓ 推奨** — 余裕を持って動く目安（`モデルサイズ × 1.2 + 2GB` が利用可能メモリ以内）
- **⚠ メモリ厳しい** — 収まらない／余裕が乏しい。スワップして大幅に遅くなる可能性

**⚙ 推奨値** ボタンは、選択中モデルとハードから起動フラグの目安を算出して「追加オプション」欄に入れます。

- `-ngl` — GPU に載るなら `99`（全レイヤー）、CPU のみなら `0`、VRAM が中途半端なら部分オフロード
- `--ctx-size` — 重み確保後の空きメモリで `4096`〜`32768` を段階選択
- `--threads` — CPU コア数 − 2

いずれも**目安**です。他プロセスのメモリ使用や量子化方式の違いまでは考慮しないため、実機で調整してください。

---

## 典型的な使い方

1. **モデルを選ぶ** — MODELS から使うモデルをクリック（モデルパスと名前が埋まる）
2. **llama.cpp / vllm を起動** — オプションプロファイルを選び「▶ llama.cpp / vllm 起動」。PROCESSES に `running` で表示される
3. **CodeRouter を起動** — 上部バーの「▶ CodeRouter 起動」。`稼働中` になる
4. **Claude Code を繋ぐ** — 接続文字列をコピーしてターミナルで実行:

   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8088 ANTHROPIC_AUTH_TOKEN=dummy claude
   ```

ウィンドウを閉じると、起動した CodeRouter と全 backend プロセスは自動的に停止します。

---

## 設定リファレンス — `launcher:` ブロック

MODELS の一覧とオプションプロファイルは `providers.yaml` の `launcher:` ブロックから読み込まれます。Web版と共通の設定です。

```yaml
# ~/.coderouter/providers.yaml

launcher:
  model_dirs:                # list[str] — .gguf 等を再帰検索
    - ~/llm/models
  backends:                  # 省略可 — バイナリのフルパス指定
    llama.cpp:
      binary: ~/llm/apps/llama.cpp/build/bin/llama-server
    vllm:
      binary: null           # null = PATH の python を使用
  option_profiles:           # 省略可 — UI ドロップダウンのプリセット
    llama.cpp:
      - name: "GPU フル活用"
        args:
          "-ngl": 99
          "--ctx-size": 32768
```

**`args` の型ルール:**

| YAML 型 | CLI 変換 |
|---|---|
| `int` / `float` / `str` | `--flag value` の 2 引数 |
| `bool: true` | `--flag` のみ（値なし） |
| `bool: false` | このフラグを省略 |

完全な設定例・llama.cpp / vllm のオプション早見表は `launcher_profiles.yaml.example` および [Web版ガイドの設定リファレンス](./launcher.md#設定リファレンス) を参照してください（フォーマットは両版で同一です）。

> **注意**: CodeRouter 起動ボタンが自動生成する `~/.coderouter/providers.yaml` には `launcher:` ブロックが含まれません。MODELS 一覧やオプションプロファイルを使うには、`launcher:` ブロックを持つ `providers.yaml` を用意してください。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| モデル一覧が空 | `launcher.model_dirs` 未設定、または設定ファイルが見つかっていない | `providers.yaml` に `launcher.model_dirs` を設定し、必要なら `--config` で明示指定 |
| オプションプロファイルが選べない | `launcher.option_profiles` が無い | `providers.yaml` に `option_profiles` を追加 |
| ▶ 起動ボタンが押せない（グレーアウト） | バックエンドのバイナリが見つからない | バックエンド欄下の表示を確認し、`launcher.backends.<name>.binary` にフルパスを設定 |
| 起動後すぐ `error` になる | モデルパスの誤り / VRAM 不足 | ログでエラー内容を確認 |
| `PyYAML が見つかりません` | 素の Python から実行した | `uv run python launcher_gui.py` で CodeRouter の venv から実行する |
| 再起動するとプロセスが消える | 仕様 — レジストリは in-memory | 常駐させたい場合は OS の launchd / systemd で管理する |

---

## 関連ドキュメント

- [Launcher クイックスタート](./launcher-quickstart.md) — llama.cpp / vLLM 導入からランチャー起動までの通し手順
- [Launcher ガイド（Web版）](./launcher.md) — `/launcher` ページ版
- [llama.cpp 直接接続ガイド](./llamacpp-direct.md)
- [利用ガイド](../guides/usage-guide.md)
