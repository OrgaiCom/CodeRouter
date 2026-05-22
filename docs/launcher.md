# Launcher ガイド — llama.cpp / vllm GUI（Web版）

CodeRouter の `/launcher` ページで llama.cpp と vllm をブラウザから起動・管理する手順と設定リファレンスです。

> **ブラウザを使わないデスクトップアプリ版**もあります。CodeRouter 自体の起動も含めて 1 ウィンドウで完結したい場合は [Launcher ガイド（デスクトップGUI版）](./launcher-gui.md) を参照してください。設定（`launcher:` ブロック）は両版で共通です。

---

## 概要

`http://localhost:8088/launcher` にアクセスすると、以下の操作がブラウザだけで完結します:

- `model_dirs` 配下の `.gguf` / `.safetensors` をスキャンしてリスト化
- オプションプロファイルをドロップダウンで選択して起動
- 複数プロセスを同時に管理 (llama.cpp + vllm 並走など)
- 各プロセスのログをブラウザ内でリアルタイム確認

---

## クイックスタート

> llama.cpp / vLLM をまだ導入していない場合は、先に [Launcher クイックスタート](./launcher-quickstart.md)（バックエンド導入〜起動の通し手順）を参照してください。

### 1. providers.yaml に launcher セクションを追加

```yaml
# ~/.coderouter/providers.yaml

# 既存の providers / profiles はそのまま
providers: ...
profiles: ...

# ↓ これを追加
launcher:
  model_dirs:
    - ~/models          # .gguf / .safetensors を再帰検索
  option_profiles:
    llama.cpp:
      - name: "GPU フル活用"
        args:
          "-ngl": 99
          "--ctx-size": 4096
```

テンプレートから始める場合は `launcher_profiles.yaml.example` をコピーしてください。

### 2. CodeRouter を起動

```bash
coderouter serve --port 8088
```

### 3. ブラウザで開く

```
http://localhost:8088/launcher
```

---

## UI の使い方

### Models パネル (左)

- `↻ スキャン` ボタンを押すと `model_dirs` を再スキャンしてリストを更新
- モデル名をクリックすると右の「モデルパス」欄に自動入力される
- サイズ (GB) が一緒に表示されるので VRAM と相談しやすい
- 各モデルに **メモリ推奨バッジ** (`✓ 推奨` / `⚠ メモリ厳しい`) を表示。判定基準は [メモリ推奨](#メモリ推奨) を参照
- ヘッダに検出ハード (例: `Metal · RAM 64GB`) を表示

### Launch パネル (右)

| 項目 | 説明 |
|---|---|
| **名前** | 管理しやすい任意の識別子 (例: `qwen-coder-8080`) |
| **ポート** | 起動するサーバーのポート番号 (デフォルト 8080) |
| **バックエンド** | `llama.cpp` か `vllm` を選択 |
| **モデルパス** | 左パネルから選択するか直接入力 |
| **オプションプロファイル** | `providers.yaml` で定義したプリセットを選択 |
| **追加オプション** | プロファイルにないフラグをその場で入力 (例: `--threads 8`) |

`▶ 起動` を押すとプロセスが起動し、Processes テーブルに表示されます。

「追加オプション」欄の横の **⚙ 推奨値** ボタンを押すと、選択中モデルと検出ハードに応じた起動フラグの目安が同欄に入ります。詳細は [メモリ推奨](#メモリ推奨) を参照。

### Processes テーブル

| 列 | 意味 |
|---|---|
| NAME | 起動時に設定した名前 |
| BACKEND | llama.cpp / vllm |
| MODEL | モデルファイル名 |
| PORT | リッスンポート |
| PID | OS プロセス ID (`—` は停止済み) |
| STATUS | ● running / ● starting / ● stopped / ● error |
| ACTIONS | `■ 停止` / `📋 ログ` / `✕ 削除` |

### ログビューア

`📋 ログ` をクリックするとページ下部にログパネルが開きます。`↻ 更新` で手動リフレッシュ、`✕ 閉じる` で閉じます。プロセスが running の場合は 3 秒ごとに自動更新されます。

---

## メモリ推奨

Models 一覧の各モデルには、CodeRouter を動かしているマシンの搭載メモリ (Apple Silicon はユニファイドメモリ、NVIDIA GPU は VRAM、それ以外は RAM) と照らした判定が表示されます。

- **✓ 推奨** — 余裕を持って動く目安 (`モデルサイズ × 1.2 + 2GB` が利用可能メモリ以内)
- **⚠ メモリ厳しい** — 収まらない／余裕が乏しい。スワップして大幅に遅くなる可能性

「追加オプション」欄の横の **⚙ 推奨値** ボタンを押すと、選択中モデルとハードから起動フラグの目安 (`-ngl` / `--ctx-size` / `--threads`) を算出して同欄に入れます。`-ngl` は GPU に載るなら `99`・CPU のみなら `0`、`--ctx-size` は空きメモリ量に応じて `4096`〜`32768`、`--threads` は CPU コア数 − 2 が目安です。

いずれも**目安**で、他プロセスのメモリ使用や量子化方式の違いまでは考慮しません。実機で調整してください。

---

## 設定リファレンス

### `launcher:` ブロック全体

```yaml
launcher:
  model_dirs:           # list[str]  必須
    - ~/models
  backends:             # dict[str, BackendConfig]  省略可
    llama.cpp:
      binary: null      # null = PATH の llama-server を使用
    vllm:
      binary: null      # null = PATH の python を使用
  option_profiles:      # dict[str, list[Profile]]  省略可
    llama.cpp: [...]
    vllm: [...]
```

### `backends` — バイナリパス設定

バイナリが PATH に入っていない場合（ソースビルド、venv/conda 環境など）にフルパスを指定します。

```yaml
launcher:
  backends:
    llama.cpp:
      binary: ~/llama.cpp/build/bin/llama-server   # ソースビルド例
    vllm:
      binary: ~/.venv/bin/python                    # venv 例
      # binary: ~/miniconda3/envs/vllm/bin/python   # conda 例
```

`binary` を省略または `null` にすると、PATH からデフォルト名（`llama-server` / `python`）を探します。チルダ (`~`) 展開に対応しています。

UI の「バックエンド」セレクト下に現在解決されたパスが表示されるので、設定が正しく反映されているか確認できます。

### `model_dirs`

- チルダ (`~`) 展開あり
- 存在しないパスはスキャン時に無視 (起動エラーなし)
- 検索対象拡張子: `.gguf` `.safetensors` `.bin` `.pt` `.ggml`
- サブフォルダを再帰的に検索

### `option_profiles`

```yaml
option_profiles:
  llama.cpp:            # バックエンド名 (キー)
    - name: "わかりやすい名前"   # UI ドロップダウンに表示
      args:
        "-ngl": 99              # int → "-ngl 99"
        "--ctx-size": 4096      # int → "--ctx-size 4096"
        "--dtype": "float16"    # str → "--dtype float16"
        "--mlock": true         # bool true → "--mlock" (値なし)
        "--no-mmap": false      # bool false → 省略
```

**`args` の型ルール:**

| YAML 型 | CLI 変換 |
|---|---|
| `int` / `float` / `str` | `--flag value` の 2 引数 |
| `bool: true` | `--flag` のみ (値なし) |
| `bool: false` | このフラグを省略 |

### 追加オプション (自由入力)

UI の「追加オプション」欄に入力した文字列は `shlex.split()` でパースされてコマンドの末尾に追加されます。プロファイルに定義されていない実験的なフラグを試すときに使います。

```
# 入力例
-ngl 40 --rope-scale 2.0 --rope-freq-base 10000
```

---

## llama.cpp オプション早見表

よく使うフラグのみ抜粋。完全リストは `llama-server --help`。

| フラグ | 説明 | 推奨値例 |
|---|---|---|
| `-ngl` | GPU にオフロードするレイヤー数 | `99` (全部) / `0` (CPU のみ) |
| `--ctx-size` | コンテキスト長 (トークン) | `4096` / `8192` / `131072` |
| `--threads` | CPU スレッド数 | CPU コア数 - 2 |
| `--batch-size` | バッチサイズ | `512` |
| `--mlock` | メモリにロック (スワップ防止) | `true` |
| `--no-mmap` | mmap を無効化 | `false` (通常は不要) |
| `--embedding` | Embedding モードで起動 | `true` |
| `--port` | Launcher が自動設定するため通常不要 | — |

---

## vllm オプション早見表

完全リストは `python -m vllm.entrypoints.openai.api_server --help`。

| フラグ | 説明 | 推奨値例 |
|---|---|---|
| `--dtype` | テンソルデータ型 | `"auto"` / `"float16"` / `"bfloat16"` |
| `--max-model-len` | 最大コンテキスト長 | `4096` / `32768` |
| `--gpu-memory-utilization` | GPU メモリ使用率 (0–1) | `0.85` |
| `--quantization` | 量子化方式 | `"awq"` / `"gptq"` / `"bitsandbytes"` |
| `--tensor-parallel-size` | テンソル並列数 (GPU 台数) | `2` |
| `--max-num-batched-tokens` | バッチあたり最大トークン | `8192` |

---

## 起動後の使い方 — CodeRouter への接続

Launcher で起動した llama-server / vllm は OpenAI 互換の API を提供します。起動したサーバーを CodeRouter のプロバイダとして `providers.yaml` に登録すれば、ルーティング・ガード・フォールバックが使えます。

```yaml
providers:
  - name: local-qwen-launcher
    kind: openai_compat
    base_url: http://localhost:8080/v1   # Launcher で指定したポート
    model: Qwen2.5-Coder-7B-Instruct    # llama-server の場合はファイル名ベース

profiles:
  - name: default
    providers: [local-qwen-launcher]
```

---

## プロファイルの追加・共有

### 自分でプロファイルを追加する

```yaml
launcher:
  option_profiles:
    llama.cpp:
      - name: "既存プロファイル"
        args: ...

      # ↓ 追加するだけ — コード変更なし
      - name: "私のカスタム設定"
        args:
          "-ngl": 40
          "--ctx-size": 8192
          "--rope-scale": 2.0
```

CodeRouter を再起動すると UI に反映されます。

### GitHub での共有

`launcher_profiles.yaml.example` をリポジトリに含めてあります。新しいプロファイルは `option_profiles` に追記して PR を送るだけで共有できます。コード変更は不要です。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| 「Executable not found」エラー | llama-server / python が PATH にない | `which llama-server` / `which python` で確認。venv の場合はフルパスを追加オプションに記入 |
| モデルが一覧に出ない | `model_dirs` の拡張子が対象外 | `.gguf` / `.safetensors` / `.bin` / `.pt` / `.ggml` のみ対象 |
| 起動後すぐ `error` になる | モデルパスが間違っている / VRAM 不足 | ログビューアでエラー内容を確認 |
| ポートが衝突する | 同じポートで別プロセスが動いている | ポート番号を変える |
| 再起動後にプロセスが消える | 仕様 — レジストリは in-memory | 必要なら OS の systemd / launchd でプロセスを管理 |

---

## 関連ドキュメント

- [Launcher クイックスタート](./launcher-quickstart.md) — llama.cpp / vLLM 導入からランチャー起動までの通し手順
- [Launcher ガイド（デスクトップGUI版）](./launcher-gui.md) — ブラウザ不要の tkinter アプリ版
- [アーキテクチャ詳細 — Launcher セクション](./architecture.md#launcher--llamacpp--vllm-プロセス管理-v250)
- [利用ガイド](./usage-guide.md)
- [llama.cpp 直接接続ガイド](./llamacpp-direct.md)
