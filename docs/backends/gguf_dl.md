# gguf_dl.py

> English: [`gguf_dl.en.md`](./gguf_dl.en.md)

Hugging Face から GGUF（やその他のファイル）をローカルフォルダへダウンロードする補助スクリプトです。`huggingface.co` の URL を貼るだけで `repo_id` / リビジョン / ファイル名を自動解析し、対話モードや一括ダウンロードもサポートします。

新 CLI `hf`（旧 `huggingface-cli`）の薄いラッパーではありますが、

- **URL コピペで完結する** （`hf download` は `repo_id` 形式が必須）
- **引数なしで対話モード** に入れる
- **保存先のフォールバックチェーン** （`--dest` → `$GGUF_DL_DIR` → `./models/`）

の 3 点で、毎回コマンドを組み立てる手間を省きます。

---

## 動作要件

- Python 3.8 以上
- `huggingface_hub`（新 CLI `hf` も同パッケージで提供されます）

オプションで `hf_transfer` を入れると並列高速ダウンロードが効きます。

```bash
pip install --upgrade "huggingface_hub[hf_transfer]"
```

プライベートリポジトリの場合は次のいずれかでトークンを設定します。

```bash
hf auth login            # 推奨。対話的にトークン入力
export HF_TOKEN=hf_xxxxx # もしくは環境変数
python gguf_dl.py --token hf_xxxxx ...   # スクリプト引数で渡すことも可
```

> 旧 `huggingface-cli login` は `hf auth login` に置き換わっています。

---

## クイックスタート

```bash
# 1. 単一ファイル（URL を貼るだけ）
python gguf_dl.py https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf

# 2. repo_id とファイル名を分けて指定
python gguf_dl.py TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf

# 3. 保存先を指定
python gguf_dl.py <URL> -d ~/models/gguf

# 4. パターンで複数ファイル一括 DL（split GGUF にも対応）
python gguf_dl.py bartowski/Qwen3-30B-A3B-GGUF -p "*Q4_K_M*.gguf"

# 5. リポジトリ内のファイル一覧だけ確認
python gguf_dl.py bartowski/Qwen3-30B-A3B-GGUF --list

# 6. 引数なし → 対話モード
python gguf_dl.py
```

---

## 受け付ける入力形式

第一引数（または対話プロンプト）には次のいずれかを渡せます。

| 形式 | 例 |
| --- | --- |
| ファイル参照 URL（blob） | `https://huggingface.co/<owner>/<repo>/blob/<rev>/<path>` |
| ファイル参照 URL（resolve） | `https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path>` |
| ツリー URL | `https://huggingface.co/<owner>/<repo>/tree/<rev>` |
| リポジトリルート URL | `https://huggingface.co/<owner>/<repo>` |
| repo_id 直接 | `<owner>/<repo>` |

URL にファイルパスやリビジョンが含まれていれば自動で解析され、第二引数や `--revision` を省略できます。

---

## オプション一覧

| オプション | 説明 |
| --- | --- |
| `target` | URL もしくは `<owner>/<repo>`（省略すると対話モード） |
| `filename` | ダウンロードするファイル名（URL に含まれていれば省略可） |
| `-d, --dest <DIR>` | 保存先フォルダ |
| `-r, --revision <REV>` | ブランチ/タグ/コミット（既定: URL から or `main`） |
| `-p, --pattern <GLOB>` | ワイルドカード指定（複数指定可。例: `-p '*Q4_K_M*.gguf'`） |
| `--list` | リポジトリ内のファイル一覧を表示するだけ |
| `--nested` | HF キャッシュ形式（snapshots 構造）で保存。既定はフラット配置 |
| `--token <TOKEN>` | プライベートリポジトリ用トークン（`HF_TOKEN` 環境変数も可） |
| `-y, --yes` | 確認プロンプトをスキップ |

### 保存先の決定順

1. `--dest` で明示した場所
2. 環境変数 `GGUF_DL_DIR`
3. カレントディレクトリ配下の `./models/`

存在しない場合は自動で作成します。

---

## 対話モード

引数なしで起動すると、以下の順に質問されます。

1. URL もしくは `<owner>/<repo>`
2. リビジョン（既定: `main`）
3. ファイル指定方法（`single` / `pattern` / `all-gguf`）
4. 保存先フォルダ

URL にファイル名が入っていればステップ 3 はスキップされます。

---

## ダウンロードの再開と高速化

- **再開**: `huggingface_hub` の機構により ETag / サイズで自動的に判定されます。同じコマンドを再実行すれば未取得部分から続行します。
- **高速化**: `hf_transfer` がインストールされていれば自動で並列転送が有効になります（環境変数 `HF_HUB_ENABLE_HF_TRANSFER=1` を内部でセット）。ネットワークが速いほど効きます。
- **タイムアウト**: 遅い回線で `httpx.TimeoutException` が出る場合は `export HF_HUB_DOWNLOAD_TIMEOUT=60` などで延長してください（既定 10 秒）。

---

## ファイルの配置

既定の **フラット配置** （`local_dir` 利用）では、`<dest>/` 直下にリポジトリ内のパスがそのまま展開されます。例えば `text_encoder/model.safetensors` を取得すると `<dest>/text_encoder/model.safetensors` になります。

`--nested` を付けた場合は HF キャッシュ形式で `<dest>/.hf_cache/` 配下に保存されます。複数のリビジョンを切り替える運用や、複数モデルでブロブを共有したい場合はこちらが向いています。

---

## 使用例

### split GGUF（分割ファイル）をまとめて取得

```bash
python gguf_dl.py bartowski/Some-Big-Model-GGUF \
    -p "*Q4_K_M*.gguf-*-of-*" \
    -d ~/models/gguf
```

### プライベートリポジトリ

```bash
hf auth login          # 一度ログイン
python gguf_dl.py myorg/private-llama -p "*.gguf"
```

### 一覧だけ眺めて、欲しいファイルを選ぶ

```bash
python gguf_dl.py TheBloke/Llama-2-7B-GGUF --list
# 出力を見て、欲しい量子化を再実行
python gguf_dl.py TheBloke/Llama-2-7B-GGUF llama-2-7b.Q5_K_M.gguf
```

---

## トラブルシューティング

**`huggingface_hub がインストールされていません`**
→ `pip install --upgrade "huggingface_hub[hf_transfer]"` を実行してください。

**401 / 403 で失敗する**
→ プライベートまたはゲート付きリポジトリです。`hf auth login` でログインするか、Web でモデルカードの利用規約に同意した上で `--token` か `HF_TOKEN` を渡してください。

**途中で止まった / もう一度走らせたい**
→ そのまま同じコマンドを再実行すれば続きから再開します。完全にやり直したい時は保存先のファイル（および `.hf_cache/` 配下）を削除してください。

**パターンに何もマッチしない**
→ `--list` でファイル名を確認してください。`-p` のパターンはフルパス／ファイル名どちらにもマッチさせます（例: `sub/file.gguf` は `sub/*.gguf` でも `*.gguf` でも拾えます）。

**`hf download` だけで足りそう**
→ 単純な用途であれば新 CLI `hf download <repo> --include "*.gguf" --local-dir <dir>` でも同等のことができます。本スクリプトの差分は「URL コピペ対応」と「対話モード」と「既定保存先のフォールバック」です。

---

## 内部で何をしているか

ざっくり言うと次の処理です。

1. 入力文字列を正規表現で解析し `(repo_id, revision, filename)` に分解
2. 必要に応じて `HfApi.repo_info()` でファイル一覧を取得
3. `--pattern` で `fnmatch` フィルタ
4. 各ファイルを `hf_hub_download(local_dir=...)` で取得（再開・進捗バーは標準機能）

---

## ライセンス

このスクリプトは個人用補助ツールとして自由に改変・再配布して構いません。
