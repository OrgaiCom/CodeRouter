# Ollama v0.23.1 実機検証チェックリスト

> **目的**: Ollama v0.23.1 の Anthropic API 互換 + Gemma 4 MTP 対応が、
> CodeRouter の価値にどう影響するかを実機で確認する。
>
> **背景**: Ollama が Anthropic Messages API をネイティブでエミュレートするようになった。
> これは CodeRouter の Wire 翻訳レイヤーと機能的に重なる。
> 第 4 話（LM Studio の Anthropic 互換追加）と同じパターンの環境変化。
>
> **日付**: 2026-05-06 作成

---

## 前提条件

```bash
# Ollama バージョン確認
ollama --version
# → 0.23.1 以上であること

# Gemma 4 pull (検証で使うモデル)
ollama pull gemma4:e4b                   # 9.6GB, default/latest, Text+Image
ollama pull gemma4:26b                   # 18GB, 256K context, Text+Image
ollama pull gemma4:31b                   # 20GB, 256K context, Text+Image
# ollama pull gemma4:31b-coding-mtp-bf16 # 64GB, MTP 版 (メモリ潤沢なら)

# 比較用: 既存の検証済みモデル
ollama pull qwen2.5-coder:7b
```

### Gemma 4 利用可能タグ (参考)

| タグ | サイズ | Context | 備考 |
|---|---|---|---|
| `gemma4:e2b` | 7.2GB | 128K | 最小。Text+Image |
| `gemma4:e4b` (= `latest`) | 9.6GB | 128K | デフォルト。Text+Image |
| `gemma4:26b` | 18GB | 256K | MoE 26B-A4B。Text+Image |
| `gemma4:31b` | 20GB | 256K | Dense 31B。Text+Image |
| `gemma4:31b-coding-mtp-bf16` | 64GB | 256K | MTP (Mac MLX)。Text only。**20h ago** |
| 各種 `-q8_0` / `-bf16` / `-mlx-bf16` / `-mxfp8` / `-nvfp4` 量子化バリアントあり |||

### ローカルにあるモデル (参考)

```
qwen2.5-coder:7b      4.7GB    ← 比較用 (既存の検証済み)
qwen2.5-coder:14b     9.0GB
qwen2.5-coder:1.5b    986MB
qwen3:32b             20GB
qwen3.5:0.8b/4b/9b
qwen3.6:27b           17GB
qwen3.6:35b           23GB
```

---

## Phase 1: Ollama 直結 — Anthropic API 互換の動作確認

CodeRouter を介さず、Ollama の Anthropic API 互換を直接叩く。

### 1-1. 基本接続

| # | 項目 | コマンド / 手順 | 期待結果 | 結果 |
|---|---|---|---|---|
| 1-1a | Anthropic 形式で basic chat | `curl http://localhost:11434/v1/messages -H 'x-api-key: ollama' -H 'anthropic-version: 2023-06-01' -d '{"model":"gemma4:e4b","max_tokens":512,"messages":[{"role":"user","content":"Say hello"}]}'` | 200 + Anthropic 形式の応答 (`content[0].type: "text"`) | ✅ 全モデル PASS (e4b 272ms / 26b 11.7s / 31b 18s / qwen 2.8s) |
| 1-1b | streaming | 同上 + `"stream": true` | SSE: `event: content_block_delta` + `event: message_stop` | ✅ 全モデル PASS |
| 1-1c | system prompt | `"system": "You are a pirate"` 付き | system が効いた応答 | ✅ 全モデル PASS (Gemma 4 は thinking 後に応答。max_tokens≥1024 必須) |
| 1-1d | 長い system prompt | Claude Code 相当 (~15K tokens) の system prompt | 切り詰めなし、正常応答 | 未検証 (手動) |

### 1-2. Tool calling (Anthropic 形式)

**ここが最重要。CodeRouter の tool-call repair が不要になるか否かの分岐点。**

| # | 項目 | 手順 | 期待結果 | 結果 |
|---|---|---|---|---|
| 1-2a | 単一 tool 定義 + 呼び出し | Anthropic `tools` パラメータで Bash tool を定義、「echo hello を実行」と依頼 | `content[].type: "tool_use"` ブロックで返る | ✅ Gemma 4 全サイズ PASS / ❌ qwen2.5-coder FAIL (テキスト本文に JSON) |
| 1-2b | tool_use の JSON 品質 | 1-2a の `input` フィールド | 有効な JSON、必須引数あり | ✅ Gemma 4 全サイズ PASS (`{"command":"echo hello"}`) |
| 1-2c | 複数 tool 定義 | Bash + Read + Write の 3 tool を定義 | 適切な tool を選択して呼ぶ | ✅ Gemma 4 全サイズ PASS (Read を選択) / ❌ qwen FAIL |
| 1-2d | tool_result → 次の応答 | tool_use → tool_result を返す → 次の応答 | tool_result を踏まえた応答 | ✅ Gemma 4 全サイズ PASS |
| 1-2e | tool を呼ばない判断 | tool 定義あり + tool 不要な質問 | テキストのみで応答（誤 tool 呼び出しなし） | ✅ 全モデル PASS |
| 1-2f | 連続 tool 呼び出し | 「ファイルを読んで内容を表示して」等の多段タスク | 複数ターンで tool_use が正しく連鎖 | 未検証 (手動) |

### 1-3. Claude Code 直結

| # | 項目 | 手順 | 期待結果 | 結果 |
|---|---|---|---|---|
| 1-3a | Claude Code 起動 | `ANTHROPIC_BASE_URL=http://localhost:11434 ANTHROPIC_AUTH_TOKEN=ollama claude --model gemma4:e4b` | 正常起動、プロンプト表示 | |
| 1-3b | 単純なタスク | 「このディレクトリのファイルを一覧して」 | Bash tool で ls を実行、結果表示 | |
| 1-3c | ファイル読み書き | 「hello.txt を作成して」 | Write tool → 確認の Read tool | |
| 1-3d | コード修正タスク | 既存ファイルの軽微な修正依頼 | Edit/Write tool で正しく編集 | |
| 1-3e | 10 分間の連続使用 | 複数タスクを連続で依頼 | tool calling が安定して動作し続ける | |

---

## Phase 2: Gemma 4 の tool calling 品質 (10 話の 3 段階判定)

Phase 1 の結果を 10 話の Level で分類する。

### 2-1. Level 判定

| # | 判定項目 | Gemma 4 e4b (9.6GB) | Gemma 4 26b (18GB) | Gemma 4 31b (20GB) | 比較: Qwen2.5-Coder 7B |
|---|---|---|---|---|---|
| 2-1a | **Level 1: 呼ぶか** — tool 定義を渡して呼ぶ確率 (10 回中) | **10/10** | **10/10** | **10/10** | 0/10 (embedded 1) |
| 2-1b | **Level 2: 形式** — tool_calls フィールドに入るか or テキスト混在か | 正規 tool_use | 正規 tool_use | 正規 tool_use | テキスト混在 |
| 2-1c | **Level 2 補足** — Ollama Anthropic API 経由でも tool_use ブロックとして返るか | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| 2-1d | **Level 3: 引数品質** — 引数の型・値が正しいか (10 回中の正答率) | **10/10** | **10/10** | **10/10** | 0/10 |
| | **総合 Level** | **Level 3** | **Level 3** | **Level 3** | **Level 0** |

### 2-2. Gemma 4 固有の観察

| # | 観察項目 | 結果 |
|---|---|---|
| 2-2a | `<think>` タグの漏れはあるか | ✅ 漏れなし (全モデル)。Ollama が thinking を `type: "thinking"` ブロックに分離 |
| 2-2b | 独自 XML タグ (`<tool_call>` 等) の出力はあるか | ✅ なし (全モデル) |
| 2-2c | 応答の日本語品質 | ✅ 全モデル PASS (日本語文字を含む応答。ただし thinking は英語) |
| 2-2d | コーディングタスクの品質 (体感) | 未検証 (手動) |
| 2-2e | MTP 有効時の速度体感 (`31b-coding-mtp-bf16`, 64GB, Mac) | 未検証 (64GB モデル未 pull) |
| 2-2f | 26b vs 31b の品質差 (体感) | 26b が速度で有利 (tool call ~500ms vs ~2000ms)。品質は同等 |
| 2-2g | e4b の実用限界 (小さいモデルでどこまで行けるか) | tool calling Level 3 到達。速度 ~1000ms/call。日本語 OK。実用域 |

**注**: Gemma 4 は全サイズで `type: "thinking"` ブロックを返す。`max_tokens` が小さいと thinking だけで消費し text が空になる。`max_tokens ≥ 1024` 推奨。

---

## Phase 3: CodeRouter 経由 vs Ollama 直結の比較

同じモデル・同じタスクを、CodeRouter 経由と Ollama 直結で比較。

### 3-1. 経路構成

```
経路 A (直結):   Claude Code → Ollama (Anthropic API)
経路 B (CR経由): Claude Code → CodeRouter → Ollama (OpenAI API)
経路 C (CR+Anthropic): Claude Code → CodeRouter → Ollama (Anthropic API, kind: anthropic)
```

### 3-2. 比較項目

| # | 比較項目 | 経路 A (直結) | 経路 B (CR+OpenAI) | 経路 C (CR+Anthropic) | 備考 |
|---|---|---|---|---|---|
| 3-2a | tool calling 成功率 (5 回) | **Gemma4: 5/5** / qwen: 0/5 | **全モデル 5/5** | 未検証 (404) | **qwen は CR 経由でのみ動作** |
| 3-2b | tool-call repair 発動回数 | N/A | Gemma4: 0回 / qwen: 5回 | N/A | Gemma 4 は修復不要 |
| 3-2c | 初回応答レイテンシ | e4b ~1s / 26b ~0.5s / 31b ~2.3s | e4b ~1.2s / 26b ~1.1s / 31b ~2.4s | — | CR 経由で微増 (翻訳オーバーヘッド) |
| 3-2d | output filter 発動 (`<think>` 等) | N/A | strip_thinking 発動 | — | Ollama 直結は thinking ブロック分離済 |
| 3-2e | エラー発生回数 (10 分間) | — | — | — | 未検証 (手動) |

### 3-3. CodeRouter の付加価値テスト

直結では得られない CodeRouter 固有の価値を確認。

| # | 項目 | 手順 | 期待結果 | 結果 |
|---|---|---|---|---|
| 3-3a | フォールバック | Ollama を停止 → リクエスト | 直結: エラー / CR: OpenRouter に自動切替 | ✅ PASS — チェーン順に全試行 (26b→31b→qwen3.6→OpenRouter→paid skip→502)。OpenRouter は API key 未設定で 401。チェーン動作は正常 |
| 3-3b | Self-healing | Ollama 停止 → 再起動 | CR: 自動復帰 | ✅ PASS — Ollama 再起動後、自動で `try-provider` → `provider-ok` → 応答復帰 |
| 3-3c | Context Budget | 50 ラウンド以上の長い会話 | CR: 自動トリミング / 直結: 400 エラー? | 未検証 |
| 3-3d | doctor 診断 | `coderouter doctor --check-model` | Gemma 4 の capabilities を正しく検出 | ✅ PASS — 全 3 モデル Exit: 0。auth, num_ctx, tool_calls, reasoning-leak, streaming 全 OK。reasoning フィールドは adapter が strip (想定通り) |

---

## Phase 4: CodeRouter への影響判定

Phase 1〜3 の結果を踏まえて判断する。

### 4-1. 機能影響マトリクス

| CodeRouter 機能 | Ollama 直結で代替可能か | 判定 |
|---|---|---|
| Wire 翻訳 (Anthropic ↔ OpenAI) | ✅ Yes (Gemma 4 + Ollama v0.23.1) | **不要** (Level 3 モデル限定) |
| Tool-call repair | ⚠️ モデル依存 | **Gemma 4: 不要** / qwen2.5-coder: **まだ必要** (0/5→5/5) |
| Output filters (`<think>` 等) | ✅ Ollama が thinking ブロック分離 | **不要** (Anthropic API 経由時) |
| 3 層フォールバック | ❌ 代替不可 | CodeRouter 固有価値 |
| 6 系統ガード | ❌ 代替不可 | CodeRouter 固有価値 |
| Self-healing | ❌ 代替不可 | CodeRouter 固有価値 |
| Replay / Audit | ❌ 代替不可 | CodeRouter 固有価値 |
| doctor 診断 | ❌ 代替不可 | CodeRouter 固有価値 |

### 4-2. ドキュメント更新の要否

| 対象 | 更新が必要か | 内容 |
|---|---|---|
| README.md / README.en.md | ⬜ | Ollama 直結経路の追記? |
| docs/architecture.md | ⬜ | 経路図に Ollama Anthropic 直結を追加? |
| 10 話 (tool calling 3 段階) | ⬜ | Gemma 4 の Level 判定追加、フレームワーク事情更新 |
| 11 話 (CodeRouter とは) | ⬜ | 「翻訳だけじゃない」の位置づけ変更? |
| docs/when-do-i-need-coderouter.md | ⬜ | Ollama 直結で済むケースの追加 |
| providers.yaml (examples) | ⬜ | Gemma 4 のプロバイダー stanza 追加 |
| model-capabilities.yaml | ⬜ | Gemma 4 の capability 定義追加 |

### 4-3. 結論テンプレート

## 結論

Ollama v0.23.1 の Anthropic API 互換により:

- Wire 翻訳の価値: **Level 3 モデル (Gemma 4) では不要になった**
- Tool-call repair の価値: **モデル依存 — Level 0-2 モデル (qwen2.5-coder 等) ではまだ必要** (0/5→5/5)
- Output filter の価値: **Ollama Anthropic API 経由では不要** (thinking ブロック分離済)
- CodeRouter の主な価値軸: **翻訳 → 運用基盤にシフト** (フォールバック / ガード / self-healing / replay)

Gemma 4 の tool calling 品質:
- Level 判定: **Level 3 (全サイズ: e4b / 26b / 31b)**
- tool_use 成功率: **10/10 (全サイズ)**、引数正答率: **10/10 (全サイズ)**
- CodeRouter 修復の必要性: **なし**
- 既存モデル (Qwen2.5-Coder 7B) との比較: **圧倒的に優** (Level 3 vs Level 0)
- `<think>` タグ漏れ: **なし** (Ollama が thinking ブロックに分離)
- 日本語品質: **PASS** (max_tokens≥1024 必須。thinking に消費されるため)

速度比較 (tool call 1 回あたり):
- e4b: ~1.0s、26b: ~0.5s (MoE で active パラメータ少)、31b: ~2.0s
- 26b が速度と品質のバランスで最優

推奨アクション:
- [x] 検証チェックリスト記入
- [ ] 10 話に Gemma 4 の Level 判定データを追加
- [ ] 11 話に「Ollama v0.23.1 で翻訳不要になったケース」を追記
- [ ] README に Ollama 直結経路を追記
- [ ] providers.yaml に Ollama Anthropic 互換プロバイダー (kind: anthropic) を追加
- [ ] docs/when-do-i-need-coderouter.md に「Ollama v0.23.1 + Level 3 モデルなら直結で OK」を追記
- [ ] 1-3 (Claude Code 直結) と 3-3 (付加価値テスト) は手動で追加検証

---

## 実行メモ

```
# 検証開始日時: 2026-05-06 22:26 JST
# 検証完了日時: 2026-05-06 22:50 JST
# 検証環境:
#   - macOS: (要記入)
#   - チップ: (要記入)
#   - メモリ: (要記入)
#   - Ollama: v0.23.1
#   - CodeRouter: v2.2.0
#   - Python: 3.12.x
#   - スクリプト: scripts/verify_ollama_0_23.py
#
# 検証ログ: result_20260506.log
#
# 初回テスト: gemma4:e4b, qwen2.5-coder:7b (Phase 1-3 個別実行)
#   → 1-1c / 2-2c FAIL: max_tokens 不足 (thinking で消費)
#   → スクリプト修正: max_tokens 引き上げ + extract_text に thinking 対応
# 再テスト: 全 4 モデル一括 (Phase 1-3)
#   → Gemma 4 全モデル ALL PASS、qwen2.5-coder は tool calling のみ FAIL (想定通り)
```
