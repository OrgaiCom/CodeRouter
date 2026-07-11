# オーケストレーション・コンパニオン設計書（Ecosystem 層、構想）

> ステータス: **構想（Concept）**。実装前。Core/Plugin の変更は伴わない。
> 正典: [`docs/inside/future.md`](../inside/future.md) §5（2026-06-27 追記）・§1.2（三層モデル）・§2.5（2026-07-11 追記）
> 前提: agent_cli Phase 1 完結（v2.7.10、[`external-agents-adapter.md`](./external-agents-adapter.md)）
> 作成: 2026-07-11 / 出典: 作者方向性指示 `_article/direction-brief-2026-07-11.md`・解析書 `_article/analysis-direction-2026-07.md` D-補足

本書は、「作業ごとにモデルを割当・サブエージェントを利用する」という作者の方向性指示のうち、**マルチステップ・オーケストレーション（Plan → 実行 → レビューの制御ループ）** をどう実現するかの構想を1枚にまとめたものである。**作るかどうか自体は未確定**（future.md §5 が既に詳しいため、当面は §2.5 の framing 追記で足りる）。実プロダクト化する段になったとき、本書を出発点にする。

---

## 1. 位置づけ

CodeRouter の「クライアント」。**別 repo・standalone・別プロセス**として作る。CodeRouter 本体（Core）にも Plugin にも組み込まない。

三層モデル（future.md §1.2）の判定基準を当てはめると: engine 内 hook（filter/guard/observer/新 ingress/新 adapter）は不要（HTTP クライアントとしてヘッダを付けて叩くだけ）→ Plugin ではない。独立プロセス・既存の Anthropic/OpenAI 互換クライアントを再利用（`X-CodeRouter-Profile` ヘッダを足すだけ）→ **Ecosystem**（`HTTP loose` 結合）。

Voice Bridge と同型で、CodeRouter Plugin SDK は使わない。future.md §1.2 Ecosystem partners に「(将来候補) multi-agent orchestrator」として登録済み。

---

## 2. 責務境界（線引き）

### 持つもの（orchestrator 側、上位層の責務）

- **制御ループ**: planner → coder → reviewer の直列実行（将来: 並列・分岐）
- **検証ループ**: reviewer の `VERDICT: PASS/FAIL` を受けて coder に再生成させる
- **safe-edit**: SEARCH/REPLACE 適用 + `git apply --check` + AST/構文検証 + テストゲート（FS に触る処理はすべてここ。wire 層の責務外）
- **セッション状態**: 会話履歴、反復回数、コスト上界などのマルチターン状態
- **サブエージェント深度管理**: 何段目のサブタスクかの追跡（future.md §2.5 の G4 に対応）
- **並列 fan-out**: 同一リクエストを複数 backend に投げて集約する場合の制御（G5 に対応）

### 持たないもの（CodeRouter wire 層に委譲）

- routing（役割 → backend の解決）
- 修復（tool-call repair、byte-fallback 等）
- ガード（context budget、tool loop、memory pressure、drift detection、self-healing）
- 記憶（plugin-memory による透過注入）
- 圧縮（plugin-compress）

この境界により、planner も coder も reviewer も CodeRouter を通すだけで上記の恩恵を**自動で**受ける（future.md §5.2 の「CodeRouter が全サブ呼び出しの共通インフラ」という絵そのもの）。

---

## 3. CodeRouter (wire 層) への接続

各サブ呼び出しに `X-CodeRouter-Profile: {planner|coder|reviewer-audit|reviewer-light}` を明示付与して、Anthropic 互換 (`/v1/messages`) または OpenAI 互換 (`/v1/chat/completions`) で CodeRouter に投げる。precedence は既存どおり `body.profile > X-CodeRouter-Profile > X-CodeRouter-Mode > auto_router`。

```
[orchestration companion]                    ← 本書のスコープ（別プロセス）
   ├─ Conductor: planner → coder → reviewer の制御ループ + 検証ループ
   ├─ safe-edit: SEARCH/REPLACE + git apply --check + AST 検証
   └─ 各サブ呼び出しに X-CodeRouter-Profile ヘッダを付与
        ↓ HTTP (Anthropic / OpenAI 互換)
[CodeRouter (wire 層)] — 既存資産そのまま、変更不要
   ├─ auto_router / capability / fallback（役割解決・能力吸収・降格）
   ├─ guards（context/memory/tool-loop/drift/self-healing）
   └─ plugin-memory / plugin-compress（透過注入）
        ↓
[推論バックエンド] — agent_cli(claude opus 等) / ローカル / クラウド混在
```

`providers.yaml` 側の構成例は future.md §2.5「opusplan 型構成例」を参照（本書では重複させない）。

---

## 4. CodeRouter 本体に要求するもの

**現状（Phase 1 完結時点）で足りている点**:

- 役割別 profile 解決、agent_cli backend、fallback、L1〜L6 guards、plugin-memory/compress の透過適用は**すべて既存資産で足りる**。追加実装は不要（future.md §2.5 (1) 確信度 HIGH）。
- `X-CodeRouter-Profile` ヘッダの precedence 機構は v1.6 以降既存。

**G1/G3（future.md §2.5 (4) のギャップ表）が欲しくなる条件**:

- G1（複合条件マッチャ）: orchestrator が profile ヘッダを明示付与している限り不要。**auto_router の保険ルールを複雑化させたい場合のみ**（例: 「コード密度が高く *かつ* ツール宣言あり」を 1 ルールで表現したい）に欲しくなる。orchestrator 側が常にヘッダを付ける設計なら発生しない。
- G3（ヘッダ/ツール名ベースのルール条件）: 同上。orchestrator が特定 MCP ツール名に応じて動的に profile を切り替えたい場合に初めて必要になる。

**明確に CodeRouter 側に要求しないもの**: 段階判定（今が Plan か実行かの意味的分類、G2）・深度連動 routing（G4）・並列 fan-out（G5）はすべて orchestrator 側の責務であり、wire 層に持ち込まない。

---

## 5. `_OUTPUTS/04-計画-方向性/multiagent/` プロトタイプとの関係

2026-06-27 の実現可能性検証（future.md §5.4-progress・§5.4-bench）で、以下のプロトタイプが M3 Max / EVO-X2 実機で動作確認済み:

- `2026-06-27_orchestrator_v1.py` / `orchestrator_v2.py` — 標準ライブラリのみの最小 Conductor。`X-CodeRouter-Profile` 付与・VERDICT 解析・safe-edit との二段ゲート化まで実証済み
- `2026-06-27_safe_edit_v1.py` — SEARCH/REPLACE 適用 + AST/構文検証 + pytest 非依存テストランナー
- `llmbench_multiagent.py` ほか — 単一 vs 役割別のベンチハーネス（結論: 役割分離の価値はコーダーの弱さに比例、L6 難所でのみ強コーダーでも逆転）

**本書が実プロダクト化に進む場合、これらは書き直しではなく卒業（別 repo への移設）が前提**。CodeRouter 本体は無改造のまま。移設手順は「実装時に定める」段階で、本書の対象外。

---

## 6. やらないこと

- **Core への統合**: 制御ループ・マルチターン状態・並列 fan-out は 1 リクエスト = 1 変換のステートレス契約（future.md §5.1）と Core 5 deps 不変条件を破る。
- **Plugin 化**: input_filter/observer hook は 1 リクエストの前後を触る器であり、複数リクエストにまたがる制御ループを持たせると透過性が混線する。
- **wire 層への意味的判定器の追加**: 「今が Plan 段階か」の判断は orchestrator が持つ（G2 は欠陥ではなく責務境界）。
- **サブスク Consumer Terms を越える常時稼働化**: agent_cli 経由の claude/codex/grok/antigravity は個人ワークフロー限定（external-agents-adapter.md §10）。orchestrator が 24/7 サービスとして常駐しサブスク OAuth を使い回す設計にはしない。
- **完全自立型エージェント化**: ゴールの管理・チェックポイント保存・完了判断は agent（Claude Code 等）または orchestrator 自身の責務であり、CodeRouter 生態系がこれを代替することはしない（future.md §2.4・§3 の既存判定を継承）。

---

## 関連ドキュメント

- [`docs/inside/future.md`](../inside/future.md) §5・§1.2・§2.5 — 正典
- [`docs/designs/external-agents-adapter.md`](./external-agents-adapter.md) — agent_cli backend（本書が利用する CodeRouter 側の対応機能）
- `_OUTPUTS/04-計画-方向性/multiagent/` — 実機プロトタイプ一式（orchestrator/safe-edit/ベンチ）
