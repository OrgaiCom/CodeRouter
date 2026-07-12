# Launcher モデルスワップ機能 設計書（llama-swap 相当の自前実装）

> 対象バージョン: CodeRouter v2.x 系 / Phase 1（オンデマンド spawn + 保留 + TTL）・Phase 2（メモリ会計 + 排他 swap）
> 関連: [`docs/backends/launcher.md`](../backends/launcher.md)、[`ingress/launcher_routes.py`](../../coderouter/ingress/launcher_routes.py)、[`routing/fallback.py`](../../coderouter/routing/fallback.py)、[`guards/memory_budget.py`](../../coderouter/guards/memory_budget.py)、[`v1.6-auto-router.md`](./v1.6-auto-router.md)
> ステータス: **Phase 1 実装済み**（2026-07-12。§10 の決定+§10.5 の as-built 記録を参照。Phase 2 未着手）
> Owner: zephel01

本設計書は、[llama-swap](https://github.com/mostlygeek/llama-swap) が提供する「モデル名でのオンデマンド起動 / ロード完了までのリクエスト保留 / アイドル TTL アンロード / メモリ排他スワップ」を、**外部ツールへ依存せず CodeRouter 本体へ自前実装する**ための方式を定義する。実装はレビュー後に着手する。本書はコード変更を伴わない。

コード上の拡張点・既存資産はすべて commit `b2c8a52` 時点のソースを典拠とする。ファイル・行番号の参照は §11 のシンボル早見表に集約する。

---

## 1. 概要と目的

### 1.1 何を作るか

CodeRouter の埋め込み Launcher（`ingress/launcher_routes.py`）は既に複数の llama.cpp / vllm / mlx プロセスを起動・停止・ログ管理できる。しかし現状は **運用者が UI から手動で起動する**前提であり、「リクエストが来たモデルを、未起動なら自動で立ち上げ、使い終わったら落とす」という llama-swap 的なライフサイクル自動化を持たない。本機能はこの空白を埋める。

具体的には次の 4 つを追加する（事前調査で「無い機能」と確定した 4 点に対応）。

1. **オンデマンド spawn** — リクエストの `model` 名を見て、対応するモデルが未起動なら Launcher が起動する。
2. **ロード中リクエストの保留（readiness）** — 起動〜ヘルス確認完了までの間、当該リクエストを待たせてから応答する。
3. **アイドル TTL アンロード** — 最後の利用から一定時間経過したプロセスを自動停止し、メモリを解放する。
4. **メモリ会計 + 排他 swap（Phase 2）** — 複数プロセスの重み + KV のメモリを合算し、ホストに載らない場合は LRU で既存プロセスをアンロードしてから新モデルを起動する（llama-swap の `groups` 相当）。

### 1.2 なぜ作るか（セキュリティ動機 = 依存最小化）

CodeRouter の核心的な運用方針は **Core 5 deps を守る依存最小主義**である（`docs/future.md`）。llama-swap は Go 製の優れた外部プロセスだが、これを導入すると次のコストを負う。

- **攻撃面の増加**: 別バイナリが localhost にリスニングし、モデルディレクトリと子プロセス spawn 権限を持つ。CodeRouter が既に持つ token auth（`CODEROUTER_LAUNCHER_TOKEN`）・model_dir トラバーサル防止（`_resolve_within_model_dirs`）・model-flag override 拒否（`_assert_no_model_override`）といった防御を**二重に用意・整合させる**必要が生じる。
- **供給網リスク**: 外部バイナリの更新追従・脆弱性監視・署名検証という運用負荷。CodeRouter の `plugins.enabled` 二段ゲート（供給網防御）の思想とも一貫しない。
- **設定の二重化**: モデルカタログ・ポート・メモリ制約を llama-swap 側 YAML と providers.yaml の両方に持つことになり、真実の単一ソースが失われる。

CodeRouter は既に **複数プロセスのライフサイクル管理・GGUF メモリ見積り・KV 会計式・model_dir 境界検証**という部品をすべて自前で持っている（§11）。不足しているのは「オンデマンド起動のトリガと保留・TTL・合算会計」という**調停ロジック**だけであり、これは新規依存ゼロで数百行規模で実装できる。よって自前実装がセキュリティ・運用の両面で合理的である。

### 1.3 llama.cpp router mode / llama-swap との関係と出口戦略

- **llama-swap**: 本機能の機能的な参照実装。`ttl` / `groups`（swap / persistent / exclusive）/ `checkEndpoint` による readiness を持つ。本設計はこの語彙を意図的に踏襲し（§5・§6）、将来 llama-swap へ運用移行したくなった場合でも設定概念の対応が取りやすい形にする。
- **llama.cpp 本体の router mode**（`--models-dir` / `--models-max` / LRU、2025-12〜）: llama.cpp 本体が同アーキテクチャを内蔵し始めている。これが成熟し llama.cpp バックエンドで安定運用できるようになったら、**CodeRouter 自前実装は llama.cpp router mode の薄いフロントに退避できる**設計とする（出口戦略）。具体的には、SwapManager のインターフェース（§4.2）を「バックエンド非依存の調停層」に保ち、`backend == "llama.cpp"` かつ router mode 有効時は spawn/stop/メモリ会計を llama.cpp router へ委譲する分岐を後から差し込めるようにする。vllm / mlx には router mode 相当が無いため、自前実装はこれらのために残る。

### 1.4 CodeRouter の設計思想との整合

`docs/future.md` の「透過性」「1 リクエスト = 1 変換のステートレス性」は、本機能では **dispatch 前の副作用（プロセス起動待ち）**として現れる。これは変換そのものをステートフルにするものではなく、「バックエンドの可用性を要求時に整える」レイヤであり、auto_router がリクエスト本文を見てプロファイルを解決するのと同じ「dispatch 前の解決フェーズ」に属する。応答生成自体は既存の openai_compat アダプタがそのまま担い、ワイヤ上の透過性は維持される。

---

## 2. スコープ / 非スコープとフェーズ分割

### 2.1 Phase 1（MVP）: オンデマンド spawn + 保留 + アイドル TTL

| 項目 | 内容 |
|---|---|
| オンデマンド spawn | `model` 名 → swap カタログ照合 → 未起動なら `_build_cmd` + `create_subprocess_exec`（既存 `api_start` 経路を関数化して再利用） |
| リクエスト保留 | 起動〜readiness 完了まで per-model の `asyncio.Event` で待機。タイムアウトあり |
| アイドル TTL | プロセス毎の `last_used`（monotonic）を記録し、background sweeper が満了プロセスを停止 |
| provider 同期 | 起動後、既存 `register_provider` で専用チェーンへ登録。アンロード時は `deregister_provider`（新設・小）で外す |
| 受け入れ条件 | (1) 未起動モデルへの初回リクエストが、手動起動なしに 200 応答を返す (2) 同一モデルへの同時 N リクエストが**1 プロセスのみ**起動し全員が応答を受ける (3) `ttl_seconds` 経過後にプロセスが自動停止し、次リクエストで再起動される (4) 起動失敗リクエストは `AdapterError(retryable)` になり、次リクエストで再試行できる（poison 化しない） |

### 2.2 Phase 2: メモリ会計 + 排他 swap（groups 相当）

| 項目 | 内容 |
|---|---|
| メモリ会計 | GGUF から重みサイズ、`memory_budget` の KV 式で各ロード済みプロセスのメモリを見積り、**合算**して `hardware.available_budget_gb` と比較（既存 `plan_fit` は単発用のため合算関数を新設） |
| 排他 swap | 新モデルがメモリに載らない場合、`group` ポリシー（`swap` / `persistent` / `exclusive`）に従い LRU で犠牲プロセスを選定・停止してから起動 |
| 受け入れ条件 | (1) 合算メモリがホスト予算を超える起動要求で、`swap` グループの LRU プロセスが停止されてから新モデルが起動する (2) `persistent` グループのプロセスは犠牲に選ばれない (3) `exclusive` グループのモデル起動時は同グループ他モデルが全て停止される (4) メモリ不足で犠牲にしても載らない場合は `insufficient` として `AdapterError` で降格し、既存プロセスは巻き添えで殺さない |

### 2.3 非スコープ

| 項目 | 理由 / 扱い |
|---|---|
| llama.cpp router mode への委譲実装 | §1.3 の出口戦略として分岐点だけ用意。実配線は router mode 成熟後の別フェーズ |
| クラッシュ自動再起動 | **前提基盤**（§3）。別エージェントが self-healing 連携で実装中。本書は重複設計しない |
| spawn 後 readiness 待ち（ヘルス確認後に provider 登録） | **前提基盤**（§3）。本書はこれを利用するのみ |
| providers.yaml の永続書き戻し | 既存方針どおり in-memory only（`register_provider` docstring）。swap カタログは静的 config、起動状態は揮発でよい |
| GPU 単位のメモリ分割 / NUMA 配置 | 過剰。ホスト全体の usable memory 単一予算で扱う（`_usable_memory_gb` の粒度を踏襲） |

---

## 3. 前提基盤（並行作業が提供する既存機能として扱う）

本設計は、別エージェントが**現在実装中**の以下 2 機能を「**既に在る基盤**」として前提し、重複して設計しない。SwapManager はこれらのフックを呼ぶ・待つ側に徹する。

| 基盤 | 本書での扱い | 依存点 |
|---|---|---|
| **(a) クラッシュ自動再起動（self-healing 連携）** | swap で起動したプロセスがクラッシュした際の復帰は self-healing 基盤が担う。SwapManager は「プロセスが died」を検知したら readiness Event をエラー確定させ、復帰は基盤に委ねる。**swap 対象プロセスには `restart_command` を設定しない**（launcher 管理プロセスの再起動は launcher が行うため、self-healing の shell 再起動と二重管理しない。§6・§10-Q4） | `guards/self_healing.py`、`_tail_logs` の exit 検知 |
| **(b) spawn 後 readiness 待ち（ヘルス確認後に provider 登録）** | spawn 直後に即 `register_provider` する現行挙動（`api_start`）は、この基盤により「healthcheck 成功後に登録」へ置き換わる見込み。SwapManager はこの **readiness 完了シグナルを待って** per-model Event を set する。readiness 判定式（`checkEndpoint` 相当）は基盤の実装に従い、SwapManager は再実装しない | `BaseAdapter.healthcheck`（openai_compat L196）、readiness 基盤の完了通知 |

> **設計上の約束**: SwapManager は readiness の**判定**を持たず、**待機と調停**だけを持つ。判定は基盤側。これにより両作業の責務が交差しない。

---

## 4. アーキテクチャ

### 4.1 層の配置

| 層 | 既存/新規 | 責務 |
|---|---|---|
| **dispatch 層**（`routing/fallback.py` の dispatch 入口） | 既存に薄いフック追加 | リクエストの `model` が swap カタログに一致するか判定 → `SwapManager.ensure_loaded(model)` を await → 完了後は既存チェーン解決に合流 |
| **SwapManager**（`coderouter/launcher_swap.py`・**新規**） | 新規 | 調停の中核。model→プロセスのマップ、per-model Lock/Event、TTL、Phase 2 のメモリ会計と排他 swap。`app.state.swap` に格納 |
| **launcher 層**（`ingress/launcher_routes.py`） | 既存を関数抽出 | プロセスの実起動/停止/ログ。`api_start` 本体を `spawn_process(spec)` へ、`api_stop` を `stop_process(id)` へ抽出し、SwapManager と UI の双方から呼べるようにする（HTTP ルートは薄いラッパに） |
| **provider 同期**（`fallback.py`） | 既存 + 小追加 | 起動後 `register_provider`（既存）、アンロード時 `deregister_provider`（新設） |
| **TTL sweeper**（SwapManager 内 background task） | 新規 | 一定間隔で `last_used` を走査し満了プロセスを stop。`_background_tasks` 同様に強参照保持 |

TTL タイマは **SwapManager 内**に置く（新 guard にはしない）。理由: TTL の判断にはプロセスレジストリ・per-model lock・in-flight lease という SwapManager の内部状態が不可分に必要で、guard（wire 層のステートレス検査）の粒度に合わないため。

### 4.2 SwapManager インターフェース（バックエンド非依存に保つ）

```
class SwapManager:
    async def ensure_loaded(model: str) -> LoadedModel        # dispatch から。保留込み
    async def acquire_lease(model: str) -> Lease               # in-flight 保護（TTL/swap から守る）
    async def release_lease(lease: Lease) -> None
    async def unload(model: str, *, reason: str) -> None       # TTL / 排他 swap から
    def touch(model: str) -> None                              # last_used 更新
    async def sweep_once() -> None                             # TTL sweeper 本体
    # Phase 2:
    async def _plan_and_evict(spec) -> None                    # メモリ会計 + LRU 退避
```

`ensure_loaded` / `unload` の内部が「launcher へ spawn/stop を投げる」か「llama.cpp router へ委譲する」かは §1.3 の出口戦略で差し替え可能な単一分岐点にする。

### 4.3 データフロー（Phase 1・テキスト図）

```
 クライアント
   │  POST /v1/chat/completions  {model: "qwen-coder-14b", ...}
   ▼
 ingress (openai_routes / anthropic_routes)
   │  ChatRequest
   ▼
 routing/fallback.py  dispatch 入口
   │  ① swap カタログに model 一致?  ──No──►  従来どおりチェーン解決（変更なし）
   │  Yes
   ▼
 SwapManager.ensure_loaded("qwen-coder-14b")
   │  per-model Lock 取得
   │  ├─ 既にロード済み & running? → touch(last_used) → Lease 返却 → return
   │  ├─ ロード中(別リクエストが起動済み)? → readiness Event を await
   │  └─ 未起動? →
   │        launcher.spawn_process(spec)         （= 既存 api_start 経路を関数化）
   │        [前提基盤(b)] healthcheck 成功を待つ
   │        register_provider(spec → dedicated chain)
   │        readiness Event.set()
   │  Lock 解放
   ▼
 dispatch がチェーン解決 → openai_compat アダプタ → localhost:PORT へ HTTP
   ▼
 応答（TTL sweeper が後で last_used 満了を検知 → unload → deregister_provider）
```

### 4.4 既存資産の再利用マップ

| 必要な機能 | 再利用元 |
|---|---|
| プロセス起動 | `api_start` の spawn 部（`_build_cmd` → `create_subprocess_exec` → `_tail_logs` 起動）を `spawn_process(spec)` に抽出 |
| プロセス停止（SIGTERM→SIGKILL） | `api_stop` を `stop_process(id)` に抽出 |
| モデルパス境界検証 | `_resolve_within_model_dirs`（model_dir トラバーサル防止） |
| model-flag override 拒否 | `_assert_no_model_override` |
| メモリ見積り（重み） | `gguf_introspect.read_gguf_metadata` の `file_size_bytes` |
| KV 会計 | `memory_budget.kv_cache_bytes` / `max_num_ctx_for_budget` / `plan_fit`（Phase 2 で合算版を新設） |
| usable memory | `_detect_hardware` / `_usable_memory_gb`（`launcher_routes.py`）or `hardware.available_budget_gb` |
| provider 登録 | `register_provider`（in-memory、同名 replace 済み） |
| MTP/spec 解決 | `resolve_speculative`（spawn spec に draft/mtp を持てる） |

---

## 5. 設定スキーマ案

既存 `schemas.py` の流儀（`ConfigDict(extra="forbid")`、`Field(description=...)`、`model_validator(mode="after")` fast-fail）に合わせる。swap カタログは **`LauncherConfig` 配下**に置く（Launcher が起動を担うため）。

### 5.1 `LauncherSwapConfig`（新規・`launcher.swap`）

```python
class LauncherSwapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="オンデマンド swap を有効化。false のとき Launcher は従来どおり手動起動のみ。",
    )
    ttl_seconds: float | None = Field(
        default=1800.0, ge=0.0,
        description=(
            "最後の利用からこの秒数を過ぎたプロセスを自動停止する。"
            "None = TTL 無効（明示停止まで常駐）。0 = 応答完了で即アンロード。"
        ),
    )
    readiness_timeout_s: float = Field(
        default=120.0, ge=1.0, le=1800.0,
        description="spawn からヘルス確認完了まで保留リクエストが待つ上限。超過で 503。",
    )
    sweep_interval_s: float = Field(
        default=15.0, ge=1.0, le=600.0,
        description="TTL sweeper の走査間隔。",
    )
    # ---- Phase 2 ----
    memory_budget_gb: float | None = Field(
        default=None, ge=0.0,
        description=(
            "Phase 2: 合算メモリ予算の明示上書き（GB）。None なら "
            "hardware.available_budget_gb（自動検出）を使う。"
        ),
    )
    max_loaded: int | None = Field(
        default=None, ge=1,
        description="Phase 2: 同時ロード数の上限。llama.cpp router mode の --models-max 相当。None=無制限（メモリ会計のみで制御）。",
    )
    models: list[SwapModelSpec] = Field(
        default_factory=list,
        description="swap 対象モデルのカタログ。リクエストの model 名/パターンで照合する。",
    )
```

### 5.2 `SwapModelSpec`（新規・カタログ 1 エントリ）

`LauncherOptionProfile` / `StartRequest` の語彙を踏襲する。`option_profile` は既存 `launcher.option_profiles` の名前を参照。

```python
class SwapModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="リクエスト body の model と一致させる論理モデル名（provider 名にもなる: launcher-swap-<name>）。",
    )
    model_pattern: str | None = Field(
        default=None,
        description=(
            "name 完全一致に加え、re.fullmatch で照合する任意パターン。"
            "auto_router.RuleMatcher.model_pattern と同じ load 時コンパイル検証を行う。"
        ),
    )
    backend: str = Field(
        ..., description="'llama.cpp' | 'vllm' | 'mlx'（_build_cmd と同一集合）。",
    )
    model_path: str = Field(
        ...,
        description="モデルファイルの絶対/~パス。起動時に _resolve_within_model_dirs で model_dirs 境界を検証。",
    )
    port: int | None = Field(
        default=None, ge=1024, le=65535,
        description="固定ポート。None なら SwapManager が空きポートを動的割当（TOCTOU 対策は §6）。",
    )
    option_profile: str | None = Field(
        default=None,
        description="launcher.option_profiles[backend] のプリセット名を参照。追加フラグはここから解決。",
    )
    extra_args: str = Field(default="", description="ワンオフの追加フラグ（shlex.split、_assert_no_model_override 適用）。")
    draft_model_path: str | None = Field(default=None, description="MTP/draft gguf（resolve_speculative に渡す）。")
    mtp_mode: str = Field(default="auto", description="'auto' | 'off'（StartRequest と同一）。")

    # ---- Phase 2 ----
    group: Literal["swap", "persistent", "exclusive"] = Field(
        default="swap",
        description=(
            "llama-swap groups 相当。'swap'=メモリ不足時に LRU 退避対象。"
            "'persistent'=退避されない常駐。'exclusive'=起動時に同グループ他モデルを全停止。"
        ),
    )
    est_weights_gb: float | None = Field(
        default=None, ge=0.0,
        description="重みメモリの手動見積り上書き（GB）。None なら GGUF file_size から算出。非 GGUF（vllm/mlx safetensors）で有用。",
    )
    num_ctx: int = Field(
        default=8192, ge=256,
        description="Phase 2 の KV 会計に用いる想定コンテキスト長。memory_budget.plan_fit の requested_num_ctx。",
    )
```

### 5.3 `LauncherConfig` への追加

```python
class LauncherConfig(BaseModel):
    # ... 既存フィールド（model_dirs / backends / option_profiles）...
    swap: LauncherSwapConfig | None = Field(
        default=None,
        description="オンデマンドモデルスワップ設定（None = 無効、従来の手動起動のみ）。",
    )
```

### 5.4 load 時検証（`model_validator(mode="after")`、fast-fail）

`auto_router` / `mode_aliases` の既存 fast-fail パターンに倣い、起動時に全検証する。

1. `swap.models[*].name` の重複禁止。`name` と既存 `providers[*].name`（`launcher-swap-<name>` 展開後）の衝突禁止。
2. `option_profile` 指定時、`launcher.option_profiles[backend]` に該当名が存在すること。
3. `model_pattern` の `re.compile` エラーを load 時に行番号付きで報告（`AutoRouteRule` L1319 の先例）。
4. `port` 明示時の重複禁止（同一ポートを 2 spec が主張していないか）。
5. `group=="exclusive"` は同一 spec が複数の排他グループに属せない（設計上グループ名は固定 3 値なので `exclusive` は 1 プールとして扱う。§10-Q3 で粒度を要判断）。
6. `swap.enabled=True` かつ `models` が空なら警告（無意味な有効化）。

### 5.5 providers.yaml 記述例

```yaml
launcher:
  model_dirs: [~/models, /data/gguf]
  option_profiles:
    llama.cpp:
      - name: gpu-fast
        args: { "-ngl": 99, "--ctx-size": 8192 }
  swap:
    enabled: true
    ttl_seconds: 900          # 15 分アイドルでアンロード
    readiness_timeout_s: 120
    memory_budget_gb: null    # 自動検出（hardware）
    max_loaded: 2             # Phase 2
    models:
      - name: qwen-coder-14b
        backend: llama.cpp
        model_path: ~/models/qwen2.5-coder-14b-q4.gguf
        option_profile: gpu-fast
        group: swap
        num_ctx: 16384
      - name: qwen-writer-14b
        model_pattern: "qwen-writer.*"
        backend: llama.cpp
        model_path: ~/models/qwen2.5-14b-instruct-q4.gguf
        option_profile: gpu-fast
        group: swap
      - name: embed-small        # 常駐（軽量・退避させない）
        backend: llama.cpp
        model_path: ~/models/nomic-embed.gguf
        group: persistent

profiles:
  - name: swap                    # swap プロバイダの受け皿（register_provider が先頭挿入）
    providers: []
```

> リクエストは `model: qwen-coder-14b`（または `X-CodeRouter-Profile: swap` + model 名）で到達する。auto_router の `model_pattern` ルールで swap プロファイルへ振り分ける運用も可能（§3.1 external-agents と同型）。

---

## 6. 並行性の設計（本機能の肝）

swap は「プロセス起動という重い副作用」を「多数の同時リクエスト」の下で正しく 1 回だけ行うことが核心である。以下を厳密に定める。

### 6.1 プリミティブと状態

SwapManager は per-model に次を持つ（`dict[str, _ModelState]`）。

```
_ModelState:
    lock: asyncio.Lock            # この model の spawn/unload を直列化
    ready: asyncio.Event          # readiness 完了で set。エラー時も set(=待機解除)
    error: BaseException | None   # spawn/readiness 失敗を保持（waiter が読む）
    proc_id: str | None           # launcher の ManagedProcess.id
    last_used: float              # monotonic（touch で更新）
    in_flight: int                # 実行中リクエスト数（Lease カウンタ）
    status: Literal["idle","loading","ready","stopping"]
```

Phase 2 のメモリ会計・退避判断は **単一のグローバル `asyncio.Lock`（`_swap_lock`）**で直列化する（複数 per-model lock を跨ぐデッドロックを避けるため、退避を伴う起動はこの 1 本の下でのみ行う）。

### 6.2 同時リクエストで 1 プロセスだけ起動する（thundering herd）

`ensure_loaded(model)` は per-model `lock` を取る。

- ロック内で `status` を確認: `ready` かつ実プロセス running なら `touch` して即返す。
- `loading` はあり得ない（lock 保持中は自分だけ）。他リクエストは lock 待ち → 取得時には `ready` になっているので即返る。
- `idle` なら `status="loading"` にして **lock を保持したまま** spawn を開始…**しない**。長時間の spawn を lock 内で待つと、同 model の後続 lock 待ちが readiness_timeout まで詰まるのは許容だが、**他 model の処理を妨げない**ため lock は per-model で十分。ただし spawn 完了まで lock を握ると sweeper 等が触れないので、**spawn 起動（プロセス fork）だけを lock 内で行い、readiness 待ちは Event ベース**にして lock を解放する（下記シーケンス）。

シーケンス（未起動時）:

```
async with state.lock:
    if state.status == "ready" and running: touch; return lease
    state.status = "loading"; state.ready.clear(); state.error = None
    [Phase2] async with _swap_lock: await _plan_and_evict(spec)   # メモリ確保
    proc_id = await launcher.spawn_process(spec)                  # fork のみ（速い）
    state.proc_id = proc_id
# lock 解放後に readiness を待つ（他リクエストは state.ready を待つ）
try:
    await asyncio.wait_for(_await_readiness(proc_id), timeout=readiness_timeout_s)
    register_provider(spec)          # 前提基盤(b)：healthcheck 後に登録
    state.status = "ready"
except Exception as e:
    state.error = e; state.status = "idle"; await launcher.stop_process(proc_id)
finally:
    state.ready.set()                # 成否に関わらず waiter を解放
```

待つ側（`loading` を観測したリクエスト）:

```
await asyncio.wait_for(state.ready.wait(), timeout=readiness_timeout_s)
if state.error: raise AdapterError(retryable=True, ...)   # poison しない（§6.4）
touch; return lease
```

> **ロック内で await するのは fork（`create_subprocess_exec`）まで**。readiness の長い待ちは Event に逃がすことで、per-model lock の保持時間を最小化する。

### 6.3 TTL 満了と新着リクエストの競合

**在庫リース方式**で解決する。

- リクエスト実行中は `acquire_lease` で `in_flight += 1`、応答完了（ストリーム終端含む）で `release_lease` で `-1` かつ `touch`。
- sweeper は `sweep_once` で各 model について **per-model lock を取ってから**判定する:
  - `in_flight > 0` → スキップ（実行中は絶対に殺さない）。
  - `now - last_used < ttl` → スキップ。
  - 満了 → `status="stopping"` にし、lock 内で `stop_process` + `deregister_provider`、`status="idle"`、`proc_id=None`。
- **競合の要**: sweeper が lock を保持している間に到着した新リクエストは lock 待ち。sweeper 完了後に lock を得ると `status` は `idle` なので**再 spawn 経路に入る**。逆にリクエストが先に lock を取り `touch`+`in_flight++` していれば、sweeper は `in_flight>0` で退避しない。lock により TOCTOU が消える。
- `ttl_seconds == 0`（応答完了で即アンロード）は、`release_lease` で `in_flight==0` になった時点で sweeper 次サイクルが即回収。ストリーミング応答は**最後のチャンク到達まで** lease を保持することが必須（途中解放すると生成中に殺す）。

### 6.4 spawn 失敗 / クラッシュ時のリカバリ（poison 回避）

- spawn 失敗・readiness タイムアウト・起動直後クラッシュ → `state.error` を set、`status="idle"`、`ready.set()`、プロセスは stop。
- **待機中の全 waiter は `AdapterError(retryable=True)` を受ける**（`state.error` を読む）。fallback エンジンが chain の次プロバイダへ降格できる。
- **次のリクエストは `status=="idle"` を見て通常の再 spawn を試みる**。`error` は spawn 開始時に毎回クリアするので、一度の失敗が恒久 poison にならない（受け入れ条件 (4)）。
- 起動後の**恒常運用中のクラッシュ**は前提基盤 (a)（self-healing / launcher 再起動）が復帰を担う。SwapManager は `_tail_logs` の exit 検知で `status` を `idle` に戻し、次リクエストで再 spawn（基盤の復帰と競合しないよう、両者とも per-model lock を尊重する。§10-Q4）。

### 6.5 Phase 2 排他 swap の並行性

- 退避を伴う起動は `_swap_lock`（グローバル 1 本）の下でのみ。これにより「2 つの新モデルが同時に同じ犠牲を選ぶ」「予算を二重計上する」を防ぐ。
- 犠牲選定は `group=="swap"` のロード済みプロセスから LRU（`last_used` 昇順）。`in_flight>0` のプロセスは犠牲にしない（実行中を殺さない）。全候補が in_flight>0 で確保できなければ、新起動を `readiness_timeout` 内でリトライ待ち、なお無理なら `insufficient` として `AdapterError`。
- `exclusive` 起動は同グループ他モデルを stop してから。`persistent` は候補から常に除外。
- **デッドロック回避**: per-model lock を複数同時に取得しない。退避対象の停止は「`_swap_lock` 保持 → 対象 model の lock を**順に**1 個ずつ取得・解放」で行い、lock のネスト順序を `_swap_lock ⊃ 単一 per-model lock` に固定する（2 個以上の per-model lock を同時保持しない）。

### 6.6 既知の罠（実装時チェックリスト）

1. **ストリーミングの lease 解放漏れ**: SSE 応答は最終チャンクで必ず `release_lease`。例外・切断でも `finally` で解放。漏れると TTL が永久に効かず、リースリークで in_flight が単調増加 → 二度と回収されない。
2. **background task の GC**: sweeper と readiness 待ちタスクは `_background_tasks` 相当の強参照 set で保持（`launcher_routes.py` の既存パターン）。
3. **イベントループのブロッキング**: GGUF 読取・`stat`・`shutil.which` は `asyncio.to_thread`（既存 `api_models` / `api_start` に倣う）。lock 保持中にブロッキング I/O を呼ばない。
4. **動的ポート割当の TOCTOU**: 空きポート探索 → bind までに他プロセスが奪う窓。`port=None` 時は「候補ポートで即 spawn し、readiness 失敗なら別ポートで 1 回リトライ」か、固定ポート運用を推奨（§10-Q2）。
5. **register/deregister とチェーンの整合**: unload 後に `deregister_provider` を呼ばないと、停止済みポートがチェーンに残り死んだ backend へルーティングされる。`register_provider` は同名 replace 済みだが、**除去 API が現状無い**ため新設が必須。
6. **readiness 基盤との二重ヘルスチェック**: SwapManager は healthcheck を自前で叩かない（前提基盤 (b) の完了通知のみ待つ）。二重に叩くとロード中モデルへ余計な負荷。
7. **sweeper と ensure_loaded の lock 順序**: 両者とも per-model lock のみを取り、`_swap_lock` は Phase 2 の退避起動でだけ取る。sweeper は `_swap_lock` を取らない（TTL 停止は退避調停と独立）。
8. **`ttl_seconds=0` × ストリーミング**: 即アンロードでもストリーム中は lease で保護される前提。lease 実装が先、TTL 実装が後、という順序で入れる。
9. **shutdown 時の掃除**: 既存 `shutdown_launcher` が全 ManagedProcess を落とすため、swap プロセスも巻き取られる。SwapManager 側の sweeper task も lifespan で cancel する。

---

## 7. セキュリティ考慮

自前実装で**新たに増える攻撃面**を明示し、既存防御を流用してカバーする。

| 面 | リスク | 対応 |
|---|---|---|
| **任意コマンド spawn の拡大** | swap カタログが実行するのは既存 `_build_cmd`（llama.cpp/vllm/mlx の 3 バックエンドのみ、`shell=True` 不使用）に限定。**新たな任意コマンド実行経路を作らない** | spawn は必ず `_build_cmd` + `create_subprocess_exec`（リスト argv）を通す。`agent_cli` のような任意 command は swap の対象外 |
| **model 名 → パス解決のトラバーサル** | `model_pattern` にマッチしたリクエストが任意ファイルをロードさせる懸念 | `model_path` は **static config（SwapModelSpec）にのみ**書ける。リクエストの `model` 名は**カタログ選択キー**であり、パスとして解釈されない。起動時 `_resolve_within_model_dirs` で `model_dirs` 境界を再検証（既存の情報漏洩対策 M14 を流用） |
| **model-flag override** | `option_profile` / `extra_args` 経由で `-m` を注入し vetted モデルを差し替える | 既存 `_assert_no_model_override` を spawn 経路で必ず適用（`options.keys()` と `shlex.split(extra_args)` の両方） |
| **未認証のオンデマンド起動** | swap により**リクエスト 1 本でプロセス起動**が起きるため、`/v1/*` エンドポイントが実質 spawn 権限を持つ | swap の spawn は「static カタログに列挙されたモデル」に限定されるため、任意起動ではない。ただし DoS 面（後述）は残る。UI の start/stop は既存 `_require_launcher_token` を維持 |
| **DoS（起動フラッディング）** | 未登録 model 名の連打で毎回 spawn 試行 | カタログ**非該当**の model はそもそも swap 対象外（従来チェーンへ）。該当 model への同時要求は per-model lock で 1 起動に収束。`max_loaded`（Phase 2）とメモリ会計が総量を制限。必要なら future work でレート制限 |
| **リソース枯渇** | TTL 無効設定 + 多数モデルでメモリ枯渇 | Phase 2 のメモリ会計が既定の防波堤。Phase 1 のみ運用時は `ttl_seconds` と `max_loaded`（Phase 2）で運用者が上限管理 |

> 原則: **swap は「静的カタログに列挙済みのモデルを、要求時に起動する」ことしかできない**。リクエスト本文が新しいコマンド・新しいパス・新しいフラグを持ち込む経路を一切開かない。

---

## 8. テスト計画

llama-server 等の実バックエンド無しでテストするため、**フェイクプロセス**を軸にする（`tests/test_launcher_mtp.py` L368 の `_FakeProc` 差し替え手法を踏襲）。

### 8.1 ユニット（SwapManager 単体・純ロジック）

| # | テスト | 手法 |
|---|---|---|
| U1 | 同時 N リクエストで spawn 1 回 | `spawn_process` を counting fake に差し替え、`asyncio.gather` で N 本 `ensure_loaded` → spawn 呼び出しが 1 回 |
| U2 | ロード済み即返し + touch | 2 回目の `ensure_loaded` が spawn せず `last_used` を更新 |
| U3 | readiness タイムアウト | fake readiness を無限待ちにし `readiness_timeout_s` 超過で全 waiter が `AdapterError(retryable)` |
| U4 | spawn 失敗の非 poison | 1 回目 fake を失敗、2 回目成功 → 2 回目が再 spawn して成功 |
| U5 | TTL 満了アンロード | monotonic clock を注入（`time.monotonic` 差し替え）、`sweep_once` で満了プロセス stop + `deregister_provider` 呼び出し |
| U6 | TTL vs 新着競合 | sweeper が lock 保持中に `ensure_loaded` を割り込ませ、`in_flight>0` なら unload されないこと / lease 無しなら再 spawn すること |
| U7 | lease ライフサイクル | ストリーミング模擬で最終チャンクまで `in_flight==1`、終端後 `0` |
| U8 | Phase 2 メモリ会計 | 合算関数へ既知の GGUF メタ（fake `GGUFInfo`）を与え、予算超過で `_plan_and_evict` が LRU 対象を選ぶ |
| U9 | Phase 2 group ポリシー | `persistent` が犠牲対象外、`exclusive` 起動で同グループ全停止、`in_flight>0` は退避されない |
| U10 | load 時検証 | name 重複 / 不明 option_profile / 不正 model_pattern / port 重複が起動時 `ValueError` |

### 8.2 統合（TestClient E2E・フェイク backend）

| # | テスト | 手法 |
|---|---|---|
| I1 | オンデマンド起動 → 200 | `TestClient(create_app())`（`test_launcher_mtp.py` L340）+ swap config。`spawn_process` を fake し、fake が即 readiness OK → `register_provider` 済みの fake provider（`respx` 等で localhost:PORT をモック）へルーティングされ 200 |
| I2 | 非カタログ model は従来経路 | swap 非該当 model が SwapManager を通らずチェーン解決される |
| I3 | ストリーミング中に TTL が発火しない | SSE 応答中に sweep_interval を跨いでもプロセスが lease で保護される |
| I4 | fallback 降格 | spawn 失敗注入 → `AdapterError(retryable)` → chain 次プロバイダへ降格 |
| I5 | shutdown 掃除 | lifespan 終了で sweeper cancel + `shutdown_launcher` が swap プロセスを stop |

> フェイク backend は「readiness をコントロールでき、`/v1/chat/completions` に固定応答を返す」最小 HTTP スタブ、または `openai_compat` アダプタを `respx` でモック。前提基盤 (b) の readiness 通知はテスト用フックで即時 set できるようにする。

---

## 9. 実装見積り（ファイル・関数レベル）

| フェーズ | 対象 | 主な変更ファイル | 概算規模 |
|---|---|---|---|
| **Phase 1** | SwapManager 中核（`ensure_loaded` / lease / TTL sweeper / per-model state） | `coderouter/launcher_swap.py`（新規） | 新規 ~350–450 行 |
| | spawn/stop 関数抽出（HTTP ルートは薄いラッパへ） | `ingress/launcher_routes.py`（`api_start`→`spawn_process`、`api_stop`→`stop_process` の抽出リファクタ） | 改修 ~60–100 行（純増は小） |
| | dispatch フック（model→swap 判定 → `ensure_loaded` await） | `routing/fallback.py`（dispatch 入口に 1 分岐） | +~30–50 行 |
| | `deregister_provider`（`register_provider` の対） | `routing/fallback.py` | +~25 行 |
| | スキーマ（`LauncherSwapConfig` / `SwapModelSpec` / `LauncherConfig.swap` + validators） | `config/schemas.py` | +~120–150 行 |
| | app.state 配線 + lifespan での sweeper 起動/停止 | `ingress/app.py` | +~20 行 |
| | テスト（U1–U7・U10・I1–I5） | `tests/test_launcher_swap.py`（新規） | 新規 ~400 行 |
| **Phase 2** | 合算メモリ会計（複数プロセスの重み+KV 合算 → 予算比較） | `guards/memory_budget.py`（`plan_fit_multi` 新設・純関数）or `launcher_swap.py` | +~80–120 行 |
| | 排他 swap（`_plan_and_evict` / group ポリシー / `_swap_lock`） | `coderouter/launcher_swap.py` | +~120–180 行 |
| | スキーマ（`group` / `est_weights_gb` / `num_ctx` / `memory_budget_gb` / `max_loaded`） | `config/schemas.py` | +~40 行（§5 に含む） |
| | テスト（U8・U9） | `tests/test_launcher_swap.py` | +~150 行 |

Phase 1 純増 概算 ~600–700 行 + テスト、Phase 2 追加 ~350–450 行 + テスト。新規ランタイム依存はゼロ（asyncio 標準機能のみ）。

---

## 10. レビュー決定事項（2026-07-12 確定）

レビュー(2026-07-12)で以下の通り確定した。実装はこの決定に従う。

- **Q1: TTL** — グローバル `ttl_seconds` 単一で開始。`SwapModelSpec` 毎の上書きは Phase 1 では入れない（後方互換で後から追加可能）。
- **Q2: ポート割当** — **固定ポート推奨**。カタログ各エントリに `port` を明記する運用を既定とし、`port` 省略時の自動割当は best-effort（衝突時リトライ 1 回）に留める。ドキュメントにも固定ポート推奨を明記する。
- **Q3: グループ** — `swap` / `persistent` / `exclusive` の **3 固定値で開始**。任意グループ名への一般化は将来拡張。
- **Q4: self-healing / auto-restart との責務境界** — TTL アンロードは `api_stop` と同じ**意図的停止経路**（`ManagedProcess.stopping = True` をセットしてから SIGTERM）を通す。これにより launcher の auto-restart（2026-07-12 実装、既定無効）は発動せず、クラッシュ時の一次制御は auto-restart、寿命管理は SwapManager、と役割が重ならない。swap 対象 provider には `restart_command` を設定しない。
- **Q5: readiness 通知** — `ManagedProcess` に `ready: asyncio.Event` を追加し、`_wait_ready_and_register`（2026-07-12 実装）が register_provider 成功後に set する。SwapManager はこの Event を await する（ポーリング不要）。
- **Q6: 非 GGUF のメモリ見積り** — Phase 2 の合算会計は **GGUF（llama.cpp）のみ**を対象とし、vllm / mlx は `max_loaded` による台数制御のみ。safetensors 自動見積りはやらない。
- **Q7: ルーティング到達手段** — **auto_router ルールを自動同梱する**。swap カタログの各モデル名から `model_pattern`（`re.escape` した完全一致）ルールを自動生成し、**運用者が providers.yaml に書いた user ルールの後・`default_rule_profile` フォールスルーの前**に注入する（first-match wins のため手書きルールが常に優先）。ルール id は `swap:<model名>`。無効化フラグ `launcher.swap.inject_auto_router_rules: bool = True` を設ける。auto_router は `default_profile: auto` のときのみ発火する既存制約は変えない（明示ヘッダ / body.profile 経由の到達も従来どおり可能）。

### 10.5 Phase 1 実装記録（as-built、2026-07-12）

実装は Sonnet、敵対レビューは Opus（並行性・セキュリティ・横断影響）で実施。全指摘修正後 **1725 テストパス / ruff クリーン**。設計からの主な確定差分:

- **swap プロファイルはモデル毎専用**（`launcher-swap-<name>`）。共有チェーンは register_provider の先頭挿入仕様と衝突し誤配線するため（§5.5 の共有例は無効。設定例もモデル毎で読むこと）。プレースホルダプロファイルは load 時に `model_construct` で事前注入（ingress の profile 存在チェックを通すため）。`FallbackChain.providers` の `min_length=1` は維持。
- **dispatch フックはプロファイル解決後**に評価: 解決先が swap 専用プロファイルなら model 名に関わらずリース取得、解決先チェーンに swap provider が含まれる場合も取得、それ以外は no-op（model 名一致だけでは spawn しない）。
- **auto_router ルール自動注入**: `auto_router` 未設定かつ `default_profile: auto` の場合、合成ブロックは **swap ルール先行 + BUNDLED_RULES をマージ内包**する（swap ルールは re.escape 完全一致=明示指定でのみ発火する強いシグナルのため優先。bundled の全置換セマンティクスの例外はこの合成ブロックのみで、user 明示ブロックは従来どおり末尾追記・全置換）。
- **swap 起動プロセスは汎用 provider 登録（launcher-<backend>-<port>）を抑止**し（`ManagedProcess.swap_managed`）、TTL アンロード時の provider/adapter リークを防止。**launcher auto-restart の対象外**（SwapManager が唯一のスーパーバイザ。クラッシュ回復は次リクエストの再 spawn）。
- **readiness**: `ManagedProcess.ready: asyncio.Event`（try/finally で必ず set）。SwapManager は per-model lock を spawn+readiness 全体で保持（§6.2 の Event 方式は不採用。per-model のため他モデルを阻害しない）。
- **swap 失敗は NoProvidersAvailableError に変換**して既存 502 経路へ合流。
- **fail-fast**: `SwapModelSpec.model_path` の model_dirs 内包を load 時にも検証（spawn 時の再検証は多層防御として残置）。
- 実装ファイル: `coderouter/launcher_swap.py`（新規）、`config/schemas.py`、`routing/fallback.py`（`deregister_provider` 新設含む）、`ingress/launcher_routes.py`、`ingress/app.py`。テスト: `tests/test_launcher_swap.py`（31件）+ `tests/test_launcher_swap_review.py`（レビュー回帰15件）。
- 残課題: `docs/backends/launcher.md` への swap 設定の運用ドキュメント追記、launcher_gui.py（Tk 版）は swap 非対応のまま、Phase 2（メモリ会計+排他 swap）は未着手。

## 11. 付録: シンボル早見表（commit b2c8a52）

| 対象 | ファイル:行 |
|---|---|
| `LauncherRegistry` / `ManagedProcess` | `ingress/launcher_routes.py` L146 / L108 |
| `_build_cmd`（argv 構築・3 backend） | `ingress/launcher_routes.py` L451 |
| `api_start`（spawn 経路・provider sync） | `ingress/launcher_routes.py` L795–948 |
| `api_stop`（SIGTERM→SIGKILL） | `ingress/launcher_routes.py` L951–978 |
| `_tail_logs`（ログ drain・exit 検知・MTP fallback ループ） | `ingress/launcher_routes.py` L547–625 |
| `shutdown_launcher`（lifespan 掃除） | `ingress/launcher_routes.py` L628–653 |
| `_resolve_within_model_dirs`（トラバーサル防止 M14） | `ingress/launcher_routes.py` L412–430 |
| `_assert_no_model_override`（model-flag 拒否 H8） | `ingress/launcher_routes.py` L241–256 |
| `_require_launcher_token`（token auth H8） | `ingress/launcher_routes.py` L68–94 |
| `_detect_hardware` / `_usable_memory_gb` | `ingress/launcher_routes.py` L301 / L338 |
| `resolve_speculative`（MTP/draft） | `coderouter/launcher_speculative.py` |
| `register_provider`（in-memory 登録・同名 replace・チェーン先頭挿入） | `routing/fallback.py` L1182–1254 |
| `_adapters` キャッシュ / `build_adapter` | `routing/fallback.py` L1112 / `adapters/registry.py` |
| `BaseAdapter.healthcheck` | `adapters/base.py` L240、`adapters/openai_compat.py` L196 |
| `plan_fit` / `kv_cache_bytes` / `max_num_ctx_for_budget`（KV 会計・単発用） | `guards/memory_budget.py` L150 / L120 / L131 |
| `read_gguf_metadata`（重み見積り `file_size_bytes`） | `coderouter/gguf_introspect.py` L244 |
| `hardware.available_budget_gb`（usable memory） | `coderouter/hardware.py` |
| `SelfHealingOrchestrator.try_restart`（前提基盤 a・per-provider lock） | `guards/self_healing.py` L251 |
| `ProviderConfig.restart_command` | `config/schemas.py` L489 |
| `LauncherConfig` / `LauncherOptionProfile` | `config/schemas.py` L1455 / L1424 |
| `ProviderConfig`（`kind` 動的・`agent_cli` opt-in 先例） | `config/schemas.py` L345 |
| `AutoRouteRule.model_pattern`（load 時 regex コンパイル検証） | `config/schemas.py` L1256, L1319 |
| テスト雛形（TestClient / `_FakeProc`） | `tests/test_launcher_mtp.py` L340 / L368 |
