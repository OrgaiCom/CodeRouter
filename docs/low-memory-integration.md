# 統合ガイド — 既存パッケージへの組み込み手順

本ディレクトリの新規モジュールを既存 `coderouter/` に統合するための差分。
CLAUDE.md 規約により**既存ファイルは未編集**。実適用は承認後、`feature/low-memory-guard` ブランチで行う。

新規追加ファイル（`src/coderouter/` → 実パッケージ同位置へ配置）:

- `coderouter/hardware.py`（新規・top-level）
- `coderouter/gguf_introspect.py`（新規・top-level）
- `coderouter/guards/memory_budget.py`（新規）
- `coderouter/token_estimation_accurate.py`（新規・任意精度バックエンド）

以下は既存ファイルへの**追記差分**。

---

## 1. `pyproject.toml` — extras 追加（コア 5 個は不変）

`[project.optional-dependencies]` に追記:

```toml
[project.optional-dependencies]
# 既存の test/dev extras はそのまま
accuracy = ["tokenizers>=0.20"]   # 任意: 正確なトークン数（CJK 精度）
repair   = ["json-repair>=0.30"]  # 任意: rescue parsing の壊れ JSON 復元強化
```

注意: コア `dependencies` は `fastapi/uvicorn/httpx/pydantic/pyyaml` の 5 個のまま。
README の `deps-5` バッジは維持される（extras は opt-in）。

---

## 2. `coderouter/config/schemas.py` — 設定フィールド追加

### 2-a. ProfileConfig（`memory_pressure_action` の近くに追記）

```python
    # --- 低メモリ: 事前回避（Phase 1） ---
    memory_budget_action: Literal["off", "warn", "fit"] = Field(
        default="off",
        description=(
            "送信前メモリ予算ガード。``off``=無効（既定・後方互換）。"
            "``warn``=実機に載らない見込みを警告のみ。``fit``=有効 num_ctx を"
            "実機メモリに合わせて自動縮小し、同バジェットで履歴もトリム。"
        ),
    )
    memory_budget_headroom_gb: float = Field(
        default=1.5,
        description="OS/他プロセス用に確保するメモリ下限 (GiB)。",
    )
    memory_budget_headroom_ratio: float = Field(
        default=0.15,
        description="usable メモリに対する確保割合。下限 GB との max を採用。",
    )

    # --- 低メモリ: 段階的縮退（Phase 2） ---
    memory_pressure_retry_shrink: bool = Field(
        default=False,
        description=(
            "OOM 検知時、別 provider へ落とす前に同一 provider を num_ctx 縮小で"
            "再試行する。モデルが 1 個しかない低メモリ機向け。"
        ),
    )
    memory_pressure_retry_max: int = Field(
        default=1,
        ge=0,
        description="縮小再試行の上限回数（既定 1）。",
    )
```

### 2-b. ProviderConfig（`max_context_tokens` の近くに追記）

```python
    min_num_ctx: int = Field(
        default=2048,
        ge=256,
        description="事前回避/縮退で num_ctx を縮める際の下限。",
    )
    model_path: str | None = Field(
        default=None,
        description=(
            "GGUF ファイルパス。指定時はヘッダから layer/embd/quant を読み、"
            "メモリ予算を精密化（未指定時は保守的フォールバック形状）。"
        ),
    )
```

---

## 3. `coderouter/routing/fallback.py` — Phase 1 wiring（事前回避）

既存 `_apply_context_budget_guard`（L1 トリム）の**直後**に、新ヘルパー
`_apply_memory_budget_guard` を呼ぶ。事前回避は「有効 num_ctx の算出 → その値で
トリム + backend への num_ctx override」を行う。

### 3-a. 新ヘルパー（`_apply_context_budget_guard` の下に追加）

```python
def _apply_memory_budget_guard(
    request: AnthropicRequest,
    config: Any,
    first_provider_config: Any | None,
) -> tuple[AnthropicRequest, int | None, str | None]:
    """実機メモリに合わせて有効 num_ctx を算出し、必要なら履歴をトリム。

    Returns (request, num_ctx_override, action)
      - num_ctx_override: backend に渡す上限 num_ctx（None=変更なし）
      - action: None / "warn" / "fit" / "insufficient"
    """
    import asyncio

    from coderouter.gguf_introspect import try_read_gguf_metadata
    from coderouter.guards.context_budget import trim_to_budget
    from coderouter.guards.memory_budget import plan_fit
    from coderouter.hardware import available_budget_gb, detect_hardware
    from coderouter.logging import (
        log_memory_budget_fit,       # 新規（§6）
        log_memory_budget_insufficient,
    )

    if first_provider_config is None:
        return request, None, None
    chosen = request.profile or config.default_profile
    try:
        profile = config.profile_by_name(chosen)
    except (KeyError, ValueError):
        return request, None, None
    if getattr(profile, "memory_budget_action", "off") == "off":
        return request, None, None

    # 実機検出（ブロッキング I/O → スレッドへ）
    hw = detect_hardware()
    budget_gb = available_budget_gb(
        hw,
        floor_gb=profile.memory_budget_headroom_gb,
        ratio=profile.memory_budget_headroom_ratio,
    )
    if budget_gb <= 0.0:
        return request, None, None  # 検出不能 → no-op

    # モデル形状（GGUF があれば精密、無ければ保守的フォールバック）
    info = None
    model_path = getattr(first_provider_config, "model_path", None)
    if model_path:
        info = try_read_gguf_metadata(model_path)

    requested_ctx = _resolve_max_context_tokens(first_provider_config)
    min_ctx = getattr(first_provider_config, "min_num_ctx", 2048)

    decision = plan_fit(
        available_budget_gb=budget_gb,
        weights_bytes=(info.weights_bytes if info else 0),
        requested_num_ctx=requested_ctx,
        n_layers=(info.n_layers if info else None),
        n_embd=(info.n_embd if info else None),
        n_heads=(info.n_heads if info else None),
        n_kv_heads=(info.n_kv_heads if info else None),
        min_num_ctx=min_ctx,
    )

    if decision.action == "insufficient":
        log_memory_budget_insufficient(
            logger, provider=first_provider_config.name, profile=profile.name,
            available_bytes=decision.available_bytes,
            weights_bytes=decision.weights_bytes,
        )
        # warn のみ: 落とす判断はチェーン側に委ねる
        return request, decision.effective_num_ctx, "insufficient"

    if decision.action in ("ok", "unknown"):
        return request, None, None

    # action == "shrink": num_ctx を縮め、同バジェットで履歴トリム
    if profile.memory_budget_action == "warn":
        log_memory_budget_fit(
            logger, provider=first_provider_config.name, profile=profile.name,
            requested=decision.requested_num_ctx,
            effective=decision.effective_num_ctx, applied=False,
        )
        return request, None, "warn"

    trimmed, _ = trim_to_budget(
        request,
        max_context_tokens=decision.effective_num_ctx,
        trim_target=0.9,
        preserve_last_n=profile.context_budget_preserve_last_n,
    )
    log_memory_budget_fit(
        logger, provider=first_provider_config.name, profile=profile.name,
        requested=decision.requested_num_ctx,
        effective=decision.effective_num_ctx, applied=True,
    )
    return trimmed, decision.effective_num_ctx, "fit"
```

### 3-b. 呼び出し（`_apply_context_budget_guard` を呼んでいる箇所の直後）

```python
    request, _cb = _apply_context_budget_guard(request, config, first_provider_config)
    # ↓ 追加
    request, num_ctx_override, _mb = _apply_memory_budget_guard(
        request, config, first_provider_config
    )
```

### 3-c. num_ctx override の伝播（要・アダプタ側確認）

`num_ctx_override` を backend リクエストへ反映する。バックエンド別:

- **Ollama / OpenAI 互換**: ペイロードの `options.num_ctx`（Ollama）を override。
  既存の `options.num_ctx` 設定箇所（`adapters/` 配下、§schemas L190 付近のコメント参照）で
  `min(既存値 or 無限, num_ctx_override)` を採る。
- **llama.cpp launcher**: 起動引数 `--ctx-size` に反映（launcher 起動プロファイル）。
- **未対応 backend**: override 不可なら warn ログのみ（トリムは適用済みなので安全側）。

> 注: アダプタの該当行は本作業では未読のため、適用時に `adapters/` の
> num_ctx 設定箇所を grep（`num_ctx`）して 1 箇所追記すること。

---

## 4. `coderouter/routing/fallback.py` — Phase 2（OOM 時の同一 provider 縮小再試行）

OOM 検知サイト（`is_memory_pressure_error(exc)` が True の箇所、現行 L1062 付近）で、
`mark_pressured` の**前に**縮小再試行を試みる。再試行ループはディスパッチ実行側
（provider を呼ぶ try/except を持つ関数）に実装する:

```python
        if is_memory_pressure_error(exc):
            chain = ...  # resolved profile
            if getattr(chain, "memory_pressure_retry_shrink", False) and \
               retries_done < chain.memory_pressure_retry_max:
                # num_ctx を半減（下限 min_num_ctx）し履歴を強めにトリムして即再試行
                shrunk_ctx = max(
                    provider_config.min_num_ctx,
                    (current_num_ctx or _resolve_max_context_tokens(provider_config)) // 2,
                )
                request = trim_to_budget(
                    request, max_context_tokens=shrunk_ctx,
                    trim_target=0.85, preserve_last_n=2,
                )[0]
                current_num_ctx = shrunk_ctx
                retries_done += 1
                log_memory_pressure_retry_shrink(  # 新規（§6）
                    logger, provider=provider, new_num_ctx=shrunk_ctx,
                )
                continue  # 同一 provider を再試行
            # 既存どおり: mark_pressured → 次 provider
```

ポイント: 再試行は**同一 provider** に対して `memory_pressure_retry_max` 回まで。
使い切ったら従来の cooldown / fallthrough にフォールバック（後方互換）。

---

## 5. Phase 3 — 事前警告（`doctor` / launcher）

### 5-a. `coderouter/ingress/launcher_routes.py`

既存 `_detect_hardware()` / `_usable_memory_gb()` を **`coderouter/hardware.py` に移設**し、
launcher 側は後方互換のため re-export:

```python
# launcher_routes.py
from coderouter.hardware import detect_hardware as _detect_hw  # noqa: F401

def _detect_hardware() -> dict[str, Any]:
    hw = _detect_hw()
    return {"ram_gb": hw.ram_gb, "vram_gb": hw.vram_gb,
            "gpu": hw.gpu, "cpu_count": hw.cpu_count}
```

### 5-b. model-fit プローブ（launcher のモデルスキャン結果 / `coderouter doctor`）

```python
from coderouter.gguf_introspect import try_read_gguf_metadata
from coderouter.guards.memory_budget import plan_fit
from coderouter.hardware import available_budget_gb, detect_hardware

def model_fit_advice(model_path: str, requested_ctx: int = 8192) -> dict:
    hw = detect_hardware()
    budget = available_budget_gb(hw)
    info = try_read_gguf_metadata(model_path)
    d = plan_fit(
        available_budget_gb=budget,
        weights_bytes=(info.weights_bytes if info else 0),
        requested_num_ctx=requested_ctx,
        n_layers=(info.n_layers if info else None),
        n_embd=(info.n_embd if info else None),
        n_heads=(info.n_heads if info else None),
        n_kv_heads=(info.n_kv_heads if info else None),
    )
    advice = {
        "fits": d.fits, "action": d.action,
        "effective_num_ctx": d.effective_num_ctx,
        "quant": info.quant_name if info else None,
    }
    if d.action == "insufficient":
        advice["suggestion"] = (
            "より小さい quant（例: Q4_K_M→Q4_0 / IQ3）か、より小さいモデルを推奨"
        )
    elif d.action == "shrink":
        advice["suggestion"] = (
            f"num_ctx を {d.effective_num_ctx} に下げれば安全に動作"
        )
    return advice
```

launcher UI / `coderouter doctor` 出力にこの advice を表示する。

---

## 6. `coderouter/logging.py` — 新規ログヘルパー

既存ログ関数群と同スタイルで追加:

- `log_memory_budget_fit(logger, *, provider, profile, requested, effective, applied)`
- `log_memory_budget_insufficient(logger, *, provider, profile, available_bytes, weights_bytes)`
- `log_memory_pressure_retry_shrink(logger, *, provider, new_num_ctx)`

いずれも既存の構造化ログ（イベント名 + payload）に倣う。

---

## 7. テスト追加（既存 `tests/` へ）

- `tests/test_hardware.py` / `tests/test_gguf_introspect.py` /
  `tests/test_memory_budget.py` / `tests/test_token_estimation_accurate.py`
  （本ディレクトリ `tests/` をそのまま移植。conftest のパス挿入は不要になる）。
- `tests/test_fallback_memory_budget.py`（新規）: `_apply_memory_budget_guard` の
  off/warn/fit/insufficient 分岐と num_ctx override 伝播。
- 既存 26-scenario eval に低メモリ 3 ケース追加（計画書 §6）。

### CI ジョブ

1. **extras 無しジョブ**: コア 5 個のみで全テスト通過（char/4 フォールバック検証）→ 5-deps 不変条件を固定。
2. **extras 有りジョブ**: `pip install -e .[accuracy,repair]` で精密パス検証。
3. **`pip-audit`** を必須化、`.github` に Dependabot 設定（§セキュリティ）。
