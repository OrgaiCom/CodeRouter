# Integration guide — steps to embed into the existing package

日本語版: [`low-memory-integration.md`](./low-memory-integration.md)

The diff for integrating the new modules in this directory into the existing `coderouter/`.
Per the CLAUDE.md convention, **existing files are left unedited**. Actual application happens after approval, on the `feature/low-memory-guard` branch.

Newly added files (place `src/coderouter/` → at the same path in the real package):

- `coderouter/hardware.py` (new, top-level)
- `coderouter/gguf_introspect.py` (new, top-level)
- `coderouter/guards/memory_budget.py` (new)
- `coderouter/token_estimation_accurate.py` (new, optional accuracy backend)

The following are **append diffs** to existing files.

---

## 1. `pyproject.toml` — add extras (the core 5 stay unchanged)

Append to `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
# existing test/dev extras stay as-is
accuracy = ["tokenizers>=0.20"]   # optional: accurate token count (CJK precision)
repair   = ["json-repair>=0.30"]  # optional: stronger broken-JSON recovery for rescue parsing
```

Note: the core `dependencies` stay the 5 of `fastapi/uvicorn/httpx/pydantic/pyyaml`.
The README's `deps-5` badge is preserved (extras are opt-in).

---

## 2. `coderouter/config/schemas.py` — add config fields

### 2-a. ProfileConfig (append near `memory_pressure_action`)

```python
    # --- low memory: proactive avoidance (Phase 1) ---
    memory_budget_action: Literal["off", "warn", "fit"] = Field(
        default="off",
        description=(
            "Pre-send memory budget guard. ``off``=disabled (default, backward compatible). "
            "``warn``=warn only when it won't fit on the actual machine. ``fit``=automatically "
            "shrink the effective num_ctx to fit the real machine memory, and trim history to the same budget."
        ),
    )
    memory_budget_headroom_gb: float = Field(
        default=1.5,
        description="Minimum memory reserved for the OS/other processes (GiB).",
    )
    memory_budget_headroom_ratio: float = Field(
        default=0.15,
        description="Reservation ratio relative to usable memory. The max of this and the floor GB is used.",
    )

    # --- low memory: graceful degradation (Phase 2) ---
    memory_pressure_retry_shrink: bool = Field(
        default=False,
        description=(
            "On OOM detection, retry the same provider with a reduced num_ctx before "
            "falling to another provider. For low-memory machines with only one model."
        ),
    )
    memory_pressure_retry_max: int = Field(
        default=1,
        ge=0,
        description="Max number of shrink retries (default 1).",
    )
```

### 2-b. ProviderConfig (append near `max_context_tokens`)

```python
    min_num_ctx: int = Field(
        default=2048,
        ge=256,
        description="Lower bound when shrinking num_ctx for proactive avoidance/degradation.",
    )
    model_path: str | None = Field(
        default=None,
        description=(
            "GGUF file path. When set, reads layer/embd/quant from the header to "
            "refine the memory budget (conservative fallback shape when unset)."
        ),
    )
```

---

## 3. `coderouter/routing/fallback.py` — Phase 1 wiring (proactive avoidance)

Call the new helper `_apply_memory_budget_guard` **immediately after** the existing
`_apply_context_budget_guard` (L1 trim). Proactive avoidance does "compute the effective num_ctx →
trim + backend num_ctx override with that value".

### 3-a. New helper (add below `_apply_context_budget_guard`)

```python
def _apply_memory_budget_guard(
    request: AnthropicRequest,
    config: Any,
    first_provider_config: Any | None,
) -> tuple[AnthropicRequest, int | None, str | None]:
    """Compute the effective num_ctx to fit the real machine memory and trim history if needed.

    Returns (request, num_ctx_override, action)
      - num_ctx_override: upper-bound num_ctx to pass to the backend (None=no change)
      - action: None / "warn" / "fit" / "insufficient"
    """
    import asyncio

    from coderouter.gguf_introspect import try_read_gguf_metadata
    from coderouter.guards.context_budget import trim_to_budget
    from coderouter.guards.memory_budget import plan_fit
    from coderouter.hardware import available_budget_gb, detect_hardware
    from coderouter.logging import (
        log_memory_budget_fit,       # new (§6)
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

    # detect the real machine (blocking I/O → move to a thread)
    hw = detect_hardware()
    budget_gb = available_budget_gb(
        hw,
        floor_gb=profile.memory_budget_headroom_gb,
        ratio=profile.memory_budget_headroom_ratio,
    )
    if budget_gb <= 0.0:
        return request, None, None  # cannot detect → no-op

    # model shape (precise if GGUF is present, conservative fallback otherwise)
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
        # warn only: leave the drop decision to the chain side
        return request, decision.effective_num_ctx, "insufficient"

    if decision.action in ("ok", "unknown"):
        return request, None, None

    # action == "shrink": shrink num_ctx and trim history to the same budget
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

### 3-b. Call site (immediately after where `_apply_context_budget_guard` is called)

```python
    request, _cb = _apply_context_budget_guard(request, config, first_provider_config)
    # ↓ add
    request, num_ctx_override, _mb = _apply_memory_budget_guard(
        request, config, first_provider_config
    )
```

### 3-c. Propagating the num_ctx override (needs adapter-side confirmation)

Reflect `num_ctx_override` into the backend request. Per backend:

- **Ollama / OpenAI-compatible**: override the payload's `options.num_ctx` (Ollama).
  At the existing `options.num_ctx` set site (under `adapters/`, see the comment near §schemas L190),
  take `min(existing value or infinity, num_ctx_override)`.
- **llama.cpp launcher**: reflect into the launch argument `--ctx-size` (launcher launch profile).
- **Unsupported backend**: if override is impossible, warn-log only (safe, since the trim is already applied).

> Note: the relevant adapter lines were not read during this work, so when applying, grep
> `adapters/` for the num_ctx set site (`num_ctx`) and add one line there.

---

## 4. `coderouter/routing/fallback.py` — Phase 2 (same-provider shrink retry on OOM)

At the OOM detection site (where `is_memory_pressure_error(exc)` is True, currently near L1062),
attempt a shrink retry **before** `mark_pressured`. Implement the retry loop on the dispatch-execution side
(the function that has the try/except calling the provider):

```python
        if is_memory_pressure_error(exc):
            chain = ...  # resolved profile
            if getattr(chain, "memory_pressure_retry_shrink", False) and \
               retries_done < chain.memory_pressure_retry_max:
                # halve num_ctx (floor min_num_ctx), trim history harder, and retry immediately
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
                log_memory_pressure_retry_shrink(  # new (§6)
                    logger, provider=provider, new_num_ctx=shrunk_ctx,
                )
                continue  # retry the same provider
            # as before: mark_pressured → next provider
```

Key point: retries are against the **same provider** up to `memory_pressure_retry_max` times.
Once exhausted, fall back to the conventional cooldown / fallthrough (backward compatible).

---

## 5. Phase 3 — proactive warnings (`doctor` / launcher)

### 5-a. `coderouter/ingress/launcher_routes.py`

**Move** the existing `_detect_hardware()` / `_usable_memory_gb()` **into `coderouter/hardware.py`**,
and have the launcher side re-export them for backward compatibility:

```python
# launcher_routes.py
from coderouter.hardware import detect_hardware as _detect_hw  # noqa: F401

def _detect_hardware() -> dict[str, Any]:
    hw = _detect_hw()
    return {"ram_gb": hw.ram_gb, "vram_gb": hw.vram_gb,
            "gpu": hw.gpu, "cpu_count": hw.cpu_count}
```

### 5-b. model-fit probe (launcher model-scan results / `coderouter doctor`)

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
            "Recommend a smaller quant (e.g. Q4_K_M→Q4_0 / IQ3) or a smaller model"
        )
    elif d.action == "shrink":
        advice["suggestion"] = (
            f"Lowering num_ctx to {d.effective_num_ctx} runs safely"
        )
    return advice
```

Display this advice in the launcher UI / `coderouter doctor` output.

---

## 6. `coderouter/logging.py` — new log helpers

Add in the same style as the existing log functions:

- `log_memory_budget_fit(logger, *, provider, profile, requested, effective, applied)`
- `log_memory_budget_insufficient(logger, *, provider, profile, available_bytes, weights_bytes)`
- `log_memory_pressure_retry_shrink(logger, *, provider, new_num_ctx)`

All follow the existing structured logging (event name + payload).

---

## 7. Add tests (to the existing `tests/`)

- `tests/test_hardware.py` / `tests/test_gguf_introspect.py` /
  `tests/test_memory_budget.py` / `tests/test_token_estimation_accurate.py`
  (port this directory's `tests/` as-is; the conftest path insertion becomes unnecessary).
- `tests/test_fallback_memory_budget.py` (new): the off/warn/fit/insufficient branches of
  `_apply_memory_budget_guard` and num_ctx override propagation.
- Add 3 low-memory cases to the existing 26-scenario eval (plan doc §6).

### CI jobs

1. **No-extras job**: all tests pass with only the core 5 (verifies the char/4 fallback) → pins the 5-deps invariant.
2. **With-extras job**: `pip install -e .[accuracy,repair]` verifies the precise path.
3. Make **`pip-audit`** required, and add a Dependabot config to `.github` (§security).
