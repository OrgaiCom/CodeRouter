"""OpenAI-compatible routes: POST /v1/chat/completions (+ minimal /v1/models).

Profile selection precedence (first hit wins):
    1. JSON body field:  {"profile": "fast", ...}
    2. HTTP header:       X-CodeRouter-Profile: fast
    3. HTTP header:       X-CodeRouter-Mode: coding  (v0.6-D, via mode_aliases)
    4. auto_router       (v1.6-A, fires only when default_profile == "auto")
    5. config.default_profile

Body wins over header so that a caller who can embed the field has final say
(useful when a single client talks to multiple routers behind a proxy that
rewrites headers). Mode sits below Profile because Mode is an INTENT
(``coding`` / ``long`` / ``fast``) and Profile is the concrete
implementation — when a caller specifies the concrete profile, respect it.

The auto router slot is intentionally narrow: it only fires when the operator
opts in via ``default_profile: auto`` (the reserved sentinel). For every other
configuration the chain behaves exactly as in v0.6-D — unresolved requests fall
through to the engine, which applies ``config.default_profile``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from coderouter.adapters.base import ChatRequest
from coderouter.logging import get_logger
from coderouter.routing import FallbackEngine, NoProvidersAvailableError
from coderouter.routing.auto_router import RESERVED_PROFILE_NAME, classify

router = APIRouter()
logger = get_logger(__name__)

_PROFILE_HEADER = "x-coderouter-profile"
_MODE_HEADER = "x-coderouter-mode"

# M14: overall SSE stream ceiling — see anthropic_routes for rationale.
_STREAM_TIMEOUT_MULTIPLIER = 20.0
_STREAM_TIMEOUT_DEFAULT_S = 900.0
_STREAM_TIMEOUT_MIN_S = 60.0


def _resolve_stream_timeout_s(engine: FallbackEngine, profile: str | None) -> float:
    """M14: derive the overall stream ceiling (seconds) for a profile.

    Mirrors the Anthropic route. Uses the profile's per-call ``timeout_s``
    (or the first provider's, or the default) scaled up. Never below
    ``_STREAM_TIMEOUT_MIN_S``; any resolution failure falls back to
    ``_STREAM_TIMEOUT_DEFAULT_S``. Does not change the config schema.
    """
    per_call: float | None = None
    try:
        config = engine.config
        chosen = profile or config.default_profile
        chain_cfg = config.profile_by_name(chosen)
        per_call = getattr(chain_cfg, "timeout_s", None)
        if per_call is None:
            for pname in getattr(chain_cfg, "providers", []) or []:
                pconf = next(
                    (p for p in config.providers if p.name == pname), None
                )
                if pconf is not None:
                    per_call = getattr(pconf, "timeout_s", None)
                    break
    except (AttributeError, KeyError, ValueError):
        per_call = None

    if per_call is None:
        return _STREAM_TIMEOUT_DEFAULT_S
    return max(_STREAM_TIMEOUT_MIN_S, float(per_call) * _STREAM_TIMEOUT_MULTIPLIER)


@router.get("/models")
async def list_models(request: Request) -> dict[str, object]:
    """Minimal /v1/models so OpenAI SDKs that probe it don't choke."""
    config = request.app.state.config
    return {
        "object": "list",
        "data": [
            {
                "id": p.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "coderouter",
            }
            for p in config.providers
        ],
    }


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    payload: dict[str, Any],
    request: Request,
    x_coderouter_profile: str | None = Header(default=None, alias=_PROFILE_HEADER),
    x_coderouter_mode: str | None = Header(default=None, alias=_MODE_HEADER),
) -> StreamingResponse | dict[str, Any]:
    """OpenAI Chat Completions endpoint.

    Validates the body into :class:`ChatRequest`, resolves the profile
    per the precedence described in the module docstring, and dispatches
    to the engine. Streaming requests return a :class:`StreamingResponse`
    that serializes chunks onto the OpenAI SSE wire (``data: {json}`` +
    trailing ``data: [DONE]``); non-streaming requests return the JSON
    response body.
    """
    engine: FallbackEngine = request.app.state.engine
    config = request.app.state.config

    # Accept extension fields (e.g. "profile") without rejecting
    try:
        chat_req = ChatRequest.model_validate(payload)
    except Exception as exc:  # pydantic.ValidationError, etc.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Header-based override (body wins if both are set — see module docstring)
    if chat_req.profile is None and x_coderouter_profile:
        chat_req.profile = x_coderouter_profile

    # v0.6-D: ``X-CodeRouter-Mode`` → mode_aliases → profile. Only kicks
    # in when neither body nor X-CodeRouter-Profile already nailed down
    # the profile (profile > mode precedence).
    if chat_req.profile is None and x_coderouter_mode:
        try:
            chat_req.profile = config.resolve_mode(x_coderouter_mode)
        except KeyError as exc:
            available = sorted(config.mode_aliases.keys())
            raise HTTPException(
                status_code=400,
                detail=(f"unknown mode {x_coderouter_mode!r}. available modes: {available}"),
            ) from exc
        logger.info(
            "mode-alias-resolved",
            extra={"mode": x_coderouter_mode, "profile": chat_req.profile},
        )

    # v1.6-A: auto router slot. Only fires when the operator opted in by
    # setting ``default_profile: auto`` and no higher-priority caller signal
    # (body / profile header / mode header) already nailed down a profile.
    # When inactive, the engine still falls through to
    # ``config.default_profile`` on its own — same semantics as pre-v1.6.
    if chat_req.profile is None and config.default_profile == RESERVED_PROFILE_NAME:
        chat_req.profile = classify(payload, config)

    # Validate profile exists before we kick off any upstream call
    if chat_req.profile is not None:
        try:
            config.profile_by_name(chat_req.profile)
        except KeyError as exc:
            available = [p.name for p in config.profiles]
            raise HTTPException(
                status_code=400,
                detail=(f"unknown profile {chat_req.profile!r}. available: {available}"),
            ) from exc

    if chat_req.stream:
        # M14: overall stream timeout + client-disconnect cleanup.
        timeout_s = _resolve_stream_timeout_s(engine, chat_req.profile)
        return StreamingResponse(
            _sse_iterator(engine, chat_req, timeout_s=timeout_s),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        response = await engine.generate(chat_req)
    except NoProvidersAvailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return response.model_dump(exclude_none=True)


async def _sse_iterator(
    engine: FallbackEngine,
    chat_req: ChatRequest,
    *,
    timeout_s: float = _STREAM_TIMEOUT_DEFAULT_S,
) -> AsyncIterator[str]:
    """Wrap the engine's stream into SSE wire format.

    M14: bounded by an overall ``timeout_s`` ceiling and guarantees the
    upstream engine generator is finalized on client disconnect
    (``CancelledError``) or timeout, so the upstream connection is
    released instead of leaking.
    """
    source = engine.stream(chat_req)
    try:
        async with asyncio.timeout(timeout_s):
            async for chunk in source:
                data = chunk.model_dump(exclude_none=True)
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    except NoProvidersAvailableError as exc:
        # Encode the error inside the SSE channel — OpenAI clients handle this
        err = {"error": {"message": str(exc), "type": "no_providers_available"}}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except TimeoutError:
        # M14: overall ceiling hit — surface a terminal error frame.
        logger.warning("sse-stream-timeout", extra={"timeout_s": timeout_s})
        err = {"error": {"message": f"stream exceeded {timeout_s:.0f}s ceiling", "type": "timeout"}}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        # M14: client disconnected — re-raise after finalizing the source.
        logger.info("sse-client-disconnect")
        raise
    finally:
        # M14: ensure the engine generator's finally blocks run so the
        # adapter's httpx streaming context releases the upstream socket.
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            # Best-effort cleanup; never mask the original exit reason.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await aclose()
