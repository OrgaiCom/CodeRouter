"""Regression tests for the medium-priority routing fixes (M1, M2, M3, M11).

Each class targets one fix:

  * ``TestM1DriftVerdictScoping`` — the drift verdict is request-scoped, so
    concurrent requests never observe each other's verdict and the ingress
    header does not go stale across requests.
  * ``TestM2StreamRecordsAttempt`` — the streaming paths (OpenAI + Anthropic)
    feed ``AdaptiveAdjuster.record_attempt`` (previously only the
    non-streaming ``generate_anthropic`` did).
  * ``TestM3DriftDemotionChainOrder`` — drift demotion reorders the chain
    even when the profile is NOT adaptive.
  * ``TestM11SingleResolution`` — ``apply_context_budget`` + the engine entry
    point resolve the chain (and run the token estimate) only once.

These are self-contained (scripted fake adapters, no network).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import (
    AdapterError,
    BaseAdapter,
    ChatRequest,
    ChatResponse,
    ProviderCallOverrides,
    StreamChunk,
)
from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.guards.drift_detection import DriftVerdict, DriftWindow
from coderouter.routing import FallbackEngine, NoProvidersAvailableError
from coderouter.routing.adaptive import AdaptiveAdjuster
from coderouter.routing.fallback import _drift_verdict_ctx
from coderouter.translation import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeNativeAdapter(AnthropicAdapter):
    """Scripted native Anthropic adapter."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        output_tokens: int = 50,
        fail_with: AdapterError | None = None,
    ) -> None:
        super().__init__(config)
        self.output_tokens = output_tokens
        self.fail_with = fail_with
        self.call_count = 0

    async def healthcheck(self) -> bool:
        return self.fail_with is None

    async def generate_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AnthropicResponse:
        self.call_count += 1
        if self.fail_with:
            raise self.fail_with
        return AnthropicResponse(
            id="msg_fake",
            model=self.config.model,
            content=[{"type": "text", "text": "x" * self.output_tokens}],
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=10, output_tokens=self.output_tokens),
            coderouter_provider=self.name,
        )

    async def stream_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        self.call_count += 1
        if self.fail_with:
            raise self.fail_with
        yield AnthropicStreamEvent(
            type="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.config.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        )
        yield AnthropicStreamEvent(
            type="content_block_start",
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        yield AnthropicStreamEvent(
            type="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "x" * self.output_tokens},
            },
        )
        yield AnthropicStreamEvent(
            type="content_block_stop",
            data={"type": "content_block_stop", "index": 0},
        )
        yield AnthropicStreamEvent(
            type="message_delta",
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": self.output_tokens},
            },
        )
        yield AnthropicStreamEvent(
            type="message_stop",
            data={"type": "message_stop"},
        )


class _FakeOpenAIAdapter:
    """Duck-typed OpenAI-shaped adapter for generate() / stream() paths."""

    def __init__(self, name: str, *, fail_with: AdapterError | None = None) -> None:
        self.name = name
        self.fail_with = fail_with

    async def generate(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> ChatResponse:
        if self.fail_with:
            raise self.fail_with
        return ChatResponse.model_validate(
            {
                "id": "cmpl_fake",
                "object": "chat.completion",
                "created": 0,
                "model": self.name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async def stream(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if self.fail_with:
            raise self.fail_with
        yield StreamChunk.model_validate(
            {
                "id": "cmpl_fake",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self.name,
                "choices": [
                    {"index": 0, "delta": {"content": "hi"}, "finish_reason": None}
                ],
            }
        )


# ---------------------------------------------------------------------------
# Config / engine helpers
# ---------------------------------------------------------------------------


def _provider(name: str) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="test-model",
        api_key_env="ANTHROPIC_API_KEY",
    )


def _config(
    *,
    providers: list[str],
    adaptive: bool = False,
    drift_action: str = "off",
) -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[_provider(n) for n in providers],
        profiles=[
            FallbackChain(
                name="default",
                providers=providers,
                adaptive=adaptive,
                drift_detection_action=drift_action,
                drift_detection_window_size=10,
            ),
        ],
    )


def _engine(cfg: CodeRouterConfig, adapters: dict[str, BaseAdapter]) -> FallbackEngine:
    """Build a FallbackEngine via __new__ with just enough state."""
    engine = FallbackEngine.__new__(FallbackEngine)
    engine.config = cfg
    engine._adapters = adapters
    engine._adaptive_adjuster = AdaptiveAdjuster()
    engine._drift_window = DriftWindow(
        max_size=cfg.profiles[0].drift_detection_window_size
    )
    engine._drift_demoted = {}
    engine._last_drift_verdict = None
    return engine


def _req(*, stream: bool = False) -> AnthropicRequest:
    return AnthropicRequest(
        max_tokens=64,
        messages=[AnthropicMessage(role="user", content="hi")],
        stream=stream,
    )


async def _drain(it: AsyncIterator[AnthropicStreamEvent]) -> list[AnthropicStreamEvent]:
    return [ev async for ev in it]


# ---------------------------------------------------------------------------
# M1: drift verdict is request-scoped
# ---------------------------------------------------------------------------


class TestM1DriftVerdictScoping:
    def _severe_verdict(self) -> DriftVerdict:
        return DriftVerdict(
            drifted=True,
            severity="severe",
            reason="test",
            signals={"empty_response_rate": 1.0},
        )

    @pytest.mark.asyncio
    async def test_verdict_does_not_leak_across_concurrent_requests(self) -> None:
        """Two concurrent tasks: one sets a verdict, the other must not see it.

        Before M1 the verdict lived on a shared engine attribute, so a
        request that observed drift would flip the header for a concurrent
        request that did not. With the ContextVar each task is isolated.
        """
        cfg = _config(providers=["a"])
        engine = _engine(cfg, {"a": _FakeNativeAdapter(cfg.providers[0])})

        seen: dict[str, str | None] = {}
        start = asyncio.Event()

        async def with_drift() -> None:
            _drift_verdict_ctx.set(self._severe_verdict())
            start.set()
            await asyncio.sleep(0.01)
            seen["with"] = engine.last_drift_severity

        async def without_drift() -> None:
            await start.wait()
            # This task never set a verdict — it must read None even though
            # the sibling task set one concurrently.
            seen["without"] = engine.last_drift_severity

        await asyncio.gather(with_drift(), without_drift())

        assert seen["with"] == "severe"
        assert seen["without"] is None

    @pytest.mark.asyncio
    async def test_fresh_request_starts_with_no_verdict(self) -> None:
        """A brand-new request context defaults to no drift (no stale header).

        This is the ``cooldown early-return leaves stale verdict`` bug: a
        prior request could leave a verdict set; the next request must not
        inherit it. Each ``asyncio.create_task`` copies the context with the
        ContextVar at its default.
        """
        cfg = _config(providers=["a"])
        engine = _engine(cfg, {"a": _FakeNativeAdapter(cfg.providers[0])})

        # Simulate a prior request that ended with a verdict.
        async def prior() -> None:
            _drift_verdict_ctx.set(self._severe_verdict())

        await asyncio.create_task(prior())

        result: dict[str, str | None] = {}

        async def fresh() -> None:
            result["v"] = engine.last_drift_severity

        await asyncio.create_task(fresh())
        assert result["v"] is None


# ---------------------------------------------------------------------------
# M2: streaming paths record adaptive attempts
# ---------------------------------------------------------------------------


class TestM2StreamRecordsAttempt:
    @pytest.mark.asyncio
    async def test_stream_anthropic_records_success(self) -> None:
        cfg = _config(providers=["a"])
        engine = _engine(cfg, {"a": _FakeNativeAdapter(cfg.providers[0])})

        await _drain(engine.stream_anthropic(_req(stream=True)))

        stats = engine._adaptive.stats_for("a")
        assert stats.sample_count == 1
        assert stats.error_rate == 0.0
        assert stats.median_latency_ms is not None

    @pytest.mark.asyncio
    async def test_stream_anthropic_records_failure(self) -> None:
        cfg = _config(providers=["a"])
        engine = _engine(
            cfg,
            {
                "a": _FakeNativeAdapter(
                    cfg.providers[0],
                    fail_with=AdapterError("boom", provider="a", retryable=False),
                )
            },
        )

        with pytest.raises(NoProvidersAvailableError):
            await _drain(engine.stream_anthropic(_req(stream=True)))

        stats = engine._adaptive.stats_for("a")
        assert stats.sample_count == 1
        assert stats.error_rate == 1.0

    @pytest.mark.asyncio
    async def test_openai_stream_records_success(self) -> None:
        cfg = _config(providers=["a"])
        engine = _engine(cfg, {"a": _FakeOpenAIAdapter("a")})

        chat_req = ChatRequest.model_validate(
            {"model": "default", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        chunks = [c async for c in engine.stream(chat_req)]
        assert chunks  # at least one chunk

        stats = engine._adaptive.stats_for("a")
        assert stats.sample_count == 1
        assert stats.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_openai_generate_records_success(self) -> None:
        cfg = _config(providers=["a"])
        engine = _engine(cfg, {"a": _FakeOpenAIAdapter("a")})

        chat_req = ChatRequest.model_validate(
            {"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        )
        await engine.generate(chat_req)

        stats = engine._adaptive.stats_for("a")
        assert stats.sample_count == 1
        assert stats.error_rate == 0.0


# ---------------------------------------------------------------------------
# M3: drift demotion reorders the chain even without adaptive
# ---------------------------------------------------------------------------


class TestM3DriftDemotionChainOrder:
    @pytest.mark.asyncio
    async def test_non_adaptive_profile_honors_drift_demotion(self) -> None:
        """A drift-demoted provider is pushed back even when adaptive=False.

        Pre-M3 the demotion only took effect through the adaptive machinery
        (which is gated on ``adaptive: true`` and a sample threshold), so a
        non-adaptive profile logged "demoted" but still tried the drifting
        provider first. Now ``_resolve_anthropic_chain`` reorders directly.
        """
        cfg = _config(providers=["a", "b"], adaptive=False)
        engine = _engine(
            cfg,
            {
                "a": _FakeNativeAdapter(cfg.providers[0]),
                "b": _FakeNativeAdapter(cfg.providers[1]),
            },
        )

        # Baseline: declared order preserved.
        chain = engine._resolve_anthropic_chain(_req())
        assert [a.name for a, _ in chain] == ["a", "b"]

        # Mark "a" as drift-demoted with the cooldown far in the future
        # (the engine compares against ``time.monotonic()``).
        import time as _t

        engine._drift_demoted["a"] = _t.monotonic() + 10_000

        chain2 = engine._resolve_anthropic_chain(_req())
        assert [a.name for a, _ in chain2] == ["b", "a"]

    @pytest.mark.asyncio
    async def test_expired_demotion_restores_order(self) -> None:
        """An expired drift cooldown restores the declared order."""
        cfg = _config(providers=["a", "b"], adaptive=False)
        engine = _engine(
            cfg,
            {
                "a": _FakeNativeAdapter(cfg.providers[0]),
                "b": _FakeNativeAdapter(cfg.providers[1]),
            },
        )
        import time as _t

        # Cooldown already in the past → pruned, order restored.
        engine._drift_demoted["a"] = _t.monotonic() - 1.0

        chain = engine._resolve_anthropic_chain(_req())
        assert [a.name for a, _ in chain] == ["a", "b"]
        assert "a" not in engine._drift_demoted  # pruned lazily


# ---------------------------------------------------------------------------
# M11: chain resolution + estimate happen once, not twice
# ---------------------------------------------------------------------------


class _CountingEngine(FallbackEngine):
    """Engine subclass that counts chain resolutions + budget-guard runs."""

    def __init__(self, cfg: CodeRouterConfig, adapters: dict[str, BaseAdapter]) -> None:
        self.config = cfg
        self._adapters = adapters
        self._adaptive_adjuster = AdaptiveAdjuster()
        self._drift_window = DriftWindow(
            max_size=cfg.profiles[0].drift_detection_window_size
        )
        self._drift_demoted = {}
        self._last_drift_verdict = None
        self.resolve_count = 0

    def _resolve_anthropic_chain(self, request: AnthropicRequest):  # type: ignore[override]
        self.resolve_count += 1
        return super()._resolve_anthropic_chain(request)


class TestM11SingleResolution:
    @pytest.mark.asyncio
    async def test_generate_reuses_prepared_chain(self) -> None:
        """ingress apply_context_budget + generate_anthropic resolve once."""
        cfg = _config(providers=["a"])
        engine = _CountingEngine(cfg, {"a": _FakeNativeAdapter(cfg.providers[0])})

        # Ingress path: prepare, then dispatch with the returned request.
        prepared_req, _status = engine.apply_context_budget(_req())
        assert engine.resolve_count == 1

        await engine.generate_anthropic(prepared_req)
        # generate_anthropic must NOT resolve the chain a second time.
        assert engine.resolve_count == 1

    @pytest.mark.asyncio
    async def test_stream_reuses_prepared_chain(self) -> None:
        cfg = _config(providers=["a"])
        engine = _CountingEngine(cfg, {"a": _FakeNativeAdapter(cfg.providers[0])})

        prepared_req, _status = engine.apply_context_budget(_req(stream=True))
        assert engine.resolve_count == 1

        await _drain(engine.stream_anthropic(prepared_req))
        assert engine.resolve_count == 1

    @pytest.mark.asyncio
    async def test_direct_call_without_prepare_resolves_normally(self) -> None:
        """A direct engine call (no apply_context_budget) still resolves once."""
        cfg = _config(providers=["a"])
        engine = _CountingEngine(cfg, {"a": _FakeNativeAdapter(cfg.providers[0])})

        await engine.generate_anthropic(_req())
        assert engine.resolve_count == 1


# ---------------------------------------------------------------------------
# M14: SSE overall timeout + client-disconnect cleanup
# ---------------------------------------------------------------------------


class _TrackedSource:
    """Async iterator of SSE frames that records whether it was closed.

    Used to prove ``_guard_stream`` / ``_sse_iterator`` finalize the
    upstream generator (so the upstream connection is released) on timeout
    and on client disconnect.
    """

    def __init__(self, *, frames: list[str], hang: bool = False) -> None:
        self._frames = list(frames)
        self._hang = hang
        self.closed = False

    def __aiter__(self) -> _TrackedSource:
        return self

    async def __anext__(self) -> str:
        if self._frames:
            return self._frames.pop(0)
        if self._hang:
            # Simulate a wedged upstream: never yield another frame.
            await asyncio.sleep(3600)
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class TestM14StreamGuard:
    @pytest.mark.asyncio
    async def test_timeout_emits_error_and_closes_source(self) -> None:
        from coderouter.ingress.anthropic_routes import _guard_stream

        source = _TrackedSource(frames=["event: x\ndata: {}\n\n"], hang=True)
        out: list[str] = []
        async for frame in _guard_stream(source, timeout_s=0.05, label="test"):
            out.append(frame)

        # First real frame plus a synthesized timeout error frame.
        assert any("timeout_error" in f for f in out)
        assert source.closed is True

    @pytest.mark.asyncio
    async def test_client_disconnect_closes_source(self) -> None:
        from coderouter.ingress.anthropic_routes import _guard_stream

        source = _TrackedSource(frames=["event: x\ndata: {}\n\n"], hang=True)
        gen = _guard_stream(source, timeout_s=100.0, label="test")

        # Pull the first frame, then simulate the ASGI server cancelling the
        # generator because the client went away.
        first = await gen.__anext__()
        assert "event: x" in first
        with pytest.raises(asyncio.CancelledError):
            await gen.athrow(asyncio.CancelledError())
        assert source.closed is True

    @pytest.mark.asyncio
    async def test_openai_iterator_timeout_emits_done(self) -> None:
        from coderouter.ingress.openai_routes import _sse_iterator

        class _HangingEngine:
            async def stream(self, chat_req: ChatRequest) -> AsyncIterator[StreamChunk]:
                yield StreamChunk.model_validate(
                    {
                        "id": "c",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "m",
                        "choices": [{"index": 0, "delta": {"content": "hi"}}],
                    }
                )
                await asyncio.sleep(3600)

        chat_req = ChatRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        out = [
            frame
            async for frame in _sse_iterator(
                _HangingEngine(),  # type: ignore[arg-type]
                chat_req,
                timeout_s=0.05,
            )
        ]
        assert any('"type": "timeout"' in f for f in out)
        assert out[-1] == "data: [DONE]\n\n"
