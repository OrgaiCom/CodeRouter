"""v2.15.0: fallback reason visibility.

Three layers, bottom-up:

1. :mod:`coderouter.routing.fallback_trace` in isolation — the reason
   classifier and the hop bookkeeping that turns "A failed, then B ran"
   into a resolved ``A --timeout--> B`` record.
2. The engine — a real :class:`FallbackEngine` over scripted fake adapters
   (same ``_engine_with_adapters`` shape ``test_fallback_anthropic.py``
   uses) records the right reasons on both the non-streaming and the
   streaming Anthropic paths, and emits ``fallback-occurred`` log lines.
3. The ingress — ``POST /v1/messages`` surfaces the trace as
   ``X-CodeRouter-Fallback-*`` response headers (non-streaming, and
   pre-attempt hops on the streaming path) plus the trailing
   ``coderouter_fallback`` SSE metadata event.

The invariant every layer shares: **a request served by its first
provider produces nothing.** No headers, no SSE trailer, no log lines.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi.testclient import TestClient

from coderouter.adapters.anthropic_native import AnthropicAdapter
from coderouter.adapters.base import (
    AdapterError,
    BaseAdapter,
    ChatRequest,
    ChatResponse,
    ProviderCallOverrides,
)
from coderouter.config.schemas import (
    CodeRouterConfig,
    FallbackChain,
    ProviderConfig,
)
from coderouter.ingress.app import create_app
from coderouter.routing import FallbackEngine, NoProvidersAvailableError
from coderouter.routing.fallback_trace import (
    HEADER_FALLBACK_CHAIN,
    HEADER_FALLBACK_FROM,
    HEADER_FALLBACK_REASON,
    HEADER_FALLBACK_TO,
    REASON_AUTH,
    REASON_BUDGET_EXCEEDED,
    REASON_CONNECTION,
    REASON_PAID_GATE,
    REASON_RATE_LIMIT,
    REASON_TIMEOUT,
    REASON_UPSTREAM_4XX,
    REASON_UPSTREAM_5XX,
    SSE_FALLBACK_EVENT,
    FallbackTrace,
    classify_adapter_error,
    current_fallback_trace,
    describe_adapter_error,
    reset_fallback_trace,
)
from coderouter.translation import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)

# ----------------------------------------------------------------------
# Fakes — mirror tests/test_fallback_anthropic.py so a reader moving
# between the two files sees the same shapes.
# ----------------------------------------------------------------------


class FailingOpenAIAdapter(BaseAdapter):
    """`kind: openai_compat` adapter that always raises a scripted error."""

    def __init__(self, config: ProviderConfig, *, error: AdapterError) -> None:
        super().__init__(config)
        self.error = error
        self.calls = 0

    async def healthcheck(self) -> bool:
        return False

    async def generate(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> ChatResponse:
        self.calls += 1
        raise self.error

    async def stream(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[ChatResponse]:
        self.calls += 1
        raise self.error
        yield  # pragma: no cover  # generator protocol


class OkOpenAIAdapter(BaseAdapter):
    """`kind: openai_compat` adapter that always answers with plain text."""

    def __init__(self, config: ProviderConfig, *, text: str = "served") -> None:
        super().__init__(config)
        self.text = text
        self.calls = 0

    async def healthcheck(self) -> bool:
        return True

    async def generate(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            id=f"chatcmpl-{self.name}",
            created=int(time.time()),
            model=self.config.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.text},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            coderouter_provider=self.name,
        )

    async def stream(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[ChatResponse]:  # pragma: no cover - unused
        raise NotImplementedError
        yield


class NativeAdapter(AnthropicAdapter):
    """`kind: anthropic` adapter with no network underneath.

    ``fail_with`` applies to both entry points so one fake covers the
    "primary is down" role on the streaming and non-streaming paths.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        text: str = "native ok",
        fail_with: AdapterError | None = None,
    ) -> None:
        super().__init__(config)
        self.text = text
        self.fail_with = fail_with
        self.calls = 0

    async def healthcheck(self) -> bool:
        return self.fail_with is None

    async def generate_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AnthropicResponse:
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        return AnthropicResponse(
            id="msg_native",
            model=self.config.model,
            content=[{"type": "text", "text": self.text}],
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=1, output_tokens=2),
            coderouter_provider=self.name,
        )

    async def stream_anthropic(
        self,
        request: AnthropicRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        for ev in _native_events(self.text, self.config.model):
            yield ev


def _native_events(text: str, model: str) -> list[AnthropicStreamEvent]:
    """A minimal compliant Anthropic SSE sequence."""
    return [
        AnthropicStreamEvent(
            type="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": "msg_native",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        ),
        AnthropicStreamEvent(
            type="content_block_start",
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        AnthropicStreamEvent(
            type="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        AnthropicStreamEvent(
            type="content_block_stop",
            data={"type": "content_block_stop", "index": 0},
        ),
        AnthropicStreamEvent(
            type="message_delta",
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
        ),
        AnthropicStreamEvent(
            type="message_stop",
            data={"type": "message_stop"},
        ),
    ]


# ----------------------------------------------------------------------
# Config / engine helpers
# ----------------------------------------------------------------------


def _provider(name: str, kind: str = "openai_compat", **kwargs: object) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        base_url=(
            "https://api.anthropic.com"
            if kind == "anthropic"
            else "http://localhost:11434/v1"
        ),
        model=f"{name}-model",
        api_key_env="ANTHROPIC_API_KEY" if kind == "anthropic" else None,
        **kwargs,  # type: ignore[arg-type]
    )


def _config(
    providers: list[ProviderConfig],
    *,
    allow_paid: bool = False,
) -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=allow_paid,
        default_profile="default",
        providers=providers,
        profiles=[
            FallbackChain(name="default", providers=[p.name for p in providers])
        ],
    )


def _engine(config: CodeRouterConfig, fakes: dict[str, BaseAdapter]) -> FallbackEngine:
    engine = FallbackEngine.__new__(FallbackEngine)
    engine.config = config
    engine._adapters = fakes  # type: ignore[attr-defined]
    return engine


def _req(*, stream: bool = False) -> AnthropicRequest:
    return AnthropicRequest(
        max_tokens=64,
        messages=[AnthropicMessage(role="user", content="hi")],
        stream=stream,
        model="claude-3-5-sonnet",
    )


@pytest.fixture(autouse=True)
def _clean_trace() -> Iterator[None]:
    """Each test starts with no trace installed in the context."""
    reset_fallback_trace()
    yield
    reset_fallback_trace()


# ======================================================================
# Layer 1 — the trace object itself
# ======================================================================


class TestReasonClassification:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (408, REASON_TIMEOUT),
            (429, REASON_RATE_LIMIT),
            (401, REASON_AUTH),
            (403, REASON_AUTH),
            (400, REASON_UPSTREAM_4XX),
            (404, REASON_UPSTREAM_4XX),
            (500, REASON_UPSTREAM_5XX),
            (503, REASON_UPSTREAM_5XX),
        ],
    )
    def test_status_codes_map_to_canonical_reasons(
        self, status: int, expected: str
    ) -> None:
        exc = AdapterError("boom", provider="p", status_code=status)
        assert classify_adapter_error(exc) == expected

    def test_transport_failures_are_disambiguated_by_message(self) -> None:
        """No status code → the adapters' stable message prefixes decide."""
        timeout = AdapterError("timeout contacting http://x/v1", provider="p")
        transport = AdapterError("transport error: boom", provider="p")
        assert classify_adapter_error(timeout) == REASON_TIMEOUT
        assert classify_adapter_error(transport) == REASON_CONNECTION

    def test_detail_never_carries_upstream_text(self) -> None:
        """``detail`` reaches the client; the upstream body must not.

        The full error text stays in the existing ``provider-failed`` log
        line (which passes through secret redaction). Here we only ship a
        structural summary.
        """
        exc = AdapterError(
            "500 from upstream: sk-secret-token-leaked-in-body",
            provider="p",
            status_code=500,
        )
        detail = describe_adapter_error(exc)
        assert detail == "status=500"
        assert "secret" not in detail


class TestTraceBookkeeping:
    def test_no_hops_means_no_headers(self) -> None:
        trace = FallbackTrace()
        trace.record_attempt("local")
        assert trace.occurred is False
        assert trace.header_values() == {}

    def test_single_hop_resolves_to_provider(self) -> None:
        trace = FallbackTrace()
        trace.record_attempt("local")
        trace.record_failure("local", REASON_TIMEOUT)
        trace.record_attempt("cloud")

        assert trace.chain == ["local", "cloud"]
        assert trace.reasons == [REASON_TIMEOUT]
        assert trace.header_values() == {
            HEADER_FALLBACK_FROM: "local",
            HEADER_FALLBACK_TO: "cloud",
            HEADER_FALLBACK_REASON: REASON_TIMEOUT,
            HEADER_FALLBACK_CHAIN: "local>cloud",
        }

    def test_consecutive_skips_all_point_at_the_provider_that_ran(self) -> None:
        """Two gates filtered two providers; both hops name the survivor."""
        trace = FallbackTrace()
        trace.record_skip("local", REASON_BUDGET_EXCEEDED)
        trace.record_skip("mid", REASON_PAID_GATE)
        trace.record_attempt("cloud")

        assert trace.chain == ["local", "mid", "cloud"]
        assert trace.reasons == [REASON_BUDGET_EXCEEDED, REASON_PAID_GATE]
        headers = trace.header_values()
        assert headers[HEADER_FALLBACK_CHAIN] == "local>mid>cloud"
        assert headers[HEADER_FALLBACK_TO] == "cloud"

    def test_multi_hop_chain_is_reconstructible_from_headers(self) -> None:
        """``len(chain) == len(reasons) + 1`` once someone served."""
        trace = FallbackTrace()
        trace.record_attempt("local")
        trace.record_failure("local", REASON_TIMEOUT)
        trace.record_attempt("ollama")
        trace.record_failure("ollama", REASON_UPSTREAM_5XX)
        trace.record_attempt("openrouter")

        headers = trace.header_values()
        chain = headers[HEADER_FALLBACK_CHAIN].split(">")
        reasons = headers[HEADER_FALLBACK_REASON].split(",")
        assert chain == ["local", "ollama", "openrouter"]
        assert reasons == [REASON_TIMEOUT, REASON_UPSTREAM_5XX]
        assert len(chain) == len(reasons) + 1

    def test_exhausted_chain_omits_the_to_header(self) -> None:
        trace = FallbackTrace()
        trace.record_attempt("local")
        trace.record_failure("local", REASON_TIMEOUT)
        trace.record_attempt("cloud")
        trace.record_failure("cloud", REASON_AUTH)

        headers = trace.header_values()
        assert HEADER_FALLBACK_TO not in headers
        assert headers[HEADER_FALLBACK_CHAIN] == "local>cloud"
        assert headers[HEADER_FALLBACK_REASON] == f"{REASON_TIMEOUT},{REASON_AUTH}"

    def test_provider_names_are_sanitized_for_the_wire(self) -> None:
        """A YAML-supplied name can never break (or inject into) a header."""
        trace = FallbackTrace()
        trace.record_failure("bad name\r\nX-Evil: 1", REASON_TIMEOUT)
        trace.record_attempt("ok")

        value = trace.header_values()[HEADER_FALLBACK_FROM]
        assert "\r" not in value and "\n" not in value and " " not in value
        # ``-`` survives (it is in the allowlist); the space, the CR, the LF
        # and the ``:`` each collapse to a single ``_``.
        assert value == "bad_name__X-Evil__1"


# ======================================================================
# Layer 2 — the engine
# ======================================================================


class TestEngineRecordsReasons:
    async def test_first_provider_success_records_nothing(self) -> None:
        config = _config([_provider("first", "anthropic"), _provider("second")])
        engine = _engine(
            config,
            {
                "first": NativeAdapter(config.providers[0]),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
        )

        await engine.generate_anthropic(_req())

        trace = current_fallback_trace()
        assert trace is not None
        assert trace.occurred is False
        assert trace.header_values() == {}

    async def test_upstream_5xx_is_recorded_with_from_and_to(self) -> None:
        config = _config([_provider("first", "anthropic"), _provider("second")])
        engine = _engine(
            config,
            {
                "first": NativeAdapter(
                    config.providers[0],
                    fail_with=AdapterError(
                        "503 from upstream", provider="first", status_code=503
                    ),
                ),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
        )

        resp = await engine.generate_anthropic(_req())
        assert resp.coderouter_provider == "second"

        trace = current_fallback_trace()
        assert trace is not None
        assert trace.header_values() == {
            HEADER_FALLBACK_FROM: "first",
            HEADER_FALLBACK_TO: "second",
            HEADER_FALLBACK_REASON: REASON_UPSTREAM_5XX,
            HEADER_FALLBACK_CHAIN: "first>second",
        }
        assert trace.hops[0].pre_attempt is False
        assert trace.hops[0].detail == "status=503"

    async def test_timeout_reason_survives_a_missing_status_code(self) -> None:
        config = _config([_provider("first"), _provider("second")])
        engine = _engine(
            config,
            {
                "first": FailingOpenAIAdapter(
                    config.providers[0],
                    error=AdapterError(
                        "timeout contacting http://localhost:11434/v1",
                        provider="first",
                    ),
                ),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
        )

        await engine.generate_anthropic(_req())

        trace = current_fallback_trace()
        assert trace is not None
        assert trace.reasons == [REASON_TIMEOUT]

    async def test_streaming_path_records_the_same_reasons(self) -> None:
        config = _config([_provider("first", "anthropic"), _provider("second", "anthropic")])
        engine = _engine(
            config,
            {
                "first": NativeAdapter(
                    config.providers[0],
                    fail_with=AdapterError(
                        "429 rate limited", provider="first", status_code=429
                    ),
                ),
                "second": NativeAdapter(config.providers[1], text="second served"),
            },
        )

        events = [ev async for ev in engine.stream_anthropic(_req(stream=True))]
        assert events[-1].type == "message_stop"

        trace = current_fallback_trace()
        assert trace is not None
        assert trace.reasons == [REASON_RATE_LIMIT]
        assert trace.from_provider == "first"
        assert trace.to_provider == "second"
        assert trace.hops[0].stream is True

    async def test_paid_gate_is_a_pre_attempt_hop(self) -> None:
        """A provider filtered at chain-resolve time never gets called.

        The hop still points at whoever ran, because from the client's
        perspective "my request went somewhere else, and here is why" is
        the same question whether the primary failed or was never tried.
        """
        config = _config([_provider("paid_one", paid=True), _provider("free_one")])
        engine = _engine(
            config,
            {
                "paid_one": OkOpenAIAdapter(config.providers[0]),
                "free_one": OkOpenAIAdapter(config.providers[1], text="free"),
            },
        )

        resp = await engine.generate_anthropic(_req())
        assert resp.coderouter_provider == "free_one"

        trace = current_fallback_trace()
        assert trace is not None
        assert trace.reasons == [REASON_PAID_GATE]
        assert trace.hops[0].pre_attempt is True
        assert trace.header_values()[HEADER_FALLBACK_CHAIN] == "paid_one>free_one"

    async def test_exhausted_chain_still_records_every_departure(self) -> None:
        config = _config([_provider("first"), _provider("second")])
        engine = _engine(
            config,
            {
                "first": FailingOpenAIAdapter(
                    config.providers[0],
                    error=AdapterError("503", provider="first", status_code=503),
                ),
                "second": FailingOpenAIAdapter(
                    config.providers[1],
                    error=AdapterError("500", provider="second", status_code=500),
                ),
            },
        )

        with pytest.raises(NoProvidersAvailableError):
            await engine.generate_anthropic(_req())

        trace = current_fallback_trace()
        assert trace is not None
        assert trace.reasons == [REASON_UPSTREAM_5XX, REASON_UPSTREAM_5XX]
        assert trace.to_provider is None

    async def test_successive_requests_do_not_accumulate_hops(self) -> None:
        """The trace is request-scoped, not engine-global.

        Two dispatches in the same context must not concatenate — that was
        the exact bug M1 fixed for the drift verdict.
        """
        config = _config([_provider("first"), _provider("second")])
        engine = _engine(
            config,
            {
                "first": FailingOpenAIAdapter(
                    config.providers[0],
                    error=AdapterError("503", provider="first", status_code=503),
                ),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
        )

        await engine.generate_anthropic(_req())
        await engine.generate_anthropic(_req())

        trace = current_fallback_trace()
        assert trace is not None
        assert len(trace.hops) == 1


class TestFallbackOccurredLog:
    async def test_one_warn_line_per_hop(self, caplog: pytest.LogCaptureFixture) -> None:
        config = _config([_provider("first"), _provider("second")])
        engine = _engine(
            config,
            {
                "first": FailingOpenAIAdapter(
                    config.providers[0],
                    error=AdapterError("503", provider="first", status_code=503),
                ),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
        )

        with caplog.at_level(logging.WARNING, logger="coderouter.routing.fallback"):
            await engine.generate_anthropic(_req())

        records = [r for r in caplog.records if r.msg == "fallback-occurred"]
        assert len(records) == 1
        record = records[0]
        assert record.from_provider == "first"  # type: ignore[attr-defined]
        assert record.to_provider == "second"  # type: ignore[attr-defined]
        assert record.reason == REASON_UPSTREAM_5XX  # type: ignore[attr-defined]
        assert record.pre_attempt is False  # type: ignore[attr-defined]
        assert record.hop_index == 0  # type: ignore[attr-defined]
        # ``provider`` is duplicated so existing per-provider log queries
        # keep matching without learning the new key.
        assert record.provider == "first"  # type: ignore[attr-defined]

    async def test_healthy_request_emits_no_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = _config([_provider("first"), _provider("second")])
        engine = _engine(
            config,
            {
                "first": OkOpenAIAdapter(config.providers[0]),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
        )

        with caplog.at_level(logging.WARNING, logger="coderouter.routing.fallback"):
            await engine.generate_anthropic(_req())

        assert [r for r in caplog.records if r.msg == "fallback-occurred"] == []


# ======================================================================
# Layer 3 — the ingress
# ======================================================================


def _client(
    config: CodeRouterConfig,
    fakes: dict[str, BaseAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    monkeypatch.setattr(
        "coderouter.ingress.app.load_config", lambda path=None: config
    )
    app = create_app()
    app.state.engine = _engine(config, fakes)
    app.state.config = config
    return TestClient(app)


def _body(stream: bool = False) -> dict[str, object]:
    return {
        "model": "claude-3-5-sonnet",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }


def _openai_body(stream: bool = False) -> dict[str, object]:
    return {
        "model": "first-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }


def _sse_events(text: str) -> list[tuple[str, dict[str, object]]]:
    """Parse an Anthropic SSE body into ``(event_name, payload)`` pairs."""
    out: list[tuple[str, dict[str, object]]] = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        name = ""
        payload: dict[str, object] = {}
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
        out.append((name, payload))
    return out


class TestIngressHeaders:
    def test_non_streaming_response_carries_the_fallback_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config([_provider("first"), _provider("second")])
        client = _client(
            config,
            {
                "first": FailingOpenAIAdapter(
                    config.providers[0],
                    error=AdapterError("503", provider="first", status_code=503),
                ),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
            monkeypatch,
        )

        resp = client.post("/v1/messages", json=_body())

        assert resp.status_code == 200
        assert resp.headers[HEADER_FALLBACK_FROM] == "first"
        assert resp.headers[HEADER_FALLBACK_TO] == "second"
        assert resp.headers[HEADER_FALLBACK_REASON] == REASON_UPSTREAM_5XX
        assert resp.headers[HEADER_FALLBACK_CHAIN] == "first>second"

    def test_no_fallback_means_no_new_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward compatibility: the healthy path is unchanged."""
        config = _config([_provider("first"), _provider("second")])
        client = _client(
            config,
            {
                "first": OkOpenAIAdapter(config.providers[0]),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
            monkeypatch,
        )

        resp = client.post("/v1/messages", json=_body())

        assert resp.status_code == 200
        for header in (
            HEADER_FALLBACK_FROM,
            HEADER_FALLBACK_TO,
            HEADER_FALLBACK_REASON,
            HEADER_FALLBACK_CHAIN,
        ):
            assert header not in resp.headers

    def test_openai_route_carries_the_same_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``/v1/chat/completions`` speaks the same header vocabulary.

        The OpenAI ingress had no ``X-CodeRouter-*`` *response* header at
        all before v2.15.0 (it only ever read ``X-CodeRouter-Profile`` /
        ``-Mode`` off the request), so this is the first one — built from
        the same ``header_values()`` as the Anthropic route so the two
        surfaces cannot drift apart.
        """
        config = _config([_provider("first"), _provider("second")])
        client = _client(
            config,
            {
                "first": FailingOpenAIAdapter(
                    config.providers[0],
                    error=AdapterError(
                        "timed out", provider="first", status_code=None
                    ),
                ),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
            monkeypatch,
        )

        resp = client.post("/v1/chat/completions", json=_openai_body())

        assert resp.status_code == 200
        assert resp.headers[HEADER_FALLBACK_FROM] == "first"
        assert resp.headers[HEADER_FALLBACK_TO] == "second"
        assert resp.headers[HEADER_FALLBACK_REASON] == REASON_TIMEOUT
        assert resp.headers[HEADER_FALLBACK_CHAIN] == "first>second"
        # The body is untouched — adding headers must not reshape it.
        assert resp.json()["choices"][0]["message"]["content"] == "served"

    def test_openai_route_healthy_path_has_no_new_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The zero-fallback OpenAI response is unchanged from v2.14.0."""
        config = _config([_provider("first"), _provider("second")])
        client = _client(
            config,
            {
                "first": OkOpenAIAdapter(config.providers[0]),
                "second": OkOpenAIAdapter(config.providers[1]),
            },
            monkeypatch,
        )

        resp = client.post("/v1/chat/completions", json=_openai_body())

        assert resp.status_code == 200
        for header in (
            HEADER_FALLBACK_FROM,
            HEADER_FALLBACK_TO,
            HEADER_FALLBACK_REASON,
            HEADER_FALLBACK_CHAIN,
        ):
            assert header not in resp.headers

    def test_streaming_pre_attempt_hops_reach_the_response_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chain-resolve skips are known before the SSE body commits.

        The paid gate filters ``paid_one`` while ``apply_context_budget``
        resolves the chain, which is *before* the StreamingResponse is
        constructed — so unlike a runtime attempt failure, this one can
        legitimately ride on the HTTP headers.
        """
        config = _config(
            [_provider("paid_one", paid=True), _provider("free_one", "anthropic")]
        )
        client = _client(
            config,
            {
                "paid_one": OkOpenAIAdapter(config.providers[0]),
                "free_one": NativeAdapter(config.providers[1]),
            },
            monkeypatch,
        )

        resp = client.post("/v1/messages", json=_body(stream=True))

        assert resp.status_code == 200
        assert resp.headers[HEADER_FALLBACK_FROM] == "paid_one"
        assert resp.headers[HEADER_FALLBACK_TO] == "free_one"
        assert resp.headers[HEADER_FALLBACK_REASON] == REASON_PAID_GATE


class TestIngressSseTrailer:
    def test_runtime_fallback_is_delivered_as_a_trailing_sse_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The streaming surface for reasons discovered after headers ship."""
        config = _config(
            [_provider("first", "anthropic"), _provider("second", "anthropic")]
        )
        client = _client(
            config,
            {
                "first": NativeAdapter(
                    config.providers[0],
                    fail_with=AdapterError(
                        "500 boom", provider="first", status_code=500
                    ),
                ),
                "second": NativeAdapter(config.providers[1], text="second served"),
            },
            monkeypatch,
        )

        resp = client.post("/v1/messages", json=_body(stream=True))
        events = _sse_events(resp.text)
        names = [name for name, _ in events]

        # The Anthropic sequence is intact and complete...
        assert names[0] == "message_start"
        assert "message_stop" in names
        # ...and the extension event trails it, never interleaves.
        assert names[-1] == SSE_FALLBACK_EVENT
        assert names.index("message_stop") < names.index(SSE_FALLBACK_EVENT)

        payload = events[-1][1]
        assert payload["from"] == "first"
        assert payload["to"] == "second"
        assert payload["reason"] == [REASON_UPSTREAM_5XX]
        assert payload["chain"] == ["first", "second"]
        assert payload["hops"][0]["stream"] is True  # type: ignore[index]

    def test_healthy_stream_has_no_trailing_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(
            [_provider("first", "anthropic"), _provider("second", "anthropic")]
        )
        client = _client(
            config,
            {
                "first": NativeAdapter(config.providers[0]),
                "second": NativeAdapter(config.providers[1]),
            },
            monkeypatch,
        )

        resp = client.post("/v1/messages", json=_body(stream=True))
        names = [name for name, _ in _sse_events(resp.text)]

        assert SSE_FALLBACK_EVENT not in names
        assert names[-1] == "message_stop"
