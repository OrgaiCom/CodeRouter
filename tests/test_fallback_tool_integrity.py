"""v2.15.0 regression: tool-call integrity across a fallback hop.

The claim these tests defend is "**CodeRouter can fall back mid-conversation
without breaking your tool calls**". That claim has two halves, and both
are load-bearing:

* **Inbound.** The conversation handed to the *second* provider must still
  pair every ``tool_result`` with the ``tool_use`` it answers. A chain that
  drops, reorders, or renames one half leaves the model looking at an
  orphaned tool result — which is exactly the failure mode that makes
  agents hallucinate a re-run of a tool they already ran.
* **Outbound.** The response the client finally receives must be a
  structurally valid Anthropic ``tool_use`` block, not tool syntax leaked
  into a text block, and — on the streaming path — a properly opened and
  closed ``content_block``.

The fakes follow ``tests/test_fallback_anthropic.py``: no HTTP, adapters
replaced by scripted objects that record what the engine handed them.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

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
from coderouter.routing import FallbackEngine
from coderouter.routing.fallback_trace import (
    HEADER_FALLBACK_CHAIN,
    HEADER_FALLBACK_FROM,
    HEADER_FALLBACK_REASON,
    HEADER_FALLBACK_TO,
    REASON_UPSTREAM_5XX,
    current_fallback_trace,
    reset_fallback_trace,
)
from coderouter.translation import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicTool,
)

# The in-flight tool call the conversation is already in the middle of when
# the primary provider dies.
_TOOL_USE_ID = "toolu_01ABCDEFGHIJKLMNOPQR"
_TOOL_NAME = "read_file"
# The id the surviving provider mints for the *next* tool call.
_NEXT_CALL_ID = "call_second_provider_0"


class RecordingAdapter(BaseAdapter):
    """`kind: openai_compat` fake that records every request it is handed.

    ``fail_with`` makes it the failing primary; otherwise it answers with a
    scripted tool call so the outbound half of the invariant can be
    checked. Requests are recorded *before* the failure so both providers'
    views of the conversation can be compared.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        fail_with: AdapterError | None = None,
    ) -> None:
        super().__init__(config)
        self.fail_with = fail_with
        self.seen: list[ChatRequest] = []

    async def healthcheck(self) -> bool:
        return self.fail_with is None

    def _tool_call_response(self) -> ChatResponse:
        return ChatResponse(
            id=f"chatcmpl-{self.name}",
            created=int(time.time()),
            model=self.config.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": _NEXT_CALL_ID,
                                "type": "function",
                                "function": {
                                    "name": _TOOL_NAME,
                                    "arguments": json.dumps({"path": "next.txt"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
            coderouter_provider=self.name,
        )

    async def generate(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> ChatResponse:
        self.seen.append(request)
        if self.fail_with:
            raise self.fail_with
        return self._tool_call_response()

    async def stream(
        self,
        request: ChatRequest,
        *,
        overrides: ProviderCallOverrides | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.seen.append(request)
        if self.fail_with:
            raise self.fail_with
        # Unused on the tool path: with tools declared, the engine takes the
        # v0.3-D downgrade (non-streaming + synthesize), so ``generate`` is
        # what actually runs.
        raise NotImplementedError  # pragma: no cover
        yield  # pragma: no cover


def _config() -> CodeRouterConfig:
    return CodeRouterConfig(
        allow_paid=False,
        default_profile="default",
        providers=[
            ProviderConfig(
                name="primary",
                kind="openai_compat",
                base_url="http://localhost:11434/v1",
                model="primary-model",
            ),
            ProviderConfig(
                name="backup",
                kind="openai_compat",
                base_url="http://localhost:11435/v1",
                model="backup-model",
            ),
        ],
        profiles=[FallbackChain(name="default", providers=["primary", "backup"])],
    )


def _engine(config: CodeRouterConfig, fakes: dict[str, BaseAdapter]) -> FallbackEngine:
    engine = FallbackEngine.__new__(FallbackEngine)
    engine.config = config
    engine._adapters = fakes  # type: ignore[attr-defined]
    return engine


def _mid_tool_turn_request(*, stream: bool = False) -> AnthropicRequest:
    """A conversation paused between a ``tool_use`` and its ``tool_result``.

    This is the exact state Claude Code is in when it has asked for a file
    and just fed the contents back — the moment a fallback is most likely
    to do damage.
    """
    return AnthropicRequest(
        model="claude-3-5-sonnet",
        max_tokens=256,
        stream=stream,
        tools=[
            AnthropicTool(
                name=_TOOL_NAME,
                description="Read a file from disk",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ],
        messages=[
            AnthropicMessage(role="user", content="read a.txt then read next.txt"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "text", "text": "Reading the first file."},
                    {
                        "type": "tool_use",
                        "id": _TOOL_USE_ID,
                        "name": _TOOL_NAME,
                        "input": {"path": "a.txt"},
                    },
                ],
            ),
            AnthropicMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": _TOOL_USE_ID,
                        "content": "contents of a.txt",
                    }
                ],
            ),
        ],
    )


def _assert_tool_pairing_intact(chat_req: ChatRequest) -> None:
    """Every ``role: tool`` message answers a tool call that precedes it.

    This is the invariant an OpenAI-compatible backend enforces (and that
    a well-formed Anthropic conversation encodes): a tool result is only
    meaningful next to the call it answers. Checking it positionally —
    rather than just asserting the ids exist somewhere — is what makes the
    test catch a reordering regression, not only a dropping one.
    """
    messages = [
        m if isinstance(m, dict) else m.model_dump(exclude_none=True)
        for m in chat_req.messages
    ]
    announced: set[str] = set()
    answered: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                announced.add(str(call["id"]))
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            assert call_id, "tool result lost its tool_call_id across the hop"
            assert call_id in announced, (
                f"tool_result {call_id!r} has no preceding tool_use "
                f"(announced so far: {sorted(announced)})"
            )
            answered.append(call_id)

    assert answered == [_TOOL_USE_ID], (
        "the in-flight tool_result did not survive the hop intact"
    )
    # The tools array itself has to arrive too, or the surviving provider
    # cannot legally emit the next call.
    tool_names = [
        (t.get("function") or {}).get("name")
        for t in (chat_req.tools or [])
        if isinstance(t, dict)
    ]
    assert _TOOL_NAME in tool_names


def _blocks(content: Any) -> list[dict[str, Any]]:
    """Normalize Anthropic content blocks to plain dicts."""
    out: list[dict[str, Any]] = []
    for block in content or []:
        out.append(block if isinstance(block, dict) else block.model_dump())
    return out


@pytest.fixture(autouse=True)
def _clean_trace() -> AsyncIterator[None]:
    reset_fallback_trace()
    yield
    reset_fallback_trace()


class TestNonStreamingToolIntegrity:
    async def test_tool_use_result_pair_survives_the_fallback(self) -> None:
        config = _config()
        primary = RecordingAdapter(
            config.providers[0],
            fail_with=AdapterError(
                "503 from upstream", provider="primary", status_code=503
            ),
        )
        backup = RecordingAdapter(config.providers[1])
        engine = _engine(config, {"primary": primary, "backup": backup})

        resp = await engine.generate_anthropic(_mid_tool_turn_request())

        # The fallback really happened...
        assert primary.seen and backup.seen
        assert resp.coderouter_provider == "backup"

        # ...and both providers saw the same, intact conversation. The
        # second provider is not handed a degraded copy just because it
        # came second.
        _assert_tool_pairing_intact(primary.seen[0])
        _assert_tool_pairing_intact(backup.seen[0])

    async def test_surviving_provider_returns_a_structured_tool_use_block(
        self,
    ) -> None:
        """The next call comes back as a block, not as text.

        A tool call that degrades into prose after a fallback is silently
        broken: the client sees an assistant turn that *talks about*
        calling a tool instead of one that calls it.
        """
        config = _config()
        primary = RecordingAdapter(
            config.providers[0],
            fail_with=AdapterError("503", provider="primary", status_code=503),
        )
        backup = RecordingAdapter(config.providers[1])
        engine = _engine(config, {"primary": primary, "backup": backup})

        resp = await engine.generate_anthropic(_mid_tool_turn_request())

        blocks = _blocks(resp.content)
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        assert len(tool_uses) == 1
        assert tool_uses[0]["id"] == _NEXT_CALL_ID
        assert tool_uses[0]["name"] == _TOOL_NAME
        assert tool_uses[0]["input"] == {"path": "next.txt"}
        assert resp.stop_reason == "tool_use"
        # The new call must not collide with the one already answered.
        assert tool_uses[0]["id"] != _TOOL_USE_ID

    async def test_the_hop_is_explained_on_the_same_request(self) -> None:
        """Integrity and explainability are two views of one event.

        A client that notices the provider changed mid tool-loop should be
        able to read *why* off the same response.
        """
        config = _config()
        primary = RecordingAdapter(
            config.providers[0],
            fail_with=AdapterError("503", provider="primary", status_code=503),
        )
        backup = RecordingAdapter(config.providers[1])
        engine = _engine(config, {"primary": primary, "backup": backup})

        await engine.generate_anthropic(_mid_tool_turn_request())

        trace = current_fallback_trace()
        assert trace is not None
        assert trace.header_values() == {
            HEADER_FALLBACK_FROM: "primary",
            HEADER_FALLBACK_TO: "backup",
            HEADER_FALLBACK_REASON: REASON_UPSTREAM_5XX,
            HEADER_FALLBACK_CHAIN: "primary>backup",
        }


class TestStreamingToolIntegrity:
    async def test_downgraded_stream_keeps_the_tool_block_well_formed(self) -> None:
        """Streaming + tools takes the v0.3-D downgrade; the hop must not
        leave a ``content_block_start`` without its matching stop.
        """
        config = _config()
        primary = RecordingAdapter(
            config.providers[0],
            fail_with=AdapterError("500", provider="primary", status_code=500),
        )
        backup = RecordingAdapter(config.providers[1])
        engine = _engine(config, {"primary": primary, "backup": backup})

        events = [
            ev
            async for ev in engine.stream_anthropic(_mid_tool_turn_request(stream=True))
        ]
        names = [ev.type for ev in events]

        assert names[0] == "message_start"
        assert names[-1] == "message_stop"
        assert names.count("content_block_start") == names.count("content_block_stop")

        # The inbound conversation reached the surviving provider intact.
        _assert_tool_pairing_intact(backup.seen[0])

        # And the outbound tool call is a real block with the right id.
        started = [
            ev.data.get("content_block", {})
            for ev in events
            if ev.type == "content_block_start"
        ]
        tool_blocks = [b for b in started if b.get("type") == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["id"] == _NEXT_CALL_ID
        assert tool_blocks[0]["name"] == _TOOL_NAME

        trace = current_fallback_trace()
        assert trace is not None
        assert trace.reasons == [REASON_UPSTREAM_5XX]
        assert trace.hops[0].stream is True
