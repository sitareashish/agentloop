from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentloop import tool
from agentloop.errors import ProviderError
from agentloop.messages import Message, ToolCall, ToolResult
from agentloop.providers.anthropic import AnthropicProvider, _is_retryable
from agentloop.providers.mock import MockProvider, final


# ── MockProvider ────────────────────────────────────────────────────────────────
def test_mock_fifo_order_and_call_recording():
    provider = MockProvider([final("a"), final("b")])
    r1 = provider.complete([Message("user", text="x")], [])
    r2 = provider.complete([Message("user", text="y")], [])
    assert r1.text == "a" and r2.text == "b"
    assert len(provider.calls) == 2


def test_mock_exhausted_script_raises():
    provider = MockProvider([final("a")])
    provider.complete([], [])
    with pytest.raises(ProviderError):
        provider.complete([], [])


def test_mock_requires_exactly_one_mode():
    with pytest.raises(ValueError):
        MockProvider()
    with pytest.raises(ValueError):
        MockProvider([final("a")], handler=lambda m, t: final("b"))


def test_mock_fills_default_input_tokens():
    provider = MockProvider([final("a")], input_tokens_per_call=7)
    r = provider.complete([], [])
    assert r.input_tokens == 7


# ── AnthropicProvider (with an injected fake client) ──────────────────────────────
class FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def _fake_response():
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="let me compute"),
            SimpleNamespace(type="tool_use", id="tu_1", name="add", input={"a": 2, "b": 3}),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        stop_reason="tool_use",
    )


def test_anthropic_parses_response():
    client = FakeClient(_fake_response())
    provider = AnthropicProvider(client=client)

    @tool
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    resp = provider.complete([Message("user", text="2+3")], [add.tool_spec], system="be terse")
    assert resp.text == "let me compute"
    assert resp.tool_calls == [ToolCall(id="tu_1", name="add", arguments={"a": 2, "b": 3})]
    assert resp.input_tokens == 11 and resp.output_tokens == 7
    assert resp.stop_reason == "tool_use"


def test_anthropic_message_translation():
    client = FakeClient(_fake_response())
    provider = AnthropicProvider(client=client)

    @tool
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    call = ToolCall("tu_1", "add", {"a": 2, "b": 3})
    conversation = [
        Message("user", text="2+3"),
        Message("assistant", text="calling", tool_calls=[call]),
        Message("tool", tool_results=[ToolResult("tu_1", "5", is_error=False)]),
    ]
    provider.complete(conversation, [add.tool_spec], system="sys")

    kwargs = client.messages.last_kwargs
    assert kwargs["system"] == "sys"
    assert kwargs["tools"][0]["name"] == "add"

    msgs = kwargs["messages"]
    assert msgs[0] == {"role": "user", "content": "2+3"}
    # Assistant turn carries a text block + a tool_use block.
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0] == {"type": "text", "text": "calling"}
    assert msgs[1]["content"][1] == {
        "type": "tool_use",
        "id": "tu_1",
        "name": "add",
        "input": {"a": 2, "b": 3},
    }
    # Tool results are folded into a user turn with tool_result blocks.
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "5",
        "is_error": False,
    }


def test_anthropic_wraps_errors_as_provider_error():
    client = FakeClient(RuntimeError("boom"))
    provider = AnthropicProvider(client=client)
    with pytest.raises(ProviderError):
        provider.complete([Message("user", text="hi")], [])


def test_retryable_classification():
    assert _is_retryable(type("RateLimitError", (Exception,), {})()) is True
    assert _is_retryable(type("InternalServerError", (Exception,), {})()) is True

    err = Exception("x")
    err.status_code = 503  # type: ignore[attr-defined]
    assert _is_retryable(err) is True

    bad = Exception("bad request")
    bad.status_code = 400  # type: ignore[attr-defined]
    assert _is_retryable(bad) is False
