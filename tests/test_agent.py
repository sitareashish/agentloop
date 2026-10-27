from __future__ import annotations

import pytest

from agentloop import Agent, tool
from agentloop.errors import ProviderError
from agentloop.messages import LLMResponse, Message
from agentloop.providers.base import LLMProvider
from agentloop.providers.mock import MockProvider, call_tool, final


@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    "Multiply two integers."
    return a * b


def test_single_tool_then_final():
    provider = MockProvider([call_tool("add", {"a": 2, "b": 3}), final("The answer is 5.")])
    agent = Agent(provider, tools=[add])
    result = agent.run("What is 2 + 3?")

    assert result.status == "completed"
    assert result.output == "The answer is 5."
    assert result.steps_taken == 2
    assert result.trajectory.tool_names == ["add"]
    # The tool actually ran and its result is recorded.
    assert result.trajectory.steps[0].tool_results[0].content == "5"


def test_multi_tool_chain():
    provider = MockProvider(
        [
            call_tool("add", {"a": 2, "b": 3}),
            call_tool("multiply", {"a": 5, "b": 4}),
            final("20"),
        ]
    )
    result = Agent(provider, tools=[add, multiply]).run("(2+3)*4")
    assert result.output == "20"
    assert result.trajectory.tool_names == ["add", "multiply"]


def test_max_steps_guardrail():
    # Always asks for another tool call → never finishes on its own.
    provider = MockProvider(handler=lambda msgs, tools: call_tool("add", {"a": 1, "b": 1}))
    result = Agent(provider, tools=[add], max_steps=3).run("loop forever")
    assert result.status == "max_steps"
    assert result.steps_taken == 3
    assert result.output is None


def test_token_budget_guardrail():
    provider = MockProvider(
        handler=lambda msgs, tools: LLMResponse(
            tool_calls=call_tool("add", {"a": 1, "b": 1}).tool_calls,
            input_tokens=40,
            output_tokens=10,
            stop_reason="tool_use",
        )
    )
    result = Agent(provider, tools=[add], max_steps=100, token_budget=120).run("spend")
    assert result.status == "budget_exceeded"
    # 50 tok/step → stops once cumulative >= 120, i.e. after step 3.
    assert result.steps_taken == 3
    assert result.total_tokens >= 120


def test_tool_error_is_fed_back_and_loop_recovers():
    seen_results: list[str] = []

    def handler(messages: list[Message], tools) -> LLMResponse:
        # Record any tool results the agent fed back to us.
        for m in messages:
            for r in m.tool_results:
                seen_results.append(r.content)
        if len([m for m in messages if m.role == "assistant"]) == 0:
            return call_tool("boom", {})
        return final("recovered")

    @tool
    def boom() -> str:
        "Always fails."
        raise RuntimeError("nope")

    result = Agent(MockProvider(handler=handler), tools=[boom]).run("try it")
    assert result.status == "completed"
    assert result.output == "recovered"
    assert any("nope" in s for s in seen_results)
    assert result.trajectory.steps[0].tool_results[0].is_error is True


def test_retries_on_retryable_provider_error_then_succeeds():
    class Flaky(LLMProvider):
        def __init__(self) -> None:
            self.attempts = 0

        def complete(self, messages, tools, system=None):
            self.attempts += 1
            if self.attempts < 3:
                raise ProviderError("overloaded", retryable=True)
            return final("ok")

    provider = Flaky()
    sleeps: list[float] = []
    agent = Agent(provider, max_retries=3, _sleep=sleeps.append)
    result = agent.run("hi")

    assert result.output == "ok"
    assert provider.attempts == 3
    assert sleeps == [0.5, 1.0]  # exponential backoff between the two retries


def test_non_retryable_error_propagates():
    class Broken(LLMProvider):
        def complete(self, messages, tools, system=None):
            raise ProviderError("bad request", retryable=False)

    with pytest.raises(ProviderError):
        Agent(Broken(), _sleep=lambda _: None).run("hi")


def test_retries_exhausted_propagates():
    class AlwaysFlaky(LLMProvider):
        def complete(self, messages, tools, system=None):
            raise ProviderError("still overloaded", retryable=True)

    with pytest.raises(ProviderError):
        Agent(AlwaysFlaky(), max_retries=2, _sleep=lambda _: None).run("hi")


def test_on_step_callback_fires_per_step():
    provider = MockProvider([call_tool("add", {"a": 1, "b": 1}), final("2")])
    indices: list[int] = []
    Agent(provider, tools=[add], on_step=lambda s: indices.append(s.index)).run("1+1")
    assert indices == [1, 2]


def test_run_accepts_prewired_messages():
    provider = MockProvider([final("hello back")])
    result = Agent(provider).run([Message(role="user", text="hi")])
    assert result.output == "hello back"


def test_invalid_max_steps():
    with pytest.raises(ValueError):
        Agent(MockProvider([final("x")]), max_steps=0)


def test_trajectory_pretty_runs():
    provider = MockProvider([call_tool("add", {"a": 1, "b": 2}), final("3")])
    result = Agent(provider, tools=[add]).run("1+2")
    text = result.trajectory.pretty()
    assert "add" in text and "step 1" in text
