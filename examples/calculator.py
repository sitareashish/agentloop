"""A fully offline demo: a calculator agent driven by MockProvider.

Run it with no API key:

    python examples/calculator.py

The MockProvider's `handler` inspects the latest tool result and decides the
next move, so the agent genuinely chains tool calls — add, then multiply — and
only then produces a final answer.
"""
from __future__ import annotations

from agentloop import Agent, tool
from agentloop.messages import LLMResponse, Message
from agentloop.providers.mock import MockProvider, call_tool, final


@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    "Multiply two integers."
    return a * b


def brain(messages: list[Message], tools) -> LLMResponse:
    """Stand-in 'model': compute (2 + 3) * 4 by chaining two tools."""
    last_result = None
    for m in messages:
        for r in m.tool_results:
            last_result = r.content

    if last_result is None:
        return call_tool("add", {"a": 2, "b": 3})      # step 1
    if last_result == "5":
        return call_tool("multiply", {"a": 5, "b": 4})  # step 2
    return final(f"(2 + 3) * 4 = {last_result}")        # step 3


def main() -> None:
    agent = Agent(MockProvider(handler=brain), tools=[add, multiply], max_steps=5)
    result = agent.run("What is (2 + 3) * 4?")

    print(result.trajectory.pretty())
    print("-" * 40)
    print("status :", result.status)
    print("answer :", result.output)
    print("tokens :", result.total_tokens)
    print("tools  :", result.trajectory.tool_names)


if __name__ == "__main__":
    main()
