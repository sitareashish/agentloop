"""A fully offline provider for tests, examples, and deterministic demos.

Script the exact turns an agent should see — a fixed sequence of responses, or
a callable that reacts to the conversation so far — with no network and no API
key. This is what lets agentloop's whole loop be unit-tested.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import count

from ..errors import ProviderError
from ..messages import LLMResponse, Message, ToolCall
from ..tools import ToolSpec
from .base import LLMProvider

Handler = Callable[[list[Message], list[ToolSpec]], LLMResponse]

_ids = count(1)


def call_tool(name: str, arguments: dict, *, output_tokens: int = 5) -> LLMResponse:
    """Build a response in which the model asks to call one tool."""
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)],
        output_tokens=output_tokens,
        stop_reason="tool_use",
    )


def final(text: str, *, output_tokens: int = 5) -> LLMResponse:
    """Build a response in which the model returns a final answer."""
    return LLMResponse(text=text, output_tokens=output_tokens, stop_reason="end_turn")


class MockProvider(LLMProvider):
    """Return scripted completions.

    Pass either ``responses`` (consumed in order) or ``handler`` (called with
    the live conversation). Every call is recorded on ``self.calls`` so tests
    can assert on what the agent sent.
    """

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        *,
        handler: Handler | None = None,
        input_tokens_per_call: int = 10,
    ) -> None:
        if (responses is None) == (handler is None):
            raise ValueError("Provide exactly one of `responses` or `handler`.")
        self._responses = list(responses) if responses is not None else None
        self._handler = handler
        self._input_tokens = input_tokens_per_call
        self.calls: list[tuple[list[Message], list[ToolSpec]]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
    ) -> LLMResponse:
        self.calls.append((messages, tools))

        if self._handler is not None:
            response = self._handler(messages, tools)
        else:
            assert self._responses is not None
            if not self._responses:
                raise ProviderError(
                    "MockProvider script exhausted — the agent asked for more turns than scripted."
                )
            response = self._responses.pop(0)

        if not response.input_tokens:
            response.input_tokens = self._input_tokens
        return response
