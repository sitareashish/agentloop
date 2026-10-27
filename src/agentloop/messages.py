"""Provider-agnostic conversation primitives.

These types are the contract between the agent loop and any LLM provider. They
map cleanly onto the "content block" shape used by modern tool-calling APIs
(an assistant turn can mix free text with one or more tool calls; tool results
are fed back as a separate turn) without binding the loop to any one vendor.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A request from the model to invoke a tool."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """The outcome of executing a :class:`ToolCall`, fed back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One turn in the conversation.

    * ``role="user"``      — the task or environment input.
    * ``role="assistant"`` — model output: ``text`` and/or ``tool_calls``.
    * ``role="tool"``      — results of tool calls, carried in ``tool_results``.
    """

    role: str
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class LLMResponse:
    """A normalized single completion from a provider."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)
