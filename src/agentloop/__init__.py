"""agentloop — a minimal, dependency-free agent runtime.

    from agentloop import Agent, tool
    from agentloop.providers.mock import MockProvider, call_tool, final

    @tool
    def add(a: int, b: int) -> int:
        "Add two integers."
        return a + b

    provider = MockProvider([call_tool("add", {"a": 2, "b": 3}), final("The sum is 5.")])
    result = Agent(provider, tools=[add]).run("What is 2 + 3?")
    print(result.output)            # "The sum is 5."
    print(result.trajectory.pretty())
"""
from __future__ import annotations

from .agent import Agent
from .errors import AgentLoopError, ProviderError, SchemaError, ToolError
from .messages import LLMResponse, Message, ToolCall, ToolResult
from .providers.base import LLMProvider
from .tools import ToolRegistry, ToolSpec, build_schema, tool
from .trace import RunResult, Step, Trajectory

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "tool",
    "ToolRegistry",
    "ToolSpec",
    "build_schema",
    "LLMProvider",
    "Message",
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    "Step",
    "Trajectory",
    "RunResult",
    "AgentLoopError",
    "ProviderError",
    "ToolError",
    "SchemaError",
]
