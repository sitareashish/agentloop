"""Exception hierarchy for agentloop."""
from __future__ import annotations


class AgentLoopError(Exception):
    """Base class for every error raised by agentloop."""


class ProviderError(AgentLoopError):
    """Raised when an LLM provider call fails.

    ``retryable`` lets the agent's retry policy decide whether to back off and
    try again (transient network/rate-limit errors) or fail fast (a bad request).
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolError(AgentLoopError):
    """Raised for problems registering or dispatching a tool."""


class SchemaError(AgentLoopError):
    """Raised when a tool's signature cannot be turned into a JSON schema."""
