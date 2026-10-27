"""The provider interface every LLM backend implements."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..messages import LLMResponse, Message
from ..tools import ToolSpec


class LLMProvider(ABC):
    """Adapter from agentloop's neutral types to a concrete LLM API.

    A provider does exactly one thing: take the running conversation plus the
    available tools and return a single normalized :class:`LLMResponse`. All
    looping, retrying and tool dispatch lives in the agent, not here — which is
    why a fully offline :class:`~agentloop.providers.mock.MockProvider` can
    stand in for a real API in tests.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
    ) -> LLMResponse:
        """Produce one completion for the given conversation and tools."""
        raise NotImplementedError
