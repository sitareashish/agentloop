"""Anthropic Claude backend.

Translates agentloop's neutral message/tool types to the Claude Messages API
and normalizes the response back. The ``anthropic`` SDK is imported lazily, so
installing it is only required when you actually use this provider.
"""
from __future__ import annotations

from typing import Any

from ..errors import ProviderError
from ..messages import LLMResponse, Message, ToolCall
from ..tools import ToolSpec
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ProviderError(
                    "The `anthropic` package is required for AnthropicProvider. "
                    "Install it with `pip install agentloop[anthropic]`."
                ) from exc
            self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    # ── translation: agentloop -> Anthropic ────────────────────────────────────
    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_call_id,
                        "content": r.content,
                        "is_error": r.is_error,
                    }
                    for r in m.tool_results
                ]
                out.append({"role": "user", "content": content})
            elif m.role == "assistant":
                blocks: list[dict] = []
                if m.text:
                    blocks.append({"type": "text", "text": m.text})
                for c in m.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    )
                out.append({"role": "assistant", "content": blocks})
            else:  # user
                out.append({"role": "user", "content": m.text or ""})
        return out

    # ── the one required method ─────────────────────────────────────────────────
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [t.to_schema() for t in tools]

        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - classified and re-raised below
            raise ProviderError(str(exc), retryable=_is_retryable(exc)) from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            stop_reason=getattr(resp, "stop_reason", "") or "",
        )


def _is_retryable(exc: Exception) -> bool:
    """Transient failures (rate limits, overload, connection) are worth retrying."""
    name = type(exc).__name__
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}:
        return True
    status = getattr(exc, "status_code", None)
    return status in {408, 409, 429, 500, 502, 503, 529}
