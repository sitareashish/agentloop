"""The agent loop.

``Agent.run`` drives the classic plan → act → observe cycle: ask the model what
to do, execute any tools it requests, feed the results back, and repeat until it
produces a final answer or hits a guardrail (step limit or token budget). Every
turn is recorded in a :class:`~agentloop.trace.Trajectory`.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from .errors import ProviderError
from .messages import Message
from .providers.base import LLMProvider
from .tools import ToolRegistry
from .trace import RunResult, Step, Trajectory


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry | Sequence[Callable[..., object]] | None = None,
        *,
        system: str | None = None,
        max_steps: int = 10,
        token_budget: int | None = None,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        on_step: Callable[[Step], None] | None = None,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.provider = provider
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(list(tools or []))
        self.system = system
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.on_step = on_step
        self._sleep = _sleep

    def run(self, task: str | list[Message]) -> RunResult:
        messages: list[Message] = (
            [Message(role="user", text=task)] if isinstance(task, str) else list(task)
        )
        traj = Trajectory()
        in_tok = out_tok = 0
        output: str | None = None
        status = "max_steps"

        for i in range(1, self.max_steps + 1):
            resp = self._complete_with_retry(messages)
            in_tok += resp.input_tokens
            out_tok += resp.output_tokens

            results = [self.tools.execute(c) for c in resp.tool_calls]
            step = Step(
                index=i,
                thought=resp.text,
                tool_calls=resp.tool_calls,
                tool_results=results,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )
            traj.add(step)
            if self.on_step is not None:
                self.on_step(step)

            # Record the model's turn (text + any tool calls) in the transcript.
            messages.append(Message(role="assistant", text=resp.text, tool_calls=resp.tool_calls))

            if not resp.tool_calls:
                # No tools requested → the model is done.
                output = resp.text
                status = "completed"
                break

            # Feed tool results back for the next turn.
            messages.append(Message(role="tool", tool_results=results))

            # Enforce the budget before spending another model call.
            if self.token_budget is not None and (in_tok + out_tok) >= self.token_budget:
                status = "budget_exceeded"
                break

        return RunResult(
            output=output,
            status=status,
            trajectory=traj,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    def _complete_with_retry(self, messages: list[Message]):
        attempt = 0
        while True:
            try:
                return self.provider.complete(messages, self.tools.specs(), self.system)
            except ProviderError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = self.backoff_base * (2**attempt)
                self._sleep(delay)
                attempt += 1
