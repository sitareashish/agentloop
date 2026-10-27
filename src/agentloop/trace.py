"""Structured record of what an agent did, step by step.

Every run produces a :class:`Trajectory` you can inspect, assert on, or feed to
an evaluation harness — the difference between a black box and a system you can
debug and measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .messages import ToolCall, ToolResult


@dataclass
class Step:
    """One iteration of the loop: a model turn and any tool calls it triggered."""

    index: int
    thought: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Trajectory:
    """The ordered sequence of steps taken during a run."""

    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> None:
        self.steps.append(step)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [c for s in self.steps for c in s.tool_calls]

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]

    def pretty(self) -> str:
        """A compact, human-readable transcript of the run."""
        out: list[str] = []
        for s in self.steps:
            out.append(f"── step {s.index} ({s.input_tokens}+{s.output_tokens} tok) ──")
            if s.thought:
                out.append(f"  think: {s.thought}")
            for call, res in zip(s.tool_calls, s.tool_results, strict=False):
                flag = "✗" if res.is_error else "✓"
                out.append(f"  call : {call.name}({call.arguments})")
                out.append(f"  {flag} ret: {_truncate(res.content)}")
            # Tool calls without a paired result (shouldn't happen, but be safe).
            for call in s.tool_calls[len(s.tool_results):]:
                out.append(f"  call : {call.name}({call.arguments})")
        return "\n".join(out)


@dataclass
class RunResult:
    """The outcome of :meth:`agentloop.Agent.run`.

    ``status`` is one of ``"completed"``, ``"max_steps"``, or ``"budget_exceeded"``.
    """

    output: str | None
    status: str
    trajectory: Trajectory
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def steps_taken(self) -> int:
        return len(self.trajectory.steps)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


def _truncate(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
