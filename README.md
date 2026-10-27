# 🔁 agentloop

A minimal, **dependency-free** agent runtime in Python. It implements the parts of an LLM agent that actually matter in production — a typed tool layer, a plan→act→observe loop, retries, token budgeting, and a fully inspectable trajectory — in ~500 lines you can read in one sitting, with **zero third-party dependencies in the core**.

Built to understand (and show that you understand) how agents work *underneath* a framework like LangChain, not on top of one.

<p align="left">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/core%20deps-0-success?style=flat-square" />
  <img src="https://img.shields.io/badge/tests-offline%20(mock%20provider)-success?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-black?style=flat-square" />
</p>

---

## Why

Most agent demos are a single prompt; most agent *frameworks* hide the loop behind abstractions. `agentloop` is the small, honest middle: the real control flow of a tool-using agent, written out plainly so every decision — when to call a tool, how to feed an error back, when to stop — is visible and testable.

- **Tools are just typed functions.** A `@tool` decorator reads your type hints and docstring and generates the JSON schema. No schema files, no base classes.
- **Providers are swappable.** The loop talks to an abstract `LLMProvider`. A real `AnthropicProvider` ships in the box; a `MockProvider` lets the *entire* loop be unit-tested offline, with no API key.
- **Guardrails are first-class.** Step limits, a per-run token budget, and exponential-backoff retries on transient provider errors are built into the loop, not bolted on.
- **Runs are observable.** Every run returns a `Trajectory` — each model turn, tool call, and result — ready to print, assert on, or feed to an eval harness.

## Install

```bash
pip install -e .                 # core runtime, zero dependencies
pip install -e ".[anthropic]"    # add the Anthropic provider
pip install -e ".[dev]"          # tests + linters
```

## Quickstart (no API key needed)

```python
from agentloop import Agent, tool
from agentloop.providers.mock import MockProvider, call_tool, final

@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

# Script what the "model" does, so the demo is deterministic and offline.
provider = MockProvider([
    call_tool("add", {"a": 2, "b": 3}),
    final("The answer is 5."),
])

result = Agent(provider, tools=[add]).run("What is 2 + 3?")

print(result.output)              # The answer is 5.
print(result.status)              # completed
print(result.trajectory.pretty()) # full step-by-step transcript
```

## With a real model

```python
from agentloop import Agent, tool
from agentloop.providers.anthropic import AnthropicProvider

@tool
def get_weather(city: str, unit: str = "c") -> str:
    """Look up the current weather.

    Args:
        city: Name of the city.
        unit: Temperature unit, "c" or "f".
    """
    return f"18°{unit.upper()} and clear in {city}"

agent = Agent(
    AnthropicProvider(model="claude-sonnet-4-20250514"),  # reads ANTHROPIC_API_KEY
    tools=[get_weather],
    system="You are a concise assistant. Use tools when they help.",
    max_steps=6,
    token_budget=50_000,
)

result = agent.run("What's the weather in Kharagpur in fahrenheit?")
print(result.output)
```

## How the loop works

```
run(task)
  └─► for step in 1..max_steps:
        response = provider.complete(messages, tools)      # ← retried on transient errors
        if response has tool calls:
            results = [registry.execute(call) for call in response.tool_calls]
            append assistant turn + tool results to messages
            record Step(thought, tool_calls, results, tokens)
            if cumulative tokens ≥ token_budget: stop ("budget_exceeded")
        else:
            return response.text                            # stop ("completed")
      else: stop ("max_steps")
```

Tool failures are **not** raised — they're captured as an error result and fed back to the model, so the agent can see what went wrong and recover on the next step. That self-correction behavior is covered by tests.

## Design

| Module | Responsibility |
|---|---|
| `agentloop.tools` | `@tool` decorator, type-hint → JSON-schema generation, `ToolRegistry` dispatch |
| `agentloop.agent` | the plan→act→observe loop, retries, token budget, step limit |
| `agentloop.messages` | provider-neutral `Message` / `ToolCall` / `ToolResult` / `LLMResponse` |
| `agentloop.providers.base` | the one-method `LLMProvider` interface |
| `agentloop.providers.mock` | scriptable offline provider (powers the test suite) |
| `agentloop.providers.anthropic` | Claude Messages API adapter (lazy import) |
| `agentloop.trace` | `Step` / `Trajectory` / `RunResult` — the observable record of a run |

## Supported tool parameter types

`str` · `int` · `float` · `bool` · `list[T]` · `dict` · `Literal[...]` (→ enum) · `Optional[T]`.
Parameters without a default are marked `required`; Google-style docstring `Args:` become per-parameter descriptions.

## Tests

The whole agent loop is tested **offline** through `MockProvider` — no network, no API key — so CI is fast and deterministic.

```bash
pip install -e ".[dev]"
pytest
```

CI runs the suite on Python 3.10–3.12 (see `.github/workflows/ci.yml`).

## Roadmap

- Streaming token callbacks
- Parallel tool execution within a single turn
- Additional providers (OpenAI, local models)
- A companion eval harness that scores trajectories

## License

MIT — see [LICENSE](LICENSE).
