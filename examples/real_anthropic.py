"""A real agent backed by Claude.

Requires the Anthropic SDK and an API key:

    pip install -e ".[anthropic]"
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/real_anthropic.py
"""
from __future__ import annotations

from agentloop import Agent, tool
from agentloop.providers.anthropic import AnthropicProvider

# A tiny in-memory "database" the agent can query through a tool.
_PRICES = {"BTC": 64_000, "ETH": 3_100, "SOL": 150}


@tool
def get_price(symbol: str) -> str:
    """Return the current price of a crypto asset in USD.

    Args:
        symbol: Ticker symbol, e.g. BTC, ETH, SOL.
    """
    symbol = symbol.upper()
    if symbol not in _PRICES:
        return f"Unknown symbol {symbol!r}. Known: {list(_PRICES)}"
    return f"{symbol} = ${_PRICES[symbol]:,}"


@tool
def convert(amount_usd: float, rate: float) -> float:
    """Convert a USD amount into units of an asset given its USD price.

    Args:
        amount_usd: Amount in US dollars.
        rate: Price of one unit of the asset in USD.
    """
    return round(amount_usd / rate, 6)


def main() -> None:
    agent = Agent(
        AnthropicProvider(model="claude-sonnet-4-20250514"),
        tools=[get_price, convert],
        system="You are a precise assistant. Use the tools; show your final answer plainly.",
        max_steps=6,
        token_budget=50_000,
    )
    result = agent.run("How much ETH can I buy with $10,000 right now?")

    print(result.trajectory.pretty())
    print("-" * 40)
    print("answer:", result.output)
    print("status:", result.status, "| tokens:", result.total_tokens)


if __name__ == "__main__":
    main()
