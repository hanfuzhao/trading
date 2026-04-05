





















import json
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from openai import OpenAI

from core.config import OPENAI_API_KEY
from agent.tools import TOOL_DEFINITIONS, ToolRegistry

ET = ZoneInfo("America/New_York")

SYSTEM_PROMPT = """You are an AI stock research analyst and trading assistant managing a US equity portfolio on Alpaca Paper Trading.

## Core Capabilities
1. **Deep Stock Research** — Multi-dimensional analysis (Technical, Sentiment, Macro, Fundamental, Institutional scoring) with AI price predictions from 1 day to 1 month.
2. **Market Scanning** — Scan the entire US market with custom filters, identify opportunities with tiered recommendations (Strong Buy / Buy / Watch / Avoid).
3. **Portfolio Management** — Analyze portfolio health, per-position recommendations (Hold/Sell/Add), and sector rotation insights.
4. **Trade Execution** — Execute trades through Alpaca with full risk management.

## Research Tools (use these for analysis)
- **get_stock_analysis**: Comprehensive multi-dimensional analysis with predictions
- **predict_price**: AI price predictions for 1d through 1 month horizons
- **scan_research**: Flexible market scanning with custom filters
- **get_portfolio_analysis**: Portfolio health check with per-stock recommendations
- **get_sector_analysis**: Sector rotation and relative strength

## Trading Tools
- **get_stock_data**: Quick price and indicator lookup
- **search_news**: News with AI sentiment analysis
- **scan_overnight / scan_intraday**: Strategy-specific scans
- **get_portfolio**: Account status, positions, PDT slots
- **get_macro_environment**: Market regime and macro indicators
- **execute_trade**: Submit orders with risk validation

## Trading Strategies
1. **Overnight Mean Reversion** (does NOT consume PDT): Buy oversold stocks near close, sell next morning.
2. **Intraday Afternoon Reversal** (consumes PDT): Buy significantly down stocks, exit same day.

## Hard Constraints
- PDT Rule: max 3 day trades per rolling 5 trading days.
- Long only — no short selling.
- Daily loss limit: 2% of account value.
- Max 5 concurrent positions.

## How to Respond
- For research questions, use research tools (get_stock_analysis, predict_price, scan_research).
- For quick lookups, use basic tools (get_stock_data, get_portfolio).
- Always explain your reasoning with data.
- When asked to trade, verify risk limits first.
- Answer in the same language as the user's question.
- Be concise but thorough."""

MAX_ITERATIONS = 10


@dataclass
class ToolCall:

    tool: str
    arguments: Dict[str, Any]
    result: Any
    latency_ms: int


@dataclass
class AgentResult:

    response: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    model: str = ""


class TradingAgent:







    def __init__(self, tool_registry: ToolRegistry, model: str = "gpt-4.1-mini"):
        self.llm = OpenAI(api_key=OPENAI_API_KEY)
        self.tools = tool_registry
        self.model = model
        self._conversation: List[Dict] = []

    def run(self, user_message: str, *, keep_history: bool = False) -> AgentResult:










        start = _time.time()

        if not keep_history:
            self._conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        self._conversation.append({"role": "user", "content": user_message})

        all_tool_calls: List[ToolCall] = []
        total_prompt = 0
        total_completion = 0

        for iteration in range(1, MAX_ITERATIONS + 1):

            response = self.llm.chat.completions.create(
                model=self.model,
                messages=self._conversation,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_completion_tokens=2048,
            )

            usage = response.usage
            if usage:
                total_prompt += usage.prompt_tokens
                total_completion += usage.completion_tokens

            choice = response.choices[0]
            assistant_msg = choice.message


            self._conversation.append(_msg_to_dict(assistant_msg))


            if not assistant_msg.tool_calls:
                elapsed = int((_time.time() - start) * 1000)
                return AgentResult(
                    response=assistant_msg.content or "",
                    tool_calls=all_tool_calls,
                    iterations=iteration,
                    prompt_tokens=total_prompt,
                    completion_tokens=total_completion,
                    total_tokens=total_prompt + total_completion,
                    latency_ms=elapsed,
                    model=self.model,
                )


            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                t0 = _time.time()
                result = self.tools.call(tool_name, tool_args)
                tool_ms = int((_time.time() - t0) * 1000)

                record = ToolCall(
                    tool=tool_name,
                    arguments=tool_args,
                    result=result,
                    latency_ms=tool_ms,
                )
                all_tool_calls.append(record)


                self._conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })


        elapsed = int((_time.time() - start) * 1000)
        return AgentResult(
            response="[Agent reached maximum iterations without a final answer]",
            tool_calls=all_tool_calls,
            iterations=MAX_ITERATIONS,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            latency_ms=elapsed,
            model=self.model,
        )


def _msg_to_dict(msg) -> Dict:

    d: Dict[str, Any] = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d
