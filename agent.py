"""
Trading Agent — Custom ReAct loop with OpenAI function-calling.
No LangChain / CrewAI / AutoGen.  The planning and tool-dispatch
loop is written from scratch.

Architecture
────────────
  User message
       │
       ▼
  ┌─────────┐   tool_call?   ┌──────────┐   result   ┌─────────┐
  │   LLM   │ ───────────▶  │  Tool    │ ────────▶ │   LLM   │ ─▶ ...
  │ (plan)  │               │ Registry │           │ (reason)│
  └─────────┘               └──────────┘           └─────────┘
       │ no tool_call                                    │
       ▼                                                 ▼
  Final answer                                     Final answer

Each iteration the LLM either:
  1. Calls one or more tools  →  results are appended and we loop
  2. Returns a text answer   →  we stop and return it to the user
"""
import json
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from openai import OpenAI

from config import OPENAI_API_KEY
from tools import TOOL_DEFINITIONS, ToolRegistry

ET = ZoneInfo("America/New_York")

SYSTEM_PROMPT = """You are a professional stock trading AI assistant managing a US equity portfolio on Alpaca Paper Trading.

## Your Strategies
1. **Overnight Mean Reversion** (primary, does NOT consume PDT):
   Buy oversold stocks (RSI(2)<15 AND IBS<0.25) near market close (15:30-15:45 ET),
   sell next morning (09:45-10:15 ET).  No mechanical stop-loss — risk is controlled
   by position sizing (max ~1% of account per position).
2. **Intraday Afternoon Reversal** (secondary, consumes 1 PDT slot):
   Buy stocks that are significantly down during the day (14:00-14:45 ET),
   exit by 15:40 ET.  Uses ATR-based bracket stop-loss.

## Hard Constraints (you CANNOT override these)
- PDT Rule: max 3 day trades per rolling 5 trading days. Overnight trades are exempt.
- Long only — no short selling.
- Daily loss limit: 2% of account value → stop trading for the day.
- Max 5 concurrent positions (overnight + intraday combined).
- Max 2 same-sector overnight positions.

## How to Respond
- Use the available tools to gather data before making recommendations.
- Always explain your reasoning.
- When asked to execute a trade, verify risk limits first using get_portfolio.
- Answer in the same language as the user's question.
- Be concise but thorough."""

MAX_ITERATIONS = 10


@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    tool: str
    arguments: Dict[str, Any]
    result: Any
    latency_ms: int


@dataclass
class AgentResult:
    """Everything the agent produces for one user turn."""
    response: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    model: str = ""


class TradingAgent:
    """
    Custom ReAct agent loop.

    ``run(message)`` sends the message through a plan→act→observe loop
    until the LLM produces a final text answer (or hits MAX_ITERATIONS).
    """

    def __init__(self, tool_registry: ToolRegistry, model: str = "gpt-4.1-mini"):
        self.llm = OpenAI(api_key=OPENAI_API_KEY)
        self.tools = tool_registry
        self.model = model
        self._conversation: List[Dict] = []

    def run(self, user_message: str, *, keep_history: bool = False) -> AgentResult:
        """
        Execute the agent loop for a single user turn.

        Parameters
        ----------
        user_message : str
            The user's natural-language request.
        keep_history : bool
            If True, previous conversation is preserved for multi-turn.
        """
        start = _time.time()

        if not keep_history:
            self._conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        self._conversation.append({"role": "user", "content": user_message})

        all_tool_calls: List[ToolCall] = []
        total_prompt = 0
        total_completion = 0

        for iteration in range(1, MAX_ITERATIONS + 1):
            # ── 1. Call LLM ──────────────────────────────────────
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

            # Append the raw assistant message (may contain tool_calls)
            self._conversation.append(_msg_to_dict(assistant_msg))

            # ── 2. Check: final answer or tool calls? ────────────
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

            # ── 3. Execute each tool call ────────────────────────
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

                # Feed result back to the LLM
                self._conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

        # Exhausted iterations
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
    """Convert an OpenAI ChatCompletionMessage to a serialisable dict."""
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
