#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

"""
Trading Assistant Chat Mode
Connects to OpenAI, auto-injects current positions, scan results, market state as context
Ask any question and it responds based on your live data

Usage:
    python chat.py                  # Default uses o3
    python chat.py --model gpt-5.4  # Use gpt-5.4 (faster and cheaper)
    python chat.py --model gpt-4.1-mini  # Cheapest, sufficient for simple questions
"""
import argparse
import json
import os
import glob
from datetime import datetime, date
from typing import Dict, List, Optional
from openai import OpenAI

from core.config import (
    OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_API_SECRET,
    MODEL_RANK, MODEL_DEEP, MODEL_FAST, LOG_DIR,
)
from trading.executor import OrderExecutor
from trading.pdt_tracker import PDTTracker
from trading.risk_manager import RiskManager


class TradingChat:
    def __init__(self, model: str = MODEL_RANK):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.executor = OrderExecutor()
        self.pdt = PDTTracker()
        self.risk = RiskManager()
        self.conversation: List[Dict] = []
        self.system_prompt = ""

    # ================================================================
    # Build Context
    # ================================================================

    def _build_context(self) -> str:
        """Collect all live data and build system prompt"""
        sections = []

        # Account info
        try:
            account = self.executor.get_account()
            sections.append(f"""== Account Status ==
Portfolio: ${account['portfolio_value']:,.2f}
Cash: ${account['cash']:,.2f}
Buying Power: ${account['buying_power']:,.2f}
PDT Flag: {'Yes' if account['pattern_day_trader'] else 'No'}""")
        except Exception as e:
            sections.append(f"== Account Status ==\nFailed to retrieve: {e}")

        # PDT status
        sections.append(f"""== PDT Slots ==
{self.pdt.status()}
Trades used in rolling window: {3 - self.pdt.remaining_trades()}
Next slot unlock: {self.pdt.next_trade_unlock()}""")

        # Risk status
        try:
            pv = account['portfolio_value']
            sections.append(f"""== Risk Management ==
{self.risk.status(pv)}
Consecutive losses: {self.risk.consecutive_losses}
Max single trade loss: ${pv * 1.5 / 100:,.2f} (1.5%)
Max daily loss: ${pv * 3 / 100:,.2f} (3%)""")
        except Exception:
            pass

        # Current positions
        try:
            positions = self.executor.get_positions()
            if positions:
                pos_text = "\n".join([
                    f"  {p['ticker']:6s} | {p['qty']} shares | "
                    f"Entry ${p['entry_price']:.2f} | Current ${p['current_price']:.2f} | "
                    f"PnL: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pnl_pct']:+.1f}%)"
                    for p in positions
                ])
                sections.append(f"== Current Positions ==\n{pos_text}")
            else:
                sections.append("== Current Positions ==\nNo positions")
        except Exception:
            sections.append("== Current Positions ==\nFailed to retrieve")

        # Today's scan results
        today_report = self._load_today_report()
        if today_report:
            sections.append(f"""== Today's Scan ==
Scan count: {today_report.get('scan_count', 'N/A')}
Raw candidates: {today_report.get('raw_candidates', 'N/A')}
After news filter: {today_report.get('after_news_filter', 'N/A')}
Market environment: {today_report.get('market_context', 'Unknown')}""")

            recs = today_report.get('recommendations', [])
            if recs:
                rec_text = "\n".join([
                    f"  #{r.get('rank', '?')} {r.get('ticker', '?')} | "
                    f"{r.get('action', '?')} | Confidence:{r.get('confidence', '?')} | "
                    f"Entry ${r.get('entry_price', 0):.2f} | "
                    f"R/R:{r.get('risk_reward_ratio', 0):.1f}"
                    for r in recs
                ])
                sections.append(f"== Today's Recommendations ==\n{rec_text}")

        # Recent trades
        recent_trades = self._load_recent_trades()
        if recent_trades:
            trades_text = "\n".join([
                f"  {t.get('timestamp', '?')[:16]} | {t.get('type', '?')} | "
                f"{t.get('ticker', '?')} | "
                f"${t.get('entry_price', t.get('exit_price', 0)):.2f}"
                for t in recent_trades[-10:]  # Last 10 trades
            ])
            sections.append(f"== Recent Trades ==\n{trades_text}")

        # Virtual signals (recorded when PDT slots exhausted)
        virtual = self._load_virtual_signals()
        if virtual:
            v_text = "\n".join([
                f"  {v['ticker']:6s} | Combined:{v.get('combined_score', 0):>5.1f} | "
                f"News sentiment:{v.get('news_sentiment', 0):>5.1f}"
                for v in virtual[:10]
            ])
            sections.append(f"== Today's Virtual Signals (not traded) ==\n{v_text}")

        # Current time
        now = datetime.now()
        weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][now.weekday()]
        market_status = "Pre-market" if now.hour < 9 or (now.hour == 9 and now.minute < 30) else (
            "Market open" if now.hour < 16 else "After-hours"
        )
        sections.append(f"== Time ==\n{now.strftime('%Y-%m-%d %H:%M')} {weekday} | {market_status}")

        # Macro reminder
        sections.append("""== Current Macro Environment (late March 2026) ==
- US-Iran conflict ongoing, Brent crude >$110, Strait of Hormuz passage disrupted
- Fed held rates steady in March, only 1 rate cut projected for 2026
- Chair Powell's term expires in May, successor Warsh not yet confirmed
- S&P 500 down 5 consecutive weeks, Nasdaq and Dow Jones in correction territory
- 2026 is a midterm election year, historically higher volatility in first 3 quarters
- Tariff-driven inflation still being digested
- Sector preference: Energy/Defense benefiting from conflict, Healthcare strong defensively, Consumer Discretionary under pressure""")

        return "\n\n".join(sections)

    def _build_system_prompt(self) -> str:
        """Build the complete system prompt"""
        context = self._build_context()

        return f"""You are my personal quantitative trading assistant. You can see my full account status, positions, scan results, and trade history.

Your responsibilities:
1. Answer any questions about my positions, market, and strategy based on live data
2. If I ask "should I buy/sell a stock", combine my account status, PDT slots, and risk rules to give specific advice
3. You can challenge my decisions -- if what I'm about to do violates risk rules or is unwise, say so directly
4. Keep your answers concise and direct, no pleasantries

Key rules (you must follow):
- Max single position 15%
- Max single trade loss 1.5%
- Max daily loss 3%
- PDT limit: max 3 day trades in 5 trading days
- Must close all intraday positions by 3:50 PM
- If what I want to do would violate these rules, you must clearly refuse and explain

Here is my live data:

{context}

Be concise, direct, and data-driven."""

    # ================================================================
    # Load Data Files
    # ================================================================

    def _load_today_report(self) -> Optional[Dict]:
        path = os.path.join(LOG_DIR, f"daily_report_{date.today()}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def _load_recent_trades(self) -> List[Dict]:
        path = os.path.join(LOG_DIR, "trade_log.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def _load_virtual_signals(self) -> List[Dict]:
        path = os.path.join(LOG_DIR, f"virtual_signals_{date.today()}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    # ================================================================
    # Chat
    # ================================================================

    def chat(self, user_message: str) -> str:
        """Send message and get reply"""

        # Refresh system prompt each conversation (positions may have changed)
        self.system_prompt = self._build_system_prompt()

        # Add user message
        self.conversation.append({"role": "user", "content": user_message})

        # Build complete message list
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self.conversation[-20:]  # Keep only last 20 turns to avoid context overflow

        try:
            if self.model.startswith("o") or self.model.startswith("gpt-5"):
                response = self.openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=2000,
                )
            else:
                response = self.openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_completion_tokens=2000,
                )

            reply = response.choices[0].message.content

            # Log token usage
            usage = response.usage
            if usage:
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                # Rough cost estimate
                if self.model == MODEL_RANK:  # o3
                    cost = (input_tokens * 10 + output_tokens * 40) / 1_000_000
                elif self.model == MODEL_DEEP:  # gpt-5.4
                    cost = (input_tokens * 5 + output_tokens * 15) / 1_000_000
                else:  # mini
                    cost = (input_tokens * 0.4 + output_tokens * 1.6) / 1_000_000
                print(f"  [tokens: {input_tokens}+{output_tokens} | ~${cost:.4f}]")

            # Save assistant reply
            self.conversation.append({"role": "assistant", "content": reply})

            return reply

        except Exception as e:
            error_msg = f"API call failed: {e}"
            print(f"  ❌ {error_msg}")
            return error_msg

    # ================================================================
    # Special Commands
    # ================================================================

    def handle_command(self, user_input: str) -> Optional[str]:
        """Handle special commands (no API call)"""
        cmd = user_input.strip().lower()

        if cmd in ["/status", "/s"]:
            return self._build_context()

        if cmd in ["/positions", "/pos", "/p"]:
            try:
                positions = self.executor.get_positions()
                if not positions:
                    return "No positions"
                lines = []
                for p in positions:
                    lines.append(
                        f"{p['ticker']:6s} | {p['qty']} shares | "
                        f"Entry ${p['entry_price']:.2f} | Current ${p['current_price']:.2f} | "
                        f"PnL: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pnl_pct']:+.1f}%)"
                    )
                return "\n".join(lines)
            except Exception as e:
                return f"Failed to get positions: {e}"

        if cmd in ["/pdt"]:
            return (
                f"{self.pdt.status()}\n"
                f"Next slot unlock: {self.pdt.next_trade_unlock()}"
            )

        if cmd in ["/cost"]:
            return (
                f"Current model: {self.model}\n"
                f"Conversation turns: {len(self.conversation) // 2}\n"
                f"Tip: /model <name> to switch model"
            )

        if cmd.startswith("/model "):
            new_model = cmd.split(" ", 1)[1].strip()
            valid = [MODEL_RANK, MODEL_DEEP, MODEL_FAST, "o3", "gpt-5.4", "gpt-4.1-mini"]
            if new_model in valid:
                self.model = new_model
                return f"Switched to {new_model}"
            return f"Invalid model. Options: {', '.join(valid)}"

        if cmd in ["/clear", "/c"]:
            self.conversation = []
            return "Conversation history cleared"

        if cmd in ["/help", "/h", "?"]:
            return """Available commands:
/status, /s    - View full account status
/positions, /p - View current positions
/pdt           - View PDT slots
/model <name>  - Switch model (o3, gpt-5.4, gpt-4.1-mini)
/cost          - View conversation cost
/clear, /c     - Clear conversation history
/help, /h      - Show this help
/quit, /q      - Exit

Type any question to chat with the AI."""

        if cmd in ["/quit", "/q", "exit", "quit"]:
            return "__EXIT__"

        return None  # Not a command, proceed with normal chat

    # ================================================================
    # Main Loop
    # ================================================================

    def run(self):
        """Interactive chat main loop"""
        print("\n" + "=" * 60)
        print("Trading Assistant Chat Mode")
        print(f"   Model: {self.model}")
        print("   Type /help for commands | /quit to exit")
        print("=" * 60)

        # Show account status on startup
        print("\nLoading account data...\n")
        try:
            context_summary = self._build_context()
            # Print only the first few key lines
            for line in context_summary.split("\n")[:15]:
                print(f"  {line}")
            print("  ...")
        except Exception as e:
            print(f"  Warning: partial data load failed: {e}")

        print(f"\n{'─' * 60}")
        print("Ready. Ask me anything about your positions, market, or strategy.\n")

        while True:
            try:
                user_input = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Check special commands
            cmd_result = self.handle_command(user_input)
            if cmd_result == "__EXIT__":
                print("Goodbye!")
                break
            if cmd_result is not None:
                print(f"\n{cmd_result}\n")
                continue

            # Normal chat
            print(f"\n({self.model}) Thinking...\n")
            reply = self.chat(user_input)
            print(f"{reply}\n")


def main():
    parser = argparse.ArgumentParser(description="Trading Assistant Chat Mode")
    parser.add_argument(
        "--model",
        default=MODEL_RANK,
        choices=[MODEL_RANK, MODEL_DEEP, MODEL_FAST, "o3", "gpt-5.4", "gpt-4.1-mini"],
        help="Select model (default o3)",
    )
    args = parser.parse_args()

    chat = TradingChat(model=args.model)
    chat.run()


if __name__ == "__main__":
    main()
