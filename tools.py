"""
Tool Registry — 7 tools for the LLM Trading Agent.
Each tool wraps existing modules (scanner, executor, etc.) into
a callable function with an OpenAI function-calling JSON schema.
"""
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ================================================================
# Tool JSON Schemas (OpenAI function-calling format)
# ================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": (
                "Get current price, volume, and technical indicators for a stock. "
                "Returns RSI(2), RSI(14), IBS, NATR, ATR, Bollinger Bands, MACD, "
                "support/resistance, and 20-day average volume."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'XOM'",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search and analyze recent news for a stock. Returns headlines with "
                "AI-generated sentiment (bullish/bearish/neutral), confidence, severity, "
                "and whether the news is structurally significant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_overnight",
            "description": (
                "Scan the entire US stock market for overnight mean-reversion candidates. "
                "Filters for RSI(2)<15 AND IBS<0.25, sorted by NATR (highest volatility first). "
                "Should be called between 15:00-15:30 ET for next-day overnight trades."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_intraday",
            "description": (
                "Scan the market for intraday afternoon-reversal candidates. "
                "Looks for stocks down >1.5% on the day with above-average volume. "
                "Best used between 14:00-14:45 ET."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": (
                "Get current portfolio status: account value, cash, buying power, "
                "all open positions with P&L, PDT day-trade slots remaining, "
                "and current risk management status."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_environment",
            "description": (
                "Get current macro market environment: 3-variable Regime matrix "
                "(VIX/VIX3M ratio, SPY vs 200-day SMA, VIX level), SPY daily change, "
                "USO oil price change, and Man Group's 7-variable macro analogy."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_trade",
            "description": (
                "Submit a stock trade order. Performs risk checks before execution. "
                "For overnight trades, no stop-loss is set (position sizing controls risk). "
                "For intraday trades, a bracket order with stop-loss is used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "side": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Order side",
                    },
                    "qty": {
                        "type": "integer",
                        "description": "Number of shares",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["overnight", "intraday"],
                        "description": "Trading strategy type",
                    },
                    "limit_price": {
                        "type": "number",
                        "description": "Limit price for the order (optional, uses market price if omitted)",
                    },
                },
                "required": ["symbol", "side", "qty", "strategy"],
            },
        },
    },
]


# ================================================================
# Tool Registry — wires schemas to implementations
# ================================================================

class ToolRegistry:
    """
    Holds references to the initialized trading components and
    exposes each tool as a method.  The agent calls
    ``registry.call(tool_name, **kwargs)`` which dispatches here.
    """

    def __init__(self, scanner=None, executor=None, risk_manager=None,
                 news_analyzer=None, pdt_tracker=None):
        self.scanner = scanner
        self.executor = executor
        self.risk = risk_manager
        self.news = news_analyzer
        self.pdt = pdt_tracker

        self._dispatch: Dict[str, Callable] = {
            "get_stock_data": self.get_stock_data,
            "search_news": self.search_news,
            "scan_overnight": self.scan_overnight,
            "scan_intraday": self.scan_intraday,
            "get_portfolio": self.get_portfolio,
            "get_macro_environment": self.get_macro_environment,
            "execute_trade": self.execute_trade,
        }

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict:
        fn = self._dispatch.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**arguments)
        except Exception as e:
            return {"error": str(e)}

    # ────────────────────────────────────────────────────────────
    # Tool 1: get_stock_data
    # ────────────────────────────────────────────────────────────

    def get_stock_data(self, symbol: str) -> Dict:
        symbol = symbol.upper().strip()
        snapshots = self.scanner.get_snapshots([symbol])
        snap = snapshots.get(symbol)
        if not snap or not snap.latest_trade:
            return {"error": f"No data available for {symbol}"}

        price = float(snap.latest_trade.price)
        prev_close = float(snap.previous_daily_bar.close) if snap.previous_daily_bar else price
        change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0
        today_vol = int(snap.daily_bar.volume) if snap.daily_bar else 0

        df = self.scanner.get_daily_bars(symbol)
        indicators = {}
        if df is not None:
            indicators = self.scanner.compute_indicators(df)

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "today_volume": today_vol,
            "indicators": indicators,
            "timestamp": datetime.now(ET).isoformat(),
        }

    # ────────────────────────────────────────────────────────────
    # Tool 2: search_news
    # ────────────────────────────────────────────────────────────

    def search_news(self, symbol: str) -> Dict:
        symbol = symbol.upper().strip()
        result = self.news.check_structural_risk(symbol)
        raw_news = self.news.get_news(symbol, limit=5)

        return {
            "symbol": symbol,
            "vetoed": result.get("vetoed", False),
            "reason": result.get("reason", ""),
            "news_score": result.get("news_score", 50),
            "articles": [
                {"headline": n.get("headline", ""), "url": n.get("url", "")}
                for n in raw_news[:5]
            ],
            "analyses": result.get("analyses", []),
        }

    # ────────────────────────────────────────────────────────────
    # Tool 3: scan_overnight
    # ────────────────────────────────────────────────────────────

    def scan_overnight(self) -> Dict:
        candidates = self.scanner.scan_overnight()
        return {
            "count": len(candidates),
            "candidates": [
                {
                    "ticker": c["ticker"],
                    "price": c["price"],
                    "rsi_2": c.get("rsi_2"),
                    "ibs": c.get("ibs"),
                    "natr": c.get("natr"),
                    "signal_strength": c.get("signal_strength"),
                    "sector": c.get("sector", "Unknown"),
                    "consecutive_down": c.get("indicators", {}).get("consecutive_down", 0),
                }
                for c in candidates[:15]
            ],
            "scan_time": datetime.now(ET).isoformat(),
        }

    # ────────────────────────────────────────────────────────────
    # Tool 4: scan_intraday
    # ────────────────────────────────────────────────────────────

    def scan_intraday(self) -> Dict:
        candidates = self.scanner.scan_intraday()
        return {
            "count": len(candidates),
            "candidates": [
                {
                    "ticker": c["ticker"],
                    "price": c["price"],
                    "intraday_change": c.get("intraday_change"),
                    "volume_ratio": c.get("volume_ratio"),
                    "signal_strength": c.get("signal_strength"),
                    "sector": c.get("sector", "Unknown"),
                }
                for c in candidates[:15]
            ],
            "scan_time": datetime.now(ET).isoformat(),
        }

    # ────────────────────────────────────────────────────────────
    # Tool 5: get_portfolio
    # ────────────────────────────────────────────────────────────

    def get_portfolio(self) -> Dict:
        acct = self.executor.get_account()
        positions = self.executor.get_positions()
        pdt_remaining = self.pdt.remaining_trades() if self.pdt else 0
        risk_status = self.risk.status(acct["portfolio_value"]) if self.risk else ""

        overnight_held = [
            {"ticker": t, "shares": v["shares"], "entry": v["entry_price"],
             "stop_line": v["stop_line"]}
            for t, v in self.executor.overnight_trades.items()
            if v.get("status") == "held"
        ]

        return {
            "account": acct,
            "positions": positions,
            "overnight_held": overnight_held,
            "pdt_remaining": pdt_remaining,
            "risk_status": risk_status,
            "timestamp": datetime.now(ET).isoformat(),
        }

    # ────────────────────────────────────────────────────────────
    # Tool 6: get_macro_environment
    # ────────────────────────────────────────────────────────────

    def get_macro_environment(self) -> Dict:
        return {
            "regime": self.scanner.get_regime(),
            "vix_vix3m_ratio": round(self.scanner.get_vix_vix3m_ratio(), 3),
            "spy_above_200sma": self.scanner.get_spy_above_200sma(),
            "spy_change_pct": round(self.scanner.get_spy_change(), 2),
            "uso_change_pct": round(self.scanner.get_uso_change(), 2),
            "vixy_price": round(self.scanner.get_vixy_price(), 2),
            "man_group_vars": self.scanner.get_man_group_vars(),
            "timestamp": datetime.now(ET).isoformat(),
        }

    # ────────────────────────────────────────────────────────────
    # Tool 7: execute_trade
    # ────────────────────────────────────────────────────────────

    def execute_trade(self, symbol: str, side: str, qty: int,
                      strategy: str, limit_price: float = 0) -> Dict:
        symbol = symbol.upper().strip()
        side = side.lower()

        if side != "buy":
            if strategy == "overnight":
                for t, v in self.executor.overnight_trades.items():
                    if t == symbol and v.get("status") == "held":
                        snap = self.scanner.get_snapshots([symbol])
                        price = float(snap[symbol].latest_trade.price) if symbol in snap else 0
                        ok, pnl = self.executor.exit_overnight(symbol, price, "agent sell command")
                        return {"success": ok, "pnl": round(pnl, 2), "message": f"Sold overnight {symbol}"}
            ok = self.executor.close_intraday(symbol)
            return {"success": ok, "message": f"Closed intraday {symbol}"}

        acct = self.executor.get_account()
        pv = acct["portfolio_value"]
        regime = self.scanner.get_regime()

        ok, reason = self.risk.can_trade(pv)
        if not ok:
            return {"success": False, "message": reason}

        snap = self.scanner.get_snapshots([symbol])
        price = float(snap[symbol].latest_trade.price) if symbol in snap else limit_price
        if not price:
            return {"success": False, "message": f"Cannot determine price for {symbol}"}

        if limit_price <= 0:
            limit_price = price

        if strategy == "overnight":
            if self.pdt:
                pass  # overnight doesn't consume PDT
            positions = self.executor.get_positions()
            current_on = [t for t, v in self.executor.overnight_trades.items() if v.get("status") == "held"]
            ok, msg, params = self.risk.validate_overnight(
                symbol, limit_price, pv, current_on, regime,
            )
            if not ok:
                return {"success": False, "message": msg}
            shares = min(qty, params["shares"])
            df = self.scanner.get_daily_bars(symbol)
            atr = 1.0
            natr = 2.0
            if df is not None:
                ind = self.scanner.compute_indicators(df)
                atr = ind.get("atr", 1.0)
                natr = ind.get("natr", 2.0)
            ok, msg = self.executor.enter_overnight(symbol, shares, limit_price, atr, natr)
            return {"success": ok, "message": msg, "shares": shares, "price": limit_price}

        else:  # intraday
            if self.pdt and not self.pdt.can_day_trade():
                return {"success": False, "message": "No PDT day-trade slots remaining"}
            df = self.scanner.get_daily_bars(symbol)
            atr = 1.0
            if df is not None:
                ind = self.scanner.compute_indicators(df)
                atr = ind.get("atr", 1.0)
            stops = self.risk.calculate_intraday_stops(limit_price, atr, regime)
            ok, msg, params = self.risk.validate_intraday(
                symbol, limit_price, stops["stop_loss"], pv,
                len(self.executor.get_positions()), regime,
            )
            if not ok:
                return {"success": False, "message": msg}
            shares = min(qty, params["shares"])
            ok, msg = self.executor.enter_intraday(
                symbol, shares, limit_price, stops["stop_loss"],
                stops["take_profit_1"], stops["take_profit_2"], atr,
            )
            if ok and self.pdt:
                self.pdt.record_day_trade(symbol)
            return {"success": ok, "message": msg, "shares": shares, "price": limit_price}
