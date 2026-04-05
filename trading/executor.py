"""
Order Executor v6 - Overnight/intraday position separation | Pre-market stop | Bracket Order | Extended hours
"""
import json
import os
import time as _time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    StopLimitOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from core.config import (
    ALPACA_API_KEY, ALPACA_API_SECRET,
    LOG_DIR, MAX_CAPITAL,
    ENTRY_TIMEOUT_SECONDS, TIME_STOP_MINUTES,
    PARTIAL_EXIT_R, PARTIAL_EXIT_PCT, TRAILING_ATR_MULT,
    CATASTROPHIC_STOP_PCT,
)

ET = ZoneInfo("America/New_York")
SKIP_TICKERS = {"SGOV"}


class OrderExecutor:
    def __init__(self):
        self.client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=False)
        self.overnight_trades: Dict[str, Dict] = {}
        self.intraday_trades: Dict[str, Dict] = {}
        self._stop_order_ids: Dict[str, str] = {}

    # ================================================================
    # Account
    # ================================================================

    def get_account(self) -> Dict:
        account = self.client.get_account()
        pv, cash, bp = float(account.portfolio_value), float(account.cash), float(account.buying_power)
        if MAX_CAPITAL > 0:
            pv, cash, bp = min(pv, MAX_CAPITAL), min(cash, MAX_CAPITAL), min(bp, MAX_CAPITAL)
        return {
            "portfolio_value": pv, "cash": cash, "buying_power": bp,
            "equity": float(account.equity),
            "day_trade_count": int(account.daytrade_count),
            "pattern_day_trader": bool(account.pattern_day_trader),
        }

    def get_positions(self) -> List[Dict]:
        return [{
            "ticker": p.symbol, "qty": int(p.qty),
            "side": str(p.side) if p.side else "long",
            "entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pnl": float(p.unrealized_pl),
            "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
        } for p in self.client.get_all_positions()]

    def get_available_cash(self) -> float:
        return float(self.client.get_account().cash)

    # ================================================================
    # Overnight Entry (15:30-15:45 Limit Order, no mechanical stop)
    # ================================================================

    def enter_overnight(self, ticker: str, shares: int, limit_price: float,
                        atr: float, natr: float) -> Tuple[bool, str]:
        try:
            order = LimitOrderRequest(
                symbol=ticker, qty=shares, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, limit_price=round(limit_price, 2),
            )
            result = self.client.submit_order(order)
            order_id = str(result.id)

            filled_price = self._wait_for_fill(order_id, timeout=ENTRY_TIMEOUT_SECONDS)
            if filled_price is None:
                try:
                    self.client.cancel_order_by_id(order_id)
                except Exception:
                    pass
                return False, "Overnight entry timed out without fill"

            stop_line = round(filled_price * (1 - CATASTROPHIC_STOP_PCT / 100), 2)
            self.overnight_trades[ticker] = {
                "order_id": order_id, "shares": shares,
                "entry_price": filled_price, "stop_line": stop_line,
                "atr": atr, "natr": natr,
                "entry_time": datetime.now(ET).isoformat(),
                "status": "held",
            }
            self._log_trade("OVERNIGHT_ENTRY", ticker, {
                "shares": shares, "price": filled_price, "stop_line": stop_line,
            })
            print(f"[Executor] Overnight BUY {shares} shares {ticker} @${filled_price:.2f} | Stop line ${stop_line:.2f}")
            return True, f"Overnight filled @${filled_price:.2f}"
        except Exception as e:
            print(f"[Executor] Overnight entry failed ({ticker}): {e}")
            return False, str(e)

    # ================================================================
    # Overnight Exit (9:45-10:15, based on gap direction)
    # ================================================================

    def exit_overnight(self, ticker: str, current_price: float, reason: str = "") -> Tuple[bool, float]:
        if ticker not in self.overnight_trades:
            return False, 0
        trade = self.overnight_trades[ticker]
        try:
            order = MarketOrderRequest(
                symbol=ticker, qty=trade["shares"], side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            self.client.submit_order(order)
            pnl = (current_price - trade["entry_price"]) * trade["shares"]
            trade["status"] = "closed"
            trade["exit_price"] = current_price
            trade["pnl"] = pnl
            self._log_trade("OVERNIGHT_EXIT", ticker, {
                "pnl": pnl, "exit_price": current_price, "reason": reason,
            })
            print(f"[Executor] Overnight exit {ticker} @${current_price:.2f} | PnL${pnl:+.2f} | {reason}")
            return True, pnl
        except Exception as e:
            print(f"[Executor] Overnight exit failed ({ticker}): {e}")
            return False, 0

    # ================================================================
    # Pre-market Stop (4:00 AM, extended hours Limit Sell)
    # ================================================================

    def submit_premarket_stop(self, ticker: str, shares: int, price: float) -> bool:
        try:
            aggressive_limit = round(price * 0.995, 2)
            order = LimitOrderRequest(
                symbol=ticker, qty=shares, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY, limit_price=aggressive_limit,
                extended_hours=True,
            )
            self.client.submit_order(order)
            print(f"[Executor] Pre-market stop: {ticker} {shares} shares limit${aggressive_limit:.2f}")
            if ticker in self.overnight_trades:
                self.overnight_trades[ticker]["status"] = "premarket_stopped"
            return True
        except Exception as e:
            print(f"[Executor] Pre-market stop failed ({ticker}): {e}")
            return False

    # ================================================================
    # Intraday Entry (14:00-14:45, Limit + Stop-Limit bracket)
    # ================================================================

    def enter_intraday(self, ticker: str, shares: int, limit_price: float,
                       stop_loss: float, take_profit_1: float, take_profit_2: float,
                       atr: float) -> Tuple[bool, str]:
        try:
            order = LimitOrderRequest(
                symbol=ticker, qty=shares, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, limit_price=round(limit_price, 2),
            )
            result = self.client.submit_order(order)
            order_id = str(result.id)

            filled_price = self._wait_for_fill(order_id, timeout=ENTRY_TIMEOUT_SECONDS)
            if filled_price is None:
                try:
                    self.client.cancel_order_by_id(order_id)
                except Exception:
                    pass
                return False, "Intraday entry timed out"

            r_value = abs(filled_price - stop_loss)
            self.intraday_trades[ticker] = {
                "order_id": order_id, "shares": shares, "remaining_shares": shares,
                "entry_price": filled_price, "stop_loss": stop_loss,
                "take_profit_1": take_profit_1, "take_profit_2": take_profit_2,
                "r_value": r_value, "atr": atr,
                "entry_time": datetime.now(ET).isoformat(),
                "status": "filled", "partial_taken": False,
                "highest_price": filled_price,
            }
            self._set_stop_limit(ticker, stop_loss, shares)
            self._log_trade("INTRADAY_ENTRY", ticker, {
                "shares": shares, "price": filled_price,
                "stop_loss": stop_loss, "tp1": take_profit_1,
            })
            print(f"[Executor] Intraday BUY {shares} shares {ticker} @${filled_price:.2f} | SL${stop_loss:.2f}")
            return True, f"Intraday filled @${filled_price:.2f}"
        except Exception as e:
            print(f"[Executor] Intraday entry failed ({ticker}): {e}")
            return False, str(e)

    def _set_stop_limit(self, ticker: str, stop_price: float, shares: int):
        try:
            limit = round(stop_price * 0.995, 2)
            order = StopLimitOrderRequest(
                symbol=ticker, qty=shares, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_price, 2), limit_price=limit,
            )
            result = self.client.submit_order(order)
            self._stop_order_ids[ticker] = str(result.id)
        except Exception as e:
            print(f"[Executor] Stop-loss setup failed ({ticker}): {e}")

    # ================================================================
    # Intraday Management (partial take-profit + trailing + time stop)
    # ================================================================

    def update_trailing_stop(self, ticker: str, current_price: float):
        if ticker not in self.intraday_trades:
            return
        trade = self.intraday_trades[ticker]
        if trade["status"] != "filled":
            return
        remaining = trade["remaining_shares"]
        entry = trade["entry_price"]

        if current_price > trade["highest_price"]:
            trade["highest_price"] = current_price

        if not trade["partial_taken"] and trade["r_value"] > 0:
            if current_price >= trade["take_profit_1"]:
                partial_qty = max(remaining // 2, 1)
                if partial_qty > 0 and remaining > 1:
                    try:
                        self.client.submit_order(MarketOrderRequest(
                            symbol=ticker, qty=partial_qty,
                            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                        ))
                        trade["partial_taken"] = True
                        trade["remaining_shares"] = remaining - partial_qty
                        breakeven = round(entry + 0.02, 2)
                        self._replace_stop_limit(ticker, breakeven, trade["remaining_shares"])
                        trade["stop_loss"] = breakeven
                        print(f"[Executor] TP1: {ticker} closed {partial_qty} shares, SL->breakeven ${breakeven:.2f}")
                    except Exception as e:
                        print(f"[Executor] TP1 failed ({ticker}): {e}")
                return

        if trade["partial_taken"]:
            trail_dist = trade["atr"] * TRAILING_ATR_MULT
            new_stop = round(trade["highest_price"] - trail_dist, 2)
            if new_stop > trade["stop_loss"] and new_stop > entry:
                self._replace_stop_limit(ticker, new_stop, trade["remaining_shares"])
                trade["stop_loss"] = new_stop

    def check_time_stop(self, ticker: str, current_price: float) -> bool:
        if ticker not in self.intraday_trades:
            return False
        trade = self.intraday_trades[ticker]
        if trade["status"] != "filled":
            return False
        entry_time = datetime.fromisoformat(trade["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=ET)
        minutes = (datetime.now(ET) - entry_time).total_seconds() / 60
        if minutes < TIME_STOP_MINUTES:
            return False
        gain = current_price - trade["entry_price"]
        if trade["r_value"] > 0 and gain < trade["r_value"] * 0.5:
            print(f"[Executor] Time stop: {ticker} {minutes:.0f} min, profit below 0.5R")
            return self.close_intraday(ticker)
        return False

    def _replace_stop_limit(self, ticker: str, new_stop: float, shares: int):
        old_id = self._stop_order_ids.get(ticker)
        if old_id:
            try:
                self.client.cancel_order_by_id(old_id)
            except Exception:
                pass
        try:
            limit = round(new_stop * 0.995, 2)
            order = StopLimitOrderRequest(
                symbol=ticker, qty=shares, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(new_stop, 2), limit_price=limit,
            )
            result = self.client.submit_order(order)
            self._stop_order_ids[ticker] = str(result.id)
        except Exception as e:
            print(f"[Executor] Trailing stop update failed ({ticker}): {e}")

    # ================================================================
    # Close Positions
    # ================================================================

    def close_intraday(self, ticker: str) -> bool:
        try:
            self.client.close_position(ticker)
            if ticker in self.intraday_trades:
                self.intraday_trades[ticker]["status"] = "closed"
            if ticker in self._stop_order_ids:
                try:
                    self.client.cancel_order_by_id(self._stop_order_ids[ticker])
                except Exception:
                    pass
                del self._stop_order_ids[ticker]
            return True
        except Exception as e:
            print(f"[Executor] Intraday close failed ({ticker}): {e}")
            return False

    def close_all_intraday(self) -> List[Dict]:
        """15:48 - Market close all intraday positions (keep overnight)"""
        closed = []
        positions = self.get_positions()
        overnight_tickers = {t for t, v in self.overnight_trades.items() if v["status"] == "held"}
        for p in positions:
            if p["ticker"] in SKIP_TICKERS or p["ticker"] in overnight_tickers:
                continue
            if self.close_intraday(p["ticker"]):
                closed.append(p)
        return closed

    def confirm_zero_intraday(self) -> bool:
        """15:50 - Confirm zero intraday positions (overnight retained)"""
        positions = self.get_positions()
        overnight_tickers = {t for t, v in self.overnight_trades.items() if v["status"] == "held"}
        active_intraday = [
            p for p in positions
            if p["ticker"] not in SKIP_TICKERS and p["ticker"] not in overnight_tickers
        ]
        if active_intraday:
            print(f"[Executor] Warning: 15:50 still has {len(active_intraday)} intraday positions! Force closing")
            self.cancel_pending_orders()
            for p in active_intraday:
                self.close_intraday(p["ticker"])
            return False
        print("[Executor] 15:50 zero intraday positions confirmed (overnight positions retained)")
        return True

    def check_closed_trades(self) -> List[Dict]:
        closed = []
        positions = self.get_positions()
        held = {p["ticker"] for p in positions}
        for ticker, trade in list(self.intraday_trades.items()):
            if trade["status"] == "closed":
                continue
            if ticker not in held and trade["status"] == "filled":
                exit_price = self._get_exit_price(ticker)
                shares = trade.get("remaining_shares", trade["shares"])
                pnl = (exit_price - trade["entry_price"]) * shares if exit_price else 0
                trade["status"] = "closed"
                trade["pnl"] = pnl
                if ticker in self._stop_order_ids:
                    try:
                        self.client.cancel_order_by_id(self._stop_order_ids[ticker])
                    except Exception:
                        pass
                    del self._stop_order_ids[ticker]
                closed.append({"ticker": ticker, "pnl": pnl, "type": "intraday"})
                self._log_trade("INTRADAY_EXIT", ticker, {"pnl": pnl})
                print(f"[Executor] Intraday {ticker} auto-closed PnL${pnl:+.2f}")
        return closed

    # ================================================================
    # Utilities
    # ================================================================

    def _wait_for_fill(self, order_id: str, timeout: int = 120) -> Optional[float]:
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            try:
                order = self.client.get_order_by_id(order_id)
                if str(order.status) == "filled":
                    return float(order.filled_avg_price)
                if str(order.status) in ("canceled", "expired", "rejected"):
                    return None
            except Exception:
                pass
            _time.sleep(1)
        return None

    def _get_exit_price(self, ticker: str) -> Optional[float]:
        try:
            orders = self.client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, symbols=[ticker], limit=5,
            ))
            for o in orders:
                if str(o.status) == "filled" and o.filled_avg_price:
                    return float(o.filled_avg_price)
        except Exception:
            pass
        return None

    def cancel_pending_orders(self):
        try:
            self.client.cancel_orders()
            self._stop_order_ids.clear()
        except Exception as e:
            print(f"[Executor] Cancel pending orders failed: {e}")

    def _log_trade(self, trade_type: str, ticker: str, details: Dict):
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, "trade_log.json")
        entry = {"timestamp": datetime.now(ET).isoformat(), "type": trade_type, "ticker": ticker, **details}
        logs = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        logs.append(entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2, default=str)
