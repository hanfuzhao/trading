"""
订单执行器 v3 — 只做多 | Limit入场120s | Stop-Limit止损 | 分批止盈 | 尾盘分步平仓
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
from alpaca.trading.enums import (
    OrderSide, TimeInForce, QueryOrderStatus,
)

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET,
    LOG_DIR, MAX_CAPITAL,
    ENTRY_TIMEOUT_SECONDS, TIME_STOP_MINUTES,
    PARTIAL_EXIT_R, PARTIAL_EXIT_PCT, TRAILING_ATR_MULT,
)

ET = ZoneInfo("America/New_York")
SKIP_TICKERS = {"SGOV"}


class OrderExecutor:
    def __init__(self):
        self.client = TradingClient(
            ALPACA_API_KEY, ALPACA_API_SECRET, paper=True,
        )
        self.active_trades: Dict[str, Dict] = {}
        self._stop_order_ids: Dict[str, str] = {}

    # ================================================================
    # 账户信息
    # ================================================================

    def get_account(self) -> Dict:
        account = self.client.get_account()
        pv = float(account.portfolio_value)
        cash = float(account.cash)
        bp = float(account.buying_power)
        if MAX_CAPITAL > 0:
            pv = min(pv, MAX_CAPITAL)
            cash = min(cash, MAX_CAPITAL)
            bp = min(bp, MAX_CAPITAL)
        return {
            "portfolio_value": pv, "cash": cash, "buying_power": bp,
            "equity": float(account.equity),
            "day_trade_count": int(account.daytrade_count),
            "pattern_day_trader": account.pattern_day_trader,
        }

    def get_positions(self) -> List[Dict]:
        positions = self.client.get_all_positions()
        return [{
            "ticker": p.symbol,
            "qty": int(p.qty),
            "side": str(p.side) if p.side else "long",
            "entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pnl": float(p.unrealized_pl),
            "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
        } for p in positions]

    def get_available_cash(self) -> float:
        account = self.client.get_account()
        return float(account.cash)

    # ================================================================
    # 入场 — Limit Order，120秒超时取消（v3: 只做BUY）
    # ================================================================

    def execute_entry(
        self, ticker: str, shares: int,
        entry_price: float, stop_loss: float,
        take_profit_1: float, take_profit_2: float,
        atr: float,
    ) -> Tuple[bool, str]:
        try:
            order = LimitOrderRequest(
                symbol=ticker, qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(entry_price, 2),
            )
            result = self.client.submit_order(order)
            order_id = str(result.id)

            filled_price = self._wait_for_fill(order_id, timeout=ENTRY_TIMEOUT_SECONDS)
            if filled_price is None:
                print(f"[执行] {ticker} Limit 入场单 {ENTRY_TIMEOUT_SECONDS}s 内未成交，取消")
                try:
                    self.client.cancel_order_by_id(order_id)
                except Exception:
                    pass
                return False, "入场单超时未成交"

            actual_entry = filled_price
            r_value = abs(actual_entry - stop_loss)

            self.active_trades[ticker] = {
                "order_id": order_id,
                "shares": shares,
                "remaining_shares": shares,
                "entry_price": actual_entry,
                "stop_loss": stop_loss,
                "take_profit_1": take_profit_1,
                "take_profit_2": take_profit_2,
                "r_value": r_value,
                "atr": atr,
                "entry_time": datetime.now(ET).isoformat(),
                "status": "filled",
                "partial_taken": False,
                "highest_price": actual_entry,
            }

            # 设置 Stop-Limit 止损（全部股数）
            self._set_stop_limit(ticker, stop_loss, shares)

            self._log_trade("ENTRY", ticker, {
                "action": "BUY", "shares": shares,
                "limit_price": entry_price, "filled_price": actual_entry,
                "stop_loss": stop_loss, "tp1": take_profit_1, "tp2": take_profit_2,
            })
            print(f"[执行] BUY {shares}股 {ticker} 成交@${actual_entry:.2f} | SL${stop_loss:.2f} | TP1${take_profit_1:.2f} | TP2${take_profit_2:.2f}")
            return True, f"已成交 @${actual_entry:.2f}"

        except Exception as e:
            print(f"[执行] 下单失败 ({ticker}): {e}")
            return False, str(e)

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

    def _set_stop_limit(self, ticker: str, stop_price: float, shares: int):
        """Stop-Limit 止损：limit = stop × 0.995"""
        try:
            limit_price = round(stop_price * 0.995, 2)
            order = StopLimitOrderRequest(
                symbol=ticker, qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_price, 2),
                limit_price=limit_price,
            )
            result = self.client.submit_order(order)
            self._stop_order_ids[ticker] = str(result.id)
            print(f"[执行] Stop-Limit 止损已设: {ticker} stop${stop_price:.2f} limit${limit_price:.2f}")
        except Exception as e:
            print(f"[执行] 止损设置失败 ({ticker}): {e}")

    # ================================================================
    # 分批止盈 + Trailing Stop（v3: 1R平50%，移SL到保本，1.5xATR trailing）
    # ================================================================

    def update_trailing_stop(self, ticker: str, current_price: float):
        if ticker not in self.active_trades:
            return
        trade = self.active_trades[ticker]
        if trade["status"] != "filled":
            return

        remaining = trade["remaining_shares"]
        entry = trade["entry_price"]
        r_value = trade["r_value"]

        # 更新最高价
        if current_price > trade["highest_price"]:
            trade["highest_price"] = current_price

        # === 1R 分批止盈 ===
        if not trade["partial_taken"] and r_value > 0:
            if current_price >= trade["take_profit_1"]:
                partial_qty = max(remaining // 2, 1)
                if partial_qty > 0 and remaining > 1:
                    try:
                        sell_order = MarketOrderRequest(
                            symbol=ticker, qty=partial_qty,
                            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                        )
                        self.client.submit_order(sell_order)
                        trade["partial_taken"] = True
                        trade["remaining_shares"] = remaining - partial_qty
                        breakeven = round(entry + 0.02, 2)
                        self._replace_stop_limit(ticker, breakeven, trade["remaining_shares"])
                        trade["stop_loss"] = breakeven
                        trade["highest_price"] = current_price
                        print(f"[执行] TP1 分批止盈: {ticker} 平{partial_qty}股@${current_price:.2f} (+1R), 剩余{trade['remaining_shares']}股, SL→保本${breakeven:.2f}")
                        self._log_trade("PARTIAL_EXIT", ticker, {
                            "qty": partial_qty, "price": current_price,
                            "remaining": trade["remaining_shares"],
                        })
                    except Exception as e:
                        print(f"[执行] 分批止盈失败 ({ticker}): {e}")
                return

        # === Trailing Stop（TP1 达成后启用，1.5x ATR 距离）===
        if trade["partial_taken"]:
            atr = trade.get("atr", r_value)
            trailing_distance = atr * TRAILING_ATR_MULT
            new_stop = round(trade["highest_price"] - trailing_distance, 2)
            if new_stop > trade["stop_loss"] and new_stop > entry:
                self._replace_stop_limit(ticker, new_stop, trade["remaining_shares"])
                trade["stop_loss"] = new_stop

    # ================================================================
    # 时间止损（25分钟，盈利 < 0.5R → 市价平仓）
    # ================================================================

    def check_time_stop(self, ticker: str, current_price: float) -> bool:
        if ticker not in self.active_trades:
            return False
        trade = self.active_trades[ticker]
        if trade["status"] != "filled":
            return False

        entry_time = datetime.fromisoformat(trade["entry_time"])
        now = datetime.now(ET)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=ET)
        minutes_held = (now - entry_time).total_seconds() / 60

        if minutes_held < TIME_STOP_MINUTES:
            return False

        r_value = trade["r_value"]
        gain = current_price - trade["entry_price"]

        if r_value > 0 and gain < r_value * 0.5:
            print(f"[执行] ⏰ 时间止损: {ticker} 持仓{minutes_held:.0f}分钟，盈利不足0.5R，平仓")
            success, _ = self.close_position(ticker)
            if success:
                self._log_trade("TIME_STOP", ticker, {
                    "minutes_held": round(minutes_held), "gain": round(gain, 2),
                    "r_value": round(r_value, 2),
                })
            return success
        return False

    # ================================================================
    # Stop-Limit 替换
    # ================================================================

    def _replace_stop_limit(self, ticker: str, new_stop: float, shares: int):
        old_id = self._stop_order_ids.get(ticker)
        if old_id:
            try:
                self.client.cancel_order_by_id(old_id)
            except Exception:
                pass

        try:
            limit_price = round(new_stop * 0.995, 2)
            order = StopLimitOrderRequest(
                symbol=ticker, qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(new_stop, 2),
                limit_price=limit_price,
            )
            result = self.client.submit_order(order)
            self._stop_order_ids[ticker] = str(result.id)
            print(f"[执行] 移动止损: {ticker} stop${new_stop:.2f} limit${limit_price:.2f}")
        except Exception as e:
            print(f"[执行] 移动止损失败 ({ticker}): {e}")

    # ================================================================
    # 平仓
    # ================================================================

    def close_position(self, ticker: str) -> Tuple[bool, str]:
        try:
            self.client.close_position(ticker)
            if ticker in self.active_trades:
                self.active_trades[ticker]["status"] = "closed"
            if ticker in self._stop_order_ids:
                try:
                    self.client.cancel_order_by_id(self._stop_order_ids[ticker])
                except Exception:
                    pass
                del self._stop_order_ids[ticker]
            print(f"[执行] 已平仓: {ticker}")
            return True, f"{ticker} 已平仓"
        except Exception as e:
            print(f"[执行] 平仓失败 ({ticker}): {e}")
            return False, str(e)

    def close_all_positions(self, skip_tickers: set = None) -> List[str]:
        skip = skip_tickers or SKIP_TICKERS
        closed = []
        positions = self.get_positions()
        for p in positions:
            if p["ticker"] in skip:
                continue
            success, _ = self.close_position(p["ticker"])
            if success:
                closed.append(p["ticker"])
        return closed

    # ================================================================
    # 尾盘分步平仓（v3: 15:45盈利平 → 15:48全平 → 15:50确认）
    # ================================================================

    def close_profitable_positions(self) -> List[Dict]:
        """15:45 — 盈利持仓市价平仓"""
        closed = []
        positions = self.get_positions()
        for p in positions:
            if p["ticker"] in SKIP_TICKERS:
                continue
            if p["unrealized_pnl"] > 0:
                success, _ = self.close_position(p["ticker"])
                if success:
                    closed.append(p)
                    print(f"[执行] 15:45 锁定利润: {p['ticker']} PnL${p['unrealized_pnl']:+.2f}")
        return closed

    def close_remaining_positions(self) -> List[Dict]:
        """15:48 — 所有剩余持仓市价平仓"""
        closed = []
        positions = self.get_positions()
        for p in positions:
            if p["ticker"] in SKIP_TICKERS:
                continue
            success, _ = self.close_position(p["ticker"])
            if success:
                closed.append(p)
                print(f"[执行] 15:48 强制平仓: {p['ticker']} PnL${p['unrealized_pnl']:+.2f}")
        return closed

    def confirm_zero_positions(self) -> bool:
        """15:50 — 最终确认零持仓"""
        positions = self.get_positions()
        active = [p for p in positions if p["ticker"] not in SKIP_TICKERS]
        if active:
            print(f"[执行] ⚠️ 15:50 仍有 {len(active)} 个持仓未平！强制清仓")
            self.cancel_pending_orders()
            self.close_all_positions()
            return False
        print("[执行] ✅ 15:50 确认零持仓")
        return True

    # ================================================================
    # 检查已被broker平仓的交易
    # ================================================================

    def check_closed_trades(self) -> List[Dict]:
        closed = []
        positions = self.get_positions()
        held_tickers = {p["ticker"] for p in positions}

        for ticker, trade in list(self.active_trades.items()):
            if trade["status"] == "closed":
                continue
            if ticker not in held_tickers and trade["status"] == "filled":
                entry = trade["entry_price"]
                shares = trade.get("remaining_shares", trade["shares"])

                exit_price = self._get_exit_price(ticker)
                pnl = (exit_price - entry) * shares if exit_price else 0

                trade["status"] = "closed"
                trade["exit_price"] = exit_price
                trade["pnl"] = pnl

                if ticker in self._stop_order_ids:
                    try:
                        self.client.cancel_order_by_id(self._stop_order_ids[ticker])
                    except Exception:
                        pass
                    del self._stop_order_ids[ticker]

                closed.append({"ticker": ticker, "pnl": pnl, "exit_price": exit_price, "trade": trade})
                self._log_trade("EXIT", ticker, {"pnl": pnl, "exit_price": exit_price})
                print(f"[执行] {ticker} 已平仓 | PnL: ${pnl:+.2f}")

        return closed

    def _get_exit_price(self, ticker: str) -> Optional[float]:
        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, symbols=[ticker], limit=5,
            )
            orders = self.client.get_orders(req)
            for o in orders:
                if str(o.status) == "filled" and o.filled_avg_price:
                    return float(o.filled_avg_price)
        except Exception:
            pass
        return None

    # ================================================================
    # 订单管理
    # ================================================================

    def cancel_pending_orders(self):
        try:
            self.client.cancel_orders()
            self._stop_order_ids.clear()
            print("[执行] 已取消所有挂单")
        except Exception as e:
            print(f"[执行] 取消挂单失败: {e}")

    # ================================================================
    # 日志
    # ================================================================

    def _log_trade(self, trade_type: str, ticker: str, details: Dict):
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, "trade_log.json")
        entry = {
            "timestamp": datetime.now(ET).isoformat(),
            "type": trade_type, "ticker": ticker,
            **details,
        }
        logs = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        logs.append(entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2, default=str)
