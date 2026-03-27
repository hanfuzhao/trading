"""
订单执行器 - Alpaca下单、持仓监控、尾盘强制平仓
修复：Market Order 入场、入场确认后下止损止盈、移动止损同步 broker、
      强制平仓跳过 SGOV、使用美东时间
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
    StopLimitOrderRequest, StopOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide, TimeInForce, OrderStatus, QueryOrderStatus,
)

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL,
    FORCE_CLOSE_TIME, LOG_DIR, MAX_CAPITAL,
)

ET = ZoneInfo("America/New_York")

SKIP_TICKERS = {"SGOV"}


class OrderExecutor:
    def __init__(self):
        self.client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            paper=True,
        )
        self.active_trades: Dict[str, Dict] = {}
        self._stop_order_ids: Dict[str, str] = {}  # ticker -> broker stop order id

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
            "portfolio_value": pv,
            "cash": cash,
            "buying_power": bp,
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
        """获取实际可用现金（不受 MAX_CAPITAL 影响）"""
        account = self.client.get_account()
        return float(account.cash)

    # ================================================================
    # 下单 - Market Order 入场，等待成交后再设止损/止盈
    # ================================================================

    def execute_entry(
        self,
        ticker: str,
        action: str,
        shares: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Tuple[bool, str]:
        try:
            side = OrderSide.BUY if action == "BUY" else OrderSide.SELL

            order = MarketOrderRequest(
                symbol=ticker,
                qty=shares,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            result = self.client.submit_order(order)
            order_id = str(result.id)

            # 等待入场单成交（最多 10 秒）
            filled_price = self._wait_for_fill(order_id, timeout=10)
            if filled_price is None:
                print(f"[执行] {ticker} 入场单未在 10 秒内成交，取消")
                try:
                    self.client.cancel_order_by_id(order_id)
                except Exception:
                    pass
                return False, "入场单超时未成交"

            actual_entry = filled_price

            self.active_trades[ticker] = {
                "order_id": order_id,
                "action": action,
                "shares": shares,
                "entry_price": actual_entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": datetime.now(ET).isoformat(),
                "status": "filled",
            }

            # 入场确认后，设置止损和止盈
            is_long = action == "BUY"
            self._set_stop_and_profit(ticker, stop_loss, take_profit, shares, is_long)

            self._log_trade("ENTRY", ticker, {
                "action": action, "shares": shares,
                "requested_price": entry_price, "filled_price": actual_entry,
                "stop_loss": stop_loss, "take_profit": take_profit,
                "order_id": order_id,
            })

            print(f"[执行] {action} {shares}股 {ticker} 成交@${actual_entry:.2f} | 止损${stop_loss:.2f} | 止盈${take_profit:.2f}")
            return True, f"已成交: {order_id} @${actual_entry:.2f}"

        except Exception as e:
            print(f"[执行] 下单失败 ({ticker}): {e}")
            return False, str(e)

    def _wait_for_fill(self, order_id: str, timeout: int = 10) -> Optional[float]:
        """等待订单成交，返回成交价"""
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
            _time.sleep(0.5)
        return None

    def _set_stop_and_profit(self, ticker: str, stop_loss: float, take_profit: float,
                              shares: int, is_long: bool):
        """入场确认后设置止损和止盈"""
        exit_side = OrderSide.SELL if is_long else OrderSide.BUY

        # 止损: Stop Order (不用 StopLimit 避免极端行情不成交)
        try:
            stop_order = StopOrderRequest(
                symbol=ticker,
                qty=shares,
                side=exit_side,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_loss, 2),
            )
            result = self.client.submit_order(stop_order)
            self._stop_order_ids[ticker] = str(result.id)
            print(f"[执行] 止损已设置: {ticker} @ ${stop_loss:.2f}")
        except Exception as e:
            print(f"[执行] 止损设置失败 ({ticker}): {e}")

        # 止盈: Limit Order
        try:
            tp_order = LimitOrderRequest(
                symbol=ticker,
                qty=shares,
                side=exit_side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(take_profit, 2),
            )
            self.client.submit_order(tp_order)
            print(f"[执行] 止盈已设置: {ticker} @ ${take_profit:.2f}")
        except Exception as e:
            print(f"[执行] 止盈设置失败 ({ticker}): {e}")

    # ================================================================
    # 移动止损 - 同步更新 broker 端
    # ================================================================

    def update_trailing_stop(self, ticker: str, current_price: float):
        if ticker not in self.active_trades:
            return

        trade = self.active_trades[ticker]
        entry = trade["entry_price"]
        is_long = trade["action"] == "BUY"

        if is_long:
            pnl_pct = (current_price - entry) / entry * 100
            if pnl_pct >= 1.0 and trade["stop_loss"] < entry:
                new_stop = round(entry * 1.002, 2)
                self._replace_stop_order(ticker, new_stop, trade["shares"], is_long)
                trade["stop_loss"] = new_stop
        else:
            pnl_pct = (entry - current_price) / entry * 100
            if pnl_pct >= 1.0 and trade["stop_loss"] > entry:
                new_stop = round(entry * 0.998, 2)
                self._replace_stop_order(ticker, new_stop, trade["shares"], is_long)
                trade["stop_loss"] = new_stop

    def _replace_stop_order(self, ticker: str, new_stop: float, shares: int, is_long: bool):
        """取消旧止损单，下新止损单"""
        old_id = self._stop_order_ids.get(ticker)
        if old_id:
            try:
                self.client.cancel_order_by_id(old_id)
            except Exception:
                pass

        try:
            exit_side = OrderSide.SELL if is_long else OrderSide.BUY
            stop_order = StopOrderRequest(
                symbol=ticker,
                qty=shares,
                side=exit_side,
                time_in_force=TimeInForce.DAY,
                stop_price=round(new_stop, 2),
            )
            result = self.client.submit_order(stop_order)
            self._stop_order_ids[ticker] = str(result.id)
            print(f"[执行] 移动止损: {ticker} 新止损 ${new_stop:.2f}")
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
                del self._stop_order_ids[ticker]
            print(f"[执行] 已平仓: {ticker}")
            return True, f"{ticker} 已平仓"
        except Exception as e:
            print(f"[执行] 平仓失败 ({ticker}): {e}")
            return False, str(e)

    def close_all_positions(self, skip_tickers: set = None) -> List[str]:
        """平仓所有持仓（跳过 SGOV 等指定标的）"""
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
    # 尾盘强制平仓（美东时间）
    # ================================================================

    def check_force_close(self) -> bool:
        now = datetime.now(ET)
        h, m = map(int, FORCE_CLOSE_TIME.split(":"))
        force_time = now.replace(hour=h, minute=m, second=0, microsecond=0)

        if now >= force_time:
            positions = self.get_positions()
            active_positions = [p for p in positions if p["ticker"] not in SKIP_TICKERS]
            if active_positions:
                print(f"\n{'='*50}")
                print(f"⚠️ {FORCE_CLOSE_TIME} ET 尾盘强制平仓！")
                print(f"{'='*50}")
                self.cancel_pending_orders()
                closed = self.close_all_positions()
                for ticker in closed:
                    self._log_trade("FORCE_CLOSE", ticker, {"reason": "尾盘强制平仓"})
                return True
        return False

    # ================================================================
    # 检查已平仓的交易，返回实际 PnL
    # ================================================================

    def check_closed_trades(self) -> List[Dict]:
        """检查 active_trades 中已被 broker 平仓的（止损/止盈成交），返回 PnL"""
        closed = []
        positions = self.get_positions()
        held_tickers = {p["ticker"] for p in positions}

        for ticker, trade in list(self.active_trades.items()):
            if trade["status"] == "closed":
                continue
            if ticker not in held_tickers and trade["status"] == "filled":
                entry = trade["entry_price"]
                shares = trade["shares"]
                is_long = trade["action"] == "BUY"

                # 从 broker 获取最近的成交订单来推算平仓价
                exit_price = self._get_exit_price(ticker)
                if exit_price:
                    pnl = (exit_price - entry) * shares if is_long else (entry - exit_price) * shares
                else:
                    pnl = 0

                trade["status"] = "closed"
                trade["exit_price"] = exit_price
                trade["pnl"] = pnl

                # 清理止损单
                if ticker in self._stop_order_ids:
                    try:
                        self.client.cancel_order_by_id(self._stop_order_ids[ticker])
                    except Exception:
                        pass
                    del self._stop_order_ids[ticker]

                closed.append({"ticker": ticker, "pnl": pnl, "exit_price": exit_price, "trade": trade})
                self._log_trade("EXIT", ticker, {"pnl": pnl, "exit_price": exit_price, "action": trade["action"]})
                print(f"[执行] {ticker} 已平仓 | PnL: ${pnl:+.2f}")

        return closed

    def _get_exit_price(self, ticker: str) -> Optional[float]:
        """获取最近一笔成交的平仓价格"""
        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[ticker],
                limit=5,
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

    def check_order_status(self, order_id: str) -> str:
        try:
            order = self.client.get_order_by_id(order_id)
            return str(order.status)
        except Exception:
            return "unknown"

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
            "type": trade_type,
            "ticker": ticker,
            **details,
        }
        logs = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        logs.append(entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2, default=str)
