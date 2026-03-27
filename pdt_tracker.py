"""
PDT追踪器 - 滚动5个交易日窗口内的日内交易计数
铁律：任何情况下不允许第4次日内交易
"""
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict
from zoneinfo import ZoneInfo

from config import PDT_MAX_DAY_TRADES, PDT_WINDOW_DAYS, LOG_DIR

ET = ZoneInfo("America/New_York")


class PDTTracker:
    def __init__(self):
        self.log_file = os.path.join(LOG_DIR, "pdt_trades.json")
        os.makedirs(LOG_DIR, exist_ok=True)
        self.trades: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)

    def _get_trading_days(self, n: int) -> datetime:
        """回溯n个交易日（跳过周末）"""
        dt = datetime.now(ET)
        count = 0
        while count < n:
            dt -= timedelta(days=1)
            if dt.weekday() < 5:
                count += 1
        return dt

    def get_trades_in_window(self) -> List[Dict]:
        window_start = self._get_trading_days(PDT_WINDOW_DAYS)
        return [
            t for t in self.trades
            if datetime.fromisoformat(t["date"]).replace(tzinfo=ET) >= window_start
        ]

    def can_day_trade(self) -> bool:
        return len(self.get_trades_in_window()) < PDT_MAX_DAY_TRADES

    def remaining_trades(self) -> int:
        return PDT_MAX_DAY_TRADES - len(self.get_trades_in_window())

    def record_day_trade(self, ticker: str, pnl: float = 0):
        """记录一笔日内交易（同日开平仓）"""
        trade = {
            "date": datetime.now(ET).isoformat(),
            "ticker": ticker,
            "pnl": round(pnl, 2),
        }
        self.trades.append(trade)
        self._save()
        print(f"[PDT] 已记录日内交易: {ticker} PnL=${pnl:+.2f} | 剩余名额: {self.remaining_trades()}")

    def next_trade_unlock(self) -> str:
        window_trades = self.get_trades_in_window()
        if len(window_trades) < PDT_MAX_DAY_TRADES:
            return "现在就有名额"
        oldest = min(window_trades, key=lambda t: t["date"])
        oldest_date = datetime.fromisoformat(oldest["date"])
        unlock_date = oldest_date + timedelta(days=7)
        return f"预计 {unlock_date.strftime('%m/%d %A')} 恢复1个名额"

    def status(self) -> str:
        remaining = self.remaining_trades()
        if remaining == 0:
            return f"⛔ 日内交易名额用完 (0/{PDT_MAX_DAY_TRADES}) | {self.next_trade_unlock()}"
        elif remaining == 1:
            return f"⚠️ 仅剩1次日内交易名额 (极度挑剔模式)"
        else:
            return f"✅ 剩余 {remaining}/{PDT_MAX_DAY_TRADES} 次日内交易名额"
