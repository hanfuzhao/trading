"""
PDT Tracker - Rolling 5 trading day window day trade count
Iron rule: never allow a 4th day trade under any circumstance
"""
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict
from zoneinfo import ZoneInfo

from core.config import PDT_MAX_DAY_TRADES, PDT_WINDOW_DAYS, LOG_DIR

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
        """Go back n trading days (skip weekends)"""
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
        """Record a day trade (same-day open and close)"""
        trade = {
            "date": datetime.now(ET).isoformat(),
            "ticker": ticker,
            "pnl": round(pnl, 2),
        }
        self.trades.append(trade)
        self._save()
        print(f"[PDT] Day trade recorded: {ticker} PnL=${pnl:+.2f} | Remaining slots: {self.remaining_trades()}")

    def next_trade_unlock(self) -> str:
        window_trades = self.get_trades_in_window()
        if len(window_trades) < PDT_MAX_DAY_TRADES:
            return "Slots available now"
        oldest = min(window_trades, key=lambda t: t["date"])
        oldest_date = datetime.fromisoformat(oldest["date"])
        unlock_date = oldest_date + timedelta(days=7)
        return f"Estimated {unlock_date.strftime('%m/%d %A')} 1 slot unlocks"

    def status(self) -> str:
        remaining = self.remaining_trades()
        if remaining == 0:
            return f"Day trade slots exhausted (0/{PDT_MAX_DAY_TRADES}) | {self.next_trade_unlock()}"
        elif remaining == 1:
            return f"Only 1 day trade slot left (ultra-selective mode)"
        else:
            return f"{remaining}/{PDT_MAX_DAY_TRADES} day trade slots remaining"
