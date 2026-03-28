"""
风险管理器 v6 — 3变量Regime矩阵 | 隔夜/日内分离 | 板块关联 | 日亏损2%
"""
import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Tuple, List
from zoneinfo import ZoneInfo

from config import (
    REGIME_PARAMS, MAX_CONCURRENT_POSITIONS, MAX_SAME_SECTOR_OVERNIGHT,
    MAX_DAILY_LOSS_PCT, MAX_WEEKLY_LOSS_PCT,
    CONSECUTIVE_LOSS_COOLDOWN, COOLDOWN_HOURS, LOG_DIR,
    CATASTROPHIC_STOP_PCT,
)

ET = ZoneInfo("America/New_York")


class RiskManager:
    def __init__(self):
        self.daily_pnl: float = 0
        self.today: date = datetime.now(ET).date()
        self.trade_count_today: int = 0
        self.consecutive_losses: int = 0
        self.weekly_pnl: float = 0
        self.cooldown_until: Optional[datetime] = None
        self.consecutive_loss_weeks: int = 0
        self.half_size: bool = False
        self.paused_until: Optional[date] = None
        self._load_state()

    def _state_file(self) -> str:
        os.makedirs(LOG_DIR, exist_ok=True)
        return os.path.join(LOG_DIR, "risk_state.json")

    def _load_state(self):
        path = self._state_file()
        if os.path.exists(path):
            with open(path, "r") as f:
                s = json.load(f)
                if s.get("date") == str(datetime.now(ET).date()):
                    self.daily_pnl = s.get("daily_pnl", 0)
                    self.trade_count_today = s.get("trade_count_today", 0)
                self.consecutive_losses = s.get("consecutive_losses", 0)
                self.weekly_pnl = s.get("weekly_pnl", 0)
                self.consecutive_loss_weeks = s.get("consecutive_loss_weeks", 0)
                self.half_size = s.get("half_size", False)
                cd = s.get("cooldown_until")
                if cd:
                    self.cooldown_until = datetime.fromisoformat(cd)
                pu = s.get("paused_until")
                if pu:
                    self.paused_until = date.fromisoformat(pu)

    def _save_state(self):
        with open(self._state_file(), "w") as f:
            json.dump({
                "date": str(datetime.now(ET).date()),
                "daily_pnl": self.daily_pnl,
                "trade_count_today": self.trade_count_today,
                "consecutive_losses": self.consecutive_losses,
                "weekly_pnl": self.weekly_pnl,
                "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
                "consecutive_loss_weeks": self.consecutive_loss_weeks,
                "half_size": self.half_size,
                "paused_until": self.paused_until.isoformat() if self.paused_until else None,
            }, f, indent=2)

    # ================================================================
    # 交易前检查
    # ================================================================

    def can_trade(self, portfolio_value: float) -> Tuple[bool, str]:
        today_et = datetime.now(ET).date()
        if today_et != self.today:
            if today_et.weekday() == 0:
                if self.weekly_pnl < 0:
                    self.consecutive_loss_weeks += 1
                else:
                    self.consecutive_loss_weeks = 0
                    self.half_size = False
                if self.weekly_pnl < -(portfolio_value * MAX_WEEKLY_LOSS_PCT / 100):
                    self.half_size = True
                if self.consecutive_loss_weeks >= 2:
                    self.paused_until = today_et + timedelta(days=7)
                self.weekly_pnl = 0
            self.daily_pnl = 0
            self.trade_count_today = 0
            self.today = today_et
            self.cooldown_until = None
            self._save_state()

        if self.paused_until and today_et < self.paused_until:
            return False, f"⛔ 连续亏损暂停至 {self.paused_until}"

        if self.cooldown_until:
            now = datetime.now(ET)
            if now < self.cooldown_until:
                remaining = (self.cooldown_until - now).total_seconds() / 60
                return False, f"⛔ 冷却中，{remaining:.0f}分钟后恢复"
            else:
                self.cooldown_until = None
                self.consecutive_losses = 0
                self._save_state()

        daily_limit = portfolio_value * MAX_DAILY_LOSS_PCT / 100
        if self.daily_pnl <= -daily_limit:
            return False, f"⛔ 日亏损${abs(self.daily_pnl):.2f}达上限"

        if self.consecutive_losses >= CONSECUTIVE_LOSS_COOLDOWN:
            self.cooldown_until = datetime.now(ET) + timedelta(hours=COOLDOWN_HOURS)
            self._save_state()
            return False, f"⛔ 连续{self.consecutive_losses}笔亏损，暂停{COOLDOWN_HOURS}h"

        return True, "✅ 可以交易"

    # ================================================================
    # 隔夜仓位校验
    # ================================================================

    def validate_overnight(
        self, ticker: str, price: float, portfolio_value: float,
        current_overnight_tickers: List[str], regime: str,
        sector: str = "Unknown",
    ) -> Tuple[bool, str, Dict]:
        params = REGIME_PARAMS.get(regime, REGIME_PARAMS["cautious"])

        if len(current_overnight_tickers) >= 3:
            return False, "⛔ 隔夜持仓已满3只", {}

        same_sector = sum(1 for t in current_overnight_tickers if t == sector)
        if same_sector >= MAX_SAME_SECTOR_OVERNIGHT:
            return False, f"⛔ 同板块隔夜已达{MAX_SAME_SECTOR_OVERNIGHT}只", {}

        max_pos_pct = params["on_max_pos_pct"]
        if self.half_size:
            max_pos_pct /= 2

        max_value = portfolio_value * max_pos_pct / 100
        shares = int(max_value / price) if price > 0 else 0
        if shares <= 0:
            return False, "⛔ 资金不足", {}

        actual_pos_pct = (shares * price / portfolio_value) * 100
        risk_per_share = price * CATASTROPHIC_STOP_PCT / 100
        total_risk_pct = (risk_per_share * shares / portfolio_value) * 100

        return True, "✅ 隔夜订单通过", {
            "ticker": ticker, "shares": shares,
            "entry_price": price,
            "position_pct": round(actual_pos_pct, 2),
            "risk_pct": round(total_risk_pct, 2),
        }

    # ================================================================
    # 日内仓位校验
    # ================================================================

    def validate_intraday(
        self, ticker: str, price: float, stop_loss: float,
        portfolio_value: float, current_positions: int, regime: str,
    ) -> Tuple[bool, str, Dict]:
        params = REGIME_PARAMS.get(regime, REGIME_PARAMS["cautious"])

        if current_positions >= MAX_CONCURRENT_POSITIONS:
            return False, f"⛔ 总持仓已达{MAX_CONCURRENT_POSITIONS}", {}

        if stop_loss <= 0 or stop_loss >= price:
            return False, "⛔ 无效止损", {}

        stop_dist_pct = (price - stop_loss) / price * 100
        if stop_dist_pct > 5:
            return False, f"⛔ 止损距离{stop_dist_pct:.1f}%太大", {}

        risk_pct = params["id_risk_pct"]
        if self.half_size:
            risk_pct /= 2
        risk_amount = portfolio_value * risk_pct / 100
        stop_dist = price - stop_loss
        shares_by_risk = int(risk_amount / stop_dist) if stop_dist > 0 else 0

        max_pos_pct = params["id_max_pos_pct"]
        if self.half_size:
            max_pos_pct /= 2
        shares_by_pos = int(portfolio_value * max_pos_pct / 100 / price) if price > 0 else 0

        shares = min(shares_by_risk, shares_by_pos)
        if shares <= 0:
            return False, "⛔ 风险限制下无法交易", {}

        return True, "✅ 日内订单通过", {
            "ticker": ticker, "shares": shares,
            "entry_price": price, "stop_loss": stop_loss,
        }

    # ================================================================
    # 日内止损/止盈计算
    # ================================================================

    def calculate_intraday_stops(self, price: float, atr: float, regime: str) -> Dict:
        params = REGIME_PARAMS.get(regime, REGIME_PARAMS["cautious"])
        mult = params["id_atr_mult"]
        stop_dist = atr * mult
        stop_loss = round(price - stop_dist, 2)
        tp1 = round(price + stop_dist, 2)
        tp2 = round(price + stop_dist * 2, 2)
        return {"stop_loss": stop_loss, "take_profit_1": tp1, "take_profit_2": tp2, "stop_distance": round(stop_dist, 2)}

    # ================================================================
    # 记录
    # ================================================================

    def record_trade_result(self, pnl: float):
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.trade_count_today += 1
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            self.cooldown_until = None
        self._save_state()

    def status(self, portfolio_value: float) -> str:
        limit = portfolio_value * MAX_DAILY_LOSS_PCT / 100
        s = f"日PnL${self.daily_pnl:+.2f}(限-${limit:.0f}) | 连亏{self.consecutive_losses} | 周PnL${self.weekly_pnl:+.2f}"
        if self.half_size:
            s += " | ⚠️减仓"
        if self.cooldown_until:
            s += f" | 冷却至{self.cooldown_until.strftime('%H:%M')}"
        return s
