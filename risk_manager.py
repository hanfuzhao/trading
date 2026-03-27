"""
风险管理器 v3 — 硬编码规则，AI不可覆盖
4级regime动态调仓 | 日亏损3%停 | 连输3笔暂停2h | 周亏损5%减仓 | 连续2周亏损暂停1周
"""
import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from config import (
    VOL_REGIMES, MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_LOSS_PCT, MAX_WEEKLY_LOSS_PCT,
    CONSECUTIVE_LOSS_COOLDOWN, COOLDOWN_HOURS,
    CONSECUTIVE_WEEK_LOSS_PAUSE, LOG_DIR, LONG_ONLY,
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
                saved_date = s.get("date", "")
                if saved_date == str(datetime.now(ET).date()):
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
    # 波动率动态参数
    # ================================================================

    def get_vol_params(self, vol_regime: str = "medium") -> Dict:
        return VOL_REGIMES.get(vol_regime, VOL_REGIMES["medium"])

    # ================================================================
    # 交易前检查（硬规则，不可跳过）
    # ================================================================

    def can_trade(self, portfolio_value: float) -> Tuple[bool, str]:
        today_et = datetime.now(ET).date()

        # 新的一天重置日数据
        if today_et != self.today:
            # 周一：重置周数据
            if today_et.weekday() == 0:
                if self.weekly_pnl < 0:
                    self.consecutive_loss_weeks += 1
                else:
                    self.consecutive_loss_weeks = 0
                    self.half_size = False

                # 周亏损 > 5% → 下周减仓
                if self.weekly_pnl < -(portfolio_value * MAX_WEEKLY_LOSS_PCT / 100):
                    self.half_size = True

                # 连续2周亏损 → 暂停1周
                if self.consecutive_loss_weeks >= CONSECUTIVE_WEEK_LOSS_PAUSE:
                    self.paused_until = today_et + timedelta(days=7)

                self.weekly_pnl = 0

            self.daily_pnl = 0
            self.trade_count_today = 0
            self.today = today_et
            self.cooldown_until = None
            self._save_state()

        # 周暂停检查
        if self.paused_until and today_et < self.paused_until:
            return False, f"⛔ 连续{self.consecutive_loss_weeks}周亏损，暂停至 {self.paused_until}"

        # 冷却检查（连输3笔暂停2小时）
        if self.cooldown_until:
            now = datetime.now(ET)
            if now < self.cooldown_until:
                remaining = (self.cooldown_until - now).total_seconds() / 60
                return False, f"⛔ 连续{self.consecutive_losses}笔亏损冷却中，{remaining:.0f}分钟后恢复"
            else:
                self.cooldown_until = None
                self.consecutive_losses = 0
                self._save_state()

        # 日亏损 > 3%
        daily_loss_limit = portfolio_value * MAX_DAILY_LOSS_PCT / 100
        if self.daily_pnl <= -daily_loss_limit:
            return False, f"⛔ 日亏损${abs(self.daily_pnl):.2f}（上限${daily_loss_limit:.2f}），今日停止"

        # 连输3笔 → 触发冷却
        if self.consecutive_losses >= CONSECUTIVE_LOSS_COOLDOWN:
            self.cooldown_until = datetime.now(ET) + timedelta(hours=COOLDOWN_HOURS)
            self._save_state()
            return False, f"⛔ 连续{self.consecutive_losses}笔亏损，暂停{COOLDOWN_HOURS}小时"

        return True, "✅ 可以交易"

    # ================================================================
    # 订单校验（波动率自适应仓位）
    # ================================================================

    def validate_order(
        self, ticker: str, price: float, stop_loss: float,
        position_size_pct: float, portfolio_value: float,
        current_positions: int, vol_regime: str = "medium",
    ) -> Tuple[bool, str, Dict]:
        params = self.get_vol_params(vol_regime)
        adjusted = {}

        if LONG_ONLY and stop_loss >= price:
            return False, "⛔ 只做多，止损必须低于入场价", {}

        if current_positions >= MAX_CONCURRENT_POSITIONS:
            return False, f"⛔ 已有{current_positions}个持仓（上限{MAX_CONCURRENT_POSITIONS}）", {}

        if stop_loss <= 0:
            return False, "⛔ 必须设置止损", {}

        stop_dist_pct = abs(price - stop_loss) / price * 100
        if stop_dist_pct > 5:
            return False, f"⛔ 止损距离{stop_dist_pct:.1f}%太大（日内上限5%）", {}

        # 波动率调整仓位上限
        max_pos = params["max_pos_pct"]
        if self.half_size:
            max_pos = max_pos / 2
        max_pos = min(position_size_pct, max_pos)

        # 基于风险的股数计算
        risk_pct = params["risk_pct"]
        if self.half_size:
            risk_pct = risk_pct / 2
        risk_amount = portfolio_value * risk_pct / 100
        stop_distance = abs(price - stop_loss)
        shares_by_risk = int(risk_amount / stop_distance) if stop_distance > 0 else 0

        # 基于仓位上限的股数
        shares_by_pos = int(portfolio_value * max_pos / 100 / price) if price > 0 else 0

        shares = min(shares_by_risk, shares_by_pos)
        if shares <= 0:
            return False, "⛔ 风险限制下无法买入1股", {}

        actual_pos_pct = (shares * price / portfolio_value) * 100
        actual_risk_pct = (stop_distance * shares / portfolio_value) * 100

        if max_pos != position_size_pct or shares != shares_by_pos:
            adjusted["reason"] = f"regime={vol_regime}, 调整至{shares}股 (风险{actual_risk_pct:.2f}%, 仓位{actual_pos_pct:.1f}%)"

        result = {
            "ticker": ticker, "action": "BUY",
            "shares": shares,
            "entry_price": price, "stop_loss": stop_loss,
            "position_size_pct": round(actual_pos_pct, 2),
            "risk_pct": round(actual_risk_pct, 2),
            "vol_regime": vol_regime,
            **adjusted,
        }
        return True, "✅ 订单通过风险检查", result

    # ================================================================
    # 止损/止盈计算（v3: TP1=1R, TP2=2R）
    # ================================================================

    def calculate_stops(self, price: float, atr: float, vol_regime: str = "medium") -> Dict:
        params = self.get_vol_params(vol_regime)
        mult = params["atr_mult"]

        stop_distance = atr * mult
        stop_loss = round(price - stop_distance, 2)
        take_profit_1 = round(price + stop_distance * 1.0, 2)
        take_profit_2 = round(price + stop_distance * 2.0, 2)

        rr = 2.0

        return {
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "stop_distance": round(stop_distance, 2),
            "risk_reward_ratio": rr,
        }

    # ================================================================
    # 交易结果记录
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
        daily_limit = portfolio_value * MAX_DAILY_LOSS_PCT / 100
        s = (
            f"日PnL: ${self.daily_pnl:+.2f} (上限-${daily_limit:.2f}) | "
            f"连续亏损: {self.consecutive_losses} | "
            f"今日交易: {self.trade_count_today}次 | "
            f"周PnL: ${self.weekly_pnl:+.2f}"
        )
        if self.half_size:
            s += " | ⚠️ 减仓模式"
        if self.cooldown_until:
            s += f" | 冷却至{self.cooldown_until.strftime('%H:%M')}"
        if self.paused_until:
            s += f" | 暂停至{self.paused_until}"
        return s
