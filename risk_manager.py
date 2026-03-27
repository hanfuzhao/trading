"""
风险管理器 - 硬编码规则，AI的任何建议都不能突破
"""
import json
import os
from datetime import datetime, date
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

from config import (
    MAX_POSITION_PCT, MAX_SINGLE_LOSS_PCT, MAX_DAILY_LOSS_PCT,
    MAX_CONCURRENT_POSITIONS, STOP_LOSS_ATR_MULTIPLIER,
    TAKE_PROFIT_ATR_MULTIPLIER, LOG_DIR,
)


class RiskManager:
    def __init__(self):
        self.daily_pnl: float = 0
        self.today: date = datetime.now(ET).date()
        self.trade_count_today: int = 0
        self.consecutive_losses: int = 0
        self.weekly_pnl: float = 0
        self._load_state()

    def _state_file(self) -> str:
        os.makedirs(LOG_DIR, exist_ok=True)
        return os.path.join(LOG_DIR, "risk_state.json")

    def _load_state(self):
        path = self._state_file()
        if os.path.exists(path):
            with open(path, "r") as f:
                state = json.load(f)
                saved_date = state.get("date", "")
                if saved_date == str(datetime.now(ET).date()):
                    self.daily_pnl = state.get("daily_pnl", 0)
                    self.trade_count_today = state.get("trade_count_today", 0)
                self.consecutive_losses = state.get("consecutive_losses", 0)
                self.weekly_pnl = state.get("weekly_pnl", 0)

    def _save_state(self):
        with open(self._state_file(), "w") as f:
            json.dump({
                "date": str(datetime.now(ET).date()),
                "daily_pnl": self.daily_pnl,
                "trade_count_today": self.trade_count_today,
                "consecutive_losses": self.consecutive_losses,
                "weekly_pnl": self.weekly_pnl,
            }, f, indent=2)

    # ================================================================
    # 交易前检查
    # ================================================================

    def can_trade(self, portfolio_value: float) -> Tuple[bool, str]:
        """综合检查是否允许交易"""

        today_et = datetime.now(ET).date()
        if today_et != self.today:
            # 周一重置周 PnL
            if today_et.weekday() == 0:
                self.weekly_pnl = 0
            self.daily_pnl = 0
            self.trade_count_today = 0
            self.today = today_et

        # 单日亏损上限
        daily_loss_limit = portfolio_value * MAX_DAILY_LOSS_PCT / 100
        if self.daily_pnl <= -daily_loss_limit:
            return False, f"⛔ 单日亏损已达${abs(self.daily_pnl):.2f}（上限${daily_loss_limit:.2f}），今日停止交易"

        # 连续亏损冷却
        if self.consecutive_losses >= 2:
            return False, f"⛔ 连续{self.consecutive_losses}笔亏损，今日冷却中"

        # 单日交易次数上限（包含非日内交易）
        if self.trade_count_today >= 6:
            return False, "⛔ 今日已交易6次，达到上限"

        return True, "✅ 可以交易"

    def validate_order(
        self,
        ticker: str,
        action: str,
        price: float,
        stop_loss: float,
        position_size_pct: float,
        portfolio_value: float,
        current_positions: int,
    ) -> Tuple[bool, str, Dict]:
        """
        验证单笔订单是否符合风险规则
        返回: (是否通过, 原因, 调整后的参数)
        """
        adjusted = {}

        # 最大持仓数
        if current_positions >= MAX_CONCURRENT_POSITIONS:
            return False, f"⛔ 已有{current_positions}个持仓（上限{MAX_CONCURRENT_POSITIONS}）", {}

        # 仓位上限
        if position_size_pct > MAX_POSITION_PCT:
            position_size_pct = MAX_POSITION_PCT
            adjusted["position_size_pct"] = MAX_POSITION_PCT

        # 计算实际风险
        position_value = portfolio_value * position_size_pct / 100
        shares = int(position_value / price)
        if shares <= 0:
            return False, "⛔ 仓位太小，无法购买1股", {}

        risk_per_share = abs(price - stop_loss)
        total_risk = risk_per_share * shares
        risk_pct = (total_risk / portfolio_value) * 100

        # 单笔亏损上限
        if risk_pct > MAX_SINGLE_LOSS_PCT:
            # 缩减股数使风险在限制内
            max_risk = portfolio_value * MAX_SINGLE_LOSS_PCT / 100
            shares = int(max_risk / risk_per_share)
            if shares <= 0:
                return False, "⛔ 止损距离太大，无法在风险限制内交易", {}
            position_size_pct = (shares * price / portfolio_value) * 100
            adjusted["shares"] = shares
            adjusted["position_size_pct"] = round(position_size_pct, 2)
            adjusted["reason"] = f"缩减至{shares}股以控制风险在{MAX_SINGLE_LOSS_PCT}%内"

        # 确保有止损
        if stop_loss <= 0:
            return False, "⛔ 必须设置止损", {}

        # 止损距离合理性检查
        stop_distance_pct = abs(price - stop_loss) / price * 100
        if stop_distance_pct > 5:
            return False, f"⛔ 止损距离{stop_distance_pct:.1f}%太大（日内交易上限5%）", {}

        result = {
            "ticker": ticker,
            "action": action,
            "shares": shares if "shares" in adjusted else int(position_value / price),
            "entry_price": price,
            "stop_loss": stop_loss,
            "position_size_pct": adjusted.get("position_size_pct", position_size_pct),
            "risk_pct": round(risk_pct if "shares" not in adjusted else
                              (abs(price - stop_loss) * adjusted.get("shares", shares) / portfolio_value) * 100, 2),
        }

        return True, "✅ 订单通过风险检查", result

    # ================================================================
    # 止损/止盈计算
    # ================================================================

    def calculate_stops(self, price: float, atr: float, action: str) -> Dict:
        """基于ATR计算止损止盈"""
        if action == "BUY":
            stop_loss = round(price - atr * STOP_LOSS_ATR_MULTIPLIER, 2)
            take_profit = round(price + atr * TAKE_PROFIT_ATR_MULTIPLIER, 2)
        else:  # SHORT
            stop_loss = round(price + atr * STOP_LOSS_ATR_MULTIPLIER, 2)
            take_profit = round(price - atr * TAKE_PROFIT_ATR_MULTIPLIER, 2)

        risk = abs(price - stop_loss)
        reward = abs(take_profit - price)
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": rr_ratio,
        }

    # ================================================================
    # 交易结果记录
    # ================================================================

    def record_trade_result(self, pnl: float):
        """记录交易结果，更新状态"""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.trade_count_today += 1

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self._save_state()

    def reset_weekly(self):
        """每周重置（周一调用）"""
        self.weekly_pnl = 0
        self.consecutive_losses = 0
        self._save_state()

    def status(self, portfolio_value: float) -> str:
        daily_limit = portfolio_value * MAX_DAILY_LOSS_PCT / 100
        return (
            f"日PnL: ${self.daily_pnl:+.2f} (上限 -${daily_limit:.2f}) | "
            f"连续亏损: {self.consecutive_losses} | "
            f"今日交易: {self.trade_count_today}次"
        )
