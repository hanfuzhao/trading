"""
o3 深度排名 v6 — Man Group 7变量 | 隔夜/日内分离推荐 | 新闻≤10%权重
"""
import json
import traceback
from datetime import datetime, date
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo
from openai import OpenAI

from config import (
    OPENAI_API_KEY, MODEL_RANK, MAX_O3_CALLS_PER_DAY,
    REGIME_PARAMS, MONDAY_EXCEPTION_SCORE,
)

ET = ZoneInfo("America/New_York")

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

WEEKDAY_GUIDANCE = {
    0: "周一：原则上不交易（VIX周一平均涨2.16%），用周一数据为周二做准备。例外：信号≥85分可入场。",
    1: "周二：优先使用1次名额（Turnaround Tuesday效应，均值回归最佳日）。",
    2: "周三：使用1次名额（历史回报最高、波动最低）。",
    3: "周四：保留1次给最强信号。如果周二已用且亏了→今天R:R门槛提高到>2.0。",
    4: "周五：如果周四没用，可稍放宽到R:R>1.3（名额即将重置）。如果到今天还剩3次→本周无好机会，不强行交易。",
}


class AIRanker:
    def __init__(self):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self._o3_calls_today: int = 0
        self._o3_call_date: Optional[date] = None

    def _o3_budget(self) -> bool:
        today = datetime.now(ET).date()
        if self._o3_call_date != today:
            self._o3_call_date = today
            self._o3_calls_today = 0
        return self._o3_calls_today < MAX_O3_CALLS_PER_DAY

    # ================================================================
    # 隔夜候选排名
    # ================================================================

    def rank_overnight(
        self,
        candidates: List[Dict],
        portfolio_value: float, cash: float,
        regime: str, man_group_vars: Dict,
        spy_change: float, uso_change: float,
    ) -> List[Dict]:
        if not candidates or not self._o3_budget():
            return []
        params = REGIME_PARAMS.get(regime, REGIME_PARAMS["cautious"])
        now = datetime.now(ET)
        weekday = WEEKDAY_NAMES[now.weekday()]
        day_guidance = WEEKDAY_GUIDANCE.get(now.weekday(), "")

        man_group_text = self._format_man_group(man_group_vars)
        candidates_text = self._format_overnight_candidates(candidates[:20])

        prompt = f"""你是一个管理${portfolio_value:,.0f}账户的隔夜均值回归投资组合经理。

== 策略说明 ==
隔夜均值回归策略：今天下午买入超卖股票，明天上午卖出。
核心条件：RSI(2)<15 + IBS<0.25
按NATR排序（波动越大回归空间越大）
胜率预期65%，每笔赢利0.9%，亏损0.5%
不设机械止损，用仓位大小控制风险
每晚最多持有3只

== 宏观环境（实时）==
Regime: {regime}（全面乐观/谨慎/防御/全面防御）
SPY今日: {spy_change:+.2f}%
USO今日: {uso_change:+.2f}%

{man_group_text}

== 约束条件 ==
账户总值: ${portfolio_value:,.2f}
可用现金: ${cash:,.2f}
单只最大仓位: 账户的{params['on_max_pos_pct']}%
单笔最大风险: 账户的{params['on_risk_pct']}%
今天是{weekday}
今日Regime: {regime}

== 候选股票（已通过RSI(2)<15 + IBS<0.25筛选）==
{candidates_text}

请做以下工作：
1. 评估每只候选的均值回归潜力（NATR、RSI(2)深度、IBS、连续下跌天数、板块）
2. 检查是否有结构性利空（财报、退市、重大诉讼等）——如果有，排除
3. 横向对比，选出最优的1-3只
4. 每只给出建议入场价（Limit Order）

返回纯JSON（不要markdown标记）：
{{
  "analysis_date": "{now.strftime('%Y-%m-%d')}",
  "regime": "{regime}",
  "recommendations": [
    {{
      "rank": 1,
      "ticker": "XXX",
      "action": "BUY",
      "confidence": 0到100,
      "entry_price": 数字,
      "position_size_pct": 数字,
      "natr": 数字,
      "reasoning": "完整推理"
    }}
  ],
  "skip_reason": "不交易的原因（如果不推荐任何股票）",
  "risk_warnings": ["风险1", "风险2"]
}}"""
        return self._call_o3(prompt, "overnight")

    # ================================================================
    # 日内候选排名
    # ================================================================

    def rank_intraday(
        self,
        candidates: List[Dict],
        portfolio_value: float, cash: float,
        remaining_day_trades: int,
        regime: str, man_group_vars: Dict,
        spy_change: float, uso_change: float,
    ) -> List[Dict]:
        if not candidates or not self._o3_budget():
            return []
        if remaining_day_trades <= 0:
            return []

        params = REGIME_PARAMS.get(regime, REGIME_PARAMS["cautious"])
        now = datetime.now(ET)
        weekday = WEEKDAY_NAMES[now.weekday()]
        weekday_num = now.weekday()
        day_guidance = WEEKDAY_GUIDANCE.get(weekday_num, "")

        if weekday_num == 0:
            top_score = max((c.get("signal_strength", 0) for c in candidates), default=0)
            if top_score < MONDAY_EXCEPTION_SCORE:
                print(f"[排名] 周一最高分{top_score}<{MONDAY_EXCEPTION_SCORE}，不使用PDT名额")
                return []

        man_group_text = self._format_man_group(man_group_vars)
        candidates_text = self._format_intraday_candidates(candidates[:15])

        prompt = f"""你是一个管理${portfolio_value:,.0f}账户的日内交易投资组合经理。

== 策略说明 ==
日内IV时机均值回归：14:00-14:45买入日内超卖股票，15:40前出场。
NYSE场内经纪人14:00收到收盘不平衡信息，触发反转。
持仓60-90分钟。必须15:40 Limit出场，15:48市价清仓。

== 宏观环境 ==
Regime: {regime}
SPY今日: {spy_change:+.2f}%
USO今日: {uso_change:+.2f}%

{man_group_text}

== 约束条件 ==
PDT剩余名额: {remaining_day_trades}次
账户总值: ${portfolio_value:,.2f}
可用现金: ${cash:,.2f}
单笔最大仓位: {params['id_max_pos_pct']}%
单笔最大亏损: {params['id_risk_pct']}%
止损: {params['id_atr_mult']}x ATR
今天是{weekday}

== PDT名额分配指引 ==
{day_guidance}
- 如果今天所有候选R:R都不到1.5，保留名额
- 核心问题："这个机会是不是本周最好的3个之一？"

== 候选股票 ==
{candidates_text}

门槛：
- R:R必须 > 1.5
- 置信度必须 > 65
- 最多推荐{min(remaining_day_trades, 1)}只

返回纯JSON：
{{
  "analysis_date": "{now.strftime('%Y-%m-%d')}",
  "remaining_trades": {remaining_day_trades},
  "regime": "{regime}",
  "recommendations": [
    {{
      "rank": 1,
      "ticker": "XXX",
      "action": "BUY",
      "confidence": 0到100,
      "entry_price": 数字,
      "stop_loss": 数字,
      "take_profit_1": 数字,
      "take_profit_2": 数字,
      "position_size_pct": 数字,
      "risk_reward_ratio": 数字,
      "reasoning": "完整推理"
    }}
  ],
  "save_bullets": true或false,
  "save_reason": "不交易的原因",
  "risk_warnings": ["风险1"]
}}"""
        recs = self._call_o3(prompt, "intraday")
        return self._validate_intraday_recs(recs, remaining_day_trades)

    # ================================================================
    # o3调用
    # ================================================================

    def _call_o3(self, prompt: str, call_type: str) -> List[Dict]:
        try:
            print(f"[排名] 调用o3 ({call_type})...")
            response = self.openai.chat.completions.create(
                model=MODEL_RANK,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
                response_format={"type": "json_object"},
            )
            self._o3_calls_today += 1
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            recs = result.get("recommendations", [])
            print(f"[排名] o3返回 {len(recs)} 条推荐 (今日第{self._o3_calls_today}次)")
            return recs
        except Exception as e:
            print(f"[排名] o3调用失败: {e}")
            traceback.print_exc()
            return []

    def _validate_intraday_recs(self, recs: List[Dict], remaining: int) -> List[Dict]:
        valid = []
        for r in recs:
            if r.get("action", "").upper() != "BUY":
                continue
            if r.get("confidence", 0) < 65:
                continue
            if r.get("risk_reward_ratio", 0) < 1.5:
                continue
            if r.get("stop_loss", 0) <= 0:
                continue
            if r.get("entry_price", 0) <= 0:
                continue
            entry = r["entry_price"]
            stop = r["stop_loss"]
            if (entry - stop) / entry > 0.05:
                continue
            valid.append(r)
            if len(valid) >= min(remaining, 1):
                break
        return valid

    # ================================================================
    # 格式化
    # ================================================================

    def _format_man_group(self, vars: Dict) -> str:
        if not vars:
            return "== Man Group类比变量 ==\n数据不可用"
        lines = ["== Man Group类比变量（当前值）=="]
        mappings = [
            ("spy_20d_return", "S&P 500 20日回报", "%"),
            ("tlt_ief_ratio", "收益率曲线代理(TLT/IEF)", ""),
            ("uso_20d_return", "原油(USO) 20日回报", "%"),
            ("cper_20d_return", "铜(CPER) 20日回报", "%"),
            ("shy_price", "短期利率代理(SHY)", ""),
            ("vixy_price", "股票波动率(VIXY)", ""),
            ("spy_tlt_corr", "股债相关性(SPY vs TLT 20日)", ""),
        ]
        for key, label, suffix in mappings:
            val = vars.get(key)
            if val is not None:
                lines.append(f"{label}: {val}{suffix}")
        return "\n".join(lines)

    def _format_overnight_candidates(self, candidates: List[Dict]) -> str:
        lines = []
        for i, c in enumerate(candidates, 1):
            ind = c.get("indicators", {})
            lines.append(
                f"{i}. {c['ticker']} | ${c['price']:.2f} | RSI(2)={c.get('rsi_2', '?')} | "
                f"IBS={c.get('ibs', '?')} | NATR={c.get('natr', '?')}% | "
                f"连跌{ind.get('consecutive_down', 0)}天 | 板块={c.get('sector', '?')}"
            )
        return "\n".join(lines)

    def _format_intraday_candidates(self, candidates: List[Dict]) -> str:
        lines = []
        for i, c in enumerate(candidates, 1):
            ind = c.get("indicators", {})
            lines.append(
                f"{i}. {c['ticker']} | ${c['price']:.2f} | 日内{c.get('intraday_change', '?')}% | "
                f"量比={c.get('volume_ratio', '?')}x | RSI(14)={ind.get('rsi_14', '?')} | "
                f"ATR=${ind.get('atr', '?')} | 板块={c.get('sector', '?')}"
            )
        return "\n".join(lines)
