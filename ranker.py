"""
o3 深度排名系统 v3
只做多 | USO宏观context | 完整策略指引 | 动态名额分配
每天最多3次调用
"""
import json
from datetime import datetime
from typing import List, Dict
from zoneinfo import ZoneInfo
from openai import OpenAI

from config import (
    OPENAI_API_KEY, MODEL_RANK, VOL_REGIMES,
    MONDAY_MIN_SCORE, MAX_O3_CALLS_PER_DAY,
)

ET = ZoneInfo("America/New_York")

WEEKDAY_GUIDANCE = {
    0: (
        "周一：不交易。VIX 周一平均涨 2.16%，投机股回报最低。"
        "只观察不出手。除非出现 ≥95 分的极端信号。"
    ),
    1: (
        "周二：优先使用 1 次名额。Turnaround Tuesday 效应——"
        "周一弱势股周二反弹，均值回归最佳日。"
    ),
    2: "周三：使用 1 次名额。历史平均回报最高、波动最低。",
    3: "周四：保留 1 次给最强信号。临近周五名额重置，可以稍放宽。",
    4: (
        "周五：名额即将重置。如果还剩名额可以用在强信号上；"
        "但避免持仓过周末。"
    ),
}


class DeepRanker:
    def __init__(self):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self._o3_calls_today: int = 0
        self._o3_call_date: str = ""

    def _check_o3_budget(self) -> bool:
        today = datetime.now(ET).strftime("%Y-%m-%d")
        if today != self._o3_call_date:
            self._o3_call_date = today
            self._o3_calls_today = 0
        if self._o3_calls_today >= MAX_O3_CALLS_PER_DAY:
            print(f"[排名] o3 今日已调用 {self._o3_calls_today} 次（上限{MAX_O3_CALLS_PER_DAY}），跳过")
            return False
        return True

    def rank_candidates(
        self,
        candidates: List[Dict],
        remaining_day_trades: int,
        portfolio_value: float,
        cash: float,
        current_positions: List[Dict],
        weekday: str,
        remaining_trading_days: int,
        vol_regime: str = "medium",
        scan_mode: str = "momentum",
        spy_change_pct: float = 0,
        uso_change_pct: float = 0,
    ) -> Dict:
        if not candidates:
            return {"recommendations": [], "save_bullets": True, "save_reason": "没有候选股票"}
        if remaining_day_trades <= 0:
            return {"recommendations": [], "save_bullets": True, "save_reason": "日内交易名额已用完"}

        now = datetime.now(ET)
        weekday_num = now.weekday()

        # 周一不交易（除非极端信号）
        if weekday_num == 0:
            top_score = max((c.get("signal_strength", 0) for c in candidates), default=0)
            if top_score < MONDAY_MIN_SCORE:
                return {
                    "recommendations": [], "save_bullets": True,
                    "save_reason": f"周一策略性不交易（最高分{top_score}，需≥{MONDAY_MIN_SCORE}）",
                }

        if not self._check_o3_budget():
            return {"recommendations": [], "save_bullets": True, "save_reason": "o3 每日调用次数已达上限"}

        candidates_text = self._format_candidates(candidates)
        positions_text = json.dumps(current_positions, indent=2, ensure_ascii=False) if current_positions else "无持仓"

        params = VOL_REGIMES.get(vol_regime, VOL_REGIMES["medium"])
        strategy_pref = (
            "均值回归为主（VIX 高位）" if scan_mode == "mean_reversion"
            else "动量突破为主（VIX 低位）"
        )
        day_guidance = WEEKDAY_GUIDANCE.get(weekday_num, "")

        prompt = f"""你是一个管理${portfolio_value:,.0f}账户的日内交易投资组合经理。

== 宏观环境（实时）==
SPY今日: {spy_change_pct:+.2f}%
波动率regime: {vol_regime}（低/中/高/极端）
油价代理(USO)今日: {uso_change_pct:+.2f}%
美伊战争状态: 持续中，霍尔木兹海峡通行受阻
美联储: 利率不变，2026年可能仅降息1次
市场周期: 中期选举年，历史上4-8月波动最大

== 约束条件 ==
PDT剩余名额: {remaining_day_trades}次（5个交易日滚动窗口内最多3次）
账户总值: ${portfolio_value:,.2f}
可用现金: ${cash:,.2f}
单笔最大仓位: 账户的{params['max_pos_pct']}%（已根据波动率调整）
单笔最大亏损: 账户的{params['risk_pct']}%（已根据波动率调整）
止损距离: {params['atr_mult']}x ATR（已根据波动率调整）
今天是{weekday}，本周还剩{remaining_trading_days}个交易日
当前持仓: {positions_text}

== 策略指引 ==
当前regime为{vol_regime}：
{"- 高/极端波动 → 优先均值回归（过度下跌反弹）" if scan_mode == "mean_reversion" else "- 低/中等波动 → 优先动量突破（趋势延续）"}
{"- 极度悲观的新闻情绪是加分项（市场过度恐慌 = 均值回归入场机会）" if scan_mode == "mean_reversion" else ""}
优先板块：能源 E&P > 防务 > 网络安全 > 医疗保健
回避板块：消费可选、商业地产、航空
日内交易最佳时间：9:45-10:15 AM 和 3:00-3:30 PM
只做多（不做空）

== PDT名额分配指引 ==
{day_guidance}
- FOMC 会议日/重大事件日可以例外集中使用
- 如果周二用了1次且亏了 → 周三的门槛从R:R>1.5提高到R:R>2.0
- 如果本周已用2次且都赚了 → 第3次可以稍放宽到R:R>1.3
- 如果到周四还剩3次没用 → 说明本周确实没好机会，不要强行交易

== 候选股票 ==
{candidates_text}

请做以下工作：
1. 对每只候选评估：预期日内收益率、最大亏损风险、R:R比、时效性、流动性
2. 标注策略类型（mean_reversion / momentum）
3. 横向对比排出优先级
4. 考虑名额分配：这个机会值不值得用掉一次PDT名额？
5. 如果没有R:R > 1.5的机会，明确说不交易

门槛：
- 风险回报比必须 > 1.5
- 置信度必须 > 65
- 最多推荐{min(remaining_day_trades, 2)}只
- 每只必须标明 strategy_type

返回纯JSON（不要markdown标记）：
{{
  "analysis_date": "{now.strftime('%Y-%m-%d')}",
  "remaining_trades": {remaining_day_trades},
  "vol_regime": "{vol_regime}",
  "market_context": "一段话描述今天市场环境",
  "recommendations": [
    {{
      "rank": 1,
      "ticker": "XXX",
      "strategy_type": "mean_reversion" 或 "momentum",
      "action": "BUY",
      "confidence": 0到100,
      "entry_price": 数字,
      "stop_loss": 数字,
      "take_profit_1": 数字,
      "take_profit_2": 数字,
      "position_size_pct": 数字,
      "risk_reward_ratio": 数字,
      "time_window": "建议入场时段",
      "reasoning": "完整推理"
    }}
  ],
  "save_bullets": true或false,
  "save_reason": "不交易的原因",
  "risk_warnings": ["风险1", "风险2"]
}}"""

        try:
            response = self.openai.chat.completions.create(
                model=MODEL_RANK,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000,
            )
            self._o3_calls_today += 1

            raw = response.choices[0].message.content.strip()
            reasoning_summary = getattr(response.choices[0].message, "reasoning", None)

            cleaned = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
            result["model"] = MODEL_RANK
            result["reasoning_summary"] = reasoning_summary
            result["raw_response"] = raw[:500]

            result["recommendations"] = self._validate_recommendations(
                result.get("recommendations", []),
                portfolio_value, remaining_day_trades, vol_regime,
            )
            return result

        except Exception as e:
            print(f"[排名] o3分析失败: {e}")
            return {
                "recommendations": [], "save_bullets": True,
                "save_reason": f"o3分析失败: {str(e)}", "error": str(e),
            }

    def _format_candidates(self, candidates: List[Dict]) -> str:
        texts = []
        for i, c in enumerate(candidates[:20]):
            ind = c.get("indicators", {})
            news = c.get("news_analysis", {})
            text = (
                f"\n--- 候选 #{i+1}: {c['ticker']} ---\n"
                f"价格: ${c['price']:.2f} | 板块: {c.get('sector', 'N/A')} | Beta: {c.get('beta', 'N/A')}\n"
                f"扫描模式: {c.get('scan_mode', 'N/A')} | 信号强度: {c['signal_strength']}\n"
                f"技术信号: {', '.join(c.get('signals', []))}\n"
                f"RSI(14): {ind.get('rsi', 'N/A')} | RSI(2): {ind.get('rsi_2', 'N/A')}\n"
                f"MACD交叉: {ind.get('macd_cross', 'N/A')} | MACD柱: {ind.get('macd_hist', 'N/A')}\n"
                f"BB: 上{ind.get('bb_upper', 'N/A')} / 中{ind.get('bb_middle', 'N/A')} / 下{ind.get('bb_lower', 'N/A')}\n"
                f"ATR: {ind.get('atr', 'N/A')} | 量比: {c['volume_ratio']}x\n"
                f"Williams %R: {ind.get('williams_r', 'N/A')} | 连续下跌: {ind.get('consecutive_down', 0)}天\n"
                f"支撑: {ind.get('support', 'N/A')} | 阻力: {ind.get('resistance', 'N/A')}\n"
                f"新闻情绪分: {news.get('sentiment_score', 0)} | 新闻数: {news.get('news_count', 0)}\n"
                f"新闻方向: {c.get('news_direction', 'N/A')} | 催化剂: {news.get('top_catalyst', 'N/A')}"
            )
            texts.append(text)
        return "\n".join(texts)

    def _validate_recommendations(
        self, recs: List[Dict], portfolio_value: float,
        remaining_trades: int, vol_regime: str,
    ) -> List[Dict]:
        params = VOL_REGIMES.get(vol_regime, VOL_REGIMES["medium"])
        valid = []
        for r in recs:
            if len(valid) >= min(remaining_trades, 2):
                break
            if r.get("action", "").upper() != "BUY":
                continue
            if r.get("position_size_pct", 100) > params["max_pos_pct"]:
                r["position_size_pct"] = params["max_pos_pct"]
            if not r.get("stop_loss") or r["stop_loss"] <= 0:
                continue
            entry = r.get("entry_price", 0)
            if entry <= 0:
                continue
            stop_dist_pct = abs(entry - r["stop_loss"]) / entry * 100
            if stop_dist_pct > 5:
                continue
            if r.get("risk_reward_ratio", 0) < 1.5:
                continue
            if r.get("confidence", 0) < 65:
                continue
            valid.append(r)
        return valid
