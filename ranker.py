"""
o3 深度排名系统
从10-20只候选中选出最值得用日内交易名额的Top 3
包含"子弹分配"策略
"""
import json
from datetime import datetime
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo
from openai import OpenAI

ET = ZoneInfo("America/New_York")

from config import OPENAI_API_KEY, MODEL_RANK


class DeepRanker:
    def __init__(self):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)

    def rank_candidates(
        self,
        candidates: List[Dict],
        remaining_day_trades: int,
        portfolio_value: float,
        cash: float,
        current_positions: List[Dict],
        weekday: str,
        remaining_trading_days: int,
    ) -> Dict:
        """
        用o3对候选股票做深度排名

        参数:
            candidates: 通过前两层筛选的候选列表
            remaining_day_trades: PDT剩余名额
            portfolio_value: 账户总值
            cash: 可用现金
            current_positions: 当前持仓
            weekday: 今天星期几
            remaining_trading_days: 本周剩余交易日数
        """
        if not candidates:
            return {"recommendations": [], "save_bullets": True, "save_reason": "没有候选股票"}

        if remaining_day_trades <= 0:
            return {"recommendations": [], "save_bullets": True, "save_reason": "日内交易名额已用完"}

        # 构建候选数据
        candidates_text = self._format_candidates(candidates)

        # 构建持仓信息
        positions_text = "无持仓"
        if current_positions:
            positions_text = json.dumps(current_positions, indent=2, ensure_ascii=False)

        prompt = f"""你是一个日内交易基金的投资组合经理。

== 约束条件 ==
- 本周剩余日内交易次数：{remaining_day_trades}（5个交易日滚动窗口内最多3次）
- 账户总值：${portfolio_value:,.2f}
- 可用现金：${cash:,.2f}
- 单笔最大仓位：账户的15%
- 单笔最大亏损：账户的1.5%
- 今天是{weekday}，本周还剩{remaining_trading_days}个交易日

== 当前持仓 ==
{positions_text}

== 今日候选股票 ==
{candidates_text}

请做以下工作：

1. 对每只候选股票评估：
   - 日内交易的预期收益率（考虑方向和幅度）
   - 风险（可能的最大亏损）
   - 风险回报比（必须 > 1.5 才考虑）
   - 信号时效性（还有多少时间窗口可以入场）
   - 流动性（成交量是否够，能否顺利进出）

2. 横向对比所有候选，排出优先级

3. 子弹分配策略：
   - 剩余{remaining_day_trades}次，本周还有{remaining_trading_days}天
   - 如果今天的候选不够强（没有risk/reward > 2的），建议保留名额
   - 如果是本周最后一天且还有名额，可以稍微放宽标准

4. 输出最终推荐（最多推荐{min(remaining_day_trades, 2)}只）

展示你的完整思考过程。

返回纯JSON（不要markdown标记）：
{{
  "analysis_date": "{datetime.now(ET).strftime('%Y-%m-%d')}",
  "remaining_trades": {remaining_day_trades},
  "market_context": "今天大盘环境概述",
  "recommendations": [
    {{
      "rank": 1,
      "ticker": "XXX",
      "action": "BUY" 或 "SHORT",
      "confidence": 0到100,
      "entry_price": 数字,
      "stop_loss": 数字,
      "take_profit": 数字,
      "position_size_pct": 5到15之间的数字,
      "risk_reward_ratio": 数字,
      "time_window": "建议入场时间段",
      "reasoning": "完整推理过程"
    }}
  ],
  "save_bullets": true或false,
  "save_reason": "如果建议不交易或少交易，解释为什么",
  "risk_warnings": ["全局风险因素1", "全局风险因素2"]
}}"""

        try:
            response = self.openai.chat.completions.create(
                model=MODEL_RANK,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000,
            )

            raw = response.choices[0].message.content.strip()

            # 提取思考过程（如果o3返回了reasoning）
            reasoning_summary = None
            if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "reasoning"):
                reasoning_summary = response.choices[0].message.reasoning

            # 解析JSON
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
            result["model"] = MODEL_RANK
            result["reasoning_summary"] = reasoning_summary
            result["raw_response"] = raw[:500]  # 保留前500字符用于debug

            # 验证推荐
            result["recommendations"] = self._validate_recommendations(
                result.get("recommendations", []),
                portfolio_value,
                remaining_day_trades,
            )

            return result

        except Exception as e:
            print(f"[排名] o3分析失败: {e}")
            return {
                "recommendations": [],
                "save_bullets": True,
                "save_reason": f"o3分析失败: {str(e)}",
                "error": str(e),
            }

    def _format_candidates(self, candidates: List[Dict]) -> str:
        """格式化候选数据给o3"""
        texts = []
        for i, c in enumerate(candidates[:20]):  # 最多20只
            indicators = c.get("indicators", {})
            news = c.get("news_analysis", {})

            text = f"""
--- 候选 #{i+1}: {c['ticker']} ---
价格: ${c.get('price', 'N/A')}
信号强度: {c.get('signal_strength', 'N/A')}/100
成交量比: {c.get('volume_ratio', 'N/A')}x
触发信号: {', '.join(c.get('signals', []))}

技术指标:
  RSI(14): {indicators.get('rsi', 'N/A')}
  MACD柱: {indicators.get('macd_hist', 'N/A')} ({indicators.get('macd_cross', 'none')})
  布林带: 下轨{indicators.get('bb_lower', 'N/A')} / 中轨{indicators.get('bb_middle', 'N/A')} / 上轨{indicators.get('bb_upper', 'N/A')}
  ATR(14): {indicators.get('atr', 'N/A')}
  支撑位: {indicators.get('support', 'N/A')}
  阻力位: {indicators.get('resistance', 'N/A')}

新闻:
  新闻数量: {news.get('news_count', 0)}
  情绪分数: {news.get('sentiment_score', 0)}
  利好/利空/中性: {news.get('bullish_count', 0)}/{news.get('bearish_count', 0)}/{news.get('neutral_count', 0)}
  主要催化剂: {news.get('top_catalyst', '无')}
"""
            texts.append(text)
        return "\n".join(texts)

    def _validate_recommendations(
        self, recs: List[Dict], portfolio_value: float, remaining_trades: int
    ) -> List[Dict]:
        """验证o3的推荐，确保不违反硬规则"""
        valid = []
        for r in recs:
            # 不能超过剩余名额
            if len(valid) >= remaining_trades:
                break

            # 不能超过2只同时持仓
            if len(valid) >= 2:
                break

            # 仓位不能超过15%
            if r.get("position_size_pct", 100) > 15:
                r["position_size_pct"] = 15

            # 必须有止损
            if not r.get("stop_loss"):
                continue

            # 风险回报比必须 > 1.0
            if r.get("risk_reward_ratio", 0) < 1.0:
                continue

            # 置信度必须 > 50
            if r.get("confidence", 0) < 50:
                continue

            valid.append(r)

        return valid

    # ================================================================
    # 紧急重新评估（盘中突发新闻）
    # ================================================================

    def emergency_reassess(
        self, ticker: str, new_event: str, current_position: Dict, portfolio_value: float
    ) -> Dict:
        """盘中突发事件，o3紧急评估是否需要立即行动"""

        prompt = f"""紧急评估请求。

你当前持有 {ticker}:
- 入场价: ${current_position.get('entry_price', 'N/A')}
- 当前价: ${current_position.get('current_price', 'N/A')}
- 未实现盈亏: {current_position.get('unrealized_pnl_pct', 'N/A')}%

突发事件: {new_event}

请立即评估：
1. 这个事件是否改变了交易论点
2. 是否需要立即平仓
3. 是否需要调整止损

返回纯JSON：
{{"action": "HOLD" 或 "CLOSE_NOW" 或 "TIGHTEN_STOP", "urgency": "immediate" 或 "monitor", "new_stop_loss": 数字或null, "reasoning": "原因"}}"""

        try:
            response = self.openai.chat.completions.create(
                model=MODEL_RANK,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
            )
            result = json.loads(
                response.choices[0].message.content.strip()
                .replace("```json", "").replace("```", "")
            )
            return result
        except Exception as e:
            # 失败时保守处理：收紧止损
            return {"action": "TIGHTEN_STOP", "urgency": "immediate", "reasoning": f"分析失败({e})，保守处理"}
