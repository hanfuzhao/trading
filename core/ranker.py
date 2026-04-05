


import json
import traceback
from datetime import datetime, date
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo
from openai import OpenAI

from core.config import (
    OPENAI_API_KEY, MODEL_RANK, MAX_O3_CALLS_PER_DAY,
    REGIME_PARAMS, MONDAY_EXCEPTION_SCORE,
)

ET = ZoneInfo("America/New_York")

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

WEEKDAY_GUIDANCE = {
    0: "Monday: In principle do not trade (VIX averages +2.16% on Mondays), use Monday data to prepare for Tuesday. Exception: signal >= 85 can enter.",
    1: "Tuesday: Prioritize using 1 slot (Turnaround Tuesday effect, best mean reversion day).",
    2: "Wednesday: Use 1 slot (highest historical return, lowest volatility).",
    3: "Thursday: Reserve 1 slot for the strongest signal. If Tuesday was used and lost -> raise R:R threshold to >2.0 today.",
    4: "Friday: If Thursday unused, can relax slightly to R:R>1.3 (slots about to reset). If 3 slots remain today -> no good opportunities this week, do not force trades.",
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

        prompt = f"""You are an overnight mean reversion portfolio manager managing a ${portfolio_value:,.0f} account.

== Strategy Description ==
Overnight mean reversion: buy oversold stocks this afternoon, sell next morning.
Core conditions: RSI(2)<15 + IBS<0.25
Ranked by NATR (higher volatility = larger reversion potential)
Expected win rate 65%, avg win 0.9%, avg loss 0.5%
No mechanical stop-loss, risk controlled by position sizing
Max 3 positions per night

== Macro Environment (live) ==
Regime: {regime} (bullish/cautious/defensive/crisis)
SPY today: {spy_change:+.2f}%
USO today: {uso_change:+.2f}%

{man_group_text}

== Constraints ==
Account value: ${portfolio_value:,.2f}
Available cash: ${cash:,.2f}
Max single position: {params['on_max_pos_pct']}% of account
Max single trade risk: {params['on_risk_pct']}% of account
Today is {weekday}
Today's Regime: {regime}

== Candidate stocks (passed RSI(2)<15 + IBS<0.25 filter) ==
{candidates_text}

Please do the following:
1. Evaluate each candidate's mean reversion potential (NATR, RSI(2) depth, IBS, consecutive down days, sector)
2. Check for structural negatives (earnings, delisting, major lawsuits, etc.) - if found, exclude
3. Cross-compare and select the best 1-3 stocks
4. Provide recommended entry price (Limit Order) for each

Return pure JSON (no markdown):
{{
  "analysis_date": "{now.strftime('%Y-%m-%d')}",
  "regime": "{regime}",
  "recommendations": [
    {{
      "rank": 1,
      "ticker": "XXX",
      "action": "BUY",
      "confidence": 0 to 100,
      "entry_price": number,
      "position_size_pct": number,
      "natr": number,
      "reasoning": "full reasoning"
    }}
  ],
  "skip_reason": "reason for not trading (if no stocks recommended)",
  "risk_warnings": ["risk 1", "risk 2"]
}}"""
        return self._call_o3(prompt, "overnight")





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
                print(f"[Ranker] Monday top score {top_score}<{MONDAY_EXCEPTION_SCORE}, not using PDT slot")
                return []

        man_group_text = self._format_man_group(man_group_vars)
        candidates_text = self._format_intraday_candidates(candidates[:15])

        prompt = f"""You are an intraday trading portfolio manager managing a ${portfolio_value:,.0f} account.

== Strategy Description ==
Intraday IV timing mean reversion: buy intraday oversold stocks at 14:00-14:45, exit before 15:40.
NYSE floor brokers receive closing imbalance info at 14:00, triggering reversals.
Hold 60-90 minutes. Must exit via Limit at 15:40, market close at 15:48.

== Macro Environment ==
Regime: {regime}
SPY today: {spy_change:+.2f}%
USO today: {uso_change:+.2f}%

{man_group_text}

== Constraints ==
PDT slots remaining: {remaining_day_trades}
Account value: ${portfolio_value:,.2f}
Available cash: ${cash:,.2f}
Max single position: {params['id_max_pos_pct']}%
Max single trade loss: {params['id_risk_pct']}%
Stop-loss: {params['id_atr_mult']}x ATR
Today is {weekday}

== PDT Slot Allocation Guidance ==
{day_guidance}
- If all candidates today have R:R below 1.5, save the slot
- Core question: "Is this opportunity one of the best 3 this week?"

== Candidate stocks ==
{candidates_text}

Thresholds:
- R:R must be > 1.5
- Confidence must be > 65
- Max {min(remaining_day_trades, 1)} recommendation(s)

Return pure JSON:
{{
  "analysis_date": "{now.strftime('%Y-%m-%d')}",
  "remaining_trades": {remaining_day_trades},
  "regime": "{regime}",
  "recommendations": [
    {{
      "rank": 1,
      "ticker": "XXX",
      "action": "BUY",
      "confidence": 0 to 100,
      "entry_price": number,
      "stop_loss": number,
      "take_profit_1": number,
      "take_profit_2": number,
      "position_size_pct": number,
      "risk_reward_ratio": number,
      "reasoning": "full reasoning"
    }}
  ],
  "save_bullets": true or false,
  "save_reason": "reason for not trading",
  "risk_warnings": ["risk 1"]
}}"""
        recs = self._call_o3(prompt, "intraday")
        return self._validate_intraday_recs(recs, remaining_day_trades)





    def _call_o3(self, prompt: str, call_type: str) -> List[Dict]:
        try:
            print(f"[Ranker] Calling o3 ({call_type})...")
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
            print(f"[Ranker] o3 returned {len(recs)} recommendations (call #{self._o3_calls_today} today)")
            return recs
        except Exception as e:
            print(f"[Ranker] o3 call failed: {e}")
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





    def _format_man_group(self, vars: Dict) -> str:
        if not vars:
            return "== Man Group Analogy Variables ==\nData unavailable"
        lines = ["== Man Group Analogy Variables (current values) =="]
        mappings = [
            ("spy_20d_return", "S&P 500 20-day return", "%"),
            ("tlt_ief_ratio", "Yield curve proxy (TLT/IEF)", ""),
            ("uso_20d_return", "Crude oil (USO) 20-day return", "%"),
            ("cper_20d_return", "Copper (CPER) 20-day return", "%"),
            ("shy_price", "Short-term rate proxy (SHY)", ""),
            ("vixy_price", "Equity volatility (VIXY)", ""),
            ("spy_tlt_corr", "Stock-bond correlation (SPY vs TLT 20d)", ""),
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
                f"Down {ind.get('consecutive_down', 0)} days | Sector={c.get('sector', '?')}"
            )
        return "\n".join(lines)

    def _format_intraday_candidates(self, candidates: List[Dict]) -> str:
        lines = []
        for i, c in enumerate(candidates, 1):
            ind = c.get("indicators", {})
            lines.append(
                f"{i}. {c['ticker']} | ${c['price']:.2f} | Intraday {c.get('intraday_change', '?')}% | "
                f"Vol ratio={c.get('volume_ratio', '?')}x | RSI(14)={ind.get('rsi_14', '?')} | "
                f"ATR=${ind.get('atr', '?')} | Sector={c.get('sector', '?')}"
            )
        return "\n".join(lines)
