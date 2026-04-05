# US Stock Intraday Trading Strategy Framework v2
# OpenAI + Alpaca | Selective Trading System Under PDT Restrictions

---

## 1. Core Philosophy

**You only have 3 bullets per week -- make every shot count.**

You face 4,000+ US stocks, but the PDT rule limits you to at most 3 day trades within any 5 trading-day window. This means the system's core task is not "finding signals" -- there are dozens of signals every day -- but **ranking and filtering**, picking the 3 opportunities most worth spending a bullet on this week.

The LLM's role: not to predict prices, but to act as a market-wide information processor. It helps you digest in minutes the news and data that would take a human hours to process, then outputs a priority ranking.

---

## 2. System Architecture Overview

```
Full market: 4,000+ stocks
        |
        v
+------------------------------+
|  Layer 1: Technical Scanner   |   <-- Zero API cost
|  (Local)                      |
|  Unusual volume / Extreme RSI |
|  / Gaps / Pre-market movers   |
|  4,000 -> 30-80 candidates    |
+--------------+---------------+
               |
               v
+------------------------------+
|  Layer 2: News Sentiment      |   <-- gpt-4.1-mini
|  Quick Screen                 |
|  News lookup + sentiment      |
|  classification per ticker    |
|  30-80 -> 10-20 candidates    |
+--------------+---------------+
               |
               v
+------------------------------+
|  Layer 3: Deep Ranking        |   <-- o3
|  Composite scoring +          |
|  cross-comparison             |
|  10-20 -> Top 3               |
+--------------+---------------+
               |
               v
+------------------------------+
|  Layer 4: Execution +         |   <-- Alpaca API
|  PDT Management               |
|  Confirm remaining day trades |
|  Place orders / Set SL & TP   |
+------------------------------+
```

---

## 3. Layer 1: Market-Wide Technical Scanner (Zero API Cost)

### Purpose

Quickly filter the 4,000+ stock universe down to 30-80 names showing "unusual activity today." Purely local computation -- zero API spend.

### Scan Timing

| Time | Scan Content |
|------|-------------|
| Pre-market 4:00-9:30 AM ET | Pre-market price action, gap detection, pre-market volume |
| Open 9:30-9:45 AM ET | First-15-minute volume spikes, price breakouts |
| Intraday every 15 min | Ongoing monitoring of filtered names + new anomaly detection |

### Filter Criteria (any one triggers inclusion)

```
1. Unusual Volume
   Current volume > 20-day same-period average volume x 3

2. Extreme RSI
   RSI(14) < 25 or RSI(14) > 75 (stricter than the usual 30/70 to reduce candidate count)

3. Price Gap
   Open vs. previous close > +/-3%

4. Pre-market Movers
   Pre-market volume > 100K shares AND pre-market change > +/-2%

5. Bollinger Band Breakout
   Price breaks above the upper or below the lower Bollinger Band (20,2)

6. Intraday Momentum
   Past 30-minute change > +/-2% with rising volume

7. VWAP Deviation
   Current price deviates from VWAP by > +/-2%
```

### Pre-filter (avoid junk stocks)

Exclude before scanning:
- Price < $5 (penny stocks have high volatility but even higher slippage -- unsuitable for small accounts)
- Average daily volume < 500K shares (insufficient liquidity, difficult to enter and exit)
- Market cap < $300M (too small -- poor news coverage, not enough data for AI analysis)

### Output

Each stock that passes the filter produces a structured data packet:
```json
{
  "ticker": "AAPL",
  "trigger_type": "volume_spike",
  "current_price": 185.30,
  "change_pct": 3.2,
  "volume_ratio": 4.5,
  "rsi": 78,
  "vwap": 182.10,
  "bollinger_position": "above_upper",
  "avg_spread": 0.02,
  "market_cap": 2850000000000,
  "sector": "Technology"
}
```

---

## 4. Layer 2: News Sentiment Quick Screen (gpt-4.1-mini)

### Purpose

Look up news and run sentiment analysis on the 30-80 stocks from Layer 1, filtering out "technical anomalies without fundamental backing" (noise).

### Two-Way Validation Logic

```
Case A: Technical anomaly + matching news     -> Signal reinforced, advance to Layer 3
Case B: Technical anomaly + no news           -> Possible insider/institutional activity, keep but down-weight
Case C: Major news + no technical reaction    -> Market hasn't priced it in yet, high-priority candidate
Case D: No anomaly + no news                  -> Skip (already filtered by Layer 1)
```

**Case C is the most valuable.** News is out but the price hasn't moved -- you have a time window.

### News Sources

- **Primary**: Alpaca News API (filter by ticker, unique IDs for deduplication)
- **Supplementary**: OpenAI Web Search (hourly search for breaking news on Layer 1 stocks)

### Model Usage

| Scenario | Model | Rationale |
|----------|-------|-----------|
| Routine news sentiment classification | gpt-4.1-mini | Fast; sufficient for sentiment classification |
| Complex news with confidence < 50 | gpt-5.4 | Needs to understand second-order effects |
| Macro policy / geopolitical news | gpt-5.4 | Needs to assess differentiated impact across sectors |

### Prompt (gpt-4.1-mini)

```
You are a dedicated intraday trading news analyst for {TICKER} ({full_company_name}, {sector}).

Evaluate this news ONLY for its impact within today's trading session. Ignore long-term implications.

News: {news_text}
Current technicals: Price {price}, change {change}%, volume ratio {vol_ratio}x

Return JSON:
{
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": 0-100,
  "intraday_severity": 1-10,
  "catalyst_type": "earnings" | "guidance" | "analyst" | "macro" | "sector" | "legal" | "product" | "insider" | "other",
  "expected_move_pct": estimated intraday move range,
  "summary": "one sentence"
}

If uncertain, set confidence below 50.
```

### Filtering Criteria

To advance to Layer 3:
- sentiment is bullish or bearish (skip neutral)
- confidence >= 60
- intraday_severity >= 5
- OR: very strong technical anomaly (volume_ratio > 5), retained even without news

---

## 5. Layer 3: Deep Ranking (o3)

### Purpose

This is the most critical step in the entire system. From the 10-20 candidates, select the **Top 3 most worth spending a day-trade slot on this week**.

### Why o3 Instead of Other Models

o3's multi-step reasoning ability is crucial here -- it is not analyzing a single stock but performing **cross-comparisons**:
- Stock A has stronger news but the technicals have already priced it in
- Stock B has a slightly weaker signal but better risk-reward
- Stock C looks good overall but today's liquidity is insufficient

This kind of multi-dimensional ranking task is where o3 has a clear advantage over other models.

### When to Call

- **Pre-market assessment (primary)**: 9:00-9:25 AM ET -- aggregate overnight news + pre-market data, produce today's Top 3 candidates
- **Intraday update (supplementary)**: If a strong new signal appears mid-session (e.g., a company drops major breaking news), re-rank

### Prompt

```
You are a portfolio manager at an intraday trading fund.

Constraints:
- Remaining day trades this week: {remaining_day_trades} (max 3 within a rolling 5-trading-day window)
- Account value: ${portfolio_value}
- Max position size per trade: 15% of account
- Today is {weekday}, with {remaining_days} trading days left this week

Below are today's candidates that passed two rounds of screening, each with technical data and news analysis:

{candidates_json}

Please do the following:

1. Evaluate each candidate stock:
   - Expected intraday return (considering direction and magnitude)
   - Risk (maximum possible loss)
   - Risk-reward ratio
   - Signal urgency (how much time window remains)
   - Liquidity (can you enter and exit smoothly)

2. Cross-compare all candidates and produce a priority ranking

3. Consider the "bullet allocation strategy":
   - If today is Monday with 3 remaining trades, you can use 1
   - If today is Thursday with 1 remaining trade, be extremely selective
   - If none of today's candidates are strong enough, recommend saving the slot

4. Output final recommendations

Show your complete reasoning process.

Return JSON:
{
  "analysis_date": "2026-03-27",
  "remaining_trades": 3,
  "recommendations": [
    {
      "rank": 1,
      "ticker": "AAPL",
      "action": "BUY" | "SHORT",
      "confidence": 0-100,
      "entry_price": 185.50,
      "stop_loss": 183.00,
      "take_profit": 190.00,
      "position_size_pct": 10,
      "risk_reward_ratio": 1.8,
      "time_window": "9:30-11:00 AM",
      "reasoning": "full reasoning"
    }
  ],
  "save_bullets": true | false,
  "save_reason": "if recommending no trade, explain why",
  "market_context": "overview of today's market environment",
  "risk_warnings": ["global risk factors"]
}
```

### Key Design: Bullet Allocation Strategy

o3 is not just picking stocks -- it also helps you **manage the allocation of a scarce resource (3 trade slots)**.

```
Monday:    3 remaining -> Standard strictness, use on good opportunities
Tuesday:   2-3 remaining -> Standard strictness
Wednesday: 1-2 remaining -> Raise the bar, only take A+ setups
Thursday:  1 remaining -> Extremely selective, or recommend saving for Friday
Friday:    1-3 remaining -> Slots about to reset, can relax slightly

If all 3 slots are used this week -> Monitor only, no trading; record missed opportunities for review
```

---

## 6. Layer 4: Execution and PDT Management (Alpaca API)

### PDT Tracker (Hard-coded, cannot be overridden by AI)

```python
# System must check before every order
class PDTTracker:
    """
    Maintains a count of day trades within a rolling 5-trading-day window.
    Under no circumstances may a 4th day trade be placed.
    """
    MAX_DAY_TRADES = 3
    WINDOW_DAYS = 5  # rolling trading days

    def can_day_trade(self) -> bool:
        """Check whether any day-trade slots remain"""
        recent_trades = self.get_trades_in_window()
        return len(recent_trades) < self.MAX_DAY_TRADES

    def remaining_trades(self) -> int:
        """Return the number of remaining available day trades"""
        return self.MAX_DAY_TRADES - len(self.get_trades_in_window())
```

**Iron rule: if `remaining_trades() == 0`, the system goes completely silent -- no API calls, no stock analysis. This saves money and prevents impulsive trading.**

### Order Logic

```
1. Check PDT quota -> insufficient: skip entirely
2. Check risk management rules -> violated: skip entirely
3. Place order (prefer Limit Orders, never use Market Orders)
4. Set stop-loss (bracket order)
5. Set automatic end-of-day close (day trades MUST be closed same day)

Critical: must close before market close!
  - 3:45 PM ET: begin checking all intraday positions
  - 3:50 PM ET: force market-order close on any remaining positions
  - This is a hard rule; otherwise the position becomes overnight,
    which doesn't count as a day trade but carries overnight risk
```

### Order Types

```
Entry:          Limit Order (0.05-0.10 worse than current price to avoid chasing)
Stop-loss:      Stop-Limit Order (stop price = ATR x 1.5)
Take-profit:    Limit Order (target = entry price + ATR x 2-3)
End-of-day close: Market Order (forced at 3:50 PM ET)
```

---

## 7. Risk Management Rules (Hard-coded, AI cannot override)

### Position Rules

| Rule | Limit |
|------|-------|
| Max position size per trade | **15%** of account value |
| Max concurrent positions | **2** (intraday trading requires focused attention) |
| Max loss per trade | **1.5%** of account value (small accounts must be more conservative) |
| Max daily loss | **3%** of account value (triggers trading halt for the day) |

### Why More Conservative Than Before

Previously the limits were 2% per trade and 5% per day -- those parameters were designed for swing trading 6 stocks. Intraday trading across the full market involves higher volatility and greater uncertainty, and a $5K-$25K account cannot withstand consecutive heavy losses. A 1.5% per-trade loss cap means a $10,000 account loses at most $150 per trade -- that is manageable.

### Cool-down Rules

- 2 consecutive losing trades -> stop trading for the day
- Weekly loss > 5% of account -> halve the position size cap next week
- 2 consecutive losing weeks -> pause the system for 1 week (monitor only, no trading; use the week for strategy review)

---

## 8. OpenAI Model Usage Summary

### Estimated Daily Model Calls

| Layer | Model | Calls/Day | Cost per Call | Daily Cost |
|-------|-------|-----------|---------------|------------|
| Layer 2: News quick screen | gpt-4.1-mini | 30-80 | ~$0.0004 | $0.01-0.03 |
| Layer 2: Complex news | gpt-5.4 | 5-15 | ~$0.005-0.01 | $0.03-0.15 |
| Layer 2: Web Search supplement | gpt-4.1-mini | 5-10 | ~$0.002 | $0.01-0.02 |
| Layer 3: Deep ranking | o3 (high) | 1-3 | ~$0.05-0.15 | $0.05-0.45 |
| **Daily total** | | | | **$0.10-0.65** |
| **Monthly total** | | | | **$2-14** |
| **$6,000 budget lasts** | | | | **35-250 years** |

### What the Models Should NOT Do

- **Do not**: Predict whether a specific stock will go up or down today
- **Do not**: Generate candlestick pattern interpretations (bullish engulfing, doji, etc.) -- code detects these more reliably
- **Do**: Read news and assess sentiment and severity
- **Do**: Cross-compare multiple candidate stocks on risk-reward
- **Do**: Understand second-order effects of complex news
- **Do**: Manage the allocation of scarce trade slots

---

## 9. Full Daily Workflow

### Pre-market (4:00 - 9:30 AM ET)

```
4:00 AM   System starts, check remaining PDT slots
          -> 0 remaining? System goes silent, no trading today

4:05 AM   Pull pre-market data for all stocks
          -> Local scan for pre-market anomalies (gaps, volume spikes)
          -> Filter to 20-50 pre-market candidates

4:10 AM   Pull overnight news for candidates
          -> gpt-4.1-mini sentiment classification
          -> Complex news escalated to gpt-5.4

8:30 AM   OpenAI Web Search supplementary scan
          -> Catch any missed breaking news

9:00 AM   Aggregate all data, call o3 for pre-market ranking
          -> Output today's Top 3 candidates + bullet allocation advice
          -> Set entry price, stop-loss, and take-profit for each candidate

9:25 AM   Manual final confirmation (optional)
          -> System sends notification: today's recommendations / suggest saving slots
```

### Intraday (9:30 AM - 3:45 PM ET)

```
9:30      Market opens
          -> Monitor price action of Top 3 candidates
          -> Entry price reached -> place Limit Order

9:30-10:30  The first 30-60 minutes after open have the highest volatility
          -> Continuously monitor news flow
          -> Breaking major news -> re-trigger Layer 2 + Layer 3

10:30+    Volatility decreases
          -> Reduce scan frequency (once every 30 minutes)
          -> Monitor stop-loss / take-profit on open positions

Every 15 min  News polling
          -> New major news -> mini analysis -> if it changes the picture -> o3 re-evaluation

Real-time  Price monitoring
          -> Stop-loss triggered -> auto close
          -> Take-profit triggered -> auto close
          -> Trailing stop: after profit > 1%, move stop to breakeven
```

### End of Day (3:45 - 4:00 PM ET)

```
3:45 PM   Check all intraday positions
          -> Any still open -> force market-order close at 3:50 PM
          -> No intraday positions allowed to go overnight

4:00 PM   Market close
          -> Record all trade results for the day
          -> Update PDT counter
          -> Generate daily report (P&L, signal accuracy, model costs)
```

### After Hours (4:00 PM - 8:00 PM ET)

```
4:00-6:00 PM  After-hours news scan (lightweight)
              -> Only watch for major news, prep for tomorrow
              -> Earnings releases (many companies report after hours)
              -> o3 analyzes major earnings -> flag as tomorrow's candidates

6:00 PM       Generate daily review report
              -> Today's signals vs. actual results
              -> AI recommendations vs. actual price movements
              -> Used for long-term strategy optimization
```

---

## 10. Contingency Plan When PDT Slots Are Exhausted

After using all 3 trades within a 5-trading-day window, the system does not shut down -- it switches modes:

### Monitor Only, No Trading

- Continue running Layer 1 and Layer 2 scans
- Record "what I would have traded if I had slots available"
- These virtual trades are used for backtesting and strategy optimization

### Why This Matters

When your trade slots become available again (once the earliest day trade in the rolling window is more than 5 trading days old), the system has already accumulated several days of monitoring data and virtual signals, allowing it to resume trading immediately in peak form.

---

## 11. Data Source Inventory

### Required (Free)

| Data Source | Use | Frequency |
|-------------|-----|-----------|
| Alpaca Market Data | Real-time prices, volume, candlesticks | Real-time |
| Alpaca News API | Per-ticker news feed | Every 5-15 min |
| SEC EDGAR | Earnings report filings | Earnings days |
| OpenAI Web Search | Supplementary breaking news | Hourly |

### Recommended Additions

| Data Source | Monthly Cost | Intraday Trading Value |
|-------------|-------------|------------------------|
| Polygon.io | $29 | Faster market data, tick-level granularity |
| Unusual Whales / FlowAlgo | $30-50 | Options flow data, early detection of institutional activity |
| Financial Modeling Prep | $19 | Structured earnings data, saves parsing effort |

---

## 12. Deployment Roadmap

### Phase 1: Build the Scanner (Weeks 1-2)

- [ ] Register for Alpaca Paper Trading
- [ ] Market-wide technical scanner (local Python)
- [ ] Pre-filter rules (price > $5, volume > 500K, market cap > $300M)
- [ ] PDT tracker
- [ ] End-of-day forced close logic

### Phase 2: Integrate AI (Weeks 3-4)

- [ ] gpt-4.1-mini news sentiment pipeline
- [ ] gpt-5.4 escalation logic
- [ ] o3 deep ranking + bullet allocation
- [ ] OpenAI Web Search supplementary layer
- [ ] Automated post-close review report

### Phase 3: Paper Trading Validation (Weeks 5-8)

- [ ] Run at least 4 full weeks of paper trading
- [ ] Track metrics: win rate, average profit/loss ratio, Sharpe ratio, max drawdown
- [ ] Compare: AI recommendations vs. random selection vs. pure technical
- [ ] Tune scan thresholds, scoring weights, bullet allocation strategy

### Phase 4: Live Trading (Week 9+)

- [ ] Confirm paper trading win rate > 55%
- [ ] Confirm average profit/loss ratio > 1.5:1
- [ ] Confirm max drawdown < 10%
- [ ] Begin live trading with half position sizes (max 7.5% per trade)
- [ ] Restore normal position sizes after 2 weeks with no major issues

---

## 13. Key Reminders

1. **PDT is an iron rule.** Hard-locked at the system level: 3 trades per 5 days, no exceptions. A 4th day trade = account frozen for 90 days.

2. **Bullets are more precious than signals.** Not every good signal is worth spending a slot on. If you use 2 on Monday and a perfect setup appears Thursday with no bullets left -- that is the biggest risk. o3's "bullet allocation" feature is the soul of this system.

3. **Intraday time windows.** The first 30-60 minutes after open have the most volatility and the most opportunities -- and are also when mistakes are most likely. The 11:00 AM - 2:00 PM stretch is typically dead water, not worth spending bullets on. 2:30-3:30 PM is the second active window.

4. **End-of-day close is non-negotiable.** Any position not closed by 3:50 PM gets force-closed at market price. Overnight risk is lethal for small accounts -- a single after-hours news item can gap a stock 10% the next morning.

5. **Your edge is not speed.** High-frequency trading firms are 1,000x faster than you -- do not try to compete on milliseconds. Your edge is the breadth of information AI helps you process -- it can read news on 4,000 stocks simultaneously, while an institutional analyst can only watch about 10.

6. **The $6,000 API budget is more than enough.** Do not use "saving API costs" as an excuse to reduce analysis quality. Use o3 when it is warranted; call gpt-5.4 when the situation demands it. Your bottleneck is strategy quality and data quality, not API cost.
