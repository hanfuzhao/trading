"""
Research Module — AI-powered stock analysis, multi-dimensional scoring,
and price predictions for the Research & Trading Platform.

Components:
  • StockScorer     — 5-dimension scoring (Technical/Sentiment/Macro/Fundamental/Institutional)
  • PricePredictor  — AI price predictions for 10 time horizons (1d → 1mo)
  • PortfolioAnalyzer — Per-position health analysis with Hold/Sell/Add recommendations
"""
import json
import logging
import time as _time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from openai import OpenAI

from core.config import (
    OPENAI_API_KEY, MODEL_FAST, MODEL_DEEP,
    SCORING_WEIGHTS, PREDICTION_HORIZONS,
    PREDICTION_CACHE_TTL, MAX_PREDICTION_CALLS_PER_HOUR,
    MAX_DEEP_RESEARCH_PER_DAY,
)

ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)


import threading as _threading

_llm_lock = _threading.Lock()
_last_llm_done = 0.0

_MODEL_FALLBACK_CHAIN = ["gpt-5.4", "o3", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]
# Models that don't support temperature parameter
_NO_TEMPERATURE_MODELS = {"o3", "o4-mini", "o1", "o1-mini"}
_exhausted_models = {}  # model -> reset_time

def _llm_call_with_retry(client, max_retries=2, **kwargs):
    """Call OpenAI with model fallback: if one model's RPD is exhausted, try next."""
    global _last_llm_done
    requested_model = kwargs.get("model", "gpt-5.4")

    # Build fallback chain starting from the requested model
    chain = [requested_model]
    for m in _MODEL_FALLBACK_CHAIN:
        if m not in chain:
            chain.append(m)

    for model in chain:
        # Skip models known to be exhausted (until their reset time)
        if model in _exhausted_models:
            if _time.time() < _exhausted_models[model]:
                log.info("[llm] skipping exhausted model %s", model)
                continue
            else:
                del _exhausted_models[model]

        # Prepare kwargs for this model
        call_kwargs = dict(kwargs)
        call_kwargs["model"] = model
        # o-series models don't support temperature
        if model in _NO_TEMPERATURE_MODELS or model.startswith("o"):
            call_kwargs.pop("temperature", None)

        for attempt in range(max_retries):
            with _llm_lock:
                gap = 22 - (_time.time() - _last_llm_done)
                if gap > 0:
                    log.info("[llm] waiting %.0fs for rate limit gap (%s)", gap, model)
                    _time.sleep(gap)
                try:
                    result = client.chat.completions.create(**call_kwargs)
                    _last_llm_done = _time.time()
                    log.info("[llm] %s call succeeded", model)
                    return result
                except Exception as e:
                    _last_llm_done = _time.time()
                    err = str(e)
                    if "429" in err:
                        # Check if it's daily (RPD) exhaustion vs per-minute (RPM)
                        if "per day" in err or "RPD" in err:
                            log.warning("[llm] %s daily limit exhausted, trying next model", model)
                            _exhausted_models[model] = _time.time() + 3600  # skip for 1hr
                            break  # move to next model in chain
                        elif attempt < max_retries - 1:
                            wait = 25 + attempt * 15
                            log.warning("[llm] %s 429 RPM, waiting %ds (attempt %d/%d)", model, wait, attempt + 1, max_retries)
                            _time.sleep(wait)
                        else:
                            log.warning("[llm] %s RPM retries exhausted, trying next model", model)
                            break  # move to next model
                    else:
                        raise

    raise Exception("All models exhausted (rate limited). Try again later.")


# ================================================================
# StockScorer — Multi-dimensional scoring
# ================================================================

class StockScorer:
    """
    Scores a stock on 5 dimensions (0-100 each) and produces
    a weighted composite + recommendation tier.

    Dimensions:
      1. Technical   — algorithmic, from scanner indicators (no LLM)
      2. Sentiment   — from news_analyzer sentiment scores
      3. Macro       — from regime + Man Group variables
      4. Fundamental  — LLM-synthesized from news + price trends
      5. Institutional — LLM web-search for analyst ratings
    """

    def __init__(self, scanner, news_analyzer, openai_client: OpenAI = None):
        self.scanner = scanner
        self.news = news_analyzer
        self.llm = openai_client or OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
        self._cache: Dict[str, Tuple[float, Dict]] = {}  # symbol -> (timestamp, scores)

    def score_stock(self, symbol: str, use_cache: bool = True, fast_mode: bool = False) -> Dict:
        """
        Return multi-dimensional scores for a stock.

        Returns:
            {symbol, price, change_pct, scores: {technical, sentiment, macro, fundamental, institutional},
             composite, tier, timestamp}
        """
        now = _time.time()
        if use_cache and symbol in self._cache:
            ts, cached = self._cache[symbol]
            if now - ts < PREDICTION_CACHE_TTL:
                return cached

        # Get base data
        snapshots = self.scanner.get_snapshots([symbol])
        snap = snapshots.get(symbol)
        if not snap or not snap.latest_trade:
            return {"symbol": symbol, "error": "No data available"}

        price = float(snap.latest_trade.price)
        prev_close = float(snap.previous_daily_bar.close) if snap.previous_daily_bar else price
        change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0

        df = self.scanner.get_daily_bars(symbol, days=200)
        indicators = self.scanner.compute_indicators(df) if df is not None else {}

        # Compute each dimension (returns {score, reasoning})
        tech = self._score_technical(indicators, price, df)
        sent = self._score_sentiment(symbol)
        macro = self._score_macro()
        fund = self._score_fundamental(symbol, price, change_pct, indicators, df) if not fast_mode else {"score": 50, "reasoning": "N/A (fast scan)"}
        inst = self._score_institutional(symbol) if not fast_mode else {"score": 50, "reasoning": "N/A (fast scan)"}

        scores = {
            "technical": tech["score"],
            "sentiment": sent["score"],
            "macro": macro["score"],
            "fundamental": fund["score"],
            "institutional": inst["score"],
        }
        score_details = {
            "technical": tech,
            "sentiment": sent,
            "macro": macro,
            "fundamental": fund,
            "institutional": inst,
        }

        w = SCORING_WEIGHTS
        composite = round(
            scores["technical"] * w["technical"]
            + scores["sentiment"] * w["sentiment"]
            + scores["macro"] * w["macro"]
            + scores["fundamental"] * w["fundamental"]
            + scores["institutional"] * w["institutional"],
            1,
        )

        if composite >= 80:
            tier = "Strong Buy"
        elif composite >= 65:
            tier = "Buy"
        elif composite >= 50:
            tier = "Watch"
        else:
            tier = "Avoid"

        result = {
            "symbol": symbol,
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "scores": scores,
            "score_details": score_details,
            "composite": composite,
            "tier": tier,
            "indicators": indicators,
            "timestamp": datetime.now(ET).isoformat(),
        }

        self._cache[symbol] = (now, result)
        return result

    # ── Dimension 1: Technical (pure algorithmic) ──────────────

    def _score_technical(self, ind: Dict, price: float, df) -> Dict:
        if not ind:
            return {"score": 50, "reasoning": "No indicator data available"}

        score = 50
        factors = []

        # RSI(14) — oversold is opportunity
        rsi14 = ind.get("rsi_14", 50)
        if rsi14 < 30:
            score += 20; factors.append(f"RSI({rsi14:.0f}) oversold — bullish")
        elif rsi14 < 40:
            score += 10; factors.append(f"RSI({rsi14:.0f}) low — mildly bullish")
        elif rsi14 > 70:
            score -= 15; factors.append(f"RSI({rsi14:.0f}) overbought — bearish")
        elif rsi14 > 60:
            score -= 5; factors.append(f"RSI({rsi14:.0f}) elevated")

        # MACD cross
        if ind.get("macd_cross") == "golden":
            score += 15; factors.append("MACD golden cross — bullish")
        elif ind.get("macd_cross") == "death":
            score -= 15; factors.append("MACD death cross — bearish")

        # Bollinger Band position
        bb_lower = ind.get("bb_lower", 0)
        bb_upper = ind.get("bb_upper", 0)
        if bb_upper > bb_lower and bb_upper != bb_lower:
            bb_pct = (price - bb_lower) / (bb_upper - bb_lower)
            if bb_pct < 0.2:
                score += 15; factors.append(f"Near Bollinger lower band ({bb_pct:.0%}) — bounce likely")
            elif bb_pct > 0.8:
                score -= 10; factors.append(f"Near Bollinger upper band ({bb_pct:.0%}) — stretched")

        # Support/Resistance proximity
        support = ind.get("support", 0)
        if support > 0 and price > 0:
            dist_to_support = (price - support) / price
            if dist_to_support < 0.02:
                score += 10; factors.append(f"Near support at ${support:.2f}")

        # Trend — consecutive down days (mean reversion opportunity)
        consec = ind.get("consecutive_down", 0)
        if consec >= 3:
            score += 10; factors.append(f"{consec} consecutive down days — reversion opportunity")

        # SMA alignment
        if df is not None and len(df) >= 50:
            sma50 = df["close"].tail(50).mean()
            if price > sma50:
                score += 5; factors.append("Above SMA(50)")
            else:
                score -= 5; factors.append("Below SMA(50)")

            if len(df) >= 200:
                sma200 = df["close"].tail(200).mean()
                if price > sma200:
                    score += 5; factors.append("Above SMA(200)")
                else:
                    score -= 5; factors.append("Below SMA(200)")

        final = max(0, min(100, score))
        return {"score": final, "reasoning": "; ".join(factors) if factors else "Neutral technical setup"}

    # ── Dimension 2: Sentiment ────────────────────────────────

    def _score_sentiment(self, symbol: str) -> Dict:
        try:
            result = self.news.check_structural_risk(symbol)
            if result.get("vetoed"):
                return {"score": 10, "reasoning": "Structural risk detected — news vetoed this stock"}
            ns = max(0, min(100, result.get("news_score", 50)))
            analyses = result.get("analyses", [])
            summary = "; ".join(
                f"[{a.get('sentiment','?')}] {a.get('summary','')[:60]}"
                for a in analyses[:3]
            ) if analyses else "No significant news"
            return {"score": ns, "reasoning": summary}
        except Exception:
            return {"score": 50, "reasoning": "Unable to fetch news sentiment"}

    # ── Dimension 3: Macro ────────────────────────────────────

    def _score_macro(self) -> Dict:
        regime = self.scanner.get_regime()
        regime_scores = {"bullish": 80, "cautious": 60, "defensive": 40, "crisis": 20}
        score = regime_scores.get(regime, 50)
        factors = [f"Regime: {regime}"]

        spy_chg = self.scanner.get_spy_change()
        if spy_chg > 0.5:
            score += 10; factors.append(f"SPY +{spy_chg:.1f}% — positive market")
        elif spy_chg < -0.5:
            score -= 10; factors.append(f"SPY {spy_chg:.1f}% — negative market")

        ratio = self.scanner.get_vix_vix3m_ratio()
        if ratio < 0.95:
            score += 5; factors.append(f"VIX ratio {ratio:.2f} — contango (calm)")
        elif ratio > 1.1:
            score -= 10; factors.append(f"VIX ratio {ratio:.2f} — backwardation (stressed)")

        return {"score": max(0, min(100, score)), "reasoning": "; ".join(factors)}

    # ── Dimension 4: Fundamental (LLM-synthesized) ────────────

    def _score_fundamental(self, symbol: str, price: float,
                           change_pct: float, indicators: Dict,
                           df) -> int:
        try:
            news_items = self.news.get_news(symbol, limit=5)
            headlines = "\n".join(
                f"- {n.get('headline', '')}" for n in news_items[:5]
            ) if news_items else "No recent news available."

            # Build price trend context
            trend_ctx = ""
            if df is not None and len(df) >= 20:
                closes = df["close"].tail(20).tolist()
                pct_20d = (closes[-1] / closes[0] - 1) * 100
                trend_ctx = f"20-day price change: {pct_20d:+.1f}%\nRecent closes: {[round(c, 2) for c in closes[-5:]]}"

            prompt = f"""Rate the fundamental outlook for {symbol} (current price: ${price:.2f}, today: {change_pct:+.1f}%) on a scale of 0-100.

Consider: earnings trajectory, revenue growth signals, institutional interest, competitive position.

Recent headlines:
{headlines}

{trend_ctx}

Return pure JSON: {{"score": <0-100>, "reasoning": "<one sentence>"}}"""

            response = _llm_call_with_retry(
                self.llm,
                model=MODEL_DEEP,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            log.info("[fundamental] %s raw LLM response: %s", symbol, raw[:300])
            # Extract JSON from response (may be wrapped in markdown code block)
            import re as _re
            json_match = _re.search(r'\{[^{}]*\}', raw)
            data = json.loads(json_match.group()) if json_match else json.loads(raw)
            result = {"score": max(0, min(100, int(data.get("score", 50)))), "reasoning": data.get("reasoning", "")}
            log.info("[fundamental] %s result: score=%s reasoning=%r", symbol, result["score"], result["reasoning"][:100])
            return result
        except Exception as e:
            log.error("[fundamental] %s FAILED: %s", symbol, e, exc_info=True)
            return {"score": 50, "reasoning": "Analysis unavailable"}

    # ── Dimension 5: Institutional (web search) ───────────────

    def _score_institutional(self, symbol: str) -> Dict:
        try:
            response = _llm_call_with_retry(
                self.llm,
                model=MODEL_DEEP,
                messages=[{"role": "user", "content": (
                    f"Based on your knowledge, provide the latest analyst consensus "
                    f"for stock {symbol}. Consider recent upgrades, downgrades, "
                    f"price targets, and institutional sentiment. "
                    f"Return pure JSON: {{\"score\": <0-100>, \"consensus\": \"buy/hold/sell\", "
                    f"\"recent_actions\": \"<summary of recent analyst actions>\", "
                    f"\"avg_price_target\": <number or null>}}"
                )}],
                max_completion_tokens=300,
                temperature=0,
            )
            content = response.choices[0].message.content
            log.info("[institutional] %s raw LLM response: %s", symbol, (content or "")[:300])
            if content:
                import re as _re
                json_match = _re.search(r'\{[^{}]*\}', content.strip())
                data = json.loads(json_match.group()) if json_match else json.loads(content.strip())
                result = {
                    "score": max(0, min(100, int(data.get("score", 50)))),
                    "reasoning": data.get("recent_actions", data.get("consensus", "")),
                }
                log.info("[institutional] %s result: score=%s reasoning=%r", symbol, result["score"], result["reasoning"][:100])
                return result
        except Exception as e:
            log.error("[institutional] %s FAILED: %s", symbol, e, exc_info=True)
        return {"score": 50, "reasoning": "Analysis unavailable"}


# ================================================================
# PricePredictor — AI price predictions
# ================================================================

class PricePredictor:
    """
    Generates AI price predictions for 10 time horizons (1d through 1mo).
    Uses gpt-5.4 with structured data (technicals + news + macro).
    Results are cached per symbol for PREDICTION_CACHE_TTL seconds.
    """

    def __init__(self, scanner, news_analyzer, openai_client: OpenAI = None):
        self.scanner = scanner
        self.news = news_analyzer
        self.llm = openai_client or OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
        self._cache: Dict[str, Tuple[float, List[Dict]]] = {}
        self._calls_this_hour: List[float] = []

    def predict_fast(self, symbol: str) -> List[Dict]:
        """Algorithmic predictions based on technicals only — no LLM, instant."""
        snapshots = self.scanner.get_snapshots([symbol])
        snap = snapshots.get(symbol)
        if not snap or not snap.latest_trade:
            return []
        price = float(snap.latest_trade.price)
        prev_close = float(snap.previous_daily_bar.close) if snap.previous_daily_bar else price
        df = self.scanner.get_daily_bars(symbol, days=200)
        ind = self.scanner.compute_indicators(df) if df is not None else {}

        # Calculate base daily move from ATR
        atr = ind.get("atr", 0) or 0
        natr = ind.get("natr", 0) or (atr / price * 100 if price else 0)
        daily_pct = natr if natr else 1.0  # fallback 1%

        # Directional bias from technicals
        rsi14 = ind.get("rsi_14", 50) or 50
        macd_cross = ind.get("macd_cross", "none")
        ibs = ind.get("ibs", 0.5) or 0.5
        consec_down = ind.get("consecutive_down", 0) or 0

        bias = 0.0
        reasons = []
        if rsi14 < 30:
            bias += 0.4; reasons.append(f"RSI({rsi14:.0f}) oversold")
        elif rsi14 < 40:
            bias += 0.2; reasons.append(f"RSI({rsi14:.0f}) low")
        elif rsi14 > 70:
            bias -= 0.4; reasons.append(f"RSI({rsi14:.0f}) overbought")
        elif rsi14 > 60:
            bias -= 0.1; reasons.append(f"RSI({rsi14:.0f}) elevated")

        if macd_cross == "golden":
            bias += 0.3; reasons.append("MACD golden cross")
        elif macd_cross == "death":
            bias -= 0.3; reasons.append("MACD death cross")

        if ibs < 0.2:
            bias += 0.3; reasons.append(f"IBS({ibs:.2f}) mean-reversion bullish")
        elif ibs > 0.8:
            bias -= 0.2; reasons.append(f"IBS({ibs:.2f}) extended")

        if consec_down >= 3:
            bias += 0.2; reasons.append(f"{consec_down} consecutive down days — bounce likely")

        # Ensure a minimum bias so predictions are never all zero
        if abs(bias) < 0.05:
            # Default slight upward drift (historical market avg ~0.04%/day)
            bias = 0.05
            reasons.append("slight mean drift")

        reason_str = "; ".join(reasons) if reasons else "Neutral technicals"

        # Map horizons to trading days
        horizon_days = {"1d":1,"2d":2,"3d":3,"4d":4,"5d":5,"1w":5,"2w":10,"3w":15,"4w":20,"1mo":22}
        predictions = []
        for h in PREDICTION_HORIZONS:
            days = horizon_days.get(h, 5)
            expected = bias * daily_pct * (days ** 0.5)  # sqrt scaling
            expected = round(expected, 2) or 0.01  # never exactly 0
            conf = max(30, 80 - days * 2)  # confidence decays with horizon
            direction = "up" if expected > 0.1 else ("down" if expected < -0.1 else "flat")
            predictions.append({
                "horizon": h,
                "expected_pct": round(expected, 2),
                "confidence": conf,
                "direction": direction,
                "reasoning": f"[Algorithmic] {reason_str}",
            })
        return predictions

    def predict(self, symbol: str, use_cache: bool = True) -> List[Dict]:
        """
        Return AI price predictions for all horizons.

        Returns:
            [{horizon, expected_pct, confidence, direction, reasoning}, ...]
        """
        now = _time.time()

        if use_cache and symbol in self._cache:
            ts, cached = self._cache[symbol]
            if now - ts < PREDICTION_CACHE_TTL:
                return cached

        # Rate limit
        self._calls_this_hour = [t for t in self._calls_this_hour if now - t < 3600]
        if len(self._calls_this_hour) >= MAX_PREDICTION_CALLS_PER_HOUR:
            return [{"error": "Rate limit reached. Try again later."}]

        # Gather data
        snapshots = self.scanner.get_snapshots([symbol])
        snap = snapshots.get(symbol)
        if not snap or not snap.latest_trade:
            return [{"error": f"No data for {symbol}"}]

        price = float(snap.latest_trade.price)
        prev_close = float(snap.previous_daily_bar.close) if snap.previous_daily_bar else price
        change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0

        df = self.scanner.get_daily_bars(symbol, days=200)
        indicators = self.scanner.compute_indicators(df) if df is not None else {}

        # Price history for context
        price_history = ""
        high_52w, low_52w = price, price
        if df is not None and len(df) >= 20:
            closes = df["close"].tolist()
            price_history = f"Last 20 closes: {[round(c, 2) for c in closes[-20:]]}"
            high_52w = max(df["high"].tolist())
            low_52w = min(df["low"].tolist())

        # News context
        news_result = self.news.check_structural_risk(symbol, price)
        news_ctx = f"News score: {news_result.get('news_score', 50)}/100, Vetoed: {news_result.get('vetoed', False)}"
        analyses = news_result.get("analyses", [])
        if analyses:
            headlines = "\n".join(
                f"  - [{a.get('sentiment', '?')}] {a.get('summary', '')}"
                for a in analyses[:3]
            )
            news_ctx += f"\nRecent:\n{headlines}"

        # Macro context
        regime = self.scanner.get_regime()
        spy_chg = self.scanner.get_spy_change()
        man_vars = self.scanner.get_man_group_vars()

        prompt = f"""You are a quantitative analyst. Given the following data for {symbol}, predict price movements.

== Current State ==
Price: ${price:.2f} | Today: {change_pct:+.2f}%
52-week range: ${low_52w:.2f} - ${high_52w:.2f}

== Technical Indicators ==
RSI(2): {indicators.get('rsi_2', 'N/A')} | RSI(14): {indicators.get('rsi_14', 'N/A')}
MACD histogram: {indicators.get('macd_hist', 'N/A')} | Cross: {indicators.get('macd_cross', 'N/A')}
Bollinger: {indicators.get('bb_lower', 'N/A')} / {indicators.get('bb_middle', 'N/A')} / {indicators.get('bb_upper', 'N/A')}
ATR: {indicators.get('atr', 'N/A')} | NATR: {indicators.get('natr', 'N/A')}%
Support: {indicators.get('support', 'N/A')} | Resistance: {indicators.get('resistance', 'N/A')}
Consecutive down days: {indicators.get('consecutive_down', 'N/A')} | IBS: {indicators.get('ibs', 'N/A')}

== Price History ==
{price_history}

== News & Sentiment ==
{news_ctx}

== Macro Environment ==
Regime: {regime} | SPY today: {spy_chg:+.2f}%
Man Group vars: {json.dumps(man_vars, default=str)}

Predict the expected price change (%) for each time horizon. Be realistic — use the ATR and historical volatility to calibrate your estimates. Express uncertainty via the confidence score (0-100).

Return pure JSON array:
[
  {{"horizon": "1d", "expected_pct": <float>, "confidence": <0-100>, "direction": "up"/"down"/"flat", "reasoning": "<1-2 sentences>"}},
  ...for each horizon: {', '.join(PREDICTION_HORIZONS)}
]"""

        try:
            response = _llm_call_with_retry(
                self.llm,
                model=MODEL_DEEP,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000,
            )
            content = response.choices[0].message.content.strip()
            # Extract JSON from response (may have markdown fences)
            if "```" in content:
                import re
                m = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
                if m:
                    content = m.group(1).strip()
            data = json.loads(content)

            # Handle both direct array and wrapped object
            if isinstance(data, list):
                predictions = data
            elif isinstance(data, dict):
                predictions = data.get("predictions", data.get("data", [data]))
                if not isinstance(predictions, list):
                    predictions = [predictions]
            else:
                predictions = []

            self._calls_this_hour.append(now)
            self._cache[symbol] = (now, predictions)
            return predictions

        except Exception as e:
            traceback.print_exc()
            return [{"error": str(e)}]


# ================================================================
# PortfolioAnalyzer — Per-position health + recommendations
# ================================================================

class PortfolioAnalyzer:
    """
    Analyzes all held positions: runs scoring + prediction for each,
    then produces a Hold/Sell/Add recommendation.
    """

    def __init__(self, scorer: StockScorer, predictor: PricePredictor, executor):
        self.scorer = scorer
        self.predictor = predictor
        self.executor = executor
        self._cache: Optional[Tuple[float, List[Dict]]] = None

    def analyze(self, use_cache: bool = True, fast_mode: bool = False) -> List[Dict]:
        """
        Return analysis for every position in the portfolio.
        fast_mode=True: instant results using algorithmic scoring only (no LLM).
        fast_mode=False: full analysis with LLM scoring + predictions (slow).
        """
        now = _time.time()
        if use_cache and self._cache:
            ts, cached = self._cache
            ttl = 3600 if not fast_mode else 120  # 1hr for full, 2min for fast
            if now - ts < ttl:
                return cached

        positions = self.executor.get_positions()
        if not positions:
            return []

        def _analyze_one(pos):
            symbol = pos["ticker"]
            try:
                score_data = self.scorer.score_stock(symbol, use_cache=fast_mode, fast_mode=fast_mode)
                if "error" in score_data:
                    return {**pos, "symbol": symbol, "error": score_data["error"]}
                predictions = self.predictor.predict(symbol) if not fast_mode else self.predictor.predict_fast(symbol)
                rec, reason = self._recommend(pos, score_data, predictions)
                return {
                    "symbol": symbol,
                    "qty": pos.get("qty", 0),
                    "entry_price": pos.get("entry_price", 0),
                    "current_price": pos.get("current_price", 0),
                    "market_value": pos.get("market_value", 0),
                    "change_pct": score_data.get("change_pct", 0),
                    "unrealized_pnl": pos.get("unrealized_pnl", 0),
                    "unrealized_pnl_pct": pos.get("unrealized_pnl_pct", 0),
                    "sector": pos.get("sector", "Unknown"),
                    "scores": score_data.get("scores", {}),
                    "score_details": score_data.get("score_details", {}),
                    "composite": score_data.get("composite", 50),
                    "tier": score_data.get("tier", "Watch"),
                    "indicators": score_data.get("indicators", {}),
                    "predictions": predictions,
                    "recommendation": rec,
                    "recommendation_reason": reason,
                }
            except Exception as e:
                return {**pos, "symbol": symbol, "error": str(e)}

        if fast_mode:
            # Fast: run in parallel, no LLM
            results = []
            with ThreadPoolExecutor(max_workers=min(len(positions), 4)) as pool:
                futures = {pool.submit(_analyze_one, p): p for p in positions}
                for f in as_completed(futures):
                    results.append(f.result())
        else:
            # Full: run in parallel but LLM calls are serialized by _llm_lock
            results = []
            with ThreadPoolExecutor(max_workers=min(len(positions), 4)) as pool:
                futures = {pool.submit(_analyze_one, p): p for p in positions}
                for f in as_completed(futures):
                    results.append(f.result())

        self._cache = (now, results)
        return results

    def _recommend(self, pos: Dict, scores: Dict, predictions: List[Dict]) -> Tuple[str, str]:
        """Generate Hold/Sell/Add recommendation with reasoning."""
        composite = scores.get("composite", 50)
        pnl_pct = pos.get("unrealized_pnl_pct", 0)

        # Check short-term prediction
        short_outlook = 0
        for p in predictions:
            if isinstance(p, dict) and p.get("horizon") in ("1d", "2d", "3d"):
                short_outlook += p.get("expected_pct", 0)

        # Decision logic
        if composite < 40 and pnl_pct < -3:
            return "Sell", f"Low score ({composite}) + significant loss ({pnl_pct:+.1f}%)"
        if composite < 40 and short_outlook < -1:
            return "Sell", f"Low score ({composite}) + negative short-term outlook ({short_outlook:+.1f}%)"
        if composite >= 75 and short_outlook > 1:
            return "Add", f"Strong score ({composite}) + positive outlook ({short_outlook:+.1f}%)"
        if composite >= 65 and pnl_pct > 0:
            return "Hold", f"Good score ({composite}) + profitable position ({pnl_pct:+.1f}%)"
        if pnl_pct > 5 and short_outlook < 0:
            return "Sell", f"Take profit ({pnl_pct:+.1f}%) — short-term outlook weakening"
        if composite >= 50:
            return "Hold", f"Moderate score ({composite}) — maintain position"

        return "Watch", f"Below-average score ({composite}) — monitor closely"


# ================================================================
# Deep Analysis — comprehensive single-stock research
# ================================================================

def deep_analyze(symbol: str, scorer: StockScorer, predictor: PricePredictor,
                 news_analyzer) -> Dict:
    """
    Run a comprehensive deep analysis on a single stock.
    Combines scoring, predictions, news, and technicals.
    """
    scores = scorer.score_stock(symbol, use_cache=False)
    if "error" in scores:
        return scores

    predictions = predictor.predict(symbol, use_cache=False)

    # Get comprehensive news
    news_items = news_analyzer.get_news(symbol, limit=10)
    news_summary = news_analyzer.check_structural_risk(symbol)
    deep_news = news_analyzer.analyze_deep(symbol, news_items) if news_items else None

    # Web search for latest info
    web_info = news_analyzer.web_search_supplement(symbol)

    return {
        "symbol": symbol,
        "price": scores.get("price"),
        "change_pct": scores.get("change_pct"),
        "scores": scores.get("scores", {}),
        "score_details": scores.get("score_details", {}),
        "composite": scores.get("composite"),
        "tier": scores.get("tier"),
        "indicators": scores.get("indicators", {}),
        "predictions": predictions,
        "news": {
            "articles": [
                {"headline": n.get("headline", ""), "url": n.get("url", ""),
                 "created_at": n.get("created_at", "")}
                for n in news_items[:10]
            ],
            "sentiment_summary": news_summary,
            "deep_analysis": deep_news,
        },
        "web_search": web_info,
        "timestamp": datetime.now(ET).isoformat(),
    }
