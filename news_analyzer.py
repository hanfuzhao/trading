"""
News Analyzer v6 - Weight <=10% | Risk filter only | is_structural veto power
"""
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo
from openai import OpenAI

from config import OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_API_SECRET, MODEL_FAST, MODEL_DEEP

ET = ZoneInfo("America/New_York")


class NewsAnalyzer:
    def __init__(self):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self._news_client = None
        self._init_news()

    def _init_news(self):
        try:
            from alpaca.data.historical.news import NewsClient
            self._news_client = NewsClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        except Exception:
            pass

    # ================================================================
    # News Retrieval
    # ================================================================

    def get_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        if not self._news_client:
            return []
        try:
            from alpaca.data.requests import NewsRequest
            request = NewsRequest(
                symbols=ticker,
                start=datetime.now(ET) - timedelta(hours=48),
                limit=limit,
            )
            news_set = self._news_client.get_news(request)
            articles = news_set.data.get("news", []) if hasattr(news_set, 'data') else (news_set.news if hasattr(news_set, 'news') else [])
            results = []
            seen = set()
            for a in articles:
                nid = getattr(a, 'id', None) or id(a)
                if nid in seen:
                    continue
                seen.add(nid)
                results.append({
                    "headline": getattr(a, 'headline', ''),
                    "summary": getattr(a, 'summary', ''),
                    "url": getattr(a, 'url', ''),
                    "created_at": str(getattr(a, 'created_at', '')),
                })
            return results
        except Exception:
            return []

    # ================================================================
    # Risk Filter (v6: news used only for veto)
    # ================================================================

    def check_structural_risk(self, ticker: str, price: float = 0) -> Dict:
        """
        Check for structural bearish news.
        Returns: {vetoed: bool, reason: str, news_score: 0-100}
        """
        news = self.get_news(ticker, limit=5)
        if not news:
            return {"vetoed": False, "reason": "No news", "news_score": 50}

        analyses = []
        for article in news[:5]:
            result = self._analyze_fast(ticker, article, price)
            if result:
                analyses.append(result)

        if not analyses:
            return {"vetoed": False, "reason": "Analysis failed", "news_score": 50}

        structural = any(
            a.get("is_structural") is True and a.get("intraday_severity", 0) >= 8
            for a in analyses
        )
        if structural:
            reason = next(
                (a.get("summary", "Structural bearish") for a in analyses if a.get("is_structural")),
                "Structural bearish"
            )
            return {"vetoed": True, "reason": f"⛔ {reason}", "news_score": 0}

        bearish_count = sum(1 for a in analyses if a.get("sentiment") == "bearish")
        bullish_count = sum(1 for a in analyses if a.get("sentiment") == "bullish")
        avg_severity = sum(a.get("intraday_severity", 3) for a in analyses) / len(analyses)

        news_score = 50
        news_score += (bullish_count - bearish_count) * 10
        if avg_severity >= 7:
            news_score -= 15

        news_score = max(0, min(100, news_score))

        return {
            "vetoed": False,
            "reason": f"{len(analyses)} articles, bullish {bullish_count}/bearish {bearish_count}",
            "news_score": news_score,
            "analyses": analyses,
        }

    # ================================================================
    # Fast Analysis (gpt-4.1-mini, No-CoT)
    # ================================================================

    def _analyze_fast(self, ticker: str, news_item: Dict, price: float = 0) -> Optional[Dict]:
        try:
            prompt = f"""You are a trading news analyst for {ticker}. Only evaluate the short-term impact on stock price.

Headline: {news_item['headline']}
Summary: {news_item.get('summary', 'None')}

Return pure JSON:
{{"sentiment": "bullish" or "bearish" or "neutral", "confidence": 0 to 100, "intraday_severity": 1 to 10, "catalyst_type": "earnings|guidance|analyst|macro|sector|legal|product|insider|geopolitical|other", "expected_move_pct": number, "is_structural": true or false, "summary": "one sentence"}}

is_structural: whether it changes company fundamentals (accounting fraud=true, analyst rating change=false)"""

            response = self.openai.chat.completions.create(
                model=MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=300,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception:
            return None

    # ================================================================
    # Deep Analysis (gpt-5.4, complex/macro/geopolitical news)
    # ================================================================

    def analyze_deep(self, ticker: str, news_items: List[Dict], price: float = 0) -> Optional[Dict]:
        if not news_items:
            return None
        try:
            combined = "\n".join(
                f"- {n['headline']}: {n.get('summary', '')}"
                for n in news_items[:5]
            )
            prompt = f"""You are a senior news analyst for {ticker}. Comprehensively evaluate the impact of the following news on the stock price.

News list:
{combined}

Return pure JSON:
{{"overall_sentiment": "bullish" or "bearish" or "neutral", "confidence": 0 to 100, "max_severity": 1 to 10, "is_structural": true or false, "key_catalyst": "one sentence", "risk_level": "low" or "medium" or "high"}}"""

            response = self.openai.chat.completions.create(
                model=MODEL_DEEP,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            traceback.print_exc()
            return None

    # ================================================================
    # Comprehensive Analysis (for research mode)
    # ================================================================

    def get_comprehensive_analysis(self, ticker: str, price: float = 0) -> Dict:
        """
        Full news analysis combining structural risk, deep analysis, and web search.
        Returns unified view for the research platform.
        """
        news_items = self.get_news(ticker, limit=10)
        structural = self.check_structural_risk(ticker, price)
        deep = self.analyze_deep(ticker, news_items, price) if news_items else None
        web = self.web_search_supplement(ticker)

        return {
            "symbol": ticker,
            "articles": [
                {"headline": n.get("headline", ""), "url": n.get("url", ""),
                 "summary": n.get("summary", ""), "created_at": n.get("created_at", "")}
                for n in news_items
            ],
            "structural_risk": structural,
            "deep_analysis": deep,
            "web_search": web,
            "overall_score": structural.get("news_score", 50),
            "vetoed": structural.get("vetoed", False),
        }

    # ================================================================
    # Analyst Ratings Search
    # ================================================================

    def search_analyst_ratings(self, ticker: str) -> Dict:
        """
        Search web for analyst ratings, upgrades/downgrades, price targets.
        Returns best-effort synthesis from web search.
        """
        try:
            response = self.openai.chat.completions.create(
                model=MODEL_FAST,
                messages=[{"role": "user", "content": (
                    f"Search for the latest Wall Street analyst ratings for {ticker} stock. "
                    f"Find: consensus rating (Buy/Hold/Sell), recent upgrades or downgrades, "
                    f"average price target, and any notable analyst actions in the past week. "
                    f"Return pure JSON: {{\"consensus\": \"buy/hold/sell/unknown\", "
                    f"\"num_analysts\": <int or null>, \"avg_price_target\": <number or null>, "
                    f"\"recent_changes\": [{{\"firm\": \"...\", \"action\": \"upgrade/downgrade/initiate\", "
                    f"\"rating\": \"...\", \"price_target\": <number or null>}}], "
                    f"\"summary\": \"<one paragraph>\"}}"
                )}],
                max_completion_tokens=500,
                tools=[{"type": "web_search_preview"}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content:
                return json.loads(content.strip())
        except Exception:
            pass
        return {"consensus": "unknown", "summary": "Unable to retrieve analyst data."}

    # ================================================================
    # Web Search Supplement (OpenAI web_search_preview)
    # ================================================================

    def web_search_supplement(self, ticker: str) -> Optional[Dict]:
        try:
            response = self.openai.chat.completions.create(
                model=MODEL_FAST,
                messages=[{"role": "user", "content": f"Search for latest news about stock {ticker} today. Return pure JSON: {{\"has_news\": true/false, \"sentiment\": \"bullish/bearish/neutral\", \"headline\": \"main news\", \"is_structural\": true/false, \"severity\": 1-10}}"}],
                max_completion_tokens=300,
                tools=[{"type": "web_search_preview"}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content:
                return json.loads(content.strip())
        except Exception:
            pass
        return None
