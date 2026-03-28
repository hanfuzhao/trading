"""
新闻分析器 v6 — 权重≤10% | 仅作为风险过滤器 | is_structural一票否决
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
    # 新闻获取
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
    # 风险过滤（v6: 新闻仅用于否决）
    # ================================================================

    def check_structural_risk(self, ticker: str, price: float = 0) -> Dict:
        """
        检查是否有结构性利空新闻。
        返回: {vetoed: bool, reason: str, news_score: 0-100}
        """
        news = self.get_news(ticker, limit=5)
        if not news:
            return {"vetoed": False, "reason": "无新闻", "news_score": 50}

        analyses = []
        for article in news[:5]:
            result = self._analyze_fast(ticker, article, price)
            if result:
                analyses.append(result)

        if not analyses:
            return {"vetoed": False, "reason": "分析失败", "news_score": 50}

        structural = any(
            a.get("is_structural") is True and a.get("intraday_severity", 0) >= 8
            for a in analyses
        )
        if structural:
            reason = next(
                (a.get("summary", "结构性利空") for a in analyses if a.get("is_structural")),
                "结构性利空"
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
            "reason": f"新闻{len(analyses)}条, 正{bullish_count}/负{bearish_count}",
            "news_score": news_score,
            "analyses": analyses,
        }

    # ================================================================
    # 快速分析（gpt-4.1-mini, No-CoT）
    # ================================================================

    def _analyze_fast(self, ticker: str, news_item: Dict, price: float = 0) -> Optional[Dict]:
        try:
            prompt = f"""你是{ticker}的交易新闻分析师。只评估对股价的短期影响。

新闻：{news_item['headline']}
摘要：{news_item.get('summary', '无')}

返回纯JSON：
{{"sentiment": "bullish"或"bearish"或"neutral", "confidence": 0到100, "intraday_severity": 1到10, "catalyst_type": "earnings|guidance|analyst|macro|sector|legal|product|insider|geopolitical|other", "expected_move_pct": 数字, "is_structural": true或false, "summary": "一句话"}}

is_structural: 是否改变公司基本面（财务造假=true, 分析师调评级=false）"""

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
    # 深度分析（gpt-5.4, 复杂/宏观/地缘新闻）
    # ================================================================

    def analyze_deep(self, ticker: str, news_items: List[Dict], price: float = 0) -> Optional[Dict]:
        if not news_items:
            return None
        try:
            combined = "\n".join(
                f"- {n['headline']}: {n.get('summary', '')}"
                for n in news_items[:5]
            )
            prompt = f"""你是{ticker}的高级新闻分析师。综合评估以下新闻对股价的影响。

新闻列表：
{combined}

返回纯JSON：
{{"overall_sentiment": "bullish"或"bearish"或"neutral", "confidence": 0到100, "max_severity": 1到10, "is_structural": true或false, "key_catalyst": "一句话", "risk_level": "low"或"medium"或"high"}}"""

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
    # Web Search补充（OpenAI web_search_preview）
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
