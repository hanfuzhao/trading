"""
新闻情绪分析管道 v3
gpt-4.1-mini 快筛 → gpt-5.4 升级 | is_structural 字段 | Web Search 补充
"""
import json
import os
import hashlib
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo
from openai import OpenAI
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

ET = ZoneInfo("America/New_York")

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET, OPENAI_API_KEY,
    MODEL_FAST, MODEL_DEEP, LOG_DIR,
)


class NewsAnalyzer:
    def __init__(self):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.news_client = NewsClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        self.seen_ids: OrderedDict = self._load_seen_ids()
        self.analysis_cache: Dict[str, Dict] = {}

    # ================================================================
    # 去重
    # ================================================================

    def _seen_ids_file(self) -> str:
        os.makedirs(LOG_DIR, exist_ok=True)
        return os.path.join(LOG_DIR, "seen_news_ids.json")

    def _load_seen_ids(self) -> OrderedDict:
        path = self._seen_ids_file()
        if os.path.exists(path):
            with open(path, "r") as f:
                ids = json.load(f)
                return OrderedDict.fromkeys(ids)
        return OrderedDict()

    def _save_seen_ids(self):
        items = list(self.seen_ids.keys())
        if len(items) > 10000:
            for k in items[:-10000]:
                del self.seen_ids[k]
        with open(self._seen_ids_file(), "w") as f:
            json.dump(list(self.seen_ids.keys()), f)

    # ================================================================
    # 获取新闻
    # ================================================================

    def fetch_news(self, ticker: str, hours: int = 48) -> List[Dict]:
        try:
            request = NewsRequest(
                symbols=ticker,
                start=datetime.now(ET) - timedelta(hours=hours),
                limit=20, sort="desc",
            )
            news_set = self.news_client.get_news(request)

            articles = []
            if hasattr(news_set, 'data') and isinstance(news_set.data, dict):
                articles = news_set.data.get('news', [])
            elif hasattr(news_set, 'news'):
                articles = news_set.news
            elif hasattr(news_set, 'data') and isinstance(news_set.data, list):
                articles = news_set.data

            if not articles:
                return []

            new_articles = []
            for article in articles:
                _get = (
                    (lambda k, d=None: article.get(k, d))
                    if isinstance(article, dict)
                    else (lambda k, d=None: getattr(article, k, d))
                )

                article_id = str(_get('id', ''))
                if not article_id:
                    content = f"{_get('headline', '')}_{_get('created_at', '')}"
                    article_id = hashlib.md5(content.encode()).hexdigest()

                if article_id in self.seen_ids:
                    continue

                self.seen_ids[article_id] = None
                new_articles.append({
                    "id": article_id,
                    "headline": _get('headline', ''),
                    "summary": _get('summary', '') or "",
                    "source": _get('source', ''),
                    "created_at": str(_get('created_at', '')),
                    "symbols": list(_get('symbols', []) or []),
                    "url": _get('url', '') or "",
                })

            self._save_seen_ids()
            return new_articles

        except Exception as e:
            print(f"[新闻] 获取 {ticker} 新闻失败: {e}")
            return []

    # ================================================================
    # gpt-4.1-mini 快速分类（v3: 含 is_structural + geopolitical）
    # ================================================================

    def analyze_fast(self, ticker: str, news_item: Dict, tech_context: Dict = None) -> Dict:
        tech_info = ""
        if tech_context:
            tech_info = (
                f"\n技术面：价格${tech_context.get('price', 'N/A')}，"
                f"涨跌{tech_context.get('change_pct', 'N/A')}%，"
                f"量比{tech_context.get('volume_ratio', 'N/A')}x"
            )

        prompt = f"""你是{ticker}的日内交易新闻分析师。

只评估今天交易日内的影响。

新闻：{news_item['headline']}
摘要：{news_item.get('summary', '无')}
技术面：{tech_info}

返回纯JSON：
{{"sentiment": "bullish" 或 "bearish" 或 "neutral", "confidence": 0到100, "intraday_severity": 1到10, "catalyst_type": "earnings|guidance|analyst|macro|sector|legal|product|insider|geopolitical|other", "expected_move_pct": 数字, "is_structural": true 或 false, "summary": "一句话"}}

is_structural含义：这条新闻是否改变了公司的基本面
（财务造假=true，分析师调评级=false）
不确定时confidence设50以下。"""

        try:
            response = self.openai.chat.completions.create(
                model=MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=300,
            )
            result = json.loads(
                response.choices[0].message.content
                .strip().replace("```json", "").replace("```", "")
            )
            result["model"] = MODEL_FAST
            result["ticker"] = ticker
            result["news_id"] = news_item["id"]
            result["headline"] = news_item.get("headline", "")
            result["created_at"] = news_item.get("created_at", "")
            return result

        except Exception as e:
            print(f"[新闻] mini分析失败 ({ticker}): {e}")
            return {
                "sentiment": "neutral", "confidence": 0,
                "intraday_severity": 0, "is_structural": False,
                "model": MODEL_FAST, "ticker": ticker,
                "created_at": news_item.get("created_at", ""),
                "error": str(e),
            }

    # ================================================================
    # gpt-5.4 深度分析
    # ================================================================

    def analyze_deep(self, ticker: str, news_item: Dict, mini_result: Dict, tech_context: Dict = None) -> Dict:
        tech_info = ""
        if tech_context:
            tech_info = f"""
== 技术面 ==
价格: ${tech_context.get('price', 'N/A')}
RSI: {tech_context.get('rsi', 'N/A')}
MACD: {tech_context.get('macd_hist', 'N/A')}
量比: {tech_context.get('volume_ratio', 'N/A')}x"""

        prompt = f"""你是高级金融分析师。以下新闻初步分析置信度较低，需要深度判断。

目标股票：{ticker}
新闻标题：{news_item['headline']}
新闻摘要：{news_item.get('summary', '无')}
来源：{news_item.get('source', '未知')}

初步分析（mini）：
- 情绪: {mini_result.get('sentiment', 'N/A')}
- 置信度: {mini_result.get('confidence', 'N/A')}
- 严重程度: {mini_result.get('intraday_severity', 'N/A')}
- is_structural: {mini_result.get('is_structural', 'N/A')}{tech_info}

请分析：
1. 对{ticker}的直接日内影响
2. 二阶效应（供应链、竞争对手、行业传导）
3. 是否改变公司基本面（is_structural）

返回纯JSON：
{{"sentiment": "bullish" 或 "bearish" 或 "neutral", "confidence": 0到100, "intraday_severity": 1到10, "is_structural": true 或 false, "direct_impact": "直接影响", "second_order_effects": "二阶效应", "expected_move_pct": 数字, "summary": "一句话结论"}}"""

        try:
            response = self.openai.chat.completions.create(
                model=MODEL_DEEP,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=500,
            )
            result = json.loads(
                response.choices[0].message.content
                .strip().replace("```json", "").replace("```", "")
            )
            result["model"] = MODEL_DEEP
            result["ticker"] = ticker
            result["news_id"] = news_item["id"]
            result["headline"] = news_item.get("headline", "")
            result["created_at"] = news_item.get("created_at", "")
            result["escalated"] = True
            return result

        except Exception as e:
            print(f"[新闻] 5.4深度分析失败 ({ticker}): {e}")
            return mini_result

    # ================================================================
    # 完整分析管道
    # ================================================================

    def analyze_ticker(self, ticker: str, tech_context: Dict = None) -> Dict:
        news_items = self.fetch_news(ticker, hours=48)

        use_web_search = tech_context and tech_context.get("signal_strength", 0) >= 15
        if not news_items and use_web_search:
            print(f"[新闻] {ticker} Alpaca 无新闻，启动 Web Search...")
            news_items = self.web_search_supplement(ticker)

        if not news_items:
            return {
                "ticker": ticker, "news_count": 0,
                "sentiment_score": 0, "analyses": [],
                "has_news": False,
            }

        analyses = []
        for news_item in news_items:
            result = self.analyze_fast(ticker, news_item, tech_context)

            needs_escalation = (
                result.get("confidence", 100) < 50
                or result.get("catalyst_type") in ["macro", "legal", "sector", "geopolitical"]
                or result.get("intraday_severity", 0) >= 8
            )
            if needs_escalation and result.get("sentiment") != "neutral":
                print(f"[新闻] {ticker} 升级到gpt-5.4: {news_item['headline'][:50]}...")
                result = self.analyze_deep(ticker, news_item, result, tech_context)

            analyses.append(result)

        sentiment_score = self._aggregate_sentiment(analyses)

        return {
            "ticker": ticker,
            "news_count": len(analyses),
            "sentiment_score": sentiment_score,
            "analyses": analyses,
            "has_news": True,
            "bullish_count": sum(1 for a in analyses if a.get("sentiment") == "bullish"),
            "bearish_count": sum(1 for a in analyses if a.get("sentiment") == "bearish"),
            "neutral_count": sum(1 for a in analyses if a.get("sentiment") == "neutral"),
            "top_catalyst": self._get_top_catalyst(analyses),
            "has_structural": any(a.get("is_structural") for a in analyses),
        }

    def _aggregate_sentiment(self, analyses: List[Dict]) -> float:
        if not analyses:
            return 0
        total_score = 0
        total_weight = 0

        for a in analyses:
            direction = {"bullish": 1, "bearish": -1, "neutral": 0}.get(a.get("sentiment", "neutral"), 0)
            if direction == 0:
                continue

            severity = a.get("intraday_severity", 5)
            confidence = a.get("confidence", 50)
            raw_score = direction * severity * confidence / 100

            try:
                news_time = datetime.fromisoformat(a.get("created_at", "").replace("Z", "+00:00"))
                hours_ago = (datetime.now(news_time.tzinfo) - news_time).total_seconds() / 3600
            except Exception:
                hours_ago = 12

            if hours_ago <= 1:
                time_weight = 1.0
            elif hours_ago <= 6:
                time_weight = 0.7
            else:
                time_weight = 0.3

            total_score += raw_score * time_weight
            total_weight += time_weight

        if total_weight == 0:
            return 0
        return round(total_score / total_weight * 10, 2)

    def _get_top_catalyst(self, analyses: List[Dict]) -> Optional[str]:
        if not analyses:
            return None
        sorted_a = sorted(analyses, key=lambda x: x.get("intraday_severity", 0), reverse=True)
        return sorted_a[0].get("catalyst_type")

    # ================================================================
    # OpenAI Web Search 补充
    # ================================================================

    def web_search_supplement(self, ticker: str) -> List[Dict]:
        try:
            response = self.openai.responses.create(
                model=MODEL_FAST,
                tools=[{"type": "web_search_preview"}],
                input=f"Latest breaking news about {ticker} stock today. Focus on events that could move the stock price.",
            )
            text = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for c in item.content:
                        if hasattr(c, "text"):
                            text += c.text + "\n"
                elif hasattr(item, "text"):
                    text += item.text + "\n"

            if not text.strip():
                return []

            return [{
                "id": f"websearch_{ticker}_{datetime.now(ET).strftime('%Y%m%d%H')}",
                "headline": f"Web Search: {ticker} latest news",
                "summary": text[:1000],
                "source": "OpenAI Web Search",
                "created_at": datetime.now(ET).isoformat(),
                "symbols": [ticker],
            }]

        except Exception as e:
            print(f"[新闻] Web Search失败 ({ticker}): {e}")
            return []
