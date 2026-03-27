"""
全市场技术面扫描器
从4000+只美股中筛出30-80只"今天有异动"的候选
纯本地计算，零API成本
"""
import pandas as pd
import ta
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient

ET = ZoneInfo("America/New_York")
from alpaca.data.requests import (
    StockBarsRequest, StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET,
    MIN_STOCK_PRICE, MIN_AVG_VOLUME, MIN_MARKET_CAP,
    VOLUME_SPIKE_RATIO, RSI_OVERSOLD, RSI_OVERBOUGHT,
    GAP_THRESHOLD_PCT, INTRADAY_MOMENTUM_PCT, VWAP_DEVIATION_PCT,
)


class MarketScanner:
    def __init__(self):
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        self._universe: Optional[List[str]] = None

    # ================================================================
    # 股票池构建
    # ================================================================

    def get_tradeable_universe(self, force_refresh: bool = False) -> List[str]:
        """获取可交易的美股列表（每天刷新一次即可）"""
        if self._universe and not force_refresh:
            return self._universe

        print("[扫描器] 获取可交易股票列表...")
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        assets = self.trading_client.get_all_assets(request)

        # 预过滤：只要可交易、可做空、非OTC的
        self._universe = [
            a.symbol for a in assets
            if a.tradable and a.shortable and not a.symbol.isdigit()
            and "." not in a.symbol and len(a.symbol) <= 5
        ]
        print(f"[扫描器] 可交易股票: {len(self._universe)} 只")
        return self._universe

    # ================================================================
    # 批量获取快照数据
    # ================================================================

    def get_snapshots(self, symbols: List[str]) -> Dict:
        """批量获取股票快照（当前价格、成交量等）"""
        # Alpaca限制每次请求的symbol数量，分批处理
        all_snapshots = {}
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                request = StockSnapshotRequest(symbol_or_symbols=batch)
                snapshots = self.data_client.get_stock_snapshot(request)
                all_snapshots.update(snapshots)
            except Exception as e:
                print(f"[扫描器] 快照获取失败 (batch {i}): {e}")
                continue
        return all_snapshots

    # ================================================================
    # 获取历史数据（用于计算技术指标）
    # ================================================================

    def get_daily_bars(self, symbol: str, days: int = 50) -> Optional[pd.DataFrame]:
        """获取单只股票的日线数据（至少需要35根以支撑MACD(26)计算）"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now(ET) - timedelta(days=days + 10),
                limit=days,
            )
            bars = self.data_client.get_stock_bars(request)
            bar_data = bars.data if hasattr(bars, 'data') else bars

            if symbol not in bar_data or len(bar_data[symbol]) < 30:
                return None

            df = pd.DataFrame([{
                "timestamp": b.timestamp,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume),
                "vwap": float(b.vwap) if b.vwap else None,
            } for b in bar_data[symbol]])
            return df
        except Exception:
            return None

    # ================================================================
    # 技术指标计算
    # ================================================================

    def compute_indicators(self, df: pd.DataFrame) -> Dict:
        """从日线数据计算所有技术指标"""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

        # MACD
        macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd_obj.macd().iloc[-1]
        macd_signal = macd_obj.macd_signal().iloc[-1]
        macd_hist = macd_obj.macd_diff().iloc[-1]
        prev_hist = macd_obj.macd_diff().iloc[-2] if len(df) > 1 else 0

        # 布林带
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_middle = bb.bollinger_mavg().iloc[-1]

        # ATR
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

        # 均量
        avg_volume_20 = volume.tail(20).mean()

        # 支撑/阻力（20日高低点）
        support = low.tail(20).min()
        resistance = high.tail(20).max()

        # VWAP（如果有）
        latest_vwap = df["vwap"].iloc[-1] if df["vwap"].iloc[-1] else None

        return {
            "rsi": round(rsi, 2),
            "macd_line": round(macd_line, 4),
            "macd_signal": round(macd_signal, 4),
            "macd_hist": round(macd_hist, 4),
            "macd_cross": "golden" if prev_hist <= 0 < macd_hist else ("death" if prev_hist >= 0 > macd_hist else "none"),
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "bb_middle": round(bb_middle, 2),
            "atr": round(atr, 2),
            "avg_volume_20": int(avg_volume_20),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "vwap": round(latest_vwap, 2) if latest_vwap else None,
        }

    # ================================================================
    # 核心扫描逻辑
    # ================================================================

    def scan(self) -> List[Dict]:
        """
        全市场扫描主函数
        返回通过筛选的候选股票列表
        """
        universe = self.get_tradeable_universe()
        print(f"[扫描器] 开始扫描 {len(universe)} 只股票...")

        # 第一步：批量获取快照，做初筛
        snapshots = self.get_snapshots(universe)
        candidates = []

        for symbol, snap in snapshots.items():
            try:
                if not snap or not snap.daily_bar or not snap.latest_trade:
                    continue

                price = float(snap.latest_trade.price)
                daily_bar = snap.daily_bar
                prev_bar = snap.previous_daily_bar

                # === 预过滤 ===
                if price < MIN_STOCK_PRICE:
                    continue

                today_volume = int(daily_bar.volume) if daily_bar.volume else 0

                # === 快速信号检测 ===
                signals = []

                # 跳空检测
                if prev_bar and prev_bar.close:
                    gap_pct = ((float(daily_bar.open) - float(prev_bar.close))
                               / float(prev_bar.close)) * 100
                    if abs(gap_pct) >= GAP_THRESHOLD_PCT:
                        signals.append(f"gap_{gap_pct:+.1f}%")

                # VWAP偏离
                if daily_bar.vwap and float(daily_bar.vwap) > 0:
                    vwap_dev = ((price - float(daily_bar.vwap))
                                / float(daily_bar.vwap)) * 100
                    if abs(vwap_dev) >= VWAP_DEVIATION_PCT:
                        signals.append(f"vwap_dev_{vwap_dev:+.1f}%")

                # 日内涨跌幅
                if daily_bar.open and float(daily_bar.open) > 0:
                    intraday_change = ((price - float(daily_bar.open))
                                       / float(daily_bar.open)) * 100
                    if abs(intraday_change) >= INTRADAY_MOMENTUM_PCT:
                        signals.append(f"momentum_{intraday_change:+.1f}%")

                if not signals:
                    continue

                candidates.append({
                    "ticker": symbol,
                    "price": price,
                    "today_volume": today_volume,
                    "signals": signals,
                })

            except Exception:
                continue

        print(f"[扫描器] 快照初筛通过: {len(candidates)} 只")

        # 第二步：对初筛候选做详细技术分析
        detailed_candidates = []
        for c in candidates[:150]:
            df = self.get_daily_bars(c["ticker"])
            if df is None:
                continue

            avg_vol = df["volume"].tail(20).mean()
            if avg_vol < MIN_AVG_VOLUME:
                continue
            indicators = self.compute_indicators(df)

            # 盘中时间校正：今日成交量按已过交易时间比例推算全天量（美东时间）
            now = datetime.now(ET)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            total_minutes = (market_close - market_open).total_seconds() / 60
            elapsed = max((now - market_open).total_seconds() / 60, 1)
            day_progress = min(elapsed / total_minutes, 1.0)
            projected_volume = c["today_volume"] / day_progress if day_progress > 0 else c["today_volume"]
            volume_ratio = projected_volume / avg_vol if avg_vol > 0 else 0

            # === 详细信号检测 ===
            detail_signals = list(c["signals"])

            if indicators["rsi"] <= RSI_OVERSOLD:
                detail_signals.append(f"rsi_oversold_{indicators['rsi']}")
            elif indicators["rsi"] >= RSI_OVERBOUGHT:
                detail_signals.append(f"rsi_overbought_{indicators['rsi']}")

            if volume_ratio >= VOLUME_SPIKE_RATIO:
                detail_signals.append(f"volume_spike_{volume_ratio:.1f}x")

            if indicators["macd_cross"] != "none":
                detail_signals.append(f"macd_{indicators['macd_cross']}_cross")

            if c["price"] >= indicators["bb_upper"]:
                detail_signals.append("above_bollinger")
            elif c["price"] <= indicators["bb_lower"]:
                detail_signals.append("below_bollinger")

            # 支撑/阻力附近
            if abs(c["price"] - indicators["support"]) / c["price"] < 0.01:
                detail_signals.append("near_support")
            if abs(c["price"] - indicators["resistance"]) / c["price"] < 0.01:
                detail_signals.append("near_resistance")

            if len(detail_signals) < 2:
                continue

            # 计算信号强度分
            strength = self._calculate_signal_strength(indicators, volume_ratio, detail_signals)

            detailed_candidates.append({
                "ticker": c["ticker"],
                "price": c["price"],
                "volume_ratio": round(volume_ratio, 2),
                "indicators": indicators,
                "signals": detail_signals,
                "signal_strength": strength,
            })

        # 按信号强度排序
        detailed_candidates.sort(key=lambda x: x["signal_strength"], reverse=True)

        print(f"[扫描器] 详细分析通过: {len(detailed_candidates)} 只")
        return detailed_candidates[:80]  # 最多80只进入第二层

    # ================================================================
    # 信号强度评分
    # ================================================================

    def _calculate_signal_strength(self, indicators: Dict, volume_ratio: float, signals: List[str]) -> float:
        """计算综合信号强度（0-100）"""
        score = 0

        # RSI得分
        rsi = indicators["rsi"]
        if rsi <= 20:
            score += 30
        elif rsi <= RSI_OVERSOLD:
            score += 20
        elif rsi >= 80:
            score += 30
        elif rsi >= RSI_OVERBOUGHT:
            score += 20

        # MACD交叉
        if indicators["macd_cross"] == "golden":
            score += 15
        elif indicators["macd_cross"] == "death":
            score += 15

        # 成交量
        if volume_ratio >= 5:
            score += 25
        elif volume_ratio >= VOLUME_SPIKE_RATIO:
            score += 15
        elif volume_ratio >= 2:
            score += 5

        # 信号数量bonus
        score += min(len(signals) * 3, 15)

        # 缩量惩罚
        if volume_ratio < 0.5:
            score *= 0.5

        return min(round(score, 1), 100)
