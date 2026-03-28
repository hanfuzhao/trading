"""
全市场扫描器 v6
隔夜扫描: RSI(2)<15 + IBS<0.25 + NATR排序
日内扫描: 午后均值回归（14:00-14:45窗口）
3变量Regime: VIX/VIX3M + SPY/200SMA + VIX绝对水平
Man Group 7变量宏观类比
"""
import json
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET,
    MIN_STOCK_PRICE, MIN_AVG_VOLUME,
    RSI_2_THRESHOLD, RSI_2_EXTREME, IBS_THRESHOLD,
    CONSECUTIVE_DOWN_BONUS, INTRADAY_DROP_PCT, INTRADAY_VOL_RATIO,
)

ET = ZoneInfo("America/New_York")

SECTOR_BONUS_OVERNIGHT = {
    "Consumer Staples": 0.15, "Energy E&P": 0.20, "Oil Services": 0.15,
    "Healthcare": 0.15, "Defense": 0.10, "Utilities": 0.10,
}
SECTOR_PENALTY_OVERNIGHT = {
    "Technology": -0.10, "Consumer Discretionary": -0.15,
}
SECTOR_BONUS_INTRADAY = {
    "Consumer Staples": 0.20, "Energy E&P": 0.15, "Oil Services": 0.15,
    "Healthcare": 0.10, "Defense": 0.10, "Utilities": 0.10,
}
SECTOR_PENALTY_INTRADAY = {
    "Technology": -0.15, "Consumer Discretionary": -0.20,
}


def _load_sectors() -> Dict[str, str]:
    try:
        with open("sectors.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}


SECTORS = _load_sectors()


class MarketScanner:
    def __init__(self):
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        self._universe: Optional[List[str]] = None
        self._spy_df: Optional[pd.DataFrame] = None
        self._regime: str = "cautious"
        self._vixy_price: float = 0
        self._vix_vix3m_ratio: float = 1.0
        self._spy_above_200sma: bool = True
        self._spy_change: float = 0
        self._uso_change: float = 0
        self._man_group_vars: Dict = {}

    # ================================================================
    # 股票池
    # ================================================================

    def get_tradeable_universe(self, force_refresh: bool = False) -> List[str]:
        if self._universe and not force_refresh:
            return self._universe
        print("[扫描器] 获取可交易股票列表...")
        request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        assets = self.trading_client.get_all_assets(request)
        self._universe = [
            a.symbol for a in assets
            if a.tradable and not a.symbol.isdigit()
            and "." not in a.symbol and len(a.symbol) <= 5
        ]
        print(f"[扫描器] 可交易股票: {len(self._universe)} 只")
        return self._universe

    # ================================================================
    # 数据获取
    # ================================================================

    def get_snapshots(self, symbols: List[str]) -> Dict:
        all_snapshots = {}
        for i in range(0, len(symbols), 100):
            batch = symbols[i:i + 100]
            try:
                request = StockSnapshotRequest(symbol_or_symbols=batch)
                snapshots = self.data_client.get_stock_snapshot(request)
                all_snapshots.update(snapshots)
            except Exception as e:
                print(f"[扫描器] 快照失败 (batch {i}): {e}")
        return all_snapshots

    def get_daily_bars(self, symbol: str, days: int = 50) -> Optional[pd.DataFrame]:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                start=datetime.now(ET) - timedelta(days=days + 10), limit=days,
            )
            bars = self.data_client.get_stock_bars(request)
            bar_data = bars.data if hasattr(bars, 'data') else bars
            if symbol not in bar_data or len(bar_data[symbol]) < 20:
                return None
            df = pd.DataFrame([{
                "timestamp": b.timestamp,
                "open": float(b.open), "high": float(b.high),
                "low": float(b.low), "close": float(b.close),
                "volume": int(b.volume),
                "vwap": float(b.vwap) if b.vwap else None,
            } for b in bar_data[symbol]])
            return df
        except Exception:
            return None

    # ================================================================
    # 市场环境（每日刷新）
    # ================================================================

    def refresh_market_data(self):
        self._refresh_spy_data()
        self._refresh_uso_data()
        self._detect_regime()
        self._compute_man_group_vars()

    def _refresh_spy_data(self):
        self._spy_df = self.get_daily_bars("SPY", days=220)
        if self._spy_df is not None and len(self._spy_df) >= 2:
            self._spy_change = (
                (self._spy_df["close"].iloc[-1] - self._spy_df["close"].iloc[-2])
                / self._spy_df["close"].iloc[-2] * 100
            )
            if len(self._spy_df) >= 200:
                sma200 = self._spy_df["close"].tail(200).mean()
                self._spy_above_200sma = self._spy_df["close"].iloc[-1] > sma200
                print(f"[扫描器] SPY vs 200SMA: {'上方' if self._spy_above_200sma else '下方'}")

    def _refresh_uso_data(self):
        try:
            snapshots = self.get_snapshots(["USO"])
            if "USO" in snapshots:
                snap = snapshots["USO"]
                if snap.latest_trade and snap.previous_daily_bar:
                    today = float(snap.latest_trade.price)
                    prev = float(snap.previous_daily_bar.close)
                    if prev > 0:
                        self._uso_change = (today - prev) / prev * 100
        except Exception:
            pass

    # ================================================================
    # 3变量Regime检测（v6第五章）
    # ================================================================

    def _detect_regime(self):
        vixy_df = self.get_daily_bars("VIXY", days=25)
        if vixy_df is not None and len(vixy_df) >= 20:
            self._vixy_price = vixy_df["close"].iloc[-1]
            vixy_5d = vixy_df["close"].tail(5).mean()
            vixy_20d = vixy_df["close"].tail(20).mean()
            self._vix_vix3m_ratio = vixy_5d / vixy_20d if vixy_20d > 0 else 1.0
        else:
            self._vix_vix3m_ratio = 1.0

        vix_calm = self._vix_vix3m_ratio < 1.0
        vix_low = self._vixy_price < 25 if self._vixy_price > 0 else True

        if self._spy_above_200sma and vix_calm and vix_low:
            self._regime = "bullish"
        elif self._spy_above_200sma:
            self._regime = "cautious"
        elif vix_calm and vix_low:
            self._regime = "defensive"
        else:
            self._regime = "crisis"

        print(f"[扫描器] Regime: {self._regime} | VIX/VIX3M={self._vix_vix3m_ratio:.3f} | SPY>200SMA={self._spy_above_200sma} | VIXY={self._vixy_price:.2f}")

    # ================================================================
    # Man Group 7变量
    # ================================================================

    def _compute_man_group_vars(self):
        v = {}
        if self._spy_df is not None and len(self._spy_df) >= 20:
            v["spy_20d_return"] = round((self._spy_df["close"].iloc[-1] / self._spy_df["close"].iloc[-20] - 1) * 100, 2)

        for sym, key in [("TLT", "tlt"), ("IEF", "ief"), ("SHY", "shy"), ("CPER", "cper")]:
            df = self.get_daily_bars(sym, days=25)
            if df is not None and len(df) >= 20:
                v[f"{key}_price"] = round(df["close"].iloc[-1], 2)
                v[f"{key}_20d_return"] = round((df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100, 2)

        if "tlt" in v and "ief_price" in v:
            v["tlt_ief_ratio"] = round(v.get("tlt_price", 0) / v["ief_price"], 4) if v.get("ief_price") else None

        uso_df = self.get_daily_bars("USO", days=25)
        if uso_df is not None and len(uso_df) >= 20:
            v["uso_20d_return"] = round((uso_df["close"].iloc[-1] / uso_df["close"].iloc[-20] - 1) * 100, 2)

        if self._spy_df is not None:
            tlt_df = self.get_daily_bars("TLT", days=25)
            if tlt_df is not None and len(tlt_df) >= 20:
                spy_ret = self._spy_df["close"].pct_change().dropna().tail(20)
                tlt_ret = tlt_df["close"].pct_change().dropna().tail(20)
                min_len = min(len(spy_ret), len(tlt_ret))
                if min_len >= 10:
                    v["spy_tlt_corr"] = round(spy_ret.tail(min_len).corr(tlt_ret.tail(min_len)), 3)

        v["vixy_price"] = round(self._vixy_price, 2)
        self._man_group_vars = v

    # ================================================================
    # 技术指标
    # ================================================================

    def compute_indicators(self, df: pd.DataFrame) -> Dict:
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
        rsi_2 = ta.momentum.RSIIndicator(close, window=2).rsi().iloc[-1]
        rsi_14 = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        natr = atr / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0

        ibs = 0.5
        h, l, c = high.iloc[-1], low.iloc[-1], close.iloc[-1]
        if h != l:
            ibs = (c - l) / (h - l)

        consecutive_down = 0
        for i in range(len(close) - 1, 0, -1):
            if close.iloc[i] < close.iloc[i - 1]:
                consecutive_down += 1
            else:
                break

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        macd_obj = ta.trend.MACD(close)
        macd_hist = macd_obj.macd_diff().iloc[-1]
        prev_hist = macd_obj.macd_diff().iloc[-2] if len(df) > 1 else 0
        macd_cross = "golden" if prev_hist <= 0 < macd_hist else ("death" if prev_hist >= 0 > macd_hist else "none")

        support = low.tail(20).min()
        resistance = high.tail(20).max()
        vwap = df["vwap"].iloc[-1] if df["vwap"].iloc[-1] else None

        return {
            "rsi_2": round(rsi_2, 2), "rsi_14": round(rsi_14, 2),
            "atr": round(atr, 2), "natr": round(natr, 3),
            "ibs": round(ibs, 3),
            "consecutive_down": consecutive_down,
            "bb_upper": round(bb.bollinger_hband().iloc[-1], 2),
            "bb_lower": round(bb.bollinger_lband().iloc[-1], 2),
            "bb_middle": round(bb.bollinger_mavg().iloc[-1], 2),
            "macd_hist": round(macd_hist, 4), "macd_cross": macd_cross,
            "support": round(support, 2), "resistance": round(resistance, 2),
            "vwap": round(vwap, 2) if vwap else None,
            "avg_volume_20": int(volume.tail(20).mean()),
        }

    # ================================================================
    # 隔夜扫描（v6第一章: RSI(2)<15 + IBS<0.25 + NATR排序）
    # ================================================================

    def scan_overnight(self) -> List[Dict]:
        universe = self.get_tradeable_universe()
        print(f"[扫描器] 隔夜扫描 {len(universe)} 只...")

        snapshots = self.get_snapshots(universe)
        candidates = []

        for symbol, snap in snapshots.items():
            try:
                if not snap or not snap.daily_bar or not snap.latest_trade:
                    continue
                price = float(snap.latest_trade.price)
                if price < MIN_STOCK_PRICE:
                    continue
                candidates.append({"ticker": symbol, "price": price})
            except Exception:
                continue

        print(f"[扫描器] 价格过滤后: {len(candidates)} 只")

        detailed = []
        for c in candidates[:300]:
            df = self.get_daily_bars(c["ticker"])
            if df is None:
                continue

            ind = self.compute_indicators(df)
            if ind["avg_volume_20"] < MIN_AVG_VOLUME:
                continue

            if ind["rsi_2"] >= RSI_2_THRESHOLD:
                continue
            if ind["ibs"] >= IBS_THRESHOLD:
                continue

            score = 50
            if ind["rsi_2"] < RSI_2_EXTREME:
                score += 30
            elif ind["rsi_2"] < 10:
                score += 20
            else:
                score += 10

            score += (0.25 - ind["ibs"]) * 40
            score += min(ind["natr"] * 5, 20)

            if ind["consecutive_down"] >= CONSECUTIVE_DOWN_BONUS:
                score += 15

            if c["price"] <= ind["bb_lower"]:
                score += 10
            if ind["vwap"] and c["price"] < ind["vwap"] * 0.98:
                score += 10

            sector = SECTORS.get(c["ticker"])
            if sector:
                mult = SECTOR_BONUS_OVERNIGHT.get(sector, 0) + SECTOR_PENALTY_OVERNIGHT.get(sector, 0)
                score *= (1 + mult)

            detailed.append({
                "ticker": c["ticker"], "price": c["price"],
                "indicators": ind, "signal_strength": round(min(max(score, 0), 100), 1),
                "natr": ind["natr"], "ibs": ind["ibs"], "rsi_2": ind["rsi_2"],
                "sector": sector or "Unknown",
                "strategy": "overnight",
            })

        detailed.sort(key=lambda x: x["natr"], reverse=True)
        print(f"[扫描器] 隔夜候选: {len(detailed)} 只 (RSI(2)<{RSI_2_THRESHOLD} AND IBS<{IBS_THRESHOLD})")
        return detailed

    # ================================================================
    # 日内扫描（v6: 午后均值回归, 14:00-14:45）
    # ================================================================

    def scan_intraday(self) -> List[Dict]:
        universe = self.get_tradeable_universe()
        print(f"[扫描器] 日内扫描 {len(universe)} 只...")

        snapshots = self.get_snapshots(universe)
        candidates = []

        for symbol, snap in snapshots.items():
            try:
                if not snap or not snap.daily_bar or not snap.latest_trade:
                    continue
                price = float(snap.latest_trade.price)
                if price < MIN_STOCK_PRICE:
                    continue

                daily = snap.daily_bar
                if not daily.open or float(daily.open) <= 0:
                    continue

                intraday_chg = (price - float(daily.open)) / float(daily.open) * 100
                if intraday_chg > -INTRADAY_DROP_PCT:
                    continue

                today_vol = int(daily.volume) if daily.volume else 0
                candidates.append({
                    "ticker": symbol, "price": price,
                    "intraday_change": intraday_chg, "today_volume": today_vol,
                })
            except Exception:
                continue

        print(f"[扫描器] 日内初筛: {len(candidates)} 只 (日内跌>{INTRADAY_DROP_PCT}%)")

        detailed = []
        for c in candidates[:100]:
            df = self.get_daily_bars(c["ticker"])
            if df is None:
                continue

            ind = self.compute_indicators(df)
            if ind["avg_volume_20"] < MIN_AVG_VOLUME:
                continue

            now = datetime.now(ET)
            mkt_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            elapsed = max((now - mkt_open).total_seconds() / 60, 1)
            progress = max(elapsed / 390, 0.15)
            vol_ratio = (c["today_volume"] / progress) / ind["avg_volume_20"] if ind["avg_volume_20"] > 0 else 0

            if vol_ratio < INTRADAY_VOL_RATIO:
                continue

            score = 50
            score += min(abs(c["intraday_change"]) * 5, 25)
            score += min(vol_ratio * 3, 15)
            if ind["rsi_14"] < 30:
                score += 15
            if c["price"] <= ind["bb_lower"]:
                score += 10

            sector = SECTORS.get(c["ticker"])
            if sector:
                mult = SECTOR_BONUS_INTRADAY.get(sector, 0) + SECTOR_PENALTY_INTRADAY.get(sector, 0)
                score *= (1 + mult)

            detailed.append({
                "ticker": c["ticker"], "price": c["price"],
                "indicators": ind, "signal_strength": round(min(max(score, 0), 100), 1),
                "volume_ratio": round(vol_ratio, 2),
                "intraday_change": round(c["intraday_change"], 2),
                "sector": sector or "Unknown",
                "strategy": "intraday",
            })

        detailed.sort(key=lambda x: x["signal_strength"], reverse=True)
        print(f"[扫描器] 日内候选: {len(detailed)} 只")
        return detailed

    # ================================================================
    # 公开查询
    # ================================================================

    def get_regime(self) -> str:
        return self._regime

    def get_spy_above_200sma(self) -> bool:
        return self._spy_above_200sma

    def get_vix_vix3m_ratio(self) -> float:
        return self._vix_vix3m_ratio

    def get_vixy_price(self) -> float:
        return self._vixy_price

    def get_spy_change(self) -> float:
        return self._spy_change

    def get_uso_change(self) -> float:
        return self._uso_change

    def get_spy_data(self) -> Optional[pd.DataFrame]:
        return self._spy_df

    def get_man_group_vars(self) -> Dict:
        return self._man_group_vars
