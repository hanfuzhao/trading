"""
Market Scanner v6
Overnight scan: RSI(2)<15 + IBS<0.25 + NATR ranking
Intraday scan: afternoon mean reversion (14:00-14:45 window)
3-variable Regime: VIX/VIX3M + SPY/200SMA + VIX absolute level
Man Group 7-variable macro analogy
"""
import json
import os
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from core.config import (
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
        _here = os.path.dirname(os.path.abspath(__file__))
        _sectors_path = os.path.join(_here, os.pardir, "sectors.json")
        with open(_sectors_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


SECTORS = _load_sectors()


class MarketScanner:
    def __init__(self):
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=False)
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
    # Stock Universe
    # ================================================================

    def get_tradeable_universe(self, force_refresh: bool = False) -> List[str]:
        if self._universe and not force_refresh:
            return self._universe
        print("[Scanner] Fetching tradeable stock list...")
        request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        assets = self.trading_client.get_all_assets(request)
        self._universe = [
            a.symbol for a in assets
            if a.tradable and not a.symbol.isdigit()
            and "." not in a.symbol and len(a.symbol) <= 5
        ]
        print(f"[Scanner] Tradeable stocks: {len(self._universe)}")
        return self._universe

    # ================================================================
    # Data Retrieval
    # ================================================================

    def get_snapshots(self, symbols: List[str]) -> Dict:
        all_snapshots = {}
        for i in range(0, len(symbols), 100):
            batch = symbols[i:i + 100]
            try:
                request = StockSnapshotRequest(symbol_or_symbols=batch, feed=DataFeed.IEX)
                snapshots = self.data_client.get_stock_snapshot(request)
                all_snapshots.update(snapshots)
            except Exception as e:
                print(f"[Scanner] Snapshot failed (batch {i}): {e}")
        return all_snapshots

    def get_daily_bars(self, symbol: str, days: int = 50) -> Optional[pd.DataFrame]:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                start=datetime.now(ET) - timedelta(days=days + 10), limit=days,
                feed=DataFeed.IEX,
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
    # Market Environment (daily refresh)
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
                print(f"[Scanner] SPY vs 200SMA: {'above' if self._spy_above_200sma else 'below'}")

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
    # 3-Variable Regime Detection (v6 Chapter 5)
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

        print(f"[Scanner] Regime: {self._regime} | VIX/VIX3M={self._vix_vix3m_ratio:.3f} | SPY>200SMA={self._spy_above_200sma} | VIXY={self._vixy_price:.2f}")

    # ================================================================
    # Man Group 7 Variables
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
    # Technical Indicators
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
    # Overnight Scan (v6 Ch.1: RSI(2)<15 + IBS<0.25 + NATR ranking)
    # ================================================================

    def scan_overnight(self) -> List[Dict]:
        universe = self.get_tradeable_universe()
        print(f"[Scanner] Overnight scan {len(universe)} stocks...")

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

        print(f"[Scanner] After price filter: {len(candidates)}")

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
        print(f"[Scanner] Overnight candidates: {len(detailed)} (RSI(2)<{RSI_2_THRESHOLD} AND IBS<{IBS_THRESHOLD})")
        return detailed

    # ================================================================
    # Intraday Scan (v6: afternoon mean reversion, 14:00-14:45)
    # ================================================================

    def scan_intraday(self) -> List[Dict]:
        universe = self.get_tradeable_universe()
        print(f"[Scanner] Intraday scan {len(universe)} stocks...")

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

        print(f"[Scanner] Intraday pre-filter: {len(candidates)} (intraday drop>{INTRADAY_DROP_PCT}%)")

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
        print(f"[Scanner] Intraday candidates: {len(detailed)}")
        return detailed

    # ================================================================
    # Extended Technical Indicators (for research mode)
    # ================================================================

    def compute_extended_indicators(self, df: pd.DataFrame) -> Dict:
        """Compute full indicator set including SMA/EMA/52w range/ROC."""
        base = self.compute_indicators(df)
        close = df["close"]

        # Moving averages
        if len(df) >= 50:
            base["sma_50"] = round(close.tail(50).mean(), 2)
        if len(df) >= 200:
            base["sma_200"] = round(close.tail(200).mean(), 2)

        base["ema_9"] = round(ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1], 2)
        base["ema_21"] = round(ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1], 2)

        # 52-week high/low (or max available)
        base["high_52w"] = round(df["high"].max(), 2)
        base["low_52w"] = round(df["low"].min(), 2)
        base["pct_from_52w_high"] = round((close.iloc[-1] / df["high"].max() - 1) * 100, 2)

        # Rate of change
        if len(df) >= 10:
            base["roc_10"] = round((close.iloc[-1] / close.iloc[-10] - 1) * 100, 2)
        if len(df) >= 20:
            base["roc_20"] = round((close.iloc[-1] / close.iloc[-20] - 1) * 100, 2)

        # Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14)
        base["stoch_rsi_k"] = round(stoch_rsi.stochrsi_k().iloc[-1], 2)
        base["stoch_rsi_d"] = round(stoch_rsi.stochrsi_d().iloc[-1], 2)

        return base

    # ================================================================
    # Research Scan (general purpose, strategy-agnostic)
    # ================================================================

    def scan_research(self, filters: Dict = None) -> List[Dict]:
        """
        General-purpose market scan for the research page.
        Accepts filters: sectors, min_price, max_price, sort_by, limit,
                         traits (list), horizons (list).
        Returns stocks with price data and basic indicators.
        """
        filters = filters or {}
        universe = self.get_tradeable_universe()
        sectors_filter = filters.get("sectors", [])
        min_price = filters.get("min_price", MIN_STOCK_PRICE)
        max_price = filters.get("max_price", 999999)
        sort_by = filters.get("sort_by", "change_pct")
        limit = filters.get("limit", 50)
        traits = set(filters.get("traits", []))

        # Filter by sector first if specified
        if sectors_filter:
            universe = [s for s in universe if SECTORS.get(s, "Unknown") in sectors_filter]

        # Filter by market cap traits before snapshot (narrow universe)
        if "large_cap" in traits and "small_cap" not in traits:
            # Only keep well-known large caps (rough heuristic: price > 50 or in known list)
            pass  # applied after snapshot with price filter
        if "small_cap" in traits and "large_cap" not in traits:
            pass  # applied after snapshot

        snapshots = self.get_snapshots(universe)
        candidates = []

        for symbol, snap in snapshots.items():
            try:
                if not snap or not snap.latest_trade or not snap.previous_daily_bar:
                    continue
                price = float(snap.latest_trade.price)
                if price < min_price or price > max_price:
                    continue
                prev = float(snap.previous_daily_bar.close)
                change_pct = (price - prev) / prev * 100 if prev else 0
                today_vol = int(snap.daily_bar.volume) if snap.daily_bar else 0

                # Trait-based pre-filter using snapshot data
                if "large_cap" in traits and price < 50:
                    continue
                if "small_cap" in traits and price > 50:
                    continue

                candidates.append({
                    "ticker": symbol,
                    "price": round(price, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": today_vol,
                    "sector": SECTORS.get(symbol, "Unknown"),
                })
            except Exception:
                continue

        # Sort
        reverse = sort_by not in ("change_pct_asc",)
        sort_key = sort_by.replace("_asc", "").replace("_desc", "")
        if sort_key == "change_pct":
            candidates.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
        elif sort_key == "volume":
            candidates.sort(key=lambda x: x.get("volume", 0), reverse=True)
        elif sort_key == "price":
            candidates.sort(key=lambda x: x.get("price", 0), reverse=reverse)
        else:
            candidates.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)

        return candidates[:limit]

    # ================================================================
    # Public Queries
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
