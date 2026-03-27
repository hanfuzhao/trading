"""
全市场技术面扫描器 v3
双模式评分（动量+均值回归同时扫描），USO油价追踪，VIXY波动率regime
Beta计算，板块乘数加权，成交量时间校正(0.15下限)
"""
import json
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
    GAP_THRESHOLD_PCT, VOLUME_SPIKE_RATIO,
    RSI_OVERBOUGHT, RSI_OVERBOUGHT_EXTREME,
    RSI_14_OVERSOLD, RSI_14_OVERSOLD_EXTREME,
    RSI_2_OVERSOLD, RSI_2_EXTREME,
    WILLIAMS_R_OVERSOLD, CONSECUTIVE_DOWN_DAYS,
    VWAP_DEVIATION_PCT, INTRADAY_MOMENTUM_PCT,
)

ET = ZoneInfo("America/New_York")

SECTOR_MULTIPLIER = {
    "Energy E&P": 0.20,
    "Oil Services": 0.15,
    "Defense": 0.15,
    "Healthcare": 0.10,
    "Cybersecurity": 0.10,
}
SECTOR_PENALTY = {
    "Consumer Discretionary": -0.15,
    "Commercial Real Estate": -0.20,
    "Airlines": -0.20,
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
        self._vol_regime: str = "medium"
        self._vixy_price: float = 0
        self._spy_change: float = 0
        self._uso_change: float = 0

    # ================================================================
    # 股票池
    # ================================================================

    def get_tradeable_universe(self, force_refresh: bool = False) -> List[str]:
        if self._universe and not force_refresh:
            return self._universe
        print("[扫描器] 获取可交易股票列表...")
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        assets = self.trading_client.get_all_assets(request)
        self._universe = [
            a.symbol for a in assets
            if a.tradable and a.shortable and not a.symbol.isdigit()
            and "." not in a.symbol and len(a.symbol) <= 5
        ]
        print(f"[扫描器] 可交易股票: {len(self._universe)} 只")
        return self._universe

    # ================================================================
    # 数据获取
    # ================================================================

    def get_snapshots(self, symbols: List[str]) -> Dict:
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
        return all_snapshots

    def get_daily_bars(self, symbol: str, days: int = 50) -> Optional[pd.DataFrame]:
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
                "open": float(b.open), "high": float(b.high),
                "low": float(b.low), "close": float(b.close),
                "volume": int(b.volume),
                "vwap": float(b.vwap) if b.vwap else None,
            } for b in bar_data[symbol]])
            return df
        except Exception:
            return None

    # ================================================================
    # 市场环境数据（每日刷新）
    # ================================================================

    def refresh_market_data(self):
        """每日开盘前调用：刷新 SPY/VIXY/USO 数据"""
        self._refresh_spy_data()
        self.detect_vol_regime()
        self._refresh_uso_data()

    def _refresh_spy_data(self):
        self._spy_df = self.get_daily_bars("SPY", days=30)
        if self._spy_df is not None and len(self._spy_df) >= 2:
            self._spy_change = (
                (self._spy_df["close"].iloc[-1] - self._spy_df["close"].iloc[-2])
                / self._spy_df["close"].iloc[-2] * 100
            )

    def _refresh_uso_data(self):
        """USO 油价代理日涨跌幅"""
        try:
            snapshots = self.get_snapshots(["USO"])
            if "USO" in snapshots:
                snap = snapshots["USO"]
                if snap.daily_bar and snap.previous_daily_bar:
                    today = float(snap.latest_trade.price) if snap.latest_trade else float(snap.daily_bar.close)
                    prev = float(snap.previous_daily_bar.close)
                    if prev > 0:
                        self._uso_change = (today - prev) / prev * 100
                        print(f"[扫描器] USO 日涨跌: {self._uso_change:+.2f}%")
        except Exception as e:
            print(f"[扫描器] USO 数据获取失败: {e}")

    # ================================================================
    # 波动率 Regime（VIXY 或 SPY ATR% 备用）
    # ================================================================

    def detect_vol_regime(self) -> str:
        vixy_df = self.get_daily_bars("VIXY", days=10)
        if vixy_df is not None and len(vixy_df) >= 5:
            self._vixy_price = vixy_df["close"].iloc[-1]
            vixy_5d_avg = vixy_df["close"].tail(5).mean()
            if self._vixy_price > vixy_5d_avg * 1.3:
                self._vol_regime = "extreme"
            elif self._vixy_price > vixy_5d_avg * 1.1:
                self._vol_regime = "high"
            elif self._vixy_price < vixy_5d_avg * 0.9:
                self._vol_regime = "low"
            else:
                self._vol_regime = "medium"
            print(f"[扫描器] VIXY=${self._vixy_price:.2f} (5日均${vixy_5d_avg:.2f}) → regime: {self._vol_regime}")
            return self._vol_regime

        if self._spy_df is not None and len(self._spy_df) >= 20:
            close = self._spy_df["close"]
            high = self._spy_df["high"]
            low = self._spy_df["low"]
            atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
            atr_pct = atr / close.iloc[-1] * 100
            if atr_pct > 2.5:
                self._vol_regime = "extreme"
            elif atr_pct > 1.8:
                self._vol_regime = "high"
            elif atr_pct > 1.2:
                self._vol_regime = "medium"
            else:
                self._vol_regime = "low"
            print(f"[扫描器] SPY ATR%={atr_pct:.2f}% → regime: {self._vol_regime}")
        else:
            self._vol_regime = "medium"

        return self._vol_regime

    # ================================================================
    # Beta（20日，相对SPY）
    # ================================================================

    def compute_beta(self, stock_df: pd.DataFrame) -> float:
        if self._spy_df is None or len(self._spy_df) < 10:
            return 1.0
        stock_returns = stock_df["close"].pct_change().dropna().tail(20)
        spy_returns = self._spy_df["close"].pct_change().dropna().tail(20)
        min_len = min(len(stock_returns), len(spy_returns))
        if min_len < 5:
            return 1.0
        sr = stock_returns.tail(min_len)
        spr = spy_returns.tail(min_len)
        variance = spr.var()
        if variance <= 0:
            return 1.0
        return round(sr.cov(spr) / variance, 2)

    # ================================================================
    # 技术指标
    # ================================================================

    def compute_indicators(self, df: pd.DataFrame) -> Dict:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        rsi_14 = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        rsi_2 = ta.momentum.RSIIndicator(close, window=2).rsi().iloc[-1]

        macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_hist = macd_obj.macd_diff().iloc[-1]
        prev_hist = macd_obj.macd_diff().iloc[-2] if len(df) > 1 else 0

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_middle = bb.bollinger_mavg().iloc[-1]

        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        avg_volume_20 = volume.tail(20).mean()

        williams_r = ta.momentum.WilliamsRIndicator(high, low, close, window=14).williams_r().iloc[-1]

        consecutive_down = 0
        for i in range(len(close) - 1, 0, -1):
            if close.iloc[i] < close.iloc[i - 1]:
                consecutive_down += 1
            else:
                break

        support = low.tail(20).min()
        resistance = high.tail(20).max()
        latest_vwap = df["vwap"].iloc[-1] if df["vwap"].iloc[-1] else None

        macd_cross = "none"
        if prev_hist <= 0 < macd_hist:
            macd_cross = "golden"
        elif prev_hist >= 0 > macd_hist:
            macd_cross = "death"

        return {
            "rsi": round(rsi_14, 2),
            "rsi_2": round(rsi_2, 2),
            "macd_hist": round(macd_hist, 4),
            "macd_cross": macd_cross,
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "bb_middle": round(bb_middle, 2),
            "atr": round(atr, 2),
            "avg_volume_20": int(avg_volume_20),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "vwap": round(latest_vwap, 2) if latest_vwap else None,
            "williams_r": round(williams_r, 2),
            "consecutive_down": consecutive_down,
        }

    # ================================================================
    # 信号评分（v3：所有 regime 同时扫描动量+均值回归）
    # ================================================================

    def _compute_signal_score(
        self, ind: Dict, vol_ratio: float, price: float,
        gap_pct: float, intraday_pct: float, beta: float,
    ) -> tuple:
        signals = []
        score = 0

        # === 动量类信号 ===
        if abs(gap_pct) >= GAP_THRESHOLD_PCT:
            score += 15
            signals.append(f"gap_{gap_pct:+.1f}%")

        if vol_ratio >= 3.0:
            score += 25
            signals.append(f"vol_spike_{vol_ratio:.1f}x")
        elif vol_ratio >= VOLUME_SPIKE_RATIO:
            score += 15
            signals.append(f"vol_spike_{vol_ratio:.1f}x")

        rsi14 = ind["rsi"]
        if rsi14 >= RSI_OVERBOUGHT_EXTREME:
            score += 20
            signals.append(f"rsi14_ob_{rsi14:.0f}")
        elif rsi14 >= RSI_OVERBOUGHT:
            score += 10
            signals.append(f"rsi14_ob_{rsi14:.0f}")

        if ind["macd_cross"] == "golden":
            score += 15
            signals.append("macd_golden")

        if price >= ind["bb_upper"]:
            score += 10
            signals.append("above_bb_upper")

        if price >= ind["resistance"]:
            score += 10
            signals.append("resistance_break")

        vwap = ind.get("vwap")
        if vwap and vwap > 0:
            vwap_dev = (price - vwap) / vwap * 100
            if vwap_dev > VWAP_DEVIATION_PCT:
                score += 10
                signals.append(f"vwap_above_{vwap_dev:.1f}%")

        if intraday_pct > INTRADAY_MOMENTUM_PCT:
            score += 10
            signals.append(f"momentum_{intraday_pct:+.1f}%")

        # === 均值回归类信号 ===
        rsi2 = ind["rsi_2"]
        if rsi2 < RSI_2_EXTREME:
            score += 35
            signals.append(f"rsi2_extreme_{rsi2:.1f}")
        elif rsi2 < RSI_2_OVERSOLD:
            score += 25
            signals.append(f"rsi2_oversold_{rsi2:.1f}")

        if rsi14 < RSI_14_OVERSOLD_EXTREME:
            score += 25
            signals.append(f"rsi14_deep_os_{rsi14:.0f}")
        elif rsi14 < RSI_14_OVERSOLD:
            score += 15
            signals.append(f"rsi14_os_{rsi14:.0f}")

        if ind["williams_r"] < WILLIAMS_R_OVERSOLD:
            score += 20
            signals.append(f"wr_{ind['williams_r']:.0f}")

        if ind["consecutive_down"] >= CONSECUTIVE_DOWN_DAYS and vol_ratio > 1.5:
            score += 20
            signals.append(f"consec_down_{ind['consecutive_down']}d_vol")

        if price <= ind["bb_lower"]:
            score += 15
            signals.append("below_bb_lower")

        if vwap and vwap > 0:
            vwap_dev_down = (price - vwap) / vwap * 100
            if vwap_dev_down < -2.0:
                score += 15
                signals.append(f"vwap_deep_below_{vwap_dev_down:.1f}%")

        if ind["support"] > 0 and abs(price - ind["support"]) / price < 0.01:
            score += 10
            signals.append("near_support")

        # === Beta 调整 ===
        if beta >= 1.5 and self._vol_regime in ("high", "extreme"):
            score += 10
        if beta < 0.8:
            score -= 10

        return score, signals

    # ================================================================
    # 核心扫描
    # ================================================================

    def scan(self) -> List[Dict]:
        universe = self.get_tradeable_universe()
        print(f"[扫描器] 开始扫描 {len(universe)} 只股票...")

        if self._spy_df is None:
            self._refresh_spy_data()

        regime = self._vol_regime

        snapshots = self.get_snapshots(universe)
        candidates = []

        for symbol, snap in snapshots.items():
            try:
                if not snap or not snap.daily_bar or not snap.latest_trade:
                    continue
                price = float(snap.latest_trade.price)
                daily_bar = snap.daily_bar
                prev_bar = snap.previous_daily_bar
                if price < MIN_STOCK_PRICE:
                    continue

                today_volume = int(daily_bar.volume) if daily_bar.volume else 0
                fast_signals = []

                gap_pct = 0.0
                if prev_bar and prev_bar.close:
                    gap_pct = ((float(daily_bar.open) - float(prev_bar.close)) / float(prev_bar.close)) * 100
                    if abs(gap_pct) >= GAP_THRESHOLD_PCT:
                        fast_signals.append("gap")

                intraday_pct = 0.0
                if daily_bar.open and float(daily_bar.open) > 0:
                    intraday_pct = ((price - float(daily_bar.open)) / float(daily_bar.open)) * 100
                    if abs(intraday_pct) >= INTRADAY_MOMENTUM_PCT:
                        fast_signals.append("momentum")
                    if intraday_pct < -1.0:
                        fast_signals.append("drop")

                if daily_bar.vwap and float(daily_bar.vwap) > 0:
                    vwap_dev = ((price - float(daily_bar.vwap)) / float(daily_bar.vwap)) * 100
                    if abs(vwap_dev) >= VWAP_DEVIATION_PCT:
                        fast_signals.append("vwap_dev")

                if not fast_signals:
                    continue
                candidates.append({
                    "ticker": symbol, "price": price,
                    "today_volume": today_volume,
                    "gap_pct": gap_pct, "intraday_pct": intraday_pct,
                })
            except Exception:
                continue

        print(f"[扫描器] 快照初筛通过: {len(candidates)} 只")

        detailed = []
        for c in candidates[:200]:
            df = self.get_daily_bars(c["ticker"])
            if df is None:
                continue
            avg_vol = df["volume"].tail(20).mean()
            if avg_vol < MIN_AVG_VOLUME:
                continue

            indicators = self.compute_indicators(df)
            beta = self.compute_beta(df)

            # 成交量时间校正（v3: 下限 0.15）
            now = datetime.now(ET)
            mkt_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if now < mkt_open:
                progress = 0.15
            else:
                elapsed = (now - mkt_open).total_seconds() / 60
                progress = max(elapsed / 390, 0.15)
                progress = min(progress, 1.0)

            projected_vol = c["today_volume"] / progress if progress > 0 else c["today_volume"]
            vol_ratio = projected_vol / avg_vol if avg_vol > 0 else 0

            score, signals = self._compute_signal_score(
                indicators, vol_ratio, c["price"],
                c["gap_pct"], c["intraday_pct"], beta,
            )

            if len(signals) < 2:
                continue

            # 板块乘数加权 (v3: multiplicative)
            sector = SECTORS.get(c["ticker"])
            if sector:
                mult = SECTOR_MULTIPLIER.get(sector, 0)
                penalty = SECTOR_PENALTY.get(sector, 0)
                combined_mult = mult + penalty
                # USO 油价联动
                if sector in ("Energy E&P", "Oil Services", "Energy"):
                    if self._uso_change > 3:
                        combined_mult += 0.20
                    elif self._uso_change < -3:
                        combined_mult -= 0.20
                score = score * (1 + combined_mult)

            detailed.append({
                "ticker": c["ticker"], "price": c["price"],
                "volume_ratio": round(vol_ratio, 2),
                "indicators": indicators, "signals": signals,
                "signal_strength": round(min(max(score, 0), 100), 1),
                "beta": beta,
                "sector": sector or "Unknown",
                "scan_mode": "mean_reversion" if regime in ("high", "extreme") else "momentum",
                "vol_regime": regime,
            })

        detailed.sort(key=lambda x: x["signal_strength"], reverse=True)
        print(f"[扫描器] 详细分析通过: {len(detailed)} 只 | regime: {regime}")
        return detailed[:80]

    # ================================================================
    # 公开查询
    # ================================================================

    def get_vol_regime(self) -> str:
        return self._vol_regime

    def get_vixy_price(self) -> float:
        return self._vixy_price

    def get_spy_data(self) -> Optional[pd.DataFrame]:
        return self._spy_df

    def get_spy_change(self) -> float:
        return self._spy_change

    def get_uso_change(self) -> float:
        return self._uso_change
