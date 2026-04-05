"""
Dashboard Server v6 - Full daily workflow
70% overnight mean reversion + 30% intraday | WebSocket pre-market monitoring | 3-variable Regime
"""
import json
import logging
import os
import re
import threading
import time as _time
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request, send_file
from openai import OpenAI

from config import (
    OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_WS_URL,
    REGIME_PARAMS, LOG_DIR, CATASTROPHIC_STOP_PCT,
    ON_TECH_W, ON_MACRO_W, ON_NEWS_W,
    ID_TECH_W, ID_MACRO_W, ID_NEWS_W, ID_VOL_W,
    MONDAY_EXCEPTION_SCORE,
    ENTRY_TIMEOUT_SECONDS, TRAILING_ATR_MULT,
)
from scanner import MarketScanner
from executor import OrderExecutor
from risk_manager import RiskManager
from news_analyzer import NewsAnalyzer
from ranker import AIRanker
from pdt_tracker import PDTTracker
from tools import ToolRegistry
from agent import TradingAgent
from research import StockScorer, PricePredictor, PortfolioAnalyzer, deep_analyze

ET = ZoneInfo("America/New_York")

app = Flask(__name__)

# ================================================================
# Global Components
# ================================================================
scanner: Optional[MarketScanner] = None
executor: Optional[OrderExecutor] = None
risk: Optional[RiskManager] = None
news_analyzer: Optional[NewsAnalyzer] = None
ranker: Optional[AIRanker] = None
pdt: Optional[PDTTracker] = None
openai_client: Optional[OpenAI] = None
trading_agent: Optional[TradingAgent] = None
stock_scorer: Optional[StockScorer] = None
price_predictor: Optional[PricePredictor] = None
portfolio_analyzer: Optional[PortfolioAnalyzer] = None

latest_prices: Dict[str, float] = {}
watchlist: List[str] = []
auto_trade_enabled: bool = False  # Auto-trade toggle, default off

state = {
    "bot_status": "initializing",
    "scan_count": 0,
    "candidates_count": 0,
    "last_scan_time": None,
    "news_analyzed_count": 0,
    "today_trades": [],
    "activity_log": deque(maxlen=150),
    "overnight_candidates": [],
    "intraday_candidates": [],
    "recommendations": [],
    "regime": "cautious",
}


def _log(msg: str, level: str = "info"):
    now = datetime.now(ET)
    state["activity_log"].append({
        "time": now.strftime("%H:%M:%S"), "msg": msg, "level": level,
    })
    print(f"[{now.strftime('%H:%M:%S')}] {msg}")


def _hm() -> int:
    """Current Eastern time HHMM"""
    now = datetime.now(ET)
    return now.hour * 100 + now.minute


# ================================================================
# WebSocket Price Stream (v6 Ch.6: 15-line background thread)
# ================================================================

def _start_ws_thread(symbols: List[str]):
    """Start WebSocket background thread to monitor live prices"""
    if not symbols:
        return
    try:
        import websocket
    except ImportError:
        _log("Warning: websocket-client not installed, skipping live price stream", "warning")
        return

    def on_open(ws):
        auth = {"action": "auth", "key": ALPACA_API_KEY, "secret": ALPACA_API_SECRET}
        ws.send(json.dumps(auth))

    def on_message(ws, message):
        try:
            msgs = json.loads(message)
            if not isinstance(msgs, list):
                msgs = [msgs]
            for m in msgs:
                if m.get("T") == "t":
                    latest_prices[m["S"]] = float(m["p"])
                elif m.get("T") == "success" and m.get("msg") == "authenticated":
                    sub = {"action": "subscribe", "trades": symbols}
                    ws.send(json.dumps(sub))
                    _log(f"WebSocket subscribed to {len(symbols)} symbols: {symbols[:5]}")
        except Exception:
            pass

    def on_error(ws, error):
        _log(f"WebSocket error: {error}", "warning")

    ws = websocket.WebSocketApp(
        ALPACA_WS_URL,
        on_open=on_open, on_message=on_message, on_error=on_error,
    )
    wst = threading.Thread(target=ws.run_forever, kwargs={"ping_interval": 10}, daemon=True)
    wst.start()
    _log("WebSocket background thread started")


# ================================================================
# Macro Scoring
# ================================================================

def _calc_macro_score() -> float:
    score = 50
    spy_chg = scanner.get_spy_change()
    score += max(min(spy_chg * 10, 30), -30)
    regime = scanner.get_regime()
    vixy_adj = {"bullish": 15, "cautious": 5, "defensive": -10, "crisis": -20}
    score += vixy_adj.get(regime, 0)
    uso_chg = scanner.get_uso_change()
    score += max(min(uso_chg * 4, 20), -20)
    return max(0, min(100, score))


# ================================================================
# Overnight Exit Logic (v6 Ch.4: gap direction)
# ================================================================

def _process_overnight_exits():
    """09:45-10:15: Exit overnight positions based on gap direction"""
    if not auto_trade_enabled:
        return
    if not executor.overnight_trades:
        return

    hm = _hm()
    snapshots = scanner.get_snapshots(list(executor.overnight_trades.keys()))

    for ticker, trade in list(executor.overnight_trades.items()):
        if trade["status"] != "held":
            continue
        snap = snapshots.get(ticker)
        if not snap or not snap.latest_trade:
            continue

        current = float(snap.latest_trade.price)
        entry = trade["entry_price"]
        gap_pct = (current - entry) / entry * 100

        if hm <= 945:
            if gap_pct > 1.0:
                ok, pnl = executor.exit_overnight(ticker, current, f"Favorable gap {gap_pct:+.1f}% -> sell at 9:45")
                if ok:
                    risk.record_trade_result(pnl)
                    state["today_trades"].append({"ticker": ticker, "pnl": pnl, "type": "overnight_exit"})
                    _log(f"Overnight exit {ticker} gap {gap_pct:+.1f}% PnL${pnl:+.2f}")
            elif current <= trade["stop_line"]:
                ok, pnl = executor.exit_overnight(ticker, current, f"Unfavorable gap hit stop line")
                if ok:
                    risk.record_trade_result(pnl)
                    state["today_trades"].append({"ticker": ticker, "pnl": pnl, "type": "overnight_stop"})
                    _log(f"Overnight stop {ticker} broke stop line PnL${pnl:+.2f}", "warning")
        elif hm >= 1000:
            ok, pnl = executor.exit_overnight(ticker, current, f"10:00 timed exit gap {gap_pct:+.1f}%")
            if ok:
                risk.record_trade_result(pnl)
                state["today_trades"].append({"ticker": ticker, "pnl": pnl, "type": "overnight_exit"})
                _log(f"Overnight exit {ticker} @10:00 gap {gap_pct:+.1f}% PnL${pnl:+.2f}")


# ================================================================
# Pre-market Stop Monitoring (v6 Ch.6: 4:00AM WebSocket)
# ================================================================

def _premarket_stop_check():
    """From 4:00 AM: Check if overnight positions trigger extreme gap stop"""
    if not auto_trade_enabled:
        return
    for ticker, trade in list(executor.overnight_trades.items()):
        if trade["status"] != "held":
            continue
        price = latest_prices.get(ticker)
        if price and price <= trade["stop_line"]:
            _log(f"Pre-market stop triggered: {ticker} ${price:.2f} < stop line ${trade['stop_line']:.2f}", "warning")
            executor.submit_premarket_stop(ticker, trade["shares"], price)
            risk.record_trade_result((price - trade["entry_price"]) * trade["shares"])
            state["today_trades"].append({"ticker": ticker, "type": "premarket_stop"})


# ================================================================
# Overnight Entry Flow (v6 Ch.9: 15:00-15:45)
# ================================================================

def _run_overnight_scan_and_entry():
    """15:00-15:45: Scan -> News filter -> o3 ranking -> Entry"""
    if not auto_trade_enabled:
        _log("Auto-trade disabled, skipping overnight entry")
        return
    state["bot_status"] = "scanning"
    _log("Starting overnight stock scan...")
    candidates = scanner.scan_overnight()
    state["overnight_candidates"] = candidates
    state["scan_count"] += 1
    state["candidates_count"] = len(candidates)
    state["last_scan_time"] = datetime.now(ET).isoformat()

    if not candidates:
        _log("Overnight scan: no candidates")
        return

    _log(f"Overnight scan: {len(candidates)} candidates")

    state["bot_status"] = "analyzing_news"
    macro_score = _calc_macro_score()

    enriched = []
    for c in candidates[:20]:
        news_result = news_analyzer.check_structural_risk(c["ticker"], c["price"])
        state["news_analyzed_count"] += 1
        if news_result["vetoed"]:
            _log(f"❌ {c['ticker']} vetoed by news: {news_result['reason']}", "warning")
            continue

        tech = c["signal_strength"]
        ns = news_result.get("news_score", 50)
        combined = tech * ON_TECH_W + macro_score * ON_MACRO_W + ns * ON_NEWS_W
        c["combined_score"] = round(combined, 1)
        c["news"] = news_result
        enriched.append(c)

    enriched.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    state["overnight_candidates"] = enriched

    if not enriched:
        _log("No candidates after news filter")
        return

    state["bot_status"] = "ranking"
    acct = executor.get_account()
    recs = ranker.rank_overnight(
        enriched[:10],
        acct["portfolio_value"], acct["cash"],
        scanner.get_regime(), scanner.get_man_group_vars(),
        scanner.get_spy_change(), scanner.get_uso_change(),
    )

    if not recs:
        _log("o3 did not recommend any overnight positions")
        state["bot_status"] = "running"
        return

    state["recommendations"] = recs
    _log(f"o3 recommended {len(recs)} overnight positions")

    hm = _hm()
    if hm < 1530:
        _log("Waiting for 15:30 entry window")
        return

    _execute_overnight_entries(recs, acct)
    state["bot_status"] = "running"


def _execute_overnight_entries(recs: List[Dict], acct: Dict):
    regime = scanner.get_regime()
    current_overnight = [t for t, v in executor.overnight_trades.items() if v["status"] == "held"]
    sectors = {}
    for c in state["overnight_candidates"]:
        sectors[c["ticker"]] = c.get("sector", "Unknown")

    for rec in recs[:3]:
        ticker = rec.get("ticker")
        if not ticker or ticker in current_overnight:
            continue

        ok, reason = risk.can_trade(acct["portfolio_value"])
        if not ok:
            _log(f"Risk control rejected: {reason}", "warning")
            break

        sector = sectors.get(ticker, "Unknown")
        ok, msg, params = risk.validate_overnight(
            ticker, rec.get("entry_price", 0), acct["portfolio_value"],
            current_overnight, regime, sector,
        )
        if not ok:
            _log(f"{ticker} validation failed: {msg}", "warning")
            continue

        c_match = next((c for c in state["overnight_candidates"] if c["ticker"] == ticker), None)
        natr = c_match["natr"] if c_match else 2.0
        atr = c_match["indicators"]["atr"] if c_match and "indicators" in c_match else 1.0

        ok, msg = executor.enter_overnight(
            ticker, params["shares"], params["entry_price"], atr, natr,
        )
        if ok:
            current_overnight.append(ticker)
            _log(f"Overnight entry {ticker} {params['shares']} shares position {params['position_pct']}%")
            state["today_trades"].append({"ticker": ticker, "type": "overnight_entry"})
        else:
            _log(f"Overnight entry failed {ticker}: {msg}", "warning")


# ================================================================
# Intraday Entry Flow (v6 Ch.9: 14:00-14:45)
# ================================================================

def _run_intraday_scan_and_entry():
    """14:00-14:45: Scan -> News filter -> o3 ranking -> Entry"""
    if not auto_trade_enabled:
        _log("Auto-trade disabled, skipping intraday entry")
        return
    if not pdt.can_day_trade():
        return

    state["bot_status"] = "scanning"
    _log("Starting intraday scan...")
    candidates = scanner.scan_intraday()
    state["intraday_candidates"] = candidates

    if not candidates:
        _log("Intraday scan: no candidates")
        state["bot_status"] = "running"
        return

    _log(f"Intraday scan: {len(candidates)} candidates")

    state["bot_status"] = "analyzing_news"
    macro_score = _calc_macro_score()

    enriched = []
    for c in candidates[:15]:
        news_result = news_analyzer.check_structural_risk(c["ticker"], c["price"])
        state["news_analyzed_count"] += 1
        if news_result["vetoed"]:
            _log(f"❌ {c['ticker']} vetoed by news: {news_result['reason']}", "warning")
            continue

        tech = c["signal_strength"]
        ns = news_result.get("news_score", 50)
        vol_score = min(c.get("volume_ratio", 1) * 15, 100)
        combined = tech * ID_TECH_W + macro_score * ID_MACRO_W + ns * ID_NEWS_W + vol_score * ID_VOL_W
        c["combined_score"] = round(combined, 1)
        c["news"] = news_result
        enriched.append(c)

    enriched.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    state["intraday_candidates"] = enriched

    if not enriched:
        state["bot_status"] = "running"
        return

    now = datetime.now(ET)
    weekday_num = now.weekday()
    if weekday_num <= 1:
        top_score = max((c["combined_score"] for c in enriched), default=0)
        if top_score < MONDAY_EXCEPTION_SCORE:
            _log(f"Mon/Tue observation period, top score {top_score}<{MONDAY_EXCEPTION_SCORE}, not using PDT slot")
            state["bot_status"] = "running"
            return

    state["bot_status"] = "ranking"
    acct = executor.get_account()
    recs = ranker.rank_intraday(
        enriched[:10],
        acct["portfolio_value"], acct["cash"],
        pdt.remaining_trades(),
        scanner.get_regime(), scanner.get_man_group_vars(),
        scanner.get_spy_change(), scanner.get_uso_change(),
    )

    if not recs:
        _log("o3 did not recommend intraday trades")
        state["bot_status"] = "running"
        return

    state["recommendations"].extend(recs)
    _execute_intraday_entries(recs, acct)
    state["bot_status"] = "running"


def _execute_intraday_entries(recs: List[Dict], acct: Dict):
    regime = scanner.get_regime()
    positions = executor.get_positions()
    total_pos = len([p for p in positions if p["ticker"] != "SGOV"])

    for rec in recs[:1]:
        if not pdt.can_day_trade():
            break

        ticker = rec.get("ticker")
        if not ticker:
            continue

        ok, reason = risk.can_trade(acct["portfolio_value"])
        if not ok:
            break

        entry_price = rec.get("entry_price", 0)
        stop_loss = rec.get("stop_loss", 0)
        if not entry_price or not stop_loss:
            stops = risk.calculate_intraday_stops(entry_price, rec.get("atr", 1), regime)
            stop_loss = stop_loss or stops["stop_loss"]
            rec["take_profit_1"] = rec.get("take_profit_1") or stops["take_profit_1"]
            rec["take_profit_2"] = rec.get("take_profit_2") or stops["take_profit_2"]

        ok, msg, params = risk.validate_intraday(
            ticker, entry_price, stop_loss,
            acct["portfolio_value"], total_pos, regime,
        )
        if not ok:
            _log(f"{ticker} intraday validation failed: {msg}", "warning")
            continue

        c_match = next((c for c in state["intraday_candidates"] if c["ticker"] == ticker), None)
        atr = c_match["indicators"]["atr"] if c_match and "indicators" in c_match else 1.0

        ok, msg = executor.enter_intraday(
            ticker, params["shares"], params["entry_price"], params["stop_loss"],
            rec.get("take_profit_1", entry_price * 1.02),
            rec.get("take_profit_2", entry_price * 1.04),
            atr,
        )
        if ok:
            pdt.record_day_trade(ticker)
            state["today_trades"].append({"ticker": ticker, "type": "intraday_entry"})
            _log(f"Intraday entry {ticker} {params['shares']} shares")
            total_pos += 1


# ================================================================
# Intraday Position Monitoring
# ================================================================

def _monitor_intraday():
    """Every 15s: Check intraday position stop-loss/take-profit/time stop"""
    if not executor.intraday_trades:
        return

    snapshots = scanner.get_snapshots(list(executor.intraday_trades.keys()))
    for ticker, trade in list(executor.intraday_trades.items()):
        if trade["status"] != "filled":
            continue
        snap = snapshots.get(ticker)
        if not snap or not snap.latest_trade:
            continue
        price = float(snap.latest_trade.price)
        executor.update_trailing_stop(ticker, price)
        if executor.check_time_stop(ticker, price):
            risk.record_trade_result(
                (price - trade["entry_price"]) * trade.get("remaining_shares", trade["shares"])
            )
            _log(f"Time stop {ticker}")

    closed = executor.check_closed_trades()
    for c in closed:
        risk.record_trade_result(c["pnl"])
        state["today_trades"].append(c)


# ================================================================
# Intraday Exit (15:40-15:50)
# ================================================================

def _intraday_exit_sequence():
    """15:40: Limit exit -> 15:48: Market close -> 15:50: Confirm zero intraday"""
    if not auto_trade_enabled:
        return
    hm = _hm()
    if hm >= 1548:
        closed = executor.close_all_intraday()
        for c in closed:
            pnl = c.get("unrealized_pnl", 0)
            risk.record_trade_result(pnl)
            state["today_trades"].append({"ticker": c["ticker"], "pnl": pnl, "type": "forced_close"})
            _log(f"15:48 forced close {c['ticker']} PnL${pnl:+.2f}")
    if hm >= 1550:
        executor.confirm_zero_intraday()


# ================================================================
# End of Day Report
# ================================================================

def _generate_eod_report():
    trades = state["today_trades"]
    total_pnl = sum(t.get("pnl", 0) for t in trades if "pnl" in t)
    _log(f"EOD report: {len(trades)} trades | Total PnL${total_pnl:+.2f}")
    _log(f"📊 Regime={scanner.get_regime()} | VIX/VIX3M={scanner.get_vix_vix3m_ratio():.3f}")
    _log(f"PDT remaining: {pdt.remaining_trades()}/3 | {risk.status(executor.get_account()['portfolio_value'])}")

    os.makedirs(LOG_DIR, exist_ok=True)
    report = {
        "date": datetime.now(ET).strftime("%Y-%m-%d"),
        "trades": trades,
        "total_pnl": total_pnl,
        "regime": scanner.get_regime(),
        "pdt_remaining": pdt.remaining_trades(),
        "scan_count": state["scan_count"],
    }
    with open(os.path.join(LOG_DIR, f"daily_{report['date']}.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)


# ================================================================
# Main Loop
# ================================================================

def bot_loop():
    global scanner, executor, risk, news_analyzer, ranker, pdt, openai_client, trading_agent
    global stock_scorer, price_predictor, portfolio_analyzer

    try:
        _log("Initializing components...")
        scanner = MarketScanner()
        executor = OrderExecutor()
        risk = RiskManager()
        news_analyzer = NewsAnalyzer()
        ranker = AIRanker()
        pdt = PDTTracker()
        openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
        # Research platform components
        stock_scorer = StockScorer(scanner, news_analyzer, openai_client)
        price_predictor = PricePredictor(scanner, news_analyzer, openai_client)
        portfolio_analyzer = PortfolioAnalyzer(stock_scorer, price_predictor, executor)

        tool_registry = ToolRegistry(
            scanner, executor, risk, news_analyzer, pdt,
            stock_scorer, price_predictor, portfolio_analyzer,
        )
        trading_agent = TradingAgent(tool_registry)
        _log("All components ready (Agent + Research initialized)")
        # Pre-cache portfolio analysis in background (fast first, then full)
        def _preload_portfolio():
            try:
                portfolio_analyzer.analyze(fast_mode=True)
                logging.info("[Preload] Fast portfolio done, starting full LLM...")
                portfolio_analyzer.analyze(use_cache=False, fast_mode=False)
                logging.info("[Preload] Full portfolio analysis complete.")
            except Exception as e:
                logging.error(f"[Preload] Error: {e}")
        threading.Thread(target=_preload_portfolio, daemon=True).start()
        _log("Background pre-loading portfolio analysis...")
    except Exception as e:
        _log(f"Initialization failed: {e}", "error")
        traceback.print_exc()
        state["bot_status"] = "error"
        return

    daily_init_done = False
    overnight_exit_done = False
    intraday_entry_done = False
    overnight_entry_done = False
    intraday_exit_done = False
    eod_done = False
    ws_started = False
    last_premarket_check = 0
    last_monitor = 0

    while True:
        try:
            now = datetime.now(ET)
            hm = now.hour * 100 + now.minute
            weekday = now.weekday()

            if weekday >= 5:
                state["bot_status"] = "sleeping"
                _time.sleep(300)
                continue

            # ══════════════════════════════════════
            # Night 20:01 - 03:59 - Sleep
            # ══════════════════════════════════════
            if hm > 2000 or hm < 400:
                state["bot_status"] = "sleeping"
                daily_init_done = False
                overnight_exit_done = False
                intraday_entry_done = False
                overnight_entry_done = False
                intraday_exit_done = False
                eod_done = False
                ws_started = False
                _time.sleep(60)
                continue

            # ══════════════════════════════════════
            # 04:00 - Daily initialization + WebSocket start
            # ══════════════════════════════════════
            if hm >= 400 and not daily_init_done:
                state["bot_status"] = "initializing"
                _log("=== New Trading Day Start ===")
                state["scan_count"] = 0
                state["candidates_count"] = 0
                state["news_analyzed_count"] = 0
                state["today_trades"] = []
                state["recommendations"] = []
                state["overnight_candidates"] = []
                state["intraday_candidates"] = []

                acct = executor.get_account()
                _log(f"Account: ${acct['portfolio_value']:,.2f} | {risk.status(acct['portfolio_value'])}")
                _log(f"PDT: {pdt.status()}")

                scanner.refresh_market_data()
                regime = scanner.get_regime()
                state["regime"] = regime
                _log(f"Regime: {regime} | VIX/VIX3M={scanner.get_vix_vix3m_ratio():.3f} | SPY>200SMA={scanner.get_spy_above_200sma()}")

                scanner.get_tradeable_universe(force_refresh=True)

                if executor.overnight_trades and not ws_started:
                    overnight_tickers = [t for t, v in executor.overnight_trades.items() if v["status"] == "held"]
                    if overnight_tickers:
                        _start_ws_thread(overnight_tickers)
                        ws_started = True
                        _log(f"Pre-market monitoring overnight positions: {overnight_tickers}")

                daily_init_done = True
                state["bot_status"] = "premarket"

            # ══════════════════════════════════════
            # 04:00 - 09:25 - Pre-market mode
            # ══════════════════════════════════════
            if 400 <= hm < 925:
                state["bot_status"] = "premarket"

                t = _time.time()
                if t - last_premarket_check >= 30:
                    last_premarket_check = t
                    _premarket_stop_check()

                _time.sleep(15)
                continue

            # ══════════════════════════════════════
            # 09:25 - 09:30 - Final preparation
            # ══════════════════════════════════════
            if 925 <= hm < 930:
                state["bot_status"] = "premarket"
                _time.sleep(5)
                continue

            # ══════════════════════════════════════
            # 09:30 - 09:45 - Market open observation (no orders)
            # ══════════════════════════════════════
            if 930 <= hm < 945:
                state["bot_status"] = "market_open"
                _time.sleep(10)
                continue

            # ══════════════════════════════════════
            # 09:45 - 10:15 - Overnight position exit
            # ══════════════════════════════════════
            if 945 <= hm < 1015 and not overnight_exit_done:
                state["bot_status"] = "executing"
                _process_overnight_exits()
                held = [t for t, v in executor.overnight_trades.items() if v["status"] == "held"]
                if not held or hm >= 1015:
                    overnight_exit_done = True
                    _log("Overnight exit complete")
                _time.sleep(5)
                continue

            # ══════════════════════════════════════
            # 10:15 - 14:00 - Intraday monitoring
            # ══════════════════════════════════════
            if 1015 <= hm < 1400:
                state["bot_status"] = "running"
                t = _time.time()
                if t - last_monitor >= 15:
                    last_monitor = t
                    _monitor_intraday()
                _time.sleep(10)
                continue

            # ══════════════════════════════════════
            # 14:00 - 14:45 - Intraday entry window
            # ══════════════════════════════════════
            if 1400 <= hm < 1445 and not intraday_entry_done:
                _run_intraday_scan_and_entry()
                intraday_entry_done = True
                state["bot_status"] = "running"

            # ══════════════════════════════════════
            # 14:45 - 15:00 - Monitoring
            # ══════════════════════════════════════
            if 1445 <= hm < 1500:
                _monitor_intraday()
                _time.sleep(10)
                continue

            # ══════════════════════════════════════
            # 15:00 - 15:45 - Overnight stock selection + entry
            # ══════════════════════════════════════
            if 1500 <= hm < 1545 and not overnight_entry_done:
                _run_overnight_scan_and_entry()
                overnight_entry_done = True

            # ══════════════════════════════════════
            # 15:40 - 15:50 - Intraday exit
            # ══════════════════════════════════════
            if 1540 <= hm < 1550 and not intraday_exit_done:
                _intraday_exit_sequence()
                if hm >= 1550:
                    intraday_exit_done = True

            # ══════════════════════════════════════
            # 16:00 - 20:00 - After hours
            # ══════════════════════════════════════
            if 1600 <= hm <= 2000:
                state["bot_status"] = "after_hours"
                if not eod_done:
                    _generate_eod_report()
                    eod_done = True
                _time.sleep(300)
                continue

            _time.sleep(10)

        except Exception as e:
            _log(f"Main loop error: {e}", "error")
            traceback.print_exc()
            _time.sleep(30)


# ================================================================
# API Routes
# ================================================================

@app.route("/")
def index():
    resp = send_file("dashboard.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/account")
def api_account():
    try:
        acct = executor.get_account()
        positions = executor.get_positions()
        return jsonify({
            "account": acct,
            "positions": positions,
            "pdt_remaining": pdt.remaining_trades() if pdt else 0,
            "regime": scanner.get_regime() if scanner else "unknown",
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/state")
def api_state():
    return jsonify({
        "bot_status": state["bot_status"],
        "scan_count": state["scan_count"],
        "candidates_count": state["candidates_count"],
        "last_scan_time": state["last_scan_time"],
        "news_analyzed_count": state["news_analyzed_count"],
        "today_trades": state["today_trades"],
        "activity_log": list(state["activity_log"]),
        "recommendations": state["recommendations"],
        "regime": state.get("regime", "cautious"),
        "auto_trade_enabled": auto_trade_enabled,
    })


@app.route("/api/scan/results")
def api_scan_results():
    combined = []
    for c in state["overnight_candidates"]:
        combined.append(_format_scan_result(c, "overnight"))
    for c in state["intraday_candidates"]:
        combined.append(_format_scan_result(c, "intraday"))
    return jsonify({"results": combined, "count": len(combined)})


def _format_scan_result(c: Dict, strategy: str) -> Dict:
    ind = c.get("indicators", {})
    signals = []
    if strategy == "overnight":
        signals.append(f"RSI(2)={c.get('rsi_2', '?')}")
        signals.append(f"IBS={c.get('ibs', '?')}")
        if ind.get("consecutive_down", 0) >= 3:
            signals.append(f"Down {ind['consecutive_down']} days")
        signals.append(f"NATR={c.get('natr', '?')}%")
    else:
        signals.append(f"Intraday {c.get('intraday_change', '?')}%")
        signals.append(f"Vol ratio {c.get('volume_ratio', '?')}x")

    news_data = c.get("news")
    news_formatted = None
    if news_data:
        news_formatted = {
            "has_news": not news_data.get("vetoed", False),
            "sentiment_score": news_data.get("news_score", 50) - 50,
            "analyses": news_data.get("analyses", []),
            "news_count": len(news_data.get("analyses", [])),
        }

    return {
        "ticker": c["ticker"], "price": c["price"],
        "signal_strength": c.get("signal_strength", 0),
        "volume_ratio": c.get("volume_ratio", 1.0),
        "signals": signals,
        "signal_details": [],
        "indicators": {
            "rsi": ind.get("rsi_14", 50), "rsi_2": ind.get("rsi_2", 50),
            "macd_hist": ind.get("macd_hist", 0),
            "atr": ind.get("atr", 0),
            "bb_upper": ind.get("bb_upper", 0), "bb_middle": ind.get("bb_middle", 0),
            "bb_lower": ind.get("bb_lower", 0),
            "support": ind.get("support", 0), "resistance": ind.get("resistance", 0),
            "avg_volume_20": ind.get("avg_volume_20", 0),
            "ibs": ind.get("ibs", 0.5), "natr": ind.get("natr", 0),
        },
        "news": news_formatted,
        "combined_score": c.get("combined_score"),
        "sector": c.get("sector", "Unknown"),
        "strategy": strategy,
    }


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Agent-powered chat endpoint using custom ReAct loop."""
    try:
        data = request.json
        msg = data.get("message", "")

        if not trading_agent:
            return jsonify({"error": "Agent not initialized yet"})

        result = trading_agent.run(msg)

        tool_calls_summary = [
            {"tool": tc.tool, "args": tc.arguments, "latency_ms": tc.latency_ms}
            for tc in result.tool_calls
        ]

        return jsonify({
            "reply": result.response,
            "model": result.model,
            "tokens": {
                "prompt": result.prompt_tokens,
                "completion": result.completion_tokens,
                "total": result.total_tokens,
            },
            "agent": {
                "iterations": result.iterations,
                "tool_calls": tool_calls_summary,
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ================================================================
# Research Platform API Routes
# ================================================================

_portfolio_refreshing = False

@app.route("/api/portfolio/detailed")
def api_portfolio_detailed():
    """Portfolio page: positions with scores, predictions, recommendations.
    Returns cached data immediately; triggers background refresh if stale."""
    global _portfolio_refreshing
    try:
        if not portfolio_analyzer:
            return jsonify({"error": "Research components not initialized"})

        cached = portfolio_analyzer._cache
        if cached:
            ts, analysis = cached
            # If cache older than 30 min, trigger background refresh
            if _time.time() - ts > 1800 and not _portfolio_refreshing:
                _portfolio_refreshing = True
                def _bg_refresh():
                    global _portfolio_refreshing
                    try:
                        portfolio_analyzer.analyze(use_cache=False, fast_mode=False)
                    finally:
                        _portfolio_refreshing = False
                threading.Thread(target=_bg_refresh, daemon=True).start()
        else:
            # No cache — run fast_mode (algorithmic only, no LLM) for instant results
            if not _portfolio_refreshing:
                _portfolio_refreshing = True
                def _bg_fast_then_full():
                    global _portfolio_refreshing
                    try:
                        logging.info("[Portfolio] Starting fast_mode analysis...")
                        portfolio_analyzer.analyze(fast_mode=True)
                        logging.info("[Portfolio] Fast done. Starting full LLM analysis...")
                        portfolio_analyzer.analyze(use_cache=False, fast_mode=False)
                        logging.info("[Portfolio] Full LLM analysis complete.")
                    except Exception as e:
                        logging.error(f"[Portfolio] Background analysis error: {e}")
                    finally:
                        _portfolio_refreshing = False
                threading.Thread(target=_bg_fast_then_full, daemon=True).start()
            # Return placeholder while fast analysis runs
            analysis = []
            for pos in executor.get_positions():
                analysis.append({
                    "symbol": pos["ticker"],
                    "qty": pos.get("qty", 0),
                    "entry_price": pos.get("entry_price", 0),
                    "current_price": pos.get("current_price", 0),
                    "market_value": pos.get("market_value", 0),
                    "unrealized_pnl": pos.get("unrealized_pnl", 0),
                    "unrealized_pnl_pct": pos.get("unrealized_pnl_pct", 0),
                    "scores": {},
                    "score_details": {},
                    "composite": None,
                    "tier": "Loading",
                    "indicators": {},
                    "predictions": [],
                    "recommendation": "Loading",
                    "recommendation_reason": "AI analysis in progress...",
                })

        acct = executor.get_account()
        return jsonify({
            "account": acct,
            "positions": analysis,
            "pdt_remaining": pdt.remaining_trades() if pdt else 0,
            "regime": scanner.get_regime() if scanner else "unknown",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/api/portfolio/predictions/<symbol>")
def api_portfolio_predictions(symbol):
    """AI price predictions for a single stock."""
    try:
        if not price_predictor:
            return jsonify({"error": "Predictor not initialized"})
        predictions = price_predictor.predict(symbol.upper())
        return jsonify({"symbol": symbol.upper(), "predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)})


def _ai_rerank(scored: list, traits: list, horizons: list) -> list:
    """Use LLM to re-rank scored stocks based on user's trait and horizon preferences."""
    try:
        if not openai_client:
            return scored

        # Only re-rank the scored ones (those with composite != None)
        has_score = [s for s in scored if s.get("composite") is not None]
        no_score = [s for s in scored if s.get("composite") is None]
        if not has_score:
            return scored

        trait_desc = {
            "high_return": "high expected return / strong growth potential",
            "low_beta": "low beta / low correlation with market",
            "low_drawdown": "low maximum drawdown / capital preservation",
            "high_dividend": "high dividend yield",
            "low_volatility": "low price volatility / stable",
            "high_momentum": "strong price momentum / trending up",
            "value": "undervalued / low P/E / value investing",
            "small_cap": "small market cap",
            "large_cap": "large market cap / blue chip",
        }
        horizon_desc = {
            "1d": "1 day", "1w": "1 week", "2w": "2 weeks", "1m": "1 month",
            "3m": "3 months", "6m": "6 months", "1y": "1 year", "3y": "3 years",
        }

        trait_text = ", ".join(trait_desc.get(t, t) for t in traits) if traits else "no specific preference"
        horizon_text = ", ".join(horizon_desc.get(h, h) for h in horizons) if horizons else "no specific horizon"

        stocks_summary = "\n".join(
            f"- {s['ticker']}: price=${s['price']}, change={s['change_pct']}%, "
            f"composite={s.get('composite','-')}, sector={s['sector']}, "
            f"scores={json.dumps(s.get('scores',{}))}"
            for s in has_score[:20]
        )

        prompt = f"""Given these stocks and their scores, re-rank them for an investor who wants:
- Traits: {trait_text}
- Investment horizon: {horizon_text}

Stocks:
{stocks_summary}

Return a JSON array of tickers in order from best to worst fit. Also add a short "match_reason" for each.
Format: [{{"ticker":"XXX","match_reason":"...","adjusted_score":85}}]
Only return the JSON array, no other text."""

        resp = openai_client.chat.completions.create(
            model="gpt-5.4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            rankings = json.loads(match.group())
            ticker_order = {r["ticker"]: i for i, r in enumerate(rankings)}
            reason_map = {r["ticker"]: r.get("match_reason", "") for r in rankings}
            adj_score_map = {r["ticker"]: r.get("adjusted_score") for r in rankings}

            for s in has_score:
                if s["ticker"] in reason_map:
                    s["match_reason"] = reason_map[s["ticker"]]
                if s["ticker"] in adj_score_map and adj_score_map[s["ticker"]] is not None:
                    s["adjusted_score"] = adj_score_map[s["ticker"]]

            has_score.sort(key=lambda x: ticker_order.get(x["ticker"], 999))

        return has_score + no_score
    except Exception as e:
        print(f"[AI Rerank] Error: {e}")
        return scored


@app.route("/api/research/scan", methods=["POST"])
def api_research_scan():
    """Research page: scan market with filters, return scored results."""
    try:
        if not scanner or not stock_scorer:
            return jsonify({"error": "Components not initialized"})

        data = request.json or {}
        horizons = data.get("horizons", [])
        traits = data.get("traits", [])
        filters = {
            "sectors": data.get("sectors", []),
            "min_price": data.get("min_price", 10),
            "max_price": data.get("max_price", 999999),
            "sort_by": data.get("sort_by", "change_pct"),
            "limit": min(data.get("limit", 30), 50),
            "traits": traits,
        }

        # Step 1: scan for candidates (fast — just Alpaca snapshots)
        candidates = scanner.scan_research(filters)

        # Step 2: return immediately — no scoring, no LLM
        # Users click individual stocks for deep analysis
        return jsonify({
            "results": candidates,
            "count": len(candidates),
            "scored_count": 0,
            "filters": filters,
            "horizons": horizons,
            "traits": traits,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/api/research/deep/<symbol>")
def api_research_deep(symbol):
    """Deep analysis for a single stock (uses gpt-5.4)."""
    try:
        if not stock_scorer or not price_predictor or not news_analyzer:
            return jsonify({"error": "Components not initialized"})
        result = deep_analyze(symbol.upper(), stock_scorer, price_predictor, news_analyzer)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/api/research/watchlist", methods=["GET", "POST", "DELETE"])
def api_watchlist():
    """Manage watchlist: GET returns all, POST adds, DELETE removes."""
    global watchlist
    if request.method == "POST":
        data = request.json or {}
        symbol = data.get("symbol", "").upper().strip()
        if symbol and symbol not in watchlist:
            watchlist.append(symbol)
        return jsonify({"watchlist": watchlist})
    elif request.method == "DELETE":
        data = request.json or {}
        symbol = data.get("symbol", "").upper().strip()
        watchlist = [s for s in watchlist if s != symbol]
        return jsonify({"watchlist": watchlist})
    else:
        # GET — return watchlist with latest scores
        items = []
        for s in watchlist:
            try:
                scores = stock_scorer.score_stock(s) if stock_scorer else {}
                items.append({
                    "symbol": s,
                    "price": scores.get("price"),
                    "change_pct": scores.get("change_pct"),
                    "composite": scores.get("composite"),
                    "tier": scores.get("tier"),
                })
            except Exception:
                items.append({"symbol": s, "error": "Failed to score"})
        return jsonify({"watchlist": items})


@app.route("/api/auto-trade", methods=["GET", "POST"])
def api_auto_trade():
    """Toggle auto-trading on/off. GET returns current status, POST toggles."""
    global auto_trade_enabled
    if request.method == "POST":
        data = request.json or {}
        if "enabled" in data:
            auto_trade_enabled = bool(data["enabled"])
            _log(f"{'Auto-trade enabled' if auto_trade_enabled else 'Auto-trade disabled'}")
        return jsonify({"auto_trade_enabled": auto_trade_enabled})
    return jsonify({"auto_trade_enabled": auto_trade_enabled})


@app.route("/api/sectors/list")
def api_sectors_list():
    """Return available sectors for filter dropdowns."""
    from scanner import SECTORS
    unique = sorted(set(SECTORS.values()))
    return jsonify({"sectors": unique})


# ================================================================
# Startup
# ================================================================

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()
    print("Dashboard: http://localhost:5555")
    app.run(host="0.0.0.0", port=5555, debug=False, threaded=True)
