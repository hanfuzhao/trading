"""
Dashboard Server v6 — 完整日流程
70% 隔夜均值回归 + 30% 日内 | WebSocket 盘前监控 | 3变量 Regime
"""
import json
import os
import threading
import time as _time
import traceback
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

ET = ZoneInfo("America/New_York")

app = Flask(__name__)

# ================================================================
# 全局组件
# ================================================================
scanner: Optional[MarketScanner] = None
executor: Optional[OrderExecutor] = None
risk: Optional[RiskManager] = None
news_analyzer: Optional[NewsAnalyzer] = None
ranker: Optional[AIRanker] = None
pdt: Optional[PDTTracker] = None
openai_client: Optional[OpenAI] = None

latest_prices: Dict[str, float] = {}

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
    """当前美东HHMM"""
    now = datetime.now(ET)
    return now.hour * 100 + now.minute


# ================================================================
# WebSocket 价格流（v6第六章: 15行后台线程）
# ================================================================

def _start_ws_thread(symbols: List[str]):
    """启动 WebSocket 后台线程监控实时价格"""
    if not symbols:
        return
    try:
        import websocket
    except ImportError:
        _log("⚠️ websocket-client 未安装，跳过实时价格流", "warning")
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
                    _log(f"WebSocket已订阅 {len(symbols)} 只: {symbols[:5]}")
        except Exception:
            pass

    def on_error(ws, error):
        _log(f"WebSocket错误: {error}", "warning")

    ws = websocket.WebSocketApp(
        ALPACA_WS_URL,
        on_open=on_open, on_message=on_message, on_error=on_error,
    )
    wst = threading.Thread(target=ws.run_forever, kwargs={"ping_interval": 10}, daemon=True)
    wst.start()
    _log("WebSocket后台线程已启动")


# ================================================================
# 宏观评分
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
# 隔夜出场逻辑（v6第四章: 缺口方向）
# ================================================================

def _process_overnight_exits():
    """09:45-10:15: 根据缺口方向出场隔夜持仓"""
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
                ok, pnl = executor.exit_overnight(ticker, current, f"跳空有利{gap_pct:+.1f}%→9:45卖出")
                if ok:
                    risk.record_trade_result(pnl)
                    state["today_trades"].append({"ticker": ticker, "pnl": pnl, "type": "overnight_exit"})
                    _log(f"🌙 隔夜出场 {ticker} 跳空{gap_pct:+.1f}% PnL${pnl:+.2f}")
            elif current <= trade["stop_line"]:
                ok, pnl = executor.exit_overnight(ticker, current, f"跳空不利触及止损线")
                if ok:
                    risk.record_trade_result(pnl)
                    state["today_trades"].append({"ticker": ticker, "pnl": pnl, "type": "overnight_stop"})
                    _log(f"🔴 隔夜止损 {ticker} 跌破止损线 PnL${pnl:+.2f}", "warning")
        elif hm >= 1000:
            ok, pnl = executor.exit_overnight(ticker, current, f"10:00定时出场 缺口{gap_pct:+.1f}%")
            if ok:
                risk.record_trade_result(pnl)
                state["today_trades"].append({"ticker": ticker, "pnl": pnl, "type": "overnight_exit"})
                _log(f"🌙 隔夜出场 {ticker} @10:00 缺口{gap_pct:+.1f}% PnL${pnl:+.2f}")


# ================================================================
# 盘前止损监控（v6第六章: 4:00AM WebSocket）
# ================================================================

def _premarket_stop_check():
    """4:00 AM起: 检查隔夜持仓是否触发极端缺口止损"""
    for ticker, trade in list(executor.overnight_trades.items()):
        if trade["status"] != "held":
            continue
        price = latest_prices.get(ticker)
        if price and price <= trade["stop_line"]:
            _log(f"⚠️ 盘前止损触发: {ticker} ${price:.2f} < 止损线${trade['stop_line']:.2f}", "warning")
            executor.submit_premarket_stop(ticker, trade["shares"], price)
            risk.record_trade_result((price - trade["entry_price"]) * trade["shares"])
            state["today_trades"].append({"ticker": ticker, "type": "premarket_stop"})


# ================================================================
# 隔夜入场流程（v6第九章: 15:00-15:45）
# ================================================================

def _run_overnight_scan_and_entry():
    """15:00-15:45: 扫描→新闻过滤→o3排名→入场"""
    state["bot_status"] = "scanning"
    _log("🌙 开始隔夜选股扫描...")
    candidates = scanner.scan_overnight()
    state["overnight_candidates"] = candidates
    state["scan_count"] += 1
    state["candidates_count"] = len(candidates)
    state["last_scan_time"] = datetime.now(ET).isoformat()

    if not candidates:
        _log("隔夜扫描: 无候选")
        return

    _log(f"隔夜扫描: {len(candidates)} 只候选")

    state["bot_status"] = "analyzing_news"
    macro_score = _calc_macro_score()

    enriched = []
    for c in candidates[:20]:
        news_result = news_analyzer.check_structural_risk(c["ticker"], c["price"])
        state["news_analyzed_count"] += 1
        if news_result["vetoed"]:
            _log(f"❌ {c['ticker']} 被新闻否决: {news_result['reason']}", "warning")
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
        _log("新闻过滤后无候选")
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
        _log("o3未推荐任何隔夜持仓")
        state["bot_status"] = "running"
        return

    state["recommendations"] = recs
    _log(f"o3推荐 {len(recs)} 只隔夜持仓")

    hm = _hm()
    if hm < 1530:
        _log("等待15:30入场窗口")
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
            _log(f"⛔ 风控拒绝: {reason}", "warning")
            break

        sector = sectors.get(ticker, "Unknown")
        ok, msg, params = risk.validate_overnight(
            ticker, rec.get("entry_price", 0), acct["portfolio_value"],
            current_overnight, regime, sector,
        )
        if not ok:
            _log(f"⛔ {ticker} 验证失败: {msg}", "warning")
            continue

        c_match = next((c for c in state["overnight_candidates"] if c["ticker"] == ticker), None)
        natr = c_match["natr"] if c_match else 2.0
        atr = c_match["indicators"]["atr"] if c_match and "indicators" in c_match else 1.0

        ok, msg = executor.enter_overnight(
            ticker, params["shares"], params["entry_price"], atr, natr,
        )
        if ok:
            current_overnight.append(ticker)
            _log(f"✅ 隔夜入场 {ticker} {params['shares']}股 仓位{params['position_pct']}%")
            state["today_trades"].append({"ticker": ticker, "type": "overnight_entry"})
        else:
            _log(f"❌ 隔夜入场失败 {ticker}: {msg}", "warning")


# ================================================================
# 日内入场流程（v6第九章: 14:00-14:45）
# ================================================================

def _run_intraday_scan_and_entry():
    """14:00-14:45: 扫描→新闻过滤→o3排名→入场"""
    if not pdt.can_day_trade():
        return

    state["bot_status"] = "scanning"
    _log("☀️ 开始日内扫描...")
    candidates = scanner.scan_intraday()
    state["intraday_candidates"] = candidates

    if not candidates:
        _log("日内扫描: 无候选")
        state["bot_status"] = "running"
        return

    _log(f"日内扫描: {len(candidates)} 只候选")

    state["bot_status"] = "analyzing_news"
    macro_score = _calc_macro_score()

    enriched = []
    for c in candidates[:15]:
        news_result = news_analyzer.check_structural_risk(c["ticker"], c["price"])
        state["news_analyzed_count"] += 1
        if news_result["vetoed"]:
            _log(f"❌ {c['ticker']} 被新闻否决: {news_result['reason']}", "warning")
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
            _log(f"周一/周二观察期, 最高分{top_score}<{MONDAY_EXCEPTION_SCORE}, 不使用PDT名额")
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
        _log("o3未推荐日内交易")
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
            _log(f"⛔ {ticker} 日内验证失败: {msg}", "warning")
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
            _log(f"✅ 日内入场 {ticker} {params['shares']}股")
            total_pos += 1


# ================================================================
# 日内持仓监控
# ================================================================

def _monitor_intraday():
    """每15秒: 检查日内持仓止损/止盈/时间止损"""
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
            _log(f"⏰ 时间止损 {ticker}")

    closed = executor.check_closed_trades()
    for c in closed:
        risk.record_trade_result(c["pnl"])
        state["today_trades"].append(c)


# ================================================================
# 日内出场（15:40-15:50）
# ================================================================

def _intraday_exit_sequence():
    """15:40: Limit出场 → 15:48: 市价清仓 → 15:50: 确认零日内持仓"""
    hm = _hm()
    if hm >= 1548:
        closed = executor.close_all_intraday()
        for c in closed:
            pnl = c.get("unrealized_pnl", 0)
            risk.record_trade_result(pnl)
            state["today_trades"].append({"ticker": c["ticker"], "pnl": pnl, "type": "forced_close"})
            _log(f"🔴 15:48强制平仓 {c['ticker']} PnL${pnl:+.2f}")
    if hm >= 1550:
        executor.confirm_zero_intraday()


# ================================================================
# 盘后日报
# ================================================================

def _generate_eod_report():
    trades = state["today_trades"]
    total_pnl = sum(t.get("pnl", 0) for t in trades if "pnl" in t)
    _log(f"📊 盘后日报: {len(trades)}笔交易 | 总PnL${total_pnl:+.2f}")
    _log(f"📊 Regime={scanner.get_regime()} | VIX/VIX3M={scanner.get_vix_vix3m_ratio():.3f}")
    _log(f"📊 PDT剩余: {pdt.remaining_trades()}/3 | {risk.status(executor.get_account()['portfolio_value'])}")

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
# 主循环
# ================================================================

def bot_loop():
    global scanner, executor, risk, news_analyzer, ranker, pdt, openai_client

    try:
        _log("初始化组件...")
        scanner = MarketScanner()
        executor = OrderExecutor()
        risk = RiskManager()
        news_analyzer = NewsAnalyzer()
        ranker = AIRanker()
        pdt = PDTTracker()
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        _log("✅ 全部组件就绪")
    except Exception as e:
        _log(f"❌ 初始化失败: {e}", "error")
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
            # 深夜 20:01 - 03:59 — 休眠
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
            # 04:00 — 每日初始化 + WebSocket启动
            # ══════════════════════════════════════
            if hm >= 400 and not daily_init_done:
                state["bot_status"] = "initializing"
                _log("═══ 新交易日启动 ═══")
                state["scan_count"] = 0
                state["candidates_count"] = 0
                state["news_analyzed_count"] = 0
                state["today_trades"] = []
                state["recommendations"] = []
                state["overnight_candidates"] = []
                state["intraday_candidates"] = []

                acct = executor.get_account()
                _log(f"账户: ${acct['portfolio_value']:,.2f} | {risk.status(acct['portfolio_value'])}")
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
                        _log(f"盘前监控隔夜持仓: {overnight_tickers}")

                daily_init_done = True
                state["bot_status"] = "premarket"

            # ══════════════════════════════════════
            # 04:00 - 09:25 — 盘前模式
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
            # 09:25 - 09:30 — 最后准备
            # ══════════════════════════════════════
            if 925 <= hm < 930:
                state["bot_status"] = "premarket"
                _time.sleep(5)
                continue

            # ══════════════════════════════════════
            # 09:30 - 09:45 — 开盘观察期（不下单）
            # ══════════════════════════════════════
            if 930 <= hm < 945:
                state["bot_status"] = "market_open"
                _time.sleep(10)
                continue

            # ══════════════════════════════════════
            # 09:45 - 10:15 — 隔夜持仓出场
            # ══════════════════════════════════════
            if 945 <= hm < 1015 and not overnight_exit_done:
                state["bot_status"] = "executing"
                _process_overnight_exits()
                held = [t for t, v in executor.overnight_trades.items() if v["status"] == "held"]
                if not held or hm >= 1015:
                    overnight_exit_done = True
                    _log("隔夜出场完成")
                _time.sleep(5)
                continue

            # ══════════════════════════════════════
            # 10:15 - 14:00 — 盘中监控
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
            # 14:00 - 14:45 — 日内入场窗口
            # ══════════════════════════════════════
            if 1400 <= hm < 1445 and not intraday_entry_done:
                _run_intraday_scan_and_entry()
                intraday_entry_done = True
                state["bot_status"] = "running"

            # ══════════════════════════════════════
            # 14:45 - 15:00 — 监控
            # ══════════════════════════════════════
            if 1445 <= hm < 1500:
                _monitor_intraday()
                _time.sleep(10)
                continue

            # ══════════════════════════════════════
            # 15:00 - 15:45 — 隔夜选股 + 入场
            # ══════════════════════════════════════
            if 1500 <= hm < 1545 and not overnight_entry_done:
                _run_overnight_scan_and_entry()
                overnight_entry_done = True

            # ══════════════════════════════════════
            # 15:40 - 15:50 — 日内出场
            # ══════════════════════════════════════
            if 1540 <= hm < 1550 and not intraday_exit_done:
                _intraday_exit_sequence()
                if hm >= 1550:
                    intraday_exit_done = True

            # ══════════════════════════════════════
            # 16:00 - 20:00 — 盘后
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
            _log(f"❌ 主循环异常: {e}", "error")
            traceback.print_exc()
            _time.sleep(30)


# ================================================================
# API 路由
# ================================================================

@app.route("/")
def index():
    return send_file("dashboard.html")


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
            signals.append(f"连跌{ind['consecutive_down']}天")
        signals.append(f"NATR={c.get('natr', '?')}%")
    else:
        signals.append(f"日内{c.get('intraday_change', '?')}%")
        signals.append(f"量比{c.get('volume_ratio', '?')}x")

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
    try:
        data = request.json
        msg = data.get("message", "")
        model = data.get("model", "gpt-4.1-mini")

        acct = executor.get_account() if executor else {}
        positions = executor.get_positions() if executor else []
        regime = scanner.get_regime() if scanner else "unknown"

        context = f"""你是一个AI交易助手。以下是当前实时状态：

账户: ${acct.get('portfolio_value', 0):,.2f} | 现金: ${acct.get('cash', 0):,.2f}
Regime: {regime} | PDT剩余: {pdt.remaining_trades() if pdt else 0}/3
持仓: {len(positions)}个 | 隔夜候选: {len(state['overnight_candidates'])} | 日内候选: {len(state['intraday_candidates'])}
今日交易: {len(state['today_trades'])}笔
Bot状态: {state['bot_status']}
策略: v6 隔夜均值回归(RSI(2)<15+IBS<0.25) + 日内午后反转(14:00-14:45)
风控: {risk.status(acct.get('portfolio_value', 0)) if risk else '未初始化'}

用中文回答。简洁专业。"""

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": msg},
            ],
            max_completion_tokens=1000,
        )
        reply = response.choices[0].message.content
        usage = response.usage
        return jsonify({
            "reply": reply, "model": model,
            "tokens": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens, "total": usage.total_tokens},
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ================================================================
# 启动
# ================================================================

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()
    print("Dashboard: http://localhost:5555")
    app.run(host="0.0.0.0", port=5555, debug=False)
