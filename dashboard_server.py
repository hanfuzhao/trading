"""
交易仪表盘 + 自动交易Bot v3 — 一体化服务
完整实现策略手册第十章「一天的完整流程」
"""
import json
import os
import sys
import time
import threading
import traceback
from collections import deque
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

ET = ZoneInfo("America/New_York")

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET, OPENAI_API_KEY,
    MODEL_FAST, MODEL_DEEP, MODEL_RANK, LOG_DIR,
    VOL_REGIMES, MAX_CONCURRENT_POSITIONS, MAX_CAPITAL,
    MAX_DAILY_LOSS_PCT,
    ENTRY_WINDOW_1_START, ENTRY_WINDOW_1_END,
    ENTRY_WINDOW_2_START, ENTRY_WINDOW_2_END,
    BLACKOUT_OPEN_END, MIDDAY_DEAD_START, MIDDAY_DEAD_END,
    NO_NEW_POSITION_AFTER,
    MONDAY_MIN_SCORE,
)
from scanner import MarketScanner
from news_analyzer import NewsAnalyzer
from ranker import DeepRanker
from executor import OrderExecutor
from pdt_tracker import PDTTracker
from risk_manager import RiskManager
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ================================================================
# 全局共享状态
# ================================================================
state = {
    "bot_status": "initializing",
    "scan_results": [],
    "news_results": {},
    "ranking_result": {},
    "recommendations": [],
    "last_scan_time": None,
    "last_news_time": None,
    "scan_count": 0,
    "activity_log": deque(maxlen=200),
    "chat_history": [],
    "today_trades": [],
    "error": None,
}

lock = threading.Lock()


def log_activity(msg, level="info"):
    ts = datetime.now(ET).strftime("%H:%M:%S")
    with lock:
        state["activity_log"].append({"time": ts, "msg": msg, "level": level})
    print(f"[{ts}] {msg}")


# ================================================================
# 核心组件（延迟初始化）
# ================================================================
scanner = None
news_analyzer_inst = None
ranker = None
executor = None
pdt = None
risk = None
openai_client = None


def init_components():
    global scanner, news_analyzer_inst, ranker, executor, pdt, risk, openai_client
    print("[Bot] 初始化 MarketScanner...")
    scanner = MarketScanner()
    print("[Bot] 初始化 NewsAnalyzer...")
    news_analyzer_inst = NewsAnalyzer()
    print("[Bot] 初始化 DeepRanker...")
    ranker = DeepRanker()
    print("[Bot] 初始化 OrderExecutor...")
    executor = OrderExecutor()
    print("[Bot] 初始化 PDTTracker...")
    pdt = PDTTracker()
    print("[Bot] 初始化 RiskManager...")
    risk = RiskManager()
    print("[Bot] 初始化 OpenAI...")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ================================================================
# 时间工具
# ================================================================
def _hm(t=None):
    """返回当前美东时间的 HHMM 整数，方便比较"""
    t = t or datetime.now(ET)
    return t.hour * 100 + t.minute


def _parse_hm(s: str) -> int:
    return int(s.replace(":", ""))


def _is_entry_allowed() -> bool:
    """v3第六章：仅在两个入场窗口内允许开仓"""
    hm = _hm()
    w1 = _parse_hm(ENTRY_WINDOW_1_START) <= hm < _parse_hm(ENTRY_WINDOW_1_END)
    w2 = _parse_hm(ENTRY_WINDOW_2_START) <= hm < _parse_hm(ENTRY_WINDOW_2_END)
    return w1 or w2


def _is_power_hour() -> bool:
    hm = _hm()
    return _parse_hm(ENTRY_WINDOW_2_START) <= hm < _parse_hm(ENTRY_WINDOW_2_END)


# ================================================================
# 宏观评分（v3第四章：SPY ±30 + VIXY ±20 + USO ±20 → 归一化0-100）
# ================================================================
def _calc_macro_score() -> float:
    score = 50
    spy_chg = scanner.get_spy_change()
    score += min(max(spy_chg * 10, -30), 30)

    vol_regime = scanner.get_vol_regime()
    vixy_adj = {"low": 15, "medium": 0, "high": -10, "extreme": -20}
    score += vixy_adj.get(vol_regime, 0)

    uso_chg = scanner.get_uso_change()
    score += min(max(uso_chg * 4, -20), 20)

    return max(0, min(100, score))


# ================================================================
# Bot 主循环（v3第十章完整流程）
# ================================================================
def bot_loop():
    try:
        print("[Bot] 开始初始化组件...")
        init_components()
        print("[Bot] 组件初始化完成")
    except Exception as e:
        print(f"[Bot] 初始化失败: {e}")
        traceback.print_exc()
        state["bot_status"] = "error"
        state["error"] = f"初始化失败: {e}"
        return

    log_activity("Bot v3 初始化完成")
    state["bot_status"] = "running"

    try:
        account = executor.get_account()
        log_activity(f"账户 ${account['portfolio_value']:,.2f} | 现金 ${account['cash']:,.2f}")
        log_activity(f"PDT: {pdt.status()}")
    except Exception as e:
        log_activity(f"Alpaca 连接失败: {e}", "error")
        state["bot_status"] = "error"
        state["error"] = str(e)
        return

    last_tech_scan = 0
    last_pos_monitor = 0
    last_pos_news = 0
    last_web_search = 0
    daily_init_done: str = ""
    o3_premarket_done: str = ""
    eod_done: str = ""
    close_profitable_done: str = ""
    close_all_done: str = ""
    final_check_done: str = ""

    while True:
        try:
            now = datetime.now(ET)
            today_str = now.strftime("%Y-%m-%d")
            hm = _hm(now)

            # ── 周末 ──
            if now.weekday() >= 5:
                if state["bot_status"] != "weekend":
                    state["bot_status"] = "weekend"
                    log_activity("周末休市")
                time.sleep(300)
                continue

            # ── 深夜 20:00 - 04:00 ──
            if now.hour >= 20 or now.hour < 4:
                if state["bot_status"] != "sleeping":
                    state["bot_status"] = "sleeping"
                    log_activity("深夜休眠，04:00 恢复")
                time.sleep(300)
                continue

            # ══════════════════════════════════════
            # 04:00 — 每日初始化
            # ══════════════════════════════════════
            if daily_init_done != today_str and now.hour >= 4:
                daily_init_done = today_str
                eod_done = ""
                close_profitable_done = ""
                close_all_done = ""
                final_check_done = ""
                o3_premarket_done = ""
                state["today_trades"] = []
                state["news_results"] = {}
                state["recommendations"] = []
                state["scan_count"] = 0

                log_activity("═══ 每日初始化 ═══")
                log_activity(f"PDT 剩余名额: {pdt.remaining_trades()}")

                scanner.refresh_market_data()
                log_activity(f"VIXY: ${scanner.get_vixy_price():.2f} → regime: {scanner.get_vol_regime()}")
                log_activity(f"SPY 日变化: {scanner.get_spy_change():+.2f}%")
                log_activity(f"USO 日变化: {scanner.get_uso_change():+.2f}%")

                scanner.get_tradeable_universe(force_refresh=True)

            # ══════════════════════════════════════
            # 04:05 - 09:25 — 盘前扫描
            # ══════════════════════════════════════
            if now.hour < 9 or (now.hour == 9 and now.minute < 25):
                if state["bot_status"] != "premarket":
                    state["bot_status"] = "premarket"
                    log_activity("盘前监控模式")

                # 盘前技术扫描（每30分钟）
                if time.time() - last_tech_scan >= 1800:
                    _run_tech_scan()
                    last_tech_scan = time.time()

                # 08:30 Web Search 补充
                if hm >= 830 and time.time() - last_web_search >= 3600:
                    _run_web_search_supplement()
                    last_web_search = time.time()

                time.sleep(60)
                continue

            # ══════════════════════════════════════
            # 09:00 - 09:25 — o3 盘前排名
            # ══════════════════════════════════════
            if 900 <= hm < 930 and o3_premarket_done != today_str:
                o3_premarket_done = today_str
                log_activity("═══ o3 盘前排名 ═══")
                _run_news_analysis()
                _run_o3_ranking()

            # ══════════════════════════════════════
            # 09:30 - 09:45 — 开盘观察期（不下单，只扫描）
            # ══════════════════════════════════════
            if 930 <= hm < 945:
                state["bot_status"] = "observing"
                if time.time() - last_tech_scan >= 60:
                    _run_tech_scan()
                    last_tech_scan = time.time()
                if time.time() - last_pos_monitor >= 15:
                    _monitor_positions()
                    last_pos_monitor = time.time()
                time.sleep(5)
                continue

            # ══════════════════════════════════════
            # 15:30+ — 尾盘收盘程序
            # ══════════════════════════════════════
            if hm >= 1530:
                state["bot_status"] = "closing"

                # 15:45 盈利持仓平仓
                if hm >= 1545 and close_profitable_done != today_str:
                    close_profitable_done = today_str
                    closed = executor.close_profitable_positions()
                    for p in closed:
                        risk.record_trade_result(p["unrealized_pnl"])
                        pdt.record_day_trade(p["ticker"], p["unrealized_pnl"])
                        log_activity(f"15:45 锁利: {p['ticker']} PnL${p['unrealized_pnl']:+.2f}")

                # 15:48 全部平仓
                if hm >= 1548 and close_all_done != today_str:
                    close_all_done = today_str
                    executor.cancel_pending_orders()
                    closed = executor.close_remaining_positions()
                    for p in closed:
                        risk.record_trade_result(p["unrealized_pnl"])
                        pdt.record_day_trade(p["ticker"], p["unrealized_pnl"])
                        log_activity(f"15:48 强平: {p['ticker']} PnL${p['unrealized_pnl']:+.2f}")

                # 15:50 最终确认
                if hm >= 1550 and final_check_done != today_str:
                    final_check_done = today_str
                    executor.confirm_zero_positions()

                # 持仓监控（每分钟）
                if time.time() - last_pos_monitor >= 60:
                    _monitor_positions()
                    last_pos_monitor = time.time()

                time.sleep(10)
                continue

            # ══════════════════════════════════════
            # 16:00 - 20:00 — 盘后
            # ══════════════════════════════════════
            if now.hour >= 16:
                if eod_done != today_str:
                    eod_done = today_str
                    state["bot_status"] = "after_hours"
                    _generate_eod()
                time.sleep(120)
                continue

            # ══════════════════════════════════════
            # 09:45 - 15:30 — 盘中交易
            # ══════════════════════════════════════
            state["bot_status"] = "market_open"

            # 持仓监控（每15秒）
            if time.time() - last_pos_monitor >= 15:
                _monitor_positions()
                last_pos_monitor = time.time()

            # 持仓新闻轮询（每5分钟）
            if time.time() - last_pos_news >= 300:
                _check_position_news()
                last_pos_news = time.time()

            # 技术面扫描（频率由时段决定）
            scan_interval = _get_scan_interval(hm)
            if time.time() - last_tech_scan >= scan_interval:
                _run_tech_scan()
                last_tech_scan = time.time()

            # 每小时 Web Search 对 Top20 候选
            if time.time() - last_web_search >= 3600:
                _run_web_search_supplement()
                last_web_search = time.time()

            # 入场窗口内：检查推荐并执行
            if _is_entry_allowed() and state["recommendations"]:
                _try_execute_recommendations()

            time.sleep(5)

        except Exception as e:
            log_activity(f"Bot 主循环异常: {e}", "error")
            traceback.print_exc()
            state["error"] = str(e)
            time.sleep(30)


# ================================================================
# 扫描频率（v3第三章3.2节）
# ================================================================
def _get_scan_interval(hm: int) -> int:
    if hm < 925:
        return 1800
    if hm < 945:
        return 60
    if hm < 1015:
        return 300
    if hm < 1130:
        return 900
    if hm < 1330:
        return 1800
    if hm < 1445:
        return 900
    if hm < 1530:
        return 300
    return 60


# ================================================================
# 技术面扫描（免费，不调用 OpenAI）
# ================================================================
def _run_tech_scan():
    state["bot_status"] = "scanning"
    try:
        candidates = scanner.scan()
        state["scan_results"] = candidates
        state["scan_count"] = state.get("scan_count", 0) + 1
        state["last_scan_time"] = datetime.now(ET).isoformat()
        if candidates:
            top5 = ", ".join(f"{c['ticker']}({c['signal_strength']})" for c in candidates[:5])
            log_activity(f"扫描: {len(candidates)} 只异动 | Top5: {top5}")
        else:
            log_activity("扫描完成: 无异动")
    except Exception as e:
        log_activity(f"扫描失败: {e}", "error")


# ================================================================
# 新闻分析（Top20 → mini/5.4 → 综合评分）
# ================================================================
def _run_news_analysis():
    candidates = state["scan_results"]
    if not candidates:
        _run_tech_scan()
        candidates = state["scan_results"]
    if not candidates:
        return

    state["bot_status"] = "analyzing_news"
    top20 = sorted(candidates, key=lambda x: x["signal_strength"], reverse=True)[:20]
    log_activity(f"新闻分析: Top {len(top20)} 只...")

    enriched = []
    vol_regime = scanner.get_vol_regime()
    high_vol = vol_regime in ("high", "extreme")

    for c in top20:
        ticker = c["ticker"]
        tech_ctx = {
            "price": c["price"],
            "change_pct": 0,
            "volume_ratio": c["volume_ratio"],
            "rsi": c["indicators"]["rsi"],
            "macd_hist": c["indicators"]["macd_hist"],
            "signal_strength": c["signal_strength"],
        }
        try:
            news_result = news_analyzer_inst.analyze_ticker(ticker, tech_ctx)
            c["news_analysis"] = news_result
            state["news_results"][ticker] = news_result
            state["last_news_time"] = datetime.now(ET).isoformat()

            # v3第四章4.6: 进入第三层的条件
            pass_filter = False
            analyses = news_result.get("analyses", [])

            # 条件1: sentiment≠neutral AND confidence≥60 AND severity≥5
            has_strong = any(
                a.get("confidence", 0) >= 60 and a.get("intraday_severity", 0) >= 5
                and a.get("sentiment") != "neutral"
                for a in analyses
            )
            if has_strong:
                pass_filter = True

            # 条件2: 无新闻但技术极强
            if not news_result["has_news"] and c["signal_strength"] >= 70 and c["volume_ratio"] >= 4:
                pass_filter = True

            # 条件3: 高severity但价格未反应
            has_unreacted = any(
                a.get("intraday_severity", 0) >= 7
                for a in analyses
            ) and c["volume_ratio"] < 2
            if has_unreacted:
                pass_filter = True

            if not pass_filter:
                continue

            # 综合评分
            tech_score = c["signal_strength"]
            news_score = news_result["sentiment_score"]
            macro_score = _calc_macro_score()

            # v3第四章4.5: 均值回归新闻反转
            news_score_adj = abs(news_score)
            if high_vol and news_score < -30:
                structural_bad = any(
                    a.get("is_structural") is True and a.get("intraday_severity", 0) >= 7
                    for a in analyses
                )
                if structural_bad:
                    tech_score *= 0.3
                    news_score_adj = 0
                else:
                    news_score_adj = abs(news_score) * 0.6

            combined = tech_score * 0.45 + news_score_adj * 0.35 + macro_score * 0.20

            c["combined_score"] = round(combined, 2)
            c["news_direction"] = "bullish" if news_score > 0 else ("bearish" if news_score < 0 else "neutral")
            enriched.append(c)
            log_activity(f"✅ {ticker} | 综合{combined:.1f} T{tech_score:.0f} N{news_score:.1f} M{macro_score:.0f}")
        except Exception as e:
            log_activity(f"新闻分析 {ticker} 失败: {e}", "error")

    enriched.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    state["enriched_candidates"] = enriched[:20]
    log_activity(f"新闻筛选: {len(enriched)}/{len(top20)} 通过")


# ================================================================
# Web Search 补充（每小时对Top20候选）
# ================================================================
def _run_web_search_supplement():
    candidates = state["scan_results"][:20]
    if not candidates:
        return
    log_activity(f"Web Search 补充: {len(candidates)} 只")
    for c in candidates:
        if c["ticker"] in state["news_results"]:
            continue
        try:
            news = news_analyzer_inst.web_search_supplement(c["ticker"])
            if news:
                result = news_analyzer_inst.analyze_ticker(c["ticker"])
                state["news_results"][c["ticker"]] = result
        except Exception:
            pass


# ================================================================
# o3 排名
# ================================================================
def _run_o3_ranking():
    enriched = state.get("enriched_candidates", [])
    if not enriched:
        log_activity("无候选通过新闻筛选，跳过o3")
        return

    if not pdt.can_day_trade():
        log_activity(f"PDT 名额已用完 ({pdt.status()})，仅记录信号", "warning")
        return

    state["bot_status"] = "ranking"
    log_activity(f"调用 o3 深度排名 ({len(enriched)} 只)...")

    try:
        account = executor.get_account()
        positions = executor.get_positions()
        remaining = pdt.remaining_trades()
        now = datetime.now(ET)
        weekday = ["周一", "周二", "周三", "周四", "周五"][min(now.weekday(), 4)]
        remaining_days = max(5 - now.weekday(), 1)

        vol_regime = scanner.get_vol_regime()
        scan_mode = enriched[0].get("scan_mode", "momentum")

        ranking = ranker.rank_candidates(
            candidates=enriched[:20],
            remaining_day_trades=remaining,
            portfolio_value=account["portfolio_value"],
            cash=account["cash"],
            current_positions=positions,
            weekday=weekday,
            remaining_trading_days=remaining_days,
            vol_regime=vol_regime,
            scan_mode=scan_mode,
            spy_change_pct=scanner.get_spy_change(),
            uso_change_pct=scanner.get_uso_change(),
        )
        state["ranking_result"] = ranking
        state["recommendations"] = ranking.get("recommendations", [])

        if ranking.get("save_bullets"):
            log_activity(f"💾 o3: {ranking.get('save_reason', '')}", "warning")

        for rec in ranking.get("recommendations", []):
            log_activity(
                f"🎯 #{rec['rank']} {rec['ticker']} | "
                f"置信度{rec.get('confidence', 0)} R:R{rec.get('risk_reward_ratio', 0):.1f} | "
                f"入场${rec.get('entry_price', 0):.2f} SL${rec.get('stop_loss', 0):.2f}"
            )
    except Exception as e:
        log_activity(f"o3 排名失败: {e}", "error")
        traceback.print_exc()


# ================================================================
# 执行推荐
# ================================================================
def _try_execute_recommendations():
    for rec in list(state["recommendations"]):
        _execute_recommendation(rec)
    state["recommendations"] = []


def _execute_recommendation(rec):
    ticker = rec["ticker"]

    if not _is_entry_allowed():
        return

    # Power Hour: R:R ≥ 2.0
    if _is_power_hour() and rec.get("risk_reward_ratio", 0) < 2.0:
        log_activity(f"⏳ Power Hour {ticker} R:R{rec.get('risk_reward_ratio', 0):.1f} < 2.0，跳过", "warning")
        return

    if not pdt.can_day_trade():
        log_activity(f"⛔ PDT 名额不足，跳过 {ticker}", "warning")
        return

    now_hm = _hm()
    if now_hm >= _parse_hm(NO_NEW_POSITION_AFTER):
        log_activity(f"⏳ {NO_NEW_POSITION_AFTER} 后不开新仓，跳过 {ticker}", "warning")
        return

    account = executor.get_account()
    can_trade, reason = risk.can_trade(account["portfolio_value"])
    if not can_trade:
        log_activity(f"⛔ 风控拦截 {ticker}: {reason}", "warning")
        return

    available_cash = executor.get_available_cash()
    if available_cash < rec.get("entry_price", 0):
        log_activity(f"⛔ 现金${available_cash:.2f}不足，跳过 {ticker}", "warning")
        return

    vol_regime = scanner.get_vol_regime()
    positions = executor.get_positions()
    active_pos = [p for p in positions if p["ticker"] not in {"SGOV"}]

    valid, msg, order_params = risk.validate_order(
        ticker=ticker,
        price=rec.get("entry_price", 0),
        stop_loss=rec.get("stop_loss", 0),
        position_size_pct=rec.get("position_size_pct", 10),
        portfolio_value=account["portfolio_value"],
        current_positions=len(active_pos),
        vol_regime=vol_regime,
    )
    if not valid:
        log_activity(f"⛔ 订单校验失败 {ticker}: {msg}", "warning")
        return

    shares = order_params["shares"]
    max_by_cash = int(available_cash / rec.get("entry_price", 1))
    if shares > max_by_cash:
        shares = max_by_cash
    if shares <= 0:
        log_activity(f"⛔ 现金不够买1股 {ticker}", "warning")
        return

    # 计算TP1/TP2（如果o3没提供，用风险管理器算）
    tp1 = rec.get("take_profit_1", 0)
    tp2 = rec.get("take_profit_2", 0)
    atr = 0
    for c in state.get("enriched_candidates", []):
        if c["ticker"] == ticker:
            atr = c.get("indicators", {}).get("atr", 0)
            break
    if not tp1 or not tp2:
        stops = risk.calculate_stops(rec["entry_price"], atr, vol_regime)
        tp1 = tp1 or stops["take_profit_1"]
        tp2 = tp2 or stops["take_profit_2"]

    success, msg = executor.execute_entry(
        ticker=ticker, shares=shares,
        entry_price=rec["entry_price"],
        stop_loss=rec["stop_loss"],
        take_profit_1=tp1, take_profit_2=tp2,
        atr=atr,
    )

    if success:
        log_activity(f"✅ BUY {shares}股 {ticker} | {msg}")
        state["today_trades"].append({
            "time": datetime.now(ET).isoformat(),
            "ticker": ticker, "action": "BUY", "shares": shares,
            "price": rec["entry_price"],
            "stop_loss": rec["stop_loss"],
            "tp1": tp1, "tp2": tp2,
        })
    else:
        log_activity(f"❌ 下单失败 {ticker}: {msg}", "error")


# ================================================================
# 持仓监控（v3: 每15秒）
# ================================================================
def _monitor_positions():
    try:
        positions = executor.get_positions()
        for p in positions:
            if p["ticker"] in {"SGOV"}:
                continue
            if executor.check_time_stop(p["ticker"], p["current_price"]):
                pnl = p["unrealized_pnl"]
                risk.record_trade_result(pnl)
                pdt.record_day_trade(p["ticker"], pnl)
                log_activity(f"⏰ {p['ticker']} 时间止损 PnL${pnl:+.2f}", "warning")
                continue
            executor.update_trailing_stop(p["ticker"], p["current_price"])

        closed_trades = executor.check_closed_trades()
        for ct in closed_trades:
            pnl = ct["pnl"]
            risk.record_trade_result(pnl)
            pdt.record_day_trade(ct["ticker"], pnl)
            emoji = '📈' if pnl >= 0 else '📉'
            log_activity(f"{emoji} {ct['ticker']} 自动平仓 PnL${pnl:+.2f}")
    except Exception as e:
        log_activity(f"持仓监控异常: {e}", "error")


# ================================================================
# 持仓新闻轮询（v3: 每5分钟，severity≥8 → o3紧急评估）
# ================================================================
def _check_position_news():
    try:
        positions = executor.get_positions()
        for p in positions:
            if p["ticker"] in {"SGOV"}:
                continue
            news_result = news_analyzer_inst.analyze_ticker(p["ticker"])
            if news_result["has_news"]:
                state["news_results"][p["ticker"]] = news_result
                for a in news_result.get("analyses", []):
                    if a.get("intraday_severity", 0) >= 8:
                        log_activity(f"⚠️ {p['ticker']} severity≥8 新闻: {a.get('headline', '')[:60]}", "warning")
                        _run_news_analysis()
                        _run_o3_ranking()
                        return
    except Exception:
        pass


# ================================================================
# EOD 日终报告
# ================================================================
def _generate_eod():
    try:
        account = executor.get_account()
        log_activity(
            f"═══ 日终报告 ═══\n"
            f"  总值 ${account['portfolio_value']:,.2f}\n"
            f"  扫描 {state['scan_count']} 次 | 交易 {len(state['today_trades'])} 笔\n"
            f"  风险状态: {risk.status(account['portfolio_value'])}"
        )
    except Exception:
        pass


# ================================================================
# API 路由
# ================================================================

@app.route("/")
def index():
    return send_file("dashboard.html")


@app.route("/api/account")
def api_account():
    try:
        account = executor.get_account()
        positions = executor.get_positions()
        return jsonify({
            "account": account,
            "positions": positions,
            "pdt_remaining": pdt.remaining_trades(),
            "risk_status": risk.status(account["portfolio_value"]),
            "max_capital": MAX_CAPITAL,
            "vol_regime": scanner.get_vol_regime() if scanner else "N/A",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/state")
def api_state():
    with lock:
        logs = list(state["activity_log"])
    return jsonify({
        "bot_status": state["bot_status"],
        "scan_count": state["scan_count"],
        "last_scan_time": state["last_scan_time"],
        "last_news_time": state["last_news_time"],
        "candidates_count": len(state["scan_results"]),
        "news_analyzed_count": len(state["news_results"]),
        "recommendations": state["recommendations"],
        "today_trades": state["today_trades"],
        "activity_log": logs[-50:],
        "error": state["error"],
    })


@app.route("/api/scan/results")
def api_scan_results():
    results = []
    for c in state["scan_results"]:
        item = {
            "ticker": c["ticker"],
            "price": c["price"],
            "volume_ratio": c["volume_ratio"],
            "signal_strength": c["signal_strength"],
            "signals": c["signals"],
            "indicators": c["indicators"],
            "beta": c.get("beta"),
            "sector": c.get("sector"),
            "combined_score": c.get("combined_score"),
        }
        if c["ticker"] in state["news_results"]:
            item["news"] = state["news_results"][c["ticker"]]
        results.append(item)
    return jsonify({"results": results, "count": len(results)})


@app.route("/api/news/<ticker>")
def api_news_ticker(ticker):
    if ticker in state["news_results"]:
        return jsonify(state["news_results"][ticker])
    try:
        result = news_analyzer_inst.analyze_ticker(ticker)
        state["news_results"][ticker] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    user_msg = request.json.get("message", "")
    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    try:
        account = executor.get_account()
        positions = executor.get_positions()
    except Exception:
        account = {"portfolio_value": 0, "cash": 0}
        positions = []

    with lock:
        recent_logs = [f"[{l['time']}] {l['msg']}" for l in list(state["activity_log"])[-20:]]

    params = VOL_REGIMES.get(scanner.get_vol_regime() if scanner else "medium", VOL_REGIMES["medium"])
    context = f"""你是美股日内交易助手(v3策略)。

== Bot 状态: {state['bot_status']} ==
扫描: {state['scan_count']}次 | 候选: {len(state['scan_results'])} | 新闻: {len(state['news_results'])}

== 账户 ==
总值: ${account.get('portfolio_value', 0):,.2f} | 现金: ${account.get('cash', 0):,.2f} | PDT: {pdt.remaining_trades()}/3

== 风控 ==
regime: {scanner.get_vol_regime() if scanner else 'N/A'} | 仓位上限: {params['max_pos_pct']}% | 止损: {params['atr_mult']}x ATR

== 持仓 ({len(positions)}) ==
"""
    for p in positions:
        context += f"  {p['ticker']}: {p['qty']}股 入场${p['entry_price']:.2f} 现${p['current_price']:.2f} PnL${p['unrealized_pnl']:.2f}\n"
    context += f"\n== Top 10 候选 ==\n"
    for c in state["scan_results"][:10]:
        ns = state["news_results"].get(c["ticker"], {})
        context += f"  {c['ticker']}: ${c['price']:.2f} 信号{c['signal_strength']} 量比{c['volume_ratio']}x\n"
    context += f"\n== 最近日志 ==\n" + "\n".join(recent_logs[-10:])

    state["chat_history"].append({"role": "user", "content": user_msg})
    messages = [{"role": "system", "content": context}] + state["chat_history"][-20:]

    try:
        model = request.json.get("model", MODEL_FAST)
        response = openai_client.chat.completions.create(
            model=model, messages=messages,
            temperature=0.3 if not model.startswith("o") else None,
            max_completion_tokens=2000,
        )
        reply = response.choices[0].message.content
        usage = response.usage
        state["chat_history"].append({"role": "assistant", "content": reply})
        return jsonify({
            "reply": reply, "model": model,
            "tokens": {
                "prompt": usage.prompt_tokens if usage else 0,
                "completion": usage.completion_tokens if usage else 0,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    state["chat_history"] = []
    return jsonify({"status": "cleared"})


# ================================================================
# 启动
# ================================================================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    print("=" * 55)
    print("  📊 Trading Bot v3 + Dashboard")
    print(f"  🌐 http://localhost:5555")
    print(f"  🤖 策略: 全美股日内 | PDT 3次/周 | 只做多")
    print("=" * 55)

    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

    app.run(host="0.0.0.0", port=5555, debug=False)
