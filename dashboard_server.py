"""
交易仪表盘 + 自动交易Bot - 一体化服务
启动后 Bot 在后台 24 小时自动运行，仪表盘实时展示所有活动
"""
import json
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

ET = ZoneInfo("America/New_York")

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET, OPENAI_API_KEY,
    MODEL_FAST, MODEL_DEEP, MODEL_RANK, LOG_DIR,
    RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_RATIO,
    GAP_THRESHOLD_PCT, INTRADAY_MOMENTUM_PCT, VWAP_DEVIATION_PCT,
    MAX_POSITION_PCT, MAX_SINGLE_LOSS_PCT, MAX_DAILY_LOSS_PCT,
    MAX_CAPITAL,
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
# 全局共享状态 (bot 写，dashboard 读)
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
# Bot 核心组件 (延迟初始化)
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
# Bot 自动运行循环 (后台线程)
# ================================================================
def bot_loop():
    try:
        print("[Bot] 开始初始化组件...")
        init_components()
        print("[Bot] 组件初始化完成")
    except Exception as e:
        print(f"[Bot] 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        state["bot_status"] = "error"
        state["error"] = f"初始化失败: {e}"
        return

    log_activity("Bot 初始化完成，开始自动监控")
    state["bot_status"] = "running"

    try:
        account = executor.get_account()
        log_activity(f"账户连接成功 | 总值 ${account['portfolio_value']:,.2f} | 现金 ${account['cash']:,.2f}")
        log_activity(f"PDT: {pdt.status()}")
    except Exception as e:
        log_activity(f"Alpaca 连接失败: {e}", "error")
        state["bot_status"] = "error"
        state["error"] = str(e)
        return

    last_tech_scan = 0
    last_position_monitor = 0
    completed_news_rounds: set = set()
    eod_done = False

    while True:
        try:
            now = datetime.now(ET)
            today_str = now.strftime("%Y-%m-%d")

            # 每天零点重置
            if not any(today_str in r for r in completed_news_rounds):
                completed_news_rounds = set()
                eod_done = False

            # 周末休眠
            if now.weekday() >= 5:
                if state["bot_status"] != "weekend":
                    state["bot_status"] = "weekend"
                    log_activity("周末休市")
                time.sleep(300)
                continue

            # 收盘后
            if now.hour >= 16 and now.hour < 20:
                if state["bot_status"] != "after_hours":
                    state["bot_status"] = "after_hours"
                    if not eod_done:
                        log_activity("市场已收盘，进入盘后监控")
                        _generate_eod()
                        eod_done = True
                time.sleep(120)
                continue

            # 深夜休眠
            if now.hour >= 20 or now.hour < 4:
                if state["bot_status"] != "sleeping":
                    state["bot_status"] = "sleeping"
                    log_activity("深夜休眠中，04:00 AM ET 恢复")
                time.sleep(300)
                continue

            # 凌晨 4-9:30 盘前
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                if state["bot_status"] != "premarket":
                    state["bot_status"] = "premarket"
                    log_activity("盘前监控模式")

                # 盘前技术扫描（每 30 分钟，轻量级）
                if time.time() - last_tech_scan >= 1800:
                    _run_tech_scan_only()
                    last_tech_scan = time.time()

                # 检查是否到了新闻分析时间
                _check_news_schedule(now, today_str, completed_news_rounds)

                time.sleep(60)
                continue

            # 盘中 9:30 - 16:00
            state["bot_status"] = "market_open"

            # 尾盘强制平仓
            if now.hour == 15 and now.minute >= 50:
                positions_before = executor.get_positions()
                if executor.check_force_close():
                    for p in positions_before:
                        if p["ticker"] in {"SGOV"}:
                            continue
                        risk.record_trade_result(p["unrealized_pnl"])
                        pdt.record_day_trade(p["ticker"], p["unrealized_pnl"])
                        log_activity(f"尾盘平仓 {p['ticker']} PnL: ${p['unrealized_pnl']:+.2f}", "warning")

            # 持仓监控 (每 30 秒)
            if time.time() - last_position_monitor >= 30:
                _monitor_positions()
                last_position_monitor = time.time()

            # 技术面扫描 (每 15 分钟，免费)
            if time.time() - last_tech_scan >= 900:
                _run_tech_scan_only()
                last_tech_scan = time.time()

            # 检查是否到了新闻分析时间
            _check_news_schedule(now, today_str, completed_news_rounds)

            time.sleep(15)

        except Exception as e:
            log_activity(f"Bot 主循环异常: {e}", "error")
            state["error"] = str(e)
            time.sleep(30)


def _run_tech_scan_only():
    """仅技术面扫描（免费，不调用 OpenAI），更新候选列表"""
    state["bot_status"] = "scanning"
    try:
        candidates = scanner.scan()
        state["scan_results"] = candidates
        state["scan_count"] = state.get("scan_count", 0) + 1
        state["last_scan_time"] = datetime.now(ET).isoformat()
        if candidates:
            top5 = ", ".join(f"{c['ticker']}({c['signal_strength']})" for c in candidates[:5])
            log_activity(f"技术扫描完成: {len(candidates)} 只异动 | Top5: {top5}")
        else:
            log_activity("技术扫描完成: 无异动")
    except Exception as e:
        log_activity(f"技术扫描失败: {e}", "error")
    state["bot_status"] = "market_open"


def _check_news_schedule(now, today_str, completed_rounds):
    """按时间表触发新闻分析轮次"""
    NEWS_SCHEDULE = [
        (7, 0),    # 07:00 盘前
        (10, 0),   # 10:00 开盘后
        (13, 0),   # 13:00 午盘
        (15, 30),  # 15:30 尾盘前
    ]
    for h, m in NEWS_SCHEDULE:
        round_key = f"{today_str}_{h:02d}{m:02d}"
        if round_key in completed_rounds:
            continue
        # 在时间窗口内触发（允许 15 分钟延迟）
        sched_minutes = h * 60 + m
        now_minutes = now.hour * 60 + now.minute
        if 0 <= now_minutes - sched_minutes < 15:
            log_activity(f"📰 触发新闻分析轮次 ({h:02d}:{m:02d} ET)")
            completed_rounds.add(round_key)
            _run_full_pipeline()
            return


def _run_full_pipeline():
    """完整的 扫描→新闻→排名→执行 管道"""
    # 检查是否有足够现金进行新交易
    try:
        available_cash = executor.get_available_cash()
        if available_cash < 50:
            log_activity(f"可用现金 ${available_cash:.2f} 不足，跳过扫描（仅监控持仓）", "warning")
            return
    except Exception:
        pass

    state["bot_status"] = "scanning"
    log_activity("开始全市场技术面扫描...")

    # 第一层：扫描
    try:
        candidates = scanner.scan()
        state["scan_results"] = candidates
        state["scan_count"] = state.get("scan_count", 0) + 1
        state["last_scan_time"] = datetime.now(ET).isoformat()
        log_activity(f"扫描完成: {len(candidates)} 只异动股票")

        if candidates:
            top5 = [f"{c['ticker']}({c['signal_strength']})" for c in candidates[:5]]
            log_activity(f"Top 5: {', '.join(top5)}")
    except Exception as e:
        log_activity(f"扫描失败: {e}", "error")
        state["bot_status"] = "market_open"
        return

    if not candidates:
        log_activity("无技术异动，等待下一轮")
        state["bot_status"] = "market_open"
        return

    # 第二层：新闻分析
    state["bot_status"] = "analyzing_news"
    log_activity(f"开始新闻情绪分析 ({len(candidates)} 只)...")
    enriched = []

    # 只对技术面 Top 20 做新闻分析（含 web search），节省 API 费用
    candidates_for_news = sorted(candidates, key=lambda x: x["signal_strength"], reverse=True)[:20]
    log_activity(f"从 {len(candidates)} 只中选取 Top {len(candidates_for_news)} 进行新闻分析")

    for c in candidates_for_news:
        ticker = c["ticker"]
        tech_context = {
            "price": c["price"],
            "change_pct": 0,
            "volume_ratio": c["volume_ratio"],
            "rsi": c["indicators"]["rsi"],
            "macd_hist": c["indicators"]["macd_hist"],
            "signal_strength": c["signal_strength"],
        }
        try:
            news_result = news_analyzer_inst.analyze_ticker(ticker, tech_context)
            c["news_analysis"] = news_result
            state["news_results"][ticker] = news_result
            state["last_news_time"] = datetime.now(ET).isoformat()

            has_strong_news = (
                news_result["has_news"]
                and abs(news_result["sentiment_score"]) > 0
                and any(
                    a.get("confidence", 0) >= 60 and a.get("intraday_severity", 0) >= 5
                    for a in news_result["analyses"]
                )
            )
            has_very_strong_tech = c["signal_strength"] >= 60 and c["volume_ratio"] >= 5
            has_news_no_price_action = (
                news_result["has_news"]
                and abs(news_result["sentiment_score"]) > 30
                and c["volume_ratio"] < 2
            )

            if has_strong_news or has_very_strong_tech or has_news_no_price_action:
                tech_score = c["signal_strength"]
                news_score = news_result["sentiment_score"]
                # 综合分用绝对值排序，但保留方向信息供 o3 判断
                combined = tech_score * 0.4 + abs(news_score) * 0.6
                if has_news_no_price_action:
                    combined *= 1.3
                c["combined_score"] = round(combined, 2)
                c["news_direction"] = "bullish" if news_score > 0 else ("bearish" if news_score < 0 else "neutral")
                enriched.append(c)
                log_activity(f"✅ {ticker} 通过 | 综合{combined:.1f} 技术{tech_score:.1f} 新闻{news_score:.1f}")
        except Exception as e:
            log_activity(f"新闻分析 {ticker} 失败: {e}", "error")

    enriched.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    log_activity(f"新闻筛选完成: {len(enriched)}/{len(candidates)} 只通过")

    if not enriched:
        log_activity("无候选通过新闻筛选，等待下一轮")
        state["bot_status"] = "market_open"
        return

    # 检查PDT
    if not pdt.can_day_trade():
        log_activity(f"PDT 名额已用完 ({pdt.status()})，仅记录信号", "warning")
        state["bot_status"] = "market_open"
        return

    # 第三层：o3 排名
    state["bot_status"] = "ranking"
    log_activity(f"调用 o3 深度排名 ({len(enriched[:20])} 只)...")

    try:
        account = executor.get_account()
        positions = executor.get_positions()
        remaining = pdt.remaining_trades()
        now = datetime.now(ET)
        weekday = ["周一", "周二", "周三", "周四", "周五"][min(now.weekday(), 4)]
        remaining_days = max(5 - now.weekday(), 1)

        ranking = ranker.rank_candidates(
            candidates=enriched[:20],
            remaining_day_trades=remaining,
            portfolio_value=account["portfolio_value"],
            cash=account["cash"],
            current_positions=positions,
            weekday=weekday,
            remaining_trading_days=remaining_days,
        )
        state["ranking_result"] = ranking
        state["recommendations"] = ranking.get("recommendations", [])

        if ranking.get("save_bullets"):
            log_activity(f"💾 o3 建议保留名额: {ranking.get('save_reason', '')}", "warning")

        for rec in ranking.get("recommendations", []):
            log_activity(f"🎯 #{rec['rank']} {rec['ticker']} | {rec['action']} | "
                        f"置信度{rec.get('confidence', 0)} | "
                        f"入场${rec.get('entry_price', 0):.2f} 止损${rec.get('stop_loss', 0):.2f} 止盈${rec.get('take_profit', 0):.2f}")
    except Exception as e:
        log_activity(f"o3 排名失败: {e}", "error")
        state["bot_status"] = "market_open"
        return

    # 第四层：执行
    state["bot_status"] = "executing"
    for rec in ranking.get("recommendations", []):
        _execute_recommendation(rec)

    state["bot_status"] = "market_open"
    log_activity("本轮管道完成，恢复监控")


def _execute_recommendation(rec):
    ticker = rec["ticker"]
    action = rec["action"]

    if not pdt.can_day_trade():
        log_activity(f"⛔ PDT 名额不足，跳过 {ticker}", "warning")
        return

    account = executor.get_account()
    can_trade, reason = risk.can_trade(account["portfolio_value"])
    if not can_trade:
        log_activity(f"⛔ 风控拦截 {ticker}: {reason}", "warning")
        return

    # 检查实际可用现金
    available_cash = executor.get_available_cash()
    min_order = rec.get("entry_price", 0) * 1
    if available_cash < min_order:
        log_activity(f"⛔ 现金不足 ${available_cash:.2f}，跳过 {ticker}", "warning")
        return

    positions = executor.get_positions()
    active_pos = [p for p in positions if p["ticker"] not in {"SGOV"}]
    valid, msg, order_params = risk.validate_order(
        ticker=ticker, action=action,
        price=rec.get("entry_price", 0),
        stop_loss=rec.get("stop_loss", 0),
        position_size_pct=rec.get("position_size_pct", 10),
        portfolio_value=account["portfolio_value"],
        current_positions=len(active_pos),
    )

    if not valid:
        log_activity(f"⛔ 订单校验失败 {ticker}: {msg}", "warning")
        return

    # 限制实际股数不超过现金可买数量
    shares = order_params["shares"]
    max_shares_by_cash = int(available_cash / rec.get("entry_price", 1))
    if shares > max_shares_by_cash:
        shares = max_shares_by_cash
    if shares <= 0:
        log_activity(f"⛔ 现金 ${available_cash:.2f} 不够买 1 股 {ticker}", "warning")
        return

    success, msg = executor.execute_entry(
        ticker=ticker, action=action, shares=shares,
        entry_price=rec["entry_price"],
        stop_loss=rec["stop_loss"],
        take_profit=rec["take_profit"],
    )

    if success:
        log_activity(f"✅ 已成交: {action} {shares}股 {ticker} | {msg}")
        state["today_trades"].append({
            "time": datetime.now(ET).isoformat(),
            "ticker": ticker, "action": action, "shares": shares,
            "price": rec["entry_price"],
            "stop_loss": rec["stop_loss"],
            "take_profit": rec["take_profit"],
        })
    else:
        log_activity(f"❌ 下单失败 {ticker}: {msg}", "error")


def _monitor_positions():
    try:
        positions = executor.get_positions()
        for p in positions:
            if p["ticker"] in {"SGOV"}:
                continue
            executor.update_trailing_stop(p["ticker"], p["current_price"])

        # 检测已被 broker 平仓的交易（止损/止盈成交），记录 PnL 和 PDT
        closed_trades = executor.check_closed_trades()
        for ct in closed_trades:
            pnl = ct["pnl"]
            risk.record_trade_result(pnl)
            pdt.record_day_trade(ct["ticker"], pnl)
            log_activity(
                f"{'📈' if pnl >= 0 else '📉'} {ct['ticker']} 自动平仓 | PnL: ${pnl:+.2f}",
                "info" if pnl >= 0 else "warning"
            )
    except Exception as e:
        log_activity(f"持仓监控异常: {e}", "error")


def _check_position_news():
    try:
        positions = executor.get_positions()
        for p in positions:
            if p["ticker"] == "SGOV":
                continue
            news_result = news_analyzer_inst.analyze_ticker(p["ticker"])
            if news_result["has_news"] and abs(news_result["sentiment_score"]) > 50:
                log_activity(f"⚠️ {p['ticker']} 重要新闻 | 情绪分 {news_result['sentiment_score']}", "warning")
                state["news_results"][p["ticker"]] = news_result
    except Exception:
        pass


def _generate_eod():
    try:
        account = executor.get_account()
        log_activity(f"日终报告 | 总值 ${account['portfolio_value']:,.2f} | 扫描 {state['scan_count']} 次 | 交易 {len(state['today_trades'])} 笔")
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
            "signal_details": _explain_signals(c),
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
    except:
        account = {"portfolio_value": 0, "cash": 0, "buying_power": 0}
        positions = []

    with lock:
        recent_logs = [f"[{l['time']}] {l['msg']}" for l in list(state["activity_log"])[-20:]]

    context = f"""你是一个专业的美股日内交易助手。以下是当前系统实时状态：

== Bot 状态: {state['bot_status']} ==
扫描次数: {state['scan_count']} | 候选: {len(state['scan_results'])} | 已分析新闻: {len(state['news_results'])}

== 账户 ==
总值: ${account.get('portfolio_value', 0):,.2f} | 现金: ${account.get('cash', 0):,.2f} | PDT: {pdt.remaining_trades()}/3

== 持仓 ({len(positions)} 只) ==
"""
    for p in positions:
        context += f"  {p['ticker']}: {p['qty']}股 | 入场${p['entry_price']:.2f} | 现价${p['current_price']:.2f} | PnL: ${p['unrealized_pnl']:.2f}\n"

    context += f"\n== 最新扫描结果 (Top 10) ==\n"
    for c in state["scan_results"][:10]:
        ns = state["news_results"].get(c["ticker"], {})
        news_info = f"情绪{ns.get('sentiment_score', '-')}" if ns else "未分析"
        context += f"  {c['ticker']}: ${c['price']:.2f} | 信号{c['signal_strength']} | {', '.join(c['signals'][:2])} | 新闻: {news_info}\n"

    if state["recommendations"]:
        context += f"\n== o3 最新推荐 ==\n"
        for rec in state["recommendations"]:
            context += f"  #{rec['rank']} {rec['ticker']} {rec['action']} | 入场${rec.get('entry_price',0):.2f} 止损${rec.get('stop_loss',0):.2f}\n"

    context += f"\n== 最近活动日志 ==\n" + "\n".join(recent_logs[-10:])
    context += f"\n\n== 风险参数 ==\n最大仓位{MAX_POSITION_PCT}% | 单笔亏损{MAX_SINGLE_LOSS_PCT}% | 日亏损{MAX_DAILY_LOSS_PCT}%\n"

    state["chat_history"].append({"role": "user", "content": user_msg})
    messages = [{"role": "system", "content": context}] + state["chat_history"][-20:]

    try:
        model = request.json.get("model", MODEL_FAST)
        if model.startswith("o") or model.startswith("gpt-5"):
            response = openai_client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=2000)
        else:
            response = openai_client.chat.completions.create(
                model=model, messages=messages, temperature=0.3, max_completion_tokens=2000)

        reply = response.choices[0].message.content
        usage = response.usage
        state["chat_history"].append({"role": "assistant", "content": reply})
        return jsonify({
            "reply": reply, "model": model,
            "tokens": {"prompt": usage.prompt_tokens if usage else 0,
                       "completion": usage.completion_tokens if usage else 0,
                       "total": usage.total_tokens if usage else 0},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    state["chat_history"] = []
    return jsonify({"status": "cleared"})

# ================================================================
# 信号解释 (复用)
# ================================================================
def _explain_signals(candidate):
    explanations = []
    ind = candidate.get("indicators", {})
    rsi = ind.get("rsi", 50)
    if rsi <= RSI_OVERSOLD:
        explanations.append({"signal":"RSI 超卖","value":f"{rsi}","threshold":f"≤{RSI_OVERSOLD}","measurement":"14日RSI","weight":30 if rsi<=20 else 20,"direction":"bullish","description":f"RSI={rsi} 超卖区间"})
    elif rsi >= RSI_OVERBOUGHT:
        explanations.append({"signal":"RSI 超买","value":f"{rsi}","threshold":f"≥{RSI_OVERBOUGHT}","measurement":"14日RSI","weight":30 if rsi>=80 else 20,"direction":"bearish","description":f"RSI={rsi} 超买区间"})
    vr = candidate.get("volume_ratio", 0)
    if vr >= VOLUME_SPIKE_RATIO:
        explanations.append({"signal":"成交量放大","value":f"{vr:.1f}x","threshold":f"≥{VOLUME_SPIKE_RATIO}x","measurement":"今日推算量/20日均量","weight":25 if vr>=5 else 15,"direction":"neutral","description":f"量比 {vr:.1f}x"})
    mc = ind.get("macd_cross","none")
    if mc != "none":
        explanations.append({"signal":f"MACD {'金叉' if mc=='golden' else '死叉'}","value":mc,"threshold":"MACD穿越信号线","measurement":"MACD(12,26,9)","weight":15,"direction":"bullish" if mc=="golden" else "bearish","description":f"MACD {'金叉看涨' if mc=='golden' else '死叉看跌'}"})
    price = candidate.get("price",0)
    if price >= ind.get("bb_upper",999999):
        explanations.append({"signal":"突破布林上轨","value":f"${price:.2f}","threshold":"≥上轨","measurement":"布林带(20,2σ)","weight":10,"direction":"bearish","description":"突破上轨可能回归"})
    elif price <= ind.get("bb_lower",0):
        explanations.append({"signal":"跌破布林下轨","value":f"${price:.2f}","threshold":"≤下轨","measurement":"布林带(20,2σ)","weight":10,"direction":"bullish","description":"跌破下轨可能反弹"})
    for sig in candidate.get("signals",[]):
        if sig.startswith("gap_"):
            p=sig.replace("gap_","").replace("%","")
            explanations.append({"signal":"跳空","value":f"{p}%","threshold":f"≥{GAP_THRESHOLD_PCT}%","measurement":"开盘vs昨收","weight":10,"direction":"bullish" if float(p)>0 else "bearish","description":f"跳空{p}%"})
        elif sig.startswith("momentum_"):
            p=sig.replace("momentum_","").replace("%","")
            explanations.append({"signal":"日内动量","value":f"{p}%","threshold":f"≥{INTRADAY_MOMENTUM_PCT}%","measurement":"现价vs今开","weight":8,"direction":"bullish" if float(p)>0 else "bearish","description":f"日内{p}%"})
        elif sig.startswith("vwap_dev_"):
            p=sig.replace("vwap_dev_","").replace("%","")
            explanations.append({"signal":"VWAP偏离","value":f"{p}%","threshold":f"≥{VWAP_DEVIATION_PCT}%","measurement":"价格偏离VWAP","weight":8,"direction":"bullish" if float(p)>0 else "bearish","description":f"VWAP偏离{p}%"})
    return explanations

# ================================================================
# 启动
# ================================================================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    print("=" * 55)
    print("  📊 Trading Bot + Dashboard")
    print(f"  🌐 http://localhost:5555")
    print(f"  🤖 Bot 自动运行中 (后台线程)")
    print("=" * 55)

    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

    app.run(host="0.0.0.0", port=5555, debug=False)
