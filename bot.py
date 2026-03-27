#!/usr/bin/env python3
"""
全美股日内交易Bot - 主程序
OpenAI + Alpaca | PDT限制下的精选交易系统

用法：
    python bot.py                   # 正常运行（Paper Trading）
    python bot.py --scan-only       # 只扫描，不交易
    python bot.py --status          # 查看账户和PDT状态
    python bot.py --dry-run         # 完整流程但不下单
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, date
from typing import List, Dict

from config import (
    ALPACA_API_KEY, OPENAI_API_KEY, LOG_DIR,
    CLOSE_ALL, FINAL_CHECK,
)
from scanner import MarketScanner
from news_analyzer import NewsAnalyzer
from ranker import DeepRanker
from executor import OrderExecutor
from pdt_tracker import PDTTracker
from risk_manager import RiskManager


class TradingBot:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.scanner = MarketScanner()
        self.news = NewsAnalyzer()
        self.ranker = DeepRanker()
        self.executor = OrderExecutor()
        self.pdt = PDTTracker()
        self.risk = RiskManager()

        # 今日状态
        self.today_candidates: List[Dict] = []
        self.today_recommendations: List[Dict] = []
        self.scan_count: int = 0

    # ================================================================
    # 启动检查
    # ================================================================

    def preflight_check(self) -> bool:
        """启动前检查所有连接"""
        print("\n" + "=" * 60)
        print("🦙 全美股日内交易Bot - 启动检查")
        print("=" * 60)

        # API Key检查
        if not ALPACA_API_KEY or ALPACA_API_KEY == "your_alpaca_key_here":
            print("❌ ALPACA_API_KEY 未配置，请编辑 .env 文件")
            return False
        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
            print("❌ OPENAI_API_KEY 未配置，请编辑 .env 文件")
            return False

        # Alpaca连接
        try:
            account = self.executor.get_account()
            print(f"✅ Alpaca连接成功 | 账户: ${account['portfolio_value']:,.2f} | 现金: ${account['cash']:,.2f}")
        except Exception as e:
            print(f"❌ Alpaca连接失败: {e}")
            return False

        # PDT状态
        print(f"📊 {self.pdt.status()}")

        # 风险管理状态
        print(f"🛡️ {self.risk.status(account['portfolio_value'])}")

        # 模式
        if self.dry_run:
            print("🔸 DRY RUN模式：完整分析但不下单")
        else:
            print("🔹 PAPER TRADING模式：模拟盘交易")

        print("=" * 60 + "\n")
        return True

    # ================================================================
    # 第一层：全市场扫描
    # ================================================================

    def run_scan(self) -> List[Dict]:
        """执行全市场技术面扫描"""
        print(f"\n{'─' * 40}")
        print(f"[第一层] 全市场技术面扫描 ({datetime.now().strftime('%H:%M:%S')})")
        print(f"{'─' * 40}")

        candidates = self.scanner.scan()
        self.scan_count += 1

        if not candidates:
            print("[第一层] 没有发现技术异动")
            return []

        print(f"[第一层] 发现 {len(candidates)} 只异动股票，前10只：")
        for c in candidates[:10]:
            print(f"  {c['ticker']:6s} | ${c['price']:>8.2f} | "
                  f"强度:{c['signal_strength']:>5.1f} | "
                  f"量比:{c['volume_ratio']:>5.1f}x | "
                  f"信号: {', '.join(c['signals'][:3])}")

        return candidates

    # ================================================================
    # 第二层：新闻情绪分析
    # ================================================================

    def run_news_analysis(self, candidates: List[Dict]) -> List[Dict]:
        """对候选股票做新闻情绪分析"""
        print(f"\n{'─' * 40}")
        print(f"[第二层] 新闻情绪分析 ({len(candidates)} 只候选)")
        print(f"{'─' * 40}")

        enriched = []
        for c in candidates:
            ticker = c["ticker"]
            tech_context = {
                "price": c["price"],
                "change_pct": 0,  # 从signals里提取
                "volume_ratio": c["volume_ratio"],
                "rsi": c["indicators"]["rsi"],
                "macd_hist": c["indicators"]["macd_hist"],
            }

            news_result = self.news.analyze_ticker(ticker, tech_context)
            c["news_analysis"] = news_result

            # 筛选标准
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
                # 计算综合分
                tech_score = c["signal_strength"]  # 0-100
                news_score = abs(news_result["sentiment_score"])  # 0-100
                combined = tech_score * 0.4 + news_score * 0.6  # 新闻权重更高（日内交易新闻更重要）

                if has_news_no_price_action:
                    combined *= 1.3  # 加成：市场还没反应的新闻

                c["combined_score"] = round(combined, 2)
                enriched.append(c)

                status = "📰+📊" if has_strong_news else ("📊强" if has_very_strong_tech else "📰未反应")
                print(f"  ✅ {ticker:6s} | 综合:{combined:>5.1f} | "
                      f"技术:{tech_score:>5.1f} | 新闻:{news_score:>5.1f} | {status}")
            else:
                if news_result["has_news"]:
                    print(f"  ❌ {ticker:6s} | 新闻信号不够强")

        # 按综合分排序
        enriched.sort(key=lambda x: x["combined_score"], reverse=True)

        print(f"\n[第二层] {len(enriched)} 只通过新闻筛选")
        return enriched[:20]  # 最多20只进入第三层

    # ================================================================
    # 第三层：o3深度排名
    # ================================================================

    def run_ranking(self, candidates: List[Dict]) -> Dict:
        """用o3做深度排名"""
        print(f"\n{'─' * 40}")
        print(f"[第三层] o3 深度排名 ({len(candidates)} 只候选)")
        print(f"{'─' * 40}")

        account = self.executor.get_account()
        positions = self.executor.get_positions()
        remaining = self.pdt.remaining_trades()

        now = datetime.now()
        weekday = ["周一", "周二", "周三", "周四", "周五"][now.weekday()]
        remaining_days = 5 - now.weekday()  # 本周剩余交易日

        print(f"  账户: ${account['portfolio_value']:,.2f} | 现金: ${account['cash']:,.2f}")
        print(f"  PDT剩余: {remaining}次 | {weekday} | 本周还有{remaining_days}天")
        print(f"  调用 o3 (reasoning effort=high)...")

        result = self.ranker.rank_candidates(
            candidates=candidates,
            remaining_day_trades=remaining,
            portfolio_value=account["portfolio_value"],
            cash=account["cash"],
            current_positions=positions,
            weekday=weekday,
            remaining_trading_days=remaining_days,
        )

        # 打印结果
        if result.get("save_bullets"):
            print(f"\n  💾 o3建议保留名额: {result.get('save_reason', '未说明')}")

        for rec in result.get("recommendations", []):
            print(f"\n  🎯 #{rec['rank']} {rec['ticker']}")
            print(f"     动作: {rec['action']} | 置信度: {rec['confidence']}")
            print(f"     入场: ${rec.get('entry_price', 0):.2f} | "
                  f"止损: ${rec.get('stop_loss', 0):.2f} | "
                  f"止盈: ${rec.get('take_profit', 0):.2f}")
            print(f"     风险回报比: {rec.get('risk_reward_ratio', 0):.1f}")
            print(f"     仓位: {rec.get('position_size_pct', 0)}%")
            print(f"     时间窗口: {rec.get('time_window', 'N/A')}")

        if result.get("risk_warnings"):
            print(f"\n  ⚠️ 风险提醒: {', '.join(result['risk_warnings'])}")

        return result

    # ================================================================
    # 第四层：执行
    # ================================================================

    def execute_recommendations(self, ranking_result: Dict):
        """执行o3的推荐"""
        print(f"\n{'─' * 40}")
        print(f"[第四层] 执行交易")
        print(f"{'─' * 40}")

        recommendations = ranking_result.get("recommendations", [])
        if not recommendations:
            print("  没有推荐，跳过执行")
            return

        account = self.executor.get_account()
        positions = self.executor.get_positions()

        for rec in recommendations:
            ticker = rec["ticker"]
            action = rec["action"]

            # PDT检查
            if not self.pdt.can_day_trade():
                print(f"  ⛔ PDT名额用完，跳过 {ticker}")
                continue

            # 风险检查
            can_trade, reason = self.risk.can_trade(account["portfolio_value"])
            if not can_trade:
                print(f"  {reason}")
                continue

            # 验证订单
            valid, msg, order_params = self.risk.validate_order(
                ticker=ticker,
                action=action,
                price=rec.get("entry_price", 0),
                stop_loss=rec.get("stop_loss", 0),
                position_size_pct=rec.get("position_size_pct", 10),
                portfolio_value=account["portfolio_value"],
                current_positions=len(positions),
            )

            if not valid:
                print(f"  {msg}")
                continue

            shares = order_params["shares"]

            if self.dry_run:
                print(f"  [DRY RUN] 会执行: {action} {shares}股 {ticker} "
                      f"@ ${rec['entry_price']:.2f}")
                continue

            # 下单
            success, msg = self.executor.execute_entry(
                ticker=ticker,
                action=action,
                shares=shares,
                entry_price=rec["entry_price"],
                stop_loss=rec["stop_loss"],
                take_profit=rec["take_profit"],
            )

            if success:
                # 设置止损止盈
                is_long = action == "BUY"
                self.executor.set_stop_loss(ticker, rec["stop_loss"], shares, is_long)
                self.executor.set_take_profit(ticker, rec["take_profit"], shares, is_long)
                print(f"  ✅ {action} {shares}股 {ticker} 已提交")
            else:
                print(f"  ❌ {ticker} 下单失败: {msg}")

    # ================================================================
    # 盘中监控循环
    # ================================================================

    def monitor_positions(self):
        """监控当前持仓，更新移动止损"""
        positions = self.executor.get_positions()
        if not positions:
            return

        for p in positions:
            self.executor.update_trailing_stop(p["ticker"], p["current_price"])

    # ================================================================
    # 主循环
    # ================================================================

    def run_full_pipeline(self):
        """执行完整的分析→排名→执行管道"""

        # 第一层：扫描
        candidates = self.run_scan()
        if not candidates:
            print("\n没有异动，等待下一轮扫描...")
            return

        # 检查PDT（如果没名额就只做前两层记录）
        if not self.pdt.can_day_trade():
            print(f"\n{self.pdt.status()}")
            print("继续监控，记录虚拟信号用于复盘...")
            # 仍然跑新闻分析，用于记录
            enriched = self.run_news_analysis(candidates)
            self._save_virtual_signals(enriched)
            return

        # 第二层：新闻
        enriched = self.run_news_analysis(candidates)
        if not enriched:
            print("\n没有通过新闻筛选的候选，等待下一轮...")
            return

        # 第三层：排名
        ranking = self.run_ranking(enriched)

        # 第四层：执行
        self.execute_recommendations(ranking)

        # 保存今日数据
        self.today_candidates = enriched
        self.today_recommendations = ranking.get("recommendations", [])
        self._save_daily_report(candidates, enriched, ranking)

    def run(self):
        """主运行循环"""
        if not self.preflight_check():
            sys.exit(1)

        print("🚀 Bot启动，开始监控全市场...\n")

        try:
            # 首次完整扫描
            self.run_full_pipeline()

            # 进入监控循环
            last_scan = time.time()
            last_news = time.time()

            while True:
                now = datetime.now()

                # 尾盘强制平仓检查
                if self.executor.check_force_close():
                    print("\n尾盘平仓完成，今日交易结束。")
                    self._generate_eod_report()
                    break

                # 收盘后退出
                if now.hour >= 16:
                    print("\n市场已收盘，生成日报...")
                    self._generate_eod_report()
                    break

                # 盘中还没开盘
                if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                    # 盘前：每5分钟扫一次
                    if time.time() - last_scan >= 300:
                        print(f"\n[盘前扫描] {now.strftime('%H:%M')}")
                        self.run_full_pipeline()
                        last_scan = time.time()
                    time.sleep(30)
                    continue

                # 盘中监控
                self.monitor_positions()

                # 定期重新扫描（每15分钟）
                if time.time() - last_scan >= 900:
                    self.run_full_pipeline()
                    last_scan = time.time()

                # 新闻轮询（每5分钟）
                if time.time() - last_news >= 3600:
                    self._check_news_updates()
                    last_news = time.time()

                time.sleep(15)  # 主循环每15秒检查一次

        except KeyboardInterrupt:
            print("\n\n⚠️ 手动停止Bot")
            positions = self.executor.get_positions()
            if positions:
                print(f"⚠️ 注意：还有 {len(positions)} 个持仓未平仓！")
                for p in positions:
                    print(f"  {p['ticker']}: {p['qty']}股 | PnL: ${p['unrealized_pnl']:+.2f}")

    # ================================================================
    # 辅助方法
    # ================================================================

    def _check_news_updates(self):
        """盘中新闻更新检查"""
        positions = self.executor.get_positions()
        for p in positions:
            news_result = self.news.analyze_ticker(p["ticker"])
            if news_result["has_news"] and abs(news_result["sentiment_score"]) > 50:
                print(f"\n⚠️ {p['ticker']} 出现重要新闻！情绪分: {news_result['sentiment_score']}")
                # 可选：调用o3紧急评估
                if abs(news_result["sentiment_score"]) > 70:
                    emergency = self.ranker.emergency_reassess(
                        ticker=p["ticker"],
                        new_event=news_result["analyses"][0].get("summary", ""),
                        current_position=p,
                        portfolio_value=self.executor.get_account()["portfolio_value"],
                    )
                    print(f"  o3紧急评估: {emergency.get('action')} - {emergency.get('reasoning', '')[:100]}")
                    if emergency.get("action") == "CLOSE_NOW" and not self.dry_run:
                        self.executor.close_position(p["ticker"])

    def _save_virtual_signals(self, candidates: List[Dict]):
        """PDT名额用完时，保存虚拟信号用于复盘"""
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"virtual_signals_{date.today()}.json")
        with open(path, "w") as f:
            json.dump([{
                "ticker": c["ticker"],
                "price": c["price"],
                "combined_score": c.get("combined_score", 0),
                "signals": c.get("signals", []),
                "news_sentiment": c.get("news_analysis", {}).get("sentiment_score", 0),
            } for c in candidates], f, indent=2)

    def _save_daily_report(self, raw_candidates, enriched, ranking):
        """保存每日分析数据"""
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"daily_report_{date.today()}.json")
        report = {
            "date": str(date.today()),
            "scan_count": self.scan_count,
            "raw_candidates": len(raw_candidates),
            "after_news_filter": len(enriched),
            "recommendations": ranking.get("recommendations", []),
            "save_bullets": ranking.get("save_bullets", False),
            "market_context": ranking.get("market_context", ""),
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def _generate_eod_report(self):
        """生成收盘日报"""
        account = self.executor.get_account()
        print(f"\n{'=' * 60}")
        print(f"📊 日终报告 - {date.today()}")
        print(f"{'=' * 60}")
        print(f"  账户总值: ${account['portfolio_value']:,.2f}")
        print(f"  {self.risk.status(account['portfolio_value'])}")
        print(f"  {self.pdt.status()}")
        print(f"  扫描次数: {self.scan_count}")
        print(f"  推荐交易: {len(self.today_recommendations)}")
        print(f"{'=' * 60}\n")


# ================================================================
# CLI
# ================================================================

def print_status():
    """打印当前状态"""
    executor = OrderExecutor()
    pdt = PDTTracker()
    risk = RiskManager()

    account = executor.get_account()
    positions = executor.get_positions()

    print(f"\n{'=' * 50}")
    print(f"📊 账户状态")
    print(f"{'=' * 50}")
    print(f"  总值: ${account['portfolio_value']:,.2f}")
    print(f"  现金: ${account['cash']:,.2f}")
    print(f"  购买力: ${account['buying_power']:,.2f}")
    print(f"  {pdt.status()}")
    print(f"  {risk.status(account['portfolio_value'])}")

    if positions:
        print(f"\n📈 持仓 ({len(positions)}只):")
        for p in positions:
            print(f"  {p['ticker']:6s} | {p['qty']}股 | "
                  f"入场${p['entry_price']:.2f} | "
                  f"现价${p['current_price']:.2f} | "
                  f"PnL: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pnl_pct']:+.1f}%)")
    else:
        print("\n📈 无持仓")
    print()


def main():
    parser = argparse.ArgumentParser(description="全美股日内交易Bot")
    parser.add_argument("--scan-only", action="store_true", help="只扫描，不交易")
    parser.add_argument("--status", action="store_true", help="查看账户状态")
    parser.add_argument("--dry-run", action="store_true", help="完整流程但不下单")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    bot = TradingBot(dry_run=args.dry_run or args.scan_only)

    if args.scan_only:
        if bot.preflight_check():
            candidates = bot.run_scan()
            if candidates:
                bot.run_news_analysis(candidates)
    else:
        bot.run()


if __name__ == "__main__":
    main()
