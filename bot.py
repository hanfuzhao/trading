#!/usr/bin/env python3
"""
Trading Bot v6 — CLI入口（已弃用，主入口为 dashboard_server.py）
保留此文件仅用于 --status 查看账户状态。
"""
import argparse
import sys
from datetime import date

from config import ALPACA_API_KEY, OPENAI_API_KEY, LOG_DIR
from executor import OrderExecutor
from pdt_tracker import PDTTracker
from risk_manager import RiskManager


def print_status():
    executor = OrderExecutor()
    pdt_tracker = PDTTracker()
    risk_mgr = RiskManager()

    account = executor.get_account()
    positions = executor.get_positions()

    print(f"\n{'=' * 50}")
    print(f"📊 账户状态 (v6)")
    print(f"{'=' * 50}")
    print(f"  总值: ${account['portfolio_value']:,.2f}")
    print(f"  现金: ${account['cash']:,.2f}")
    print(f"  购买力: ${account['buying_power']:,.2f}")
    print(f"  {pdt_tracker.status()}")
    print(f"  {risk_mgr.status(account['portfolio_value'])}")

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
    parser = argparse.ArgumentParser(description="Trading Bot v6 CLI")
    parser.add_argument("--status", action="store_true", help="查看账户状态")
    args = parser.parse_args()

    if args.status:
        print_status()
    else:
        print("v6 主入口已迁移到 dashboard_server.py")
        print("运行: python dashboard_server.py")
        print("或: python bot.py --status 查看账户状态")


if __name__ == "__main__":
    main()
