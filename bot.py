#!/usr/bin/env python3




import argparse
import sys
from datetime import date

from core.config import ALPACA_API_KEY, OPENAI_API_KEY, LOG_DIR
from trading.executor import OrderExecutor
from trading.pdt_tracker import PDTTracker
from trading.risk_manager import RiskManager


def print_status():
    executor = OrderExecutor()
    pdt_tracker = PDTTracker()
    risk_mgr = RiskManager()

    account = executor.get_account()
    positions = executor.get_positions()

    print(f"\n{'=' * 50}")
    print(f"Account Status (v6)")
    print(f"{'=' * 50}")
    print(f"  Portfolio: ${account['portfolio_value']:,.2f}")
    print(f"  Cash: ${account['cash']:,.2f}")
    print(f"  Buying Power: ${account['buying_power']:,.2f}")
    print(f"  {pdt_tracker.status()}")
    print(f"  {risk_mgr.status(account['portfolio_value'])}")

    if positions:
        print(f"\nPositions ({len(positions)}):")
        for p in positions:
            print(f"  {p['ticker']:6s} | {p['qty']} shares | "
                  f"Entry ${p['entry_price']:.2f} | "
                  f"Current ${p['current_price']:.2f} | "
                  f"PnL: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pnl_pct']:+.1f}%)")
    else:
        print("\nNo positions")
    print()


def main():
    parser = argparse.ArgumentParser(description="Trading Bot v6 CLI")
    parser.add_argument("--status", action="store_true", help="View account status")
    args = parser.parse_args()

    if args.status:
        print_status()
    else:
        print("v6 main entry has moved to server/dashboard_server.py")
        print("Run: python -m server.dashboard_server")
        print("Or: python bot.py --status to view account status")


if __name__ == "__main__":
    main()
