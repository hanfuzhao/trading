#!/usr/bin/env python3
"""
交易助手对话模式
连接OpenAI，自动注入当前持仓、扫描结果、市场状态作为上下文
你问任何问题，它都基于你的实时数据回答

用法：
    python chat.py                  # 默认用o3
    python chat.py --model gpt-5.4  # 用gpt-5.4（更快更便宜）
    python chat.py --model gpt-4.1-mini  # 最便宜，简单问题够用
"""
import argparse
import json
import os
import glob
from datetime import datetime, date
from typing import Dict, List, Optional
from openai import OpenAI

from config import (
    OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_API_SECRET,
    MODEL_RANK, MODEL_DEEP, MODEL_FAST, LOG_DIR,
)
from executor import OrderExecutor
from pdt_tracker import PDTTracker
from risk_manager import RiskManager


class TradingChat:
    def __init__(self, model: str = MODEL_RANK):
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.executor = OrderExecutor()
        self.pdt = PDTTracker()
        self.risk = RiskManager()
        self.conversation: List[Dict] = []
        self.system_prompt = ""

    # ================================================================
    # 构建上下文
    # ================================================================

    def _build_context(self) -> str:
        """收集所有实时数据，构建系统prompt"""
        sections = []

        # 账户信息
        try:
            account = self.executor.get_account()
            sections.append(f"""== 账户状态 ==
总值: ${account['portfolio_value']:,.2f}
现金: ${account['cash']:,.2f}
购买力: ${account['buying_power']:,.2f}
PDT标记: {'是' if account['pattern_day_trader'] else '否'}""")
        except Exception as e:
            sections.append(f"== 账户状态 ==\n获取失败: {e}")

        # PDT状态
        sections.append(f"""== PDT名额 ==
{self.pdt.status()}
滚动窗口内已用交易: {3 - self.pdt.remaining_trades()}次
下一个名额恢复: {self.pdt.next_trade_unlock()}""")

        # 风险状态
        try:
            pv = account['portfolio_value']
            sections.append(f"""== 风险管理 ==
{self.risk.status(pv)}
连续亏损: {self.risk.consecutive_losses}次
单笔最大亏损上限: ${pv * 1.5 / 100:,.2f} (1.5%)
单日最大亏损上限: ${pv * 3 / 100:,.2f} (3%)""")
        except Exception:
            pass

        # 当前持仓
        try:
            positions = self.executor.get_positions()
            if positions:
                pos_text = "\n".join([
                    f"  {p['ticker']:6s} | {p['qty']}股 | "
                    f"入场${p['entry_price']:.2f} | 现价${p['current_price']:.2f} | "
                    f"PnL: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pnl_pct']:+.1f}%)"
                    for p in positions
                ])
                sections.append(f"== 当前持仓 ==\n{pos_text}")
            else:
                sections.append("== 当前持仓 ==\n无持仓")
        except Exception:
            sections.append("== 当前持仓 ==\n获取失败")

        # 今日扫描结果
        today_report = self._load_today_report()
        if today_report:
            sections.append(f"""== 今日扫描 ==
扫描次数: {today_report.get('scan_count', 'N/A')}
原始候选: {today_report.get('raw_candidates', 'N/A')}只
新闻筛选后: {today_report.get('after_news_filter', 'N/A')}只
大盘环境: {today_report.get('market_context', '未知')}""")

            recs = today_report.get('recommendations', [])
            if recs:
                rec_text = "\n".join([
                    f"  #{r.get('rank', '?')} {r.get('ticker', '?')} | "
                    f"{r.get('action', '?')} | 置信度:{r.get('confidence', '?')} | "
                    f"入场${r.get('entry_price', 0):.2f} | "
                    f"R/R:{r.get('risk_reward_ratio', 0):.1f}"
                    for r in recs
                ])
                sections.append(f"== 今日推荐 ==\n{rec_text}")

        # 最近交易记录
        recent_trades = self._load_recent_trades()
        if recent_trades:
            trades_text = "\n".join([
                f"  {t.get('timestamp', '?')[:16]} | {t.get('type', '?')} | "
                f"{t.get('ticker', '?')} | "
                f"${t.get('entry_price', t.get('exit_price', 0)):.2f}"
                for t in recent_trades[-10:]  # 最近10笔
            ])
            sections.append(f"== 最近交易 ==\n{trades_text}")

        # 虚拟信号（PDT用完时的记录）
        virtual = self._load_virtual_signals()
        if virtual:
            v_text = "\n".join([
                f"  {v['ticker']:6s} | 综合分:{v.get('combined_score', 0):>5.1f} | "
                f"新闻情绪:{v.get('news_sentiment', 0):>5.1f}"
                for v in virtual[:10]
            ])
            sections.append(f"== 今日虚拟信号（未交易）==\n{v_text}")

        # 当前时间
        now = datetime.now()
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
        market_status = "盘前" if now.hour < 9 or (now.hour == 9 and now.minute < 30) else (
            "盘中" if now.hour < 16 else "盘后"
        )
        sections.append(f"== 时间 ==\n{now.strftime('%Y-%m-%d %H:%M')} {weekday} | {market_status}")

        # 宏观提醒
        sections.append("""== 当前宏观环境（2026年3月底）==
- 美伊战争持续，布伦特原油>$110，霍尔木兹海峡通行受阻
- 美联储3月维持利率不变，仅预计2026年降息1次
- Powell主席任期5月到期，继任者Warsh尚未确认
- S&P 500连续5周下跌，纳斯达克和道琼斯进入修正区间
- 2026为中期选举年，历史上前三季度波动加大
- 关税通胀仍在消化中
- 板块偏好：能源/防务受益于战争，医疗保健防御属性强，消费可选承压""")

        return "\n\n".join(sections)

    def _build_system_prompt(self) -> str:
        """构建完整的系统prompt"""
        context = self._build_context()

        return f"""你是我的私人量化交易助手。你可以看到我的完整账户状态、持仓、扫描结果和交易历史。

你的职责：
1. 基于实时数据回答我关于持仓、市场、策略的任何问题
2. 如果我问"该不该买/卖某只股票"，你要结合我的账户状态、PDT名额、风险规则给出具体建议
3. 你可以挑战我的决定——如果我要做的事违反风险规则或不明智，直接说
4. 你的回答要简洁直接，不要客套

关键规则（你必须遵守）：
- 单笔最大仓位15%
- 单笔最大亏损1.5%
- 单日最大亏损3%
- PDT限制：5个交易日内最多3次日内交易
- 3:50 PM必须平仓所有日内持仓
- 如果我要做的事会违反这些规则，你必须明确拒绝并解释

以下是我的实时数据：

{context}

用中文回答。简洁、直接、有数据支撑。"""

    # ================================================================
    # 加载数据文件
    # ================================================================

    def _load_today_report(self) -> Optional[Dict]:
        path = os.path.join(LOG_DIR, f"daily_report_{date.today()}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def _load_recent_trades(self) -> List[Dict]:
        path = os.path.join(LOG_DIR, "trade_log.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def _load_virtual_signals(self) -> List[Dict]:
        path = os.path.join(LOG_DIR, f"virtual_signals_{date.today()}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    # ================================================================
    # 对话
    # ================================================================

    def chat(self, user_message: str) -> str:
        """发送消息并获取回复"""

        # 每次对话刷新系统prompt（持仓可能变了）
        self.system_prompt = self._build_system_prompt()

        # 添加用户消息
        self.conversation.append({"role": "user", "content": user_message})

        # 构建完整消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self.conversation[-20:]  # 只保留最近20轮，避免context太长

        try:
            if self.model.startswith("o") or self.model.startswith("gpt-5"):
                response = self.openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=2000,
                )
            else:
                response = self.openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_completion_tokens=2000,
                )

            reply = response.choices[0].message.content

            # 记录token用量
            usage = response.usage
            if usage:
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                # 粗略成本估算
                if self.model == MODEL_RANK:  # o3
                    cost = (input_tokens * 10 + output_tokens * 40) / 1_000_000
                elif self.model == MODEL_DEEP:  # gpt-5.4
                    cost = (input_tokens * 5 + output_tokens * 15) / 1_000_000
                else:  # mini
                    cost = (input_tokens * 0.4 + output_tokens * 1.6) / 1_000_000
                print(f"  [tokens: {input_tokens}+{output_tokens} | ~${cost:.4f}]")

            # 保存助手回复
            self.conversation.append({"role": "assistant", "content": reply})

            return reply

        except Exception as e:
            error_msg = f"API调用失败: {e}"
            print(f"  ❌ {error_msg}")
            return error_msg

    # ================================================================
    # 特殊命令
    # ================================================================

    def handle_command(self, user_input: str) -> Optional[str]:
        """处理特殊命令（不调用API）"""
        cmd = user_input.strip().lower()

        if cmd in ["/status", "/s"]:
            return self._build_context()

        if cmd in ["/positions", "/pos", "/p"]:
            try:
                positions = self.executor.get_positions()
                if not positions:
                    return "当前无持仓"
                lines = []
                for p in positions:
                    lines.append(
                        f"{p['ticker']:6s} | {p['qty']}股 | "
                        f"入场${p['entry_price']:.2f} | 现价${p['current_price']:.2f} | "
                        f"PnL: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pnl_pct']:+.1f}%)"
                    )
                return "\n".join(lines)
            except Exception as e:
                return f"获取持仓失败: {e}"

        if cmd in ["/pdt"]:
            return (
                f"{self.pdt.status()}\n"
                f"下一个名额恢复: {self.pdt.next_trade_unlock()}"
            )

        if cmd in ["/cost"]:
            return (
                f"当前模型: {self.model}\n"
                f"本次对话轮数: {len(self.conversation) // 2}\n"
                f"提示: /model <模型名> 切换模型"
            )

        if cmd.startswith("/model "):
            new_model = cmd.split(" ", 1)[1].strip()
            valid = [MODEL_RANK, MODEL_DEEP, MODEL_FAST, "o3", "gpt-5.4", "gpt-4.1-mini"]
            if new_model in valid:
                self.model = new_model
                return f"已切换到 {new_model}"
            return f"无效模型。可选: {', '.join(valid)}"

        if cmd in ["/clear", "/c"]:
            self.conversation = []
            return "对话历史已清空"

        if cmd in ["/help", "/h", "?"]:
            return """可用命令：
/status, /s    — 查看完整账户状态
/positions, /p — 查看当前持仓
/pdt           — 查看PDT名额
/model <名称>  — 切换模型 (o3, gpt-5.4, gpt-4.1-mini)
/cost          — 查看本次对话成本
/clear, /c     — 清空对话历史
/help, /h      — 显示此帮助
/quit, /q      — 退出

直接输入任何问题即可与AI对话。"""

        if cmd in ["/quit", "/q", "exit", "quit"]:
            return "__EXIT__"

        return None  # 不是命令，走正常对话

    # ================================================================
    # 主循环
    # ================================================================

    def run(self):
        """交互式对话主循环"""
        print("\n" + "=" * 60)
        print("🤖 交易助手对话模式")
        print(f"   模型: {self.model}")
        print("   输入 /help 查看命令 | 输入 /quit 退出")
        print("=" * 60)

        # 启动时显示账户状态
        print("\n📊 正在加载账户数据...\n")
        try:
            context_summary = self._build_context()
            # 只打印前几行关键信息
            for line in context_summary.split("\n")[:15]:
                print(f"  {line}")
            print("  ...")
        except Exception as e:
            print(f"  ⚠️ 部分数据加载失败: {e}")

        print(f"\n{'─' * 60}")
        print("准备好了。问我任何关于你的持仓、市场、策略的问题。\n")

        while True:
            try:
                user_input = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue

            # 检查特殊命令
            cmd_result = self.handle_command(user_input)
            if cmd_result == "__EXIT__":
                print("再见！")
                break
            if cmd_result is not None:
                print(f"\n{cmd_result}\n")
                continue

            # 正常对话
            print(f"\n🤖 ({self.model}) 思考中...\n")
            reply = self.chat(user_input)
            print(f"{reply}\n")


def main():
    parser = argparse.ArgumentParser(description="交易助手对话模式")
    parser.add_argument(
        "--model",
        default=MODEL_RANK,
        choices=[MODEL_RANK, MODEL_DEEP, MODEL_FAST, "o3", "gpt-5.4", "gpt-4.1-mini"],
        help="选择模型（默认o3）",
    )
    args = parser.parse_args()

    chat = TradingChat(model=args.model)
    chat.run()


if __name__ == "__main__":
    main()
