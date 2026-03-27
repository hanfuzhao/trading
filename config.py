"""
配置文件 v3 — 全美股日内交易 | 2026年4-8月 | PDT 3次/周
"""
import os
from dotenv import load_dotenv

load_dotenv()

# === API Keys ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# === 股票池过滤 ===
MIN_STOCK_PRICE = float(os.getenv("MIN_STOCK_PRICE", "5"))
MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "500000"))

# === 资金上限（模拟小账户） ===
MAX_CAPITAL = float(os.getenv("MAX_CAPITAL", "0"))  # 0=不限制

# === 交易方向 ===
LONG_ONLY = True  # 只做多，不做空

# === 动量类信号阈值 ===
GAP_THRESHOLD_PCT = 2.0
VOLUME_SPIKE_RATIO = 2.0
RSI_OVERBOUGHT = 70
RSI_OVERBOUGHT_EXTREME = 80
MACD_CROSS_SCORE = 15
BB_UPPER_SCORE = 10
RESISTANCE_BREAK_SCORE = 10
VWAP_DEVIATION_PCT = 1.5
INTRADAY_MOMENTUM_PCT = 1.5

# === 均值回归信号阈值 ===
RSI_2_OVERSOLD = 10
RSI_2_EXTREME = 5
RSI_14_OVERSOLD = 30
RSI_14_OVERSOLD_EXTREME = 20
WILLIAMS_R_OVERSOLD = -90
CONSECUTIVE_DOWN_DAYS = 3

# === 波动率 Regime 参数 ===
# {regime: (risk_pct, max_position_pct, atr_stop_mult)}
VOL_REGIMES = {
    "low":     {"risk_pct": 1.5, "max_pos_pct": 15, "atr_mult": 1.5},
    "medium":  {"risk_pct": 1.1, "max_pos_pct": 12, "atr_mult": 2.0},
    "high":    {"risk_pct": 0.75, "max_pos_pct": 8,  "atr_mult": 2.5},
    "extreme": {"risk_pct": 0.4,  "max_pos_pct": 5,  "atr_mult": 3.0},
}

# === 风险管理 ===
MAX_CONCURRENT_POSITIONS = 2
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "3"))
MAX_WEEKLY_LOSS_PCT = 5.0
CONSECUTIVE_LOSS_COOLDOWN = 3        # 连输N笔暂停
COOLDOWN_HOURS = 2                   # 暂停时长
CONSECUTIVE_WEEK_LOSS_PAUSE = 2      # 连续N周亏损暂停

# === 入场时间窗口（美东时间） ===
ENTRY_WINDOW_1_START = "09:45"
ENTRY_WINDOW_1_END = "10:15"
ENTRY_WINDOW_2_START = "15:00"
ENTRY_WINDOW_2_END = "15:30"
BLACKOUT_OPEN_END = "09:45"          # 开盘波动期
MIDDAY_DEAD_START = "11:30"
MIDDAY_DEAD_END = "13:30"
NO_NEW_POSITION_AFTER = "15:30"

# === 执行参数 ===
ENTRY_TIMEOUT_SECONDS = 120          # Limit Order 120秒未成交取消
LIMIT_OFFSET = 0.03                  # 入场Limit比现价差$0.03
TIME_STOP_MINUTES = 25               # 持仓25分钟盈利不足0.5R平仓

# === 尾盘强制平仓时间表 ===
STOP_NEW_POSITIONS = "15:30"
CLOSE_PROFITABLE = "15:45"
CLOSE_ALL = "15:48"
FINAL_CHECK = "15:50"

# === 分批止盈 ===
PARTIAL_EXIT_R = 1.0                 # 到达1R平50%
PARTIAL_EXIT_PCT = 0.5               # 平仓比例
TRAILING_ATR_MULT = 1.5              # trailing stop用1.5x ATR

# === PDT ===
PDT_MAX_DAY_TRADES = 3
PDT_WINDOW_DAYS = 5
MONDAY_MIN_SCORE = 95                # 周一入场最低分数

# === OpenAI 模型 ===
MODEL_FAST = "gpt-4.1-mini"
MODEL_DEEP = "gpt-5.4"
MODEL_RANK = "o3"
MAX_O3_CALLS_PER_DAY = 3

# === 日志 ===
LOG_DIR = "logs"
