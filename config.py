"""
配置文件 v6 — 70%隔夜均值回归 + 30%日内 | 3变量Regime | IBS+RSI(2)<15
"""
import os
from dotenv import load_dotenv

load_dotenv()

# === API Keys ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"

# === 资金 ===
MAX_CAPITAL = float(os.getenv("MAX_CAPITAL", "0"))
LONG_ONLY = True

# === 股票池过滤 ===
MIN_STOCK_PRICE = 10         # v6: $10
MIN_AVG_VOLUME = 1_000_000   # v6: 1M

# === 隔夜策略阈值 ===
RSI_2_THRESHOLD = 15         # RSI(2) < 15 (v6: 从<10放宽)
RSI_2_EXTREME = 5            # RSI(2) < 5 优先级最高
IBS_THRESHOLD = 0.25         # IBS < 0.25
CONSECUTIVE_DOWN_BONUS = 3   # 连续下跌≥3天加分
MAX_OVERNIGHT_POSITIONS = 3  # 每晚最多3只

# === 日内策略阈值 ===
INTRADAY_DROP_PCT = 1.5      # 日内跌幅>1.5%触发均值回归信号
INTRADAY_VOL_RATIO = 1.5     # 量比>1.5确认

# === 3变量Regime矩阵 ===
# {regime: {overnight params, intraday params}}
REGIME_PARAMS = {
    "bullish": {
        "on_risk_pct": 1.0,  "on_max_pos_pct": 5,  "on_total_pct": 15,
        "id_risk_pct": 1.0,  "id_max_pos_pct": 10, "id_total_pct": 10, "id_atr_mult": 2.0,
    },
    "cautious": {
        "on_risk_pct": 0.75, "on_max_pos_pct": 4,  "on_total_pct": 12,
        "id_risk_pct": 1.0,  "id_max_pos_pct": 10, "id_total_pct": 10, "id_atr_mult": 2.0,
    },
    "defensive": {
        "on_risk_pct": 0.5,  "on_max_pos_pct": 3,  "on_total_pct": 8,
        "id_risk_pct": 0.5,  "id_max_pos_pct": 5,  "id_total_pct": 5,  "id_atr_mult": 2.5,
    },
    "crisis": {
        "on_risk_pct": 0.25, "on_max_pos_pct": 2,  "on_total_pct": 5,
        "id_risk_pct": 0.5,  "id_max_pos_pct": 5,  "id_total_pct": 5,  "id_atr_mult": 2.5,
    },
}

# === 风险管理 ===
MAX_CONCURRENT_POSITIONS = 5
MAX_SAME_SECTOR_OVERNIGHT = 2
MAX_DAILY_LOSS_PCT = 2.0     # v6: 2% (从3%降低)
MAX_WEEKLY_LOSS_PCT = 5.0
CONSECUTIVE_LOSS_COOLDOWN = 3
COOLDOWN_HOURS = 2
CATASTROPHIC_STOP_PCT = 5.0  # 盘前极端缺口止损线

# === 评分权重 ===
ON_TECH_W = 0.65
ON_MACRO_W = 0.25
ON_NEWS_W = 0.10
ID_TECH_W = 0.55
ID_MACRO_W = 0.25
ID_NEWS_W = 0.10
ID_VOL_W = 0.10

# === 时间窗口（美东时间）===
OVERNIGHT_SCAN_START = "15:00"
OVERNIGHT_SCAN_END = "15:30"
OVERNIGHT_ENTRY_START = "15:30"
OVERNIGHT_ENTRY_END = "15:45"
OVERNIGHT_EXIT_START = "09:45"
OVERNIGHT_EXIT_END = "10:15"
INTRADAY_ENTRY_START = "14:00"
INTRADAY_ENTRY_END = "14:45"
INTRADAY_EXIT_LIMIT = "15:40"
INTRADAY_EXIT_MARKET = "15:48"
INTRADAY_CONFIRM_ZERO = "15:50"
MIDDAY_DEAD_START = "12:00"
MIDDAY_DEAD_END = "13:30"

# === 执行参数 ===
ENTRY_TIMEOUT_SECONDS = 120
TIME_STOP_MINUTES = 25
PARTIAL_EXIT_R = 1.0
PARTIAL_EXIT_PCT = 0.5
TRAILING_ATR_MULT = 1.5

# === PDT ===
PDT_MAX_DAY_TRADES = 3
PDT_WINDOW_DAYS = 5
MONDAY_EXCEPTION_SCORE = 85  # v6: 85 (从95降低)

# === OpenAI 模型 ===
MODEL_FAST = "gpt-4.1-mini"
MODEL_SENTIMENT = "gpt-4.1-mini"  # v6: 情绪分类用mini(No-CoT)
MODEL_DEEP = "gpt-5.4"
MODEL_RANK = "o3"
MAX_O3_CALLS_PER_DAY = 3

# === 日志 ===
LOG_DIR = "logs"
