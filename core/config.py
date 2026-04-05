


import os
from dotenv import load_dotenv

load_dotenv()


ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPACA_BASE_URL = "https://api.alpaca.markets"
ALPACA_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"


MAX_CAPITAL = float(os.getenv("MAX_CAPITAL", "0"))
LONG_ONLY = True


MIN_STOCK_PRICE = 10
MIN_AVG_VOLUME = 1_000_000


RSI_2_THRESHOLD = 15
RSI_2_EXTREME = 5
IBS_THRESHOLD = 0.25
CONSECUTIVE_DOWN_BONUS = 3
MAX_OVERNIGHT_POSITIONS = 3


INTRADAY_DROP_PCT = 1.5
INTRADAY_VOL_RATIO = 1.5



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


MAX_CONCURRENT_POSITIONS = 5
MAX_SAME_SECTOR_OVERNIGHT = 2
MAX_DAILY_LOSS_PCT = 2.0
MAX_WEEKLY_LOSS_PCT = 5.0
CONSECUTIVE_LOSS_COOLDOWN = 3
COOLDOWN_HOURS = 2
CATASTROPHIC_STOP_PCT = 5.0


ON_TECH_W = 0.65
ON_MACRO_W = 0.25
ON_NEWS_W = 0.10
ID_TECH_W = 0.55
ID_MACRO_W = 0.25
ID_NEWS_W = 0.10
ID_VOL_W = 0.10


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


ENTRY_TIMEOUT_SECONDS = 120
TIME_STOP_MINUTES = 25
PARTIAL_EXIT_R = 1.0
PARTIAL_EXIT_PCT = 0.5
TRAILING_ATR_MULT = 1.5


PDT_MAX_DAY_TRADES = 3
PDT_WINDOW_DAYS = 5
MONDAY_EXCEPTION_SCORE = 85


MODEL_FAST = "gpt-4.1-mini"
MODEL_SENTIMENT = "gpt-4.1-mini"
MODEL_DEEP = "gpt-5.4"
MODEL_RANK = "o3"
MAX_O3_CALLS_PER_DAY = 3


PREDICTION_CACHE_TTL = 1800
PORTFOLIO_REFRESH_INTERVAL = 300
RESEARCH_SCAN_LIMIT = 50
MAX_PREDICTION_CALLS_PER_HOUR = 20
MAX_DEEP_RESEARCH_PER_DAY = 10
SCORING_WEIGHTS = {
    "technical": 0.30, "sentiment": 0.20, "macro": 0.15,
    "fundamental": 0.20, "institutional": 0.15,
}
PREDICTION_HORIZONS = ["1d", "2d", "3d", "4d", "5d", "1w", "2w", "3w", "4w", "1mo"]


LOG_DIR = "logs"
