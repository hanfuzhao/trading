"""
配置文件 - 所有可调参数集中管理
"""
import os
from dotenv import load_dotenv

load_dotenv()

# === API Keys ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper Trading

# === 扫描器参数 ===
MIN_STOCK_PRICE = float(os.getenv("MIN_STOCK_PRICE", "5"))
MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "200000"))
MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", "300000000"))

# 技术面筛选阈值
VOLUME_SPIKE_RATIO = 2.0       # 成交量 > 均量 × 2（盘中会自动时间校正）
RSI_OVERSOLD = 30              # 标准超卖
RSI_OVERBOUGHT = 70            # 标准超买
GAP_THRESHOLD_PCT = 2.0        # 跳空 > 2%
PREMARKET_VOLUME_MIN = 100000  # 盘前成交量 > 10万股
PREMARKET_CHANGE_PCT = 2.0     # 盘前涨跌 > 2%
INTRADAY_MOMENTUM_PCT = 1.5    # 日内动量 > 1.5%
VWAP_DEVIATION_PCT = 1.5       # VWAP偏离 > 1.5%

# === 资金上限（模拟小账户） ===
MAX_CAPITAL = float(os.getenv("MAX_CAPITAL", "0"))  # 0=不限制，用账户实际值

# === 风险管理参数 ===
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "15"))
MAX_SINGLE_LOSS_PCT = float(os.getenv("MAX_SINGLE_LOSS_PCT", "1.5"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "3"))
MAX_CONCURRENT_POSITIONS = 2
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_ATR_MULTIPLIER = 2.5
FORCE_CLOSE_TIME = os.getenv("FORCE_CLOSE_TIME", "15:50")

# === PDT ===
PDT_MAX_DAY_TRADES = 3
PDT_WINDOW_DAYS = 5

# === OpenAI 模型 ===
MODEL_FAST = "gpt-4.1-mini"    # 新闻快筛
MODEL_DEEP = "gpt-5.4"         # 复杂新闻
MODEL_RANK = "o3"              # 深度排名

# === 扫描频率 ===
PREMARKET_SCAN_INTERVAL = 300   # 盘前每5分钟
MARKET_SCAN_INTERVAL = 60       # 盘中每1分钟（价格）
NEWS_SCAN_INTERVAL = 300        # 新闻每5分钟
WEBSEARCH_INTERVAL = 3600       # Web Search每小时

# === 日志 ===
LOG_DIR = "logs"
TRADE_LOG = "trades.json"
SIGNAL_LOG = "signals.json"
