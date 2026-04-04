# AI Trading Agent

An LLM-based autonomous trading agent for the US stock market, built with a custom ReAct loop (no LangChain/CrewAI). Uses OpenAI models + Alpaca brokerage API to analyze markets, evaluate news, manage risk, and execute trades.

## Architecture

```
User ──▶ Dashboard UI ──▶ /api/chat ──▶ Agent Loop ──▶ LLM (plan)
                                            │              │
                                            │         tool_calls?
                                            │              │ yes
                                            │              ▼
                                            │       Tool Registry
                                            │       ┌─────────────┐
                                            │       │ 7 Tools:    │
                                            │       │  Market Data│──▶ Alpaca Data API
                                            │       │  News       │──▶ Alpaca News + OpenAI
                                            │       │  Scan O/N   │──▶ Scanner (RSI/IBS)
                                            │       │  Scan Intra │──▶ Scanner (reversal)
                                            │       │  Portfolio  │──▶ Alpaca Trading API
                                            │       │  Macro Env  │──▶ SPY/VIX/USO data
                                            │       │  Execute    │──▶ Order Execution
                                            │       └─────────────┘
                                            │              │
                                            │         result back
                                            │              │
                                            ◀──────── LLM (reason) ──▶ Final Answer
```

The agent loop (`agent.py`) is written from scratch:
1. User sends a message
2. LLM receives message + tool definitions (OpenAI function-calling format)
3. LLM either returns a text answer (done) or requests tool calls
4. If tool calls → execute via `ToolRegistry`, feed results back to LLM
5. Repeat until final answer or max 10 iterations

## Tools (7 total)

| # | Tool | Description | Data Source |
|---|------|-------------|-------------|
| 1 | `get_stock_data` | Price, RSI(2), RSI(14), IBS, NATR, ATR, MACD, Bollinger Bands | Alpaca Market Data API |
| 2 | `search_news` | Fetch + AI sentiment analysis, structural risk veto | Alpaca News API + OpenAI |
| 3 | `scan_overnight` | Full-market scan: RSI(2)<15 AND IBS<0.25, sorted by NATR | Alpaca Snapshots + `ta` library |
| 4 | `scan_intraday` | Afternoon reversal scan: stocks down >1.5% with high volume | Alpaca Snapshots |
| 5 | `get_portfolio` | Account value, positions, P&L, PDT status, risk status | Alpaca Trading API |
| 6 | `get_macro_environment` | 3-variable Regime matrix, VIX/VIX3M, SPY/200SMA, Man Group 7-var | Alpaca (SPY, VIXY, USO, TLT, etc.) |
| 7 | `execute_trade` | Submit buy/sell orders with risk validation | Alpaca Trading API |

## Trading Strategy

- **Overnight Mean Reversion** (70%, no PDT cost): Buy oversold stocks at 15:30, sell next morning at 09:45-10:15
- **Intraday Afternoon Reversal** (30%, uses PDT): Buy at 14:00-14:45, exit by 15:40
- **3-Variable Regime**: VIX/VIX3M ratio + SPY vs 200-day SMA + VIX absolute level
- **Risk**: 2% daily loss limit, max 5 positions, max 2 same-sector overnight

## Evaluation

Run the evaluation suite (20 test cases across 5 categories):

```bash
python evaluation.py          # full suite
python evaluation.py --quick  # 5 key tests
```

### Metrics Measured

| Metric | Description |
|--------|-------------|
| Tool Selection Accuracy | % of expected tools correctly called |
| Keyword Coverage | % of expected keywords present in response |
| Safety Compliance | % of unsafe requests correctly refused |
| LLM-as-Judge Score | GPT-4.1-mini rates response quality 1-5 |
| Task Completion Rate | % of queries scoring ≥ 3/5 |
| Avg Iterations | Agent loop iterations per query |
| Avg Latency | Response time in ms |
| Avg Tokens | Token usage per query |

### Test Categories

1. **Information Retrieval** (5 tests): Stock prices, portfolio status, PDT slots
2. **Analysis** (5 tests): Market scanning, multi-stock comparison, macro assessment
3. **Trading** (3 tests): Order execution with risk checks
4. **Safety** (3 tests): Oversized orders, short selling, PDT abuse
5. **Complex Multi-tool** (4 tests): Full briefings, multi-factor analysis

## Setup

### Prerequisites

- Python 3.10+
- Alpaca Paper Trading account ([sign up](https://app.alpaca.markets))
- OpenAI API key ([get one](https://platform.openai.com/api-keys))

### Installation

```bash
git clone https://github.com/hanfuzhao/trading.git
cd trading
pip install flask openai alpaca-py ta python-dotenv websocket-client
```

### Configuration

Create a `.env` file:

```env
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
OPENAI_API_KEY=your_openai_key
```

### Running

```bash
# Start the dashboard (includes autonomous bot + agent chat)
python dashboard_server.py

# Open in browser
open http://localhost:5555

# Run evaluation
python evaluation.py

# Check account status (CLI)
python bot.py --status
```

## Project Structure

```
trading_bot/
├── agent.py              # Custom ReAct agent loop (core assignment deliverable)
├── tools.py              # 7 tool definitions + implementations
├── evaluation.py         # 20-test evaluation suite with quantitative metrics
├── dashboard_server.py   # Flask server: autonomous bot + agent-powered chat
├── dashboard.html        # Web UI with chat, positions, scan results
├── scanner.py            # Market scanner (RSI, IBS, NATR, regime detection)
├── executor.py           # Order execution (overnight + intraday)
├── risk_manager.py       # Risk management (regime matrix, position limits)
├── news_analyzer.py      # News sentiment analysis (structural risk veto)
├── ranker.py             # o3 deep ranking with Man Group macro analogy
├── pdt_tracker.py        # PDT day-trade tracking (3/5-day rolling window)
├── config.py             # All configuration parameters
├── sectors.json          # Stock-to-sector mapping (~200 stocks)
├── bot.py                # CLI status tool
├── .env                  # API keys (not committed)
└── logs/                 # Trade logs, evaluation results, daily reports
```

## Design Choices

1. **Custom agent loop over frameworks**: The ReAct loop in `agent.py` is ~150 lines. LangChain would add 50+ dependencies for the same functionality. Writing it from scratch gives full control over iteration limits, error handling, and tool dispatch.

2. **OpenAI function-calling for tool selection**: The LLM sees JSON schemas and decides which tools to call. This is more reliable than text-parsing approaches (regex on "Action: tool_name").

3. **News as veto-only (≤10% weight)**: Academic research (MDPI 2025) shows text sentiment adds negligible predictive value. We use news only to block structurally dangerous trades (e.g., fraud, delisting).

4. **No mechanical stop-loss for overnight**: Connors' research on 100K+ trades shows stop-losses hurt mean-reversion strategies. Position sizing controls risk instead.

5. **3-variable regime matrix**: Simple VIX thresholds miss term structure signals. VIX/VIX3M ratio > 1.0 (backwardation) correctly identified COVID 2020, 2018 Volpocalypse, and 2025 tariff shock.

## Tech Stack

- **Python 3.13** — core language
- **OpenAI API** — gpt-4.1-mini (fast), gpt-5.4 (deep analysis), o3 (ranking)
- **Alpaca API** — market data, news, order execution (paper trading)
- **Flask** — web server
- **ta** — technical analysis indicators
- **websocket-client** — real-time price streaming
