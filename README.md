# 全美股日内交易Bot 🦙

OpenAI + Alpaca | PDT限制下的精选交易系统

## 你需要做什么

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API Key
```bash
cp .env.example .env
```
编辑 `.env` 文件，填入你的3个Key：
- `ALPACA_API_KEY` — Alpaca Paper Trading的Key
- `ALPACA_API_SECRET` — Alpaca Paper Trading的Secret
- `OPENAI_API_KEY` — OpenAI API Key

**永远不要把Key分享给任何人或上传到GitHub。**

### 3. 运行

```bash
# 查看账户状态
python bot.py --status

# 只扫描不交易（看看今天有什么异动）
python bot.py --scan-only

# 完整流程但不下单（测试所有分析逻辑）
python bot.py --dry-run

# 正式运行（Paper Trading模拟盘）
python bot.py
```

## 系统架构

```
全市场4000+只 → 技术扫描(30-80只) → 新闻分析(10-20只) → o3排名(Top 3) → 下单
```

### 四层漏斗

| 层级 | 工具 | 成本 | 输出 |
|------|------|------|------|
| 第一层：技术扫描 | 本地Python | $0 | 30-80只异动股 |
| 第二层：新闻分析 | gpt-4.1-mini + gpt-5.4 | ~$0.05/天 | 10-20只有催化剂的 |
| 第三层：深度排名 | o3 | ~$0.10/天 | Top 3推荐 |
| 第四层：执行 | Alpaca API | $0 | 下单+止损止盈 |

### OpenAI模型分工

| 模型 | 任务 | 何时调用 |
|------|------|----------|
| gpt-4.1-mini | 新闻情绪快筛 | 每条新闻 |
| gpt-5.4 | 复杂新闻深度分析 | mini置信度<50时 |
| o3 (high) | 横向排名 + 子弹分配 | 每轮筛选后 |

## 关键规则

### PDT规则（硬锁）
- 5个交易日内最多3次日内交易
- 系统自动追踪，**不可能超过3次**
- 名额用完后系统只监控不交易

### 风险管理（硬编码）
- 单笔最大仓位：15%
- 单笔最大亏损：1.5%
- 单日最大亏损：3%（触发后停止交易）
- 同时最多持2只
- 3:50 PM强制平仓所有日内持仓

### 子弹分配
o3会根据本周剩余名额和剩余交易日数，决定今天该不该用名额。
不是每个好信号都值得用——周四只剩1次的时候比周一剩3次要挑剔得多。

## 对话模式

Bot在后台跑着的时候，你可以另开一个终端跟AI对话：

```bash
# 用o3深度对话（默认）
python chat.py

# 用gpt-5.4（更快）
python chat.py --model gpt-5.4

# 用gpt-4.1-mini（最便宜，简单问题够用）
python chat.py --model gpt-4.1-mini
```

对话助手会自动读取你的持仓、扫描结果、交易记录作为上下文。你可以问：
- "RTX现在还能拿吗"
- "今天扫描出了什么好机会"
- "我这周还有几次日内交易名额"
- "帮我分析一下能源板块最近的走势"
- "如果油价跌回$90，我的持仓会怎样"

对话中输入 `/help` 查看所有命令，`/status` 查看账户，`/model gpt-4.1-mini` 切换模型。

## 文件结构

```
trading_bot/
├── bot.py              # 主程序（启动入口）
├── chat.py             # 交互对话模式（连接OpenAI）
├── scanner.py          # 全市场技术面扫描器
├── news_analyzer.py    # 新闻情绪分析管道
├── ranker.py           # o3深度排名系统
├── executor.py         # Alpaca下单执行器
├── pdt_tracker.py      # PDT名额追踪器
├── risk_manager.py     # 风险管理规则
├── config.py           # 配置参数
├── requirements.txt    # Python依赖
├── .env.example        # API Key模板
└── logs/               # 运行日志（自动生成）
    ├── trade_log.json
    ├── pdt_trades.json
    ├── risk_state.json
    ├── seen_news_ids.json
    └── daily_report_*.json
```

## 日成本估算

| 项目 | 日成本 |
|------|--------|
| gpt-4.1-mini 新闻扫描 | ~$0.03 |
| gpt-5.4 复杂新闻 | ~$0.05 |
| o3 深度排名 | ~$0.10 |
| **合计** | **~$0.18** |
| **月成本** | **~$4** |
| **$6000可用** | **~125年** |

## ⚠️ 重要提醒

1. **先跑 `--dry-run` 至少1周** 确认逻辑正确再开始模拟盘交易
2. **Paper Trading至少跑4周** 再考虑实盘
3. **永远不要把 `.env` 文件上传到GitHub**
4. 这是Paper Trading，亏的不是真钱，但要认真对待数据
5. 切换到实盘时需要修改 `config.py` 中的 `ALPACA_BASE_URL` 和 `executor.py` 中的 `paper=False`
