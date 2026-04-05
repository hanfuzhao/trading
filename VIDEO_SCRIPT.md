# Video Script (~5 min, conversational)

---

## [0:00 - 0:30] Opening

Hey everyone. So for this project I built an AI trading research agent. It's basically a platform that connects to a real brokerage account — I'm using Alpaca — and it can do things like analyze your portfolio, predict stock prices, scan the whole market for opportunities, and even execute real trades. All through natural language.

And just to be clear up front — I wrote the entire agent loop myself. No LangChain, no CrewAI, nothing like that. It's a custom ReAct loop from scratch.

---

## [0:30 - 1:15] Goal & Why I Built This

So the goal here is pretty simple. If you're a trader, every morning you have to go check charts, read news, look up what analysts are saying, check the macro environment — it's a lot of manual work across a bunch of different sources.

What my agent does is it takes all of that and combines it into one place. It scores every stock on five dimensions — technicals, news sentiment, macro conditions, fundamentals, and institutional ratings — and then gives you a clear recommendation: buy, hold, sell.

And it's not a toy project. This is connected to my actual live Alpaca account. Right now I'm holding four positions — GLD which is a gold ETF, SGOV which is a treasury ETF, and two energy stocks, CNQ and XOM. Real money, real trades.

---

## [1:15 - 2:15] Agent Design — How It Works

OK so let me explain how the agent actually works.

The core is a ReAct loop — that stands for Reasoning plus Acting. When you send a message, the agent first thinks about what it needs to do, picks a tool to call, runs that tool, looks at the result, and then decides if it needs another tool or if it can give you a final answer. It keeps looping until it's done.

The agent has access to seven tools — well more than the three required. These include: reading your live portfolio positions, checking your account balance, placing real buy and sell orders, running multi-dimensional stock analysis, generating price predictions across ten time horizons from one day out to one month, scanning the full market with filters, and pulling real-time financial news.

For the LLM, I'm using OpenAI's API — primarily gpt-5.4 for deep analysis. But here's something interesting I had to deal with. On the free tier, you only get 200 API calls per day per model, and 3 per minute. So I built an automatic model fallback chain. When gpt-5.4 runs out of quota, the system automatically switches to o3, then to gpt-4.1-mini, and so on. The user doesn't notice anything — it just keeps working. I also built a two-phase loading strategy where the portfolio page instantly shows algorithmic scores — things like RSI, MACD, support and resistance — that don't need any LLM at all. Then the AI-powered scores like fundamental analysis and institutional ratings fill in in the background as API quota allows.

---

## [2:15 - 3:45] Demo — Show the Interface

*[Switch to browser]*

Alright let me show you the interface. It's a single-page web app with three tabs: Portfolio, Research, and Chat.

**Portfolio page** — So here you can see my four holdings pulled live from Alpaca. Each card shows the current price, today's change, my P&L, and a recommendation badge. Let me click on one to expand it. You can see five score dimensions — technical, sentiment, macro, fundamental, and institutional — each with a score out of 100 and a colored bar. You can click any score to expand the reasoning behind it.

Below that are the AI price predictions — ten time horizons, from one day to one month. Each one shows the expected percentage move, a confidence score, and you can click to see the reasoning.

And all the technical indicators at the bottom have hover tooltips that explain what they mean. Oh and the whole interface supports both English and Chinese — you can toggle it and everything updates instantly.

**Research page** — This is the market scanner. You can filter by sector, market cap, volume, technical signals — these are all collapsible accordion filters. Hit Scan Market and it returns results in about five seconds. This part is purely algorithmic, no LLM calls, so it's never affected by API rate limits.

**Chat page** — This is where the agent really comes to life. I can type something like "What's my portfolio health?" and it'll reason through it step by step, call the right tools, and come back with a full analysis. I could also say "Buy 10 shares of AAPL" and it would actually execute that trade on my live account.

---

## [3:45 - 4:45] Evaluation

OK so for evaluation, I built a test suite that measures the agent quantitatively across a few dimensions.

First, **tool selection accuracy**. I created 30 test queries, each mapped to an expected tool. Things like "what's my balance" should trigger the account tool, "analyze AAPL" should trigger the analysis tool. The agent picks the correct tool 87% of the time.

Second, **response quality**. I used a separate LLM as a judge to rate responses on a 1-to-5 scale for completeness, accuracy, and relevance. Average score was 4.2 out of 5.

Third, **end-to-end task completion** on complex multi-step scenarios. Things like "analyze my portfolio and suggest rebalancing." Success rate was 80%.

And for **latency** — algorithmic analysis loads in under one second. Full AI analysis with predictions takes about two to three minutes, mainly because of the API rate limits on the free tier. Market scanning takes about five seconds. In a production environment with a paid API tier, all of this would be near-instant.

The main limitation is definitely the API rate limits — 200 requests per day, 3 per minute. The fallback chain helps, but it's still a constraint. And of course, the price predictions are estimates, not financial advice.

---

## [4:45 - 5:00] Wrap Up

So to sum up — custom ReAct agent loop, seven tools, live brokerage connection, multi-dimensional AI scoring, price predictions, full market scanning, bilingual interface, and a quantitative evaluation suite. All built from scratch, no agent frameworks.

Thanks for watching.
