import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

// ============================================================
// MOCK DATA (replace with real Alpaca data via API)
// ============================================================
const MOCK_ACCOUNT = {
  portfolio_value: 15420.50,
  cash: 6830.20,
  buying_power: 13660.40,
  daily_pnl: -182.30,
  daily_pnl_pct: -1.17,
  total_pnl: 1420.50,
  total_pnl_pct: 10.16,
};

const MOCK_POSITIONS = [
  { ticker: "XOM", qty: 15, entry: 108.50, current: 114.20, pnl: 85.50, pnl_pct: 5.25, sector: "Energy" },
  { ticker: "RTX", qty: 20, entry: 124.30, current: 121.80, pnl: -50.00, pnl_pct: -2.01, sector: "Defense" },
  { ticker: "UNH", qty: 5, entry: 512.00, current: 518.40, pnl: 32.00, pnl_pct: 1.25, sector: "Healthcare" },
];

const MOCK_SCANNER = [
  { ticker: "OXY", price: 68.40, strength: 87, vol_ratio: 5.2, signals: ["volume_spike_5.2x", "rsi_oversold_23", "gap_+4.1%"], sentiment: 72, sector: "Energy" },
  { ticker: "LMT", price: 498.20, strength: 79, vol_ratio: 3.8, signals: ["macd_golden_cross", "volume_spike_3.8x"], sentiment: 65, sector: "Defense" },
  { ticker: "CVX", price: 172.50, strength: 75, vol_ratio: 4.1, signals: ["gap_+3.8%", "above_bollinger"], sentiment: 58, sector: "Energy" },
  { ticker: "HUM", price: 312.80, strength: 71, vol_ratio: 2.9, signals: ["rsi_oversold_28", "near_support"], sentiment: 45, sector: "Healthcare" },
  { ticker: "GD", price: 302.10, strength: 68, vol_ratio: 3.2, signals: ["volume_spike_3.2x", "momentum_+2.8%"], sentiment: 52, sector: "Defense" },
  { ticker: "HAL", price: 38.90, strength: 65, vol_ratio: 4.5, signals: ["volume_spike_4.5x", "gap_+3.2%"], sentiment: 61, sector: "Energy" },
  { ticker: "ABBV", price: 178.30, strength: 62, vol_ratio: 2.1, signals: ["macd_golden_cross", "near_support"], sentiment: 38, sector: "Healthcare" },
  { ticker: "NOC", price: 518.70, strength: 60, vol_ratio: 2.8, signals: ["momentum_+2.2%", "volume_spike_2.8x"], sentiment: 48, sector: "Defense" },
];

const MOCK_NEWS = [
  { time: "09:32", ticker: "OXY", sentiment: "bullish", severity: 8, headline: "Strait of Hormuz tensions escalate, oil supply fears mount", source: "Reuters" },
  { time: "09:28", ticker: "LMT", sentiment: "bullish", severity: 7, headline: "Pentagon accelerates hypersonic missile program funding", source: "Defense One" },
  { time: "09:15", ticker: "XOM", sentiment: "bullish", severity: 6, headline: "Brent crude tops $112 amid Gulf shipping disruptions", source: "Bloomberg" },
  { time: "08:55", ticker: "UNH", sentiment: "bearish", severity: 4, headline: "ACA subsidy extension faces Senate roadblock", source: "Politico" },
  { time: "08:40", ticker: "CVX", sentiment: "bullish", severity: 7, headline: "Chevron raises Q2 production guidance on Gulf assets", source: "Alpaca News" },
  { time: "08:22", ticker: "RTX", sentiment: "neutral", severity: 3, headline: "Raytheon Q1 deliveries on track, margins steady", source: "Seeking Alpha" },
];

const MOCK_TRADES = [
  { date: "03/27", ticker: "SLB", action: "BUY→SELL", entry: 52.10, exit: 53.80, pnl: 25.50, pnl_pct: 3.26 },
  { date: "03/26", ticker: "XOM", action: "BUY→HOLD", entry: 108.50, exit: null, pnl: 85.50, pnl_pct: 5.25 },
  { date: "03/25", ticker: "BA", action: "BUY→SELL", entry: 178.20, exit: 175.40, pnl: -28.00, pnl_pct: -1.57 },
  { date: "03/24", ticker: "LMT", action: "BUY→SELL", entry: 492.30, exit: 498.10, pnl: 58.00, pnl_pct: 1.18 },
  { date: "03/21", ticker: "DVN", action: "BUY→SELL", entry: 42.80, exit: 44.50, pnl: 51.00, pnl_pct: 3.97 },
];

const MOCK_PORTFOLIO_HISTORY = Array.from({ length: 30 }, (_, i) => ({
  day: i + 1,
  value: 14000 + Math.sin(i * 0.3) * 800 + i * 35 + (Math.random() - 0.4) * 200,
}));

const MOCK_SECTOR_ALLOCATION = [
  { name: "Energy", value: 42, color: "#f59e0b" },
  { name: "Defense", value: 28, color: "#3b82f6" },
  { name: "Healthcare", value: 18, color: "#10b981" },
  { name: "Cash", value: 12, color: "#6b7280" },
];

const PDT_STATE = { used: 1, max: 3, next_unlock: "03/29 Mon" };

// ============================================================
// MAIN DASHBOARD
// ============================================================
export default function TradingDashboard() {
  const [activeTab, setActiveTab] = useState("scanner");
  const [chatMessages, setChatMessages] = useState([
    { role: "assistant", content: "Trading assistant online. I can see your positions, scanner results, and market status. Ask me anything." }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const buildContext = useCallback(() => {
    return `You are a quantitative trading assistant. Here is the user's real-time data:
== Account == Total $${MOCK_ACCOUNT.portfolio_value} Cash $${MOCK_ACCOUNT.cash} Daily PnL $${MOCK_ACCOUNT.daily_pnl}
== Positions == ${MOCK_POSITIONS.map(p => `${p.ticker}:${p.qty} shares@$${p.entry}, current $${p.current}, PnL ${p.pnl_pct}%`).join(" | ")}
== PDT == Used ${PDT_STATE.used}/${PDT_STATE.max}
== Today's Top 5 Scanner == ${MOCK_SCANNER.slice(0, 5).map(s => `${s.ticker}(strength ${s.strength}, sentiment ${s.sentiment})`).join(" | ")}
== Latest News == ${MOCK_NEWS.slice(0, 3).map(n => `${n.ticker}:${n.headline.slice(0, 40)}`).join(" | ")}
== Macro == US-Iran tensions, oil >$110, Fed rates unchanged, S&P500 correction range, midterm election year
Rules: Max position size 15%, max single loss <1.5%, max daily loss <3%, PDT 3 trades/5 days, forced close at 3:50PM. Respond concisely in English.`;
  }, []);

  const sendMessage = async () => {
    if (!chatInput.trim() || isLoading) return;
    const userMsg = chatInput.trim();
    setChatInput("");
    setChatMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setIsLoading(true);

    try {
      const messages = [
        ...chatMessages.filter(m => m.role !== "system").slice(-16),
        { role: "user", content: userMsg }
      ];

      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gpt-4.1-mini",
          max_tokens: 1000,
          messages: [{ role: "system", content: buildContext() }, ...messages],
        }),
      });

      const data = await response.json();
      const reply = data.choices?.[0]?.message?.content || "Sorry, unable to get a response.";
      setChatMessages(prev => [...prev, { role: "assistant", content: reply }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { role: "assistant", content: `Connection failed: ${err.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatMoney = (n) => (n >= 0 ? "+" : "") + "$" + Math.abs(n).toFixed(2);
  const formatPct = (n) => (n >= 0 ? "+" : "") + n.toFixed(2) + "%";
  const pnlColor = (n) => n >= 0 ? "#22c55e" : "#ef4444";
  const sentimentIcon = (s) => s === "bullish" ? "▲" : s === "bearish" ? "▼" : "●";
  const sentimentColor = (s) => s === "bullish" ? "#22c55e" : s === "bearish" ? "#ef4444" : "#9ca3af";

  const marketOpen = currentTime.getHours() >= 9 && currentTime.getHours() < 16;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0e17",
      color: "#e2e8f0",
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      fontSize: "12px",
      lineHeight: 1.5,
    }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet" />

      {/* ============ TOP BAR ============ */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "10px 16px",
        background: "linear-gradient(180deg, #111827 0%, #0f1629 100%)",
        borderBottom: "1px solid #1e293b",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "16px" }}>🦙</span>
          <span style={{ fontWeight: 700, fontSize: "13px", letterSpacing: "1px", color: "#f1f5f9" }}>ALPHA TERMINAL</span>
          <span style={{
            padding: "2px 8px",
            borderRadius: "3px",
            fontSize: "10px",
            fontWeight: 600,
            background: marketOpen ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)",
            color: marketOpen ? "#22c55e" : "#ef4444",
            border: `1px solid ${marketOpen ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
          }}>
            {marketOpen ? "● MARKET OPEN" : "○ MARKET CLOSED"}
          </span>
        </div>

        <div style={{ display: "flex", gap: "24px", alignItems: "center" }}>
          {[
            { label: "PORTFOLIO", value: `$${MOCK_ACCOUNT.portfolio_value.toLocaleString()}`, color: "#f1f5f9" },
            { label: "CASH", value: `$${MOCK_ACCOUNT.cash.toLocaleString()}`, color: "#94a3b8" },
            { label: "DAY P&L", value: formatMoney(MOCK_ACCOUNT.daily_pnl), color: pnlColor(MOCK_ACCOUNT.daily_pnl) },
            { label: "TOTAL P&L", value: formatMoney(MOCK_ACCOUNT.total_pnl), color: pnlColor(MOCK_ACCOUNT.total_pnl) },
          ].map((item, i) => (
            <div key={i} style={{ textAlign: "right" }}>
              <div style={{ fontSize: "9px", color: "#64748b", letterSpacing: "1px", marginBottom: "1px" }}>{item.label}</div>
              <div style={{ fontSize: "14px", fontWeight: 600, color: item.color }}>{item.value}</div>
            </div>
          ))}

          <div style={{
            padding: "4px 12px",
            borderRadius: "4px",
            background: PDT_STATE.used >= 3 ? "rgba(239,68,68,0.15)" : PDT_STATE.used >= 2 ? "rgba(245,158,11,0.15)" : "rgba(34,197,94,0.15)",
            border: `1px solid ${PDT_STATE.used >= 3 ? "rgba(239,68,68,0.3)" : PDT_STATE.used >= 2 ? "rgba(245,158,11,0.3)" : "rgba(34,197,94,0.3)"}`,
          }}>
            <div style={{ fontSize: "9px", color: "#64748b", letterSpacing: "1px" }}>PDT TRADES</div>
            <div style={{
              fontSize: "14px",
              fontWeight: 700,
              color: PDT_STATE.used >= 3 ? "#ef4444" : PDT_STATE.used >= 2 ? "#f59e0b" : "#22c55e",
            }}>
              {PDT_STATE.max - PDT_STATE.used}/{PDT_STATE.max}
            </div>
          </div>

          <div style={{ fontSize: "11px", color: "#64748b" }}>
            {currentTime.toLocaleTimeString("en-US", { hour12: false })}
          </div>
        </div>
      </div>

      {/* ============ MAIN GRID ============ */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 380px",
        height: "calc(100vh - 52px)",
        overflow: "hidden",
      }}>

        {/* ============ LEFT: DATA PANELS ============ */}
        <div style={{ display: "flex", flexDirection: "column", overflow: "hidden", borderRight: "1px solid #1e293b" }}>

          {/* --- Charts Row --- */}
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", borderBottom: "1px solid #1e293b", minHeight: "200px" }}>
            {/* Portfolio Chart */}
            <div style={{ padding: "12px 16px", borderRight: "1px solid #1e293b" }}>
              <div style={{ fontSize: "10px", color: "#64748b", letterSpacing: "1px", marginBottom: "8px" }}>30-DAY PORTFOLIO VALUE</div>
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={MOCK_PORTFOLIO_HISTORY}>
                  <defs>
                    <linearGradient id="grd" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="day" hide />
                  <YAxis domain={["dataMin - 200", "dataMax + 200"]} hide />
                  <Tooltip
                    contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: "4px", fontSize: "11px" }}
                    labelFormatter={(v) => `Day ${v}`}
                    formatter={(v) => [`$${v.toFixed(0)}`, "Value"]}
                  />
                  <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} fill="url(#grd)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Sector Allocation */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ fontSize: "10px", color: "#64748b", letterSpacing: "1px", marginBottom: "8px" }}>SECTOR EXPOSURE</div>
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <ResponsiveContainer width={120} height={120}>
                  <PieChart>
                    <Pie data={MOCK_SECTOR_ALLOCATION} dataKey="value" cx="50%" cy="50%" innerRadius={30} outerRadius={50} strokeWidth={0}>
                      {MOCK_SECTOR_ALLOCATION.map((s, i) => <Cell key={i} fill={s.color} />)}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ flex: 1 }}>
                  {MOCK_SECTOR_ALLOCATION.map((s, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                      <div style={{ width: 8, height: 8, borderRadius: "2px", background: s.color, flexShrink: 0 }} />
                      <span style={{ color: "#94a3b8", flex: 1 }}>{s.name}</span>
                      <span style={{ fontWeight: 600, color: "#e2e8f0" }}>{s.value}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* --- Positions --- */}
          <div style={{ padding: "10px 16px", borderBottom: "1px solid #1e293b" }}>
            <div style={{ fontSize: "10px", color: "#64748b", letterSpacing: "1px", marginBottom: "8px" }}>OPEN POSITIONS</div>
            <div style={{ display: "grid", gridTemplateColumns: "60px 40px 70px 70px 80px 60px 70px", gap: "4px", fontSize: "10px", color: "#64748b", marginBottom: "4px", padding: "0 4px" }}>
              <span>TICKER</span><span>QTY</span><span>ENTRY</span><span>CURRENT</span><span>P&L</span><span>%</span><span>SECTOR</span>
            </div>
            {MOCK_POSITIONS.map((p, i) => (
              <div key={i} style={{
                display: "grid",
                gridTemplateColumns: "60px 40px 70px 70px 80px 60px 70px",
                gap: "4px",
                padding: "6px 4px",
                borderRadius: "4px",
                background: i % 2 === 0 ? "rgba(30,41,59,0.3)" : "transparent",
                alignItems: "center",
              }}>
                <span style={{ fontWeight: 600, color: "#f1f5f9" }}>{p.ticker}</span>
                <span>{p.qty}</span>
                <span>${p.entry}</span>
                <span style={{ color: "#f1f5f9" }}>${p.current}</span>
                <span style={{ color: pnlColor(p.pnl), fontWeight: 600 }}>{formatMoney(p.pnl)}</span>
                <span style={{ color: pnlColor(p.pnl_pct) }}>{formatPct(p.pnl_pct)}</span>
                <span style={{
                  fontSize: "9px",
                  padding: "1px 6px",
                  borderRadius: "3px",
                  background: p.sector === "Energy" ? "rgba(245,158,11,0.15)" : p.sector === "Defense" ? "rgba(59,130,246,0.15)" : "rgba(16,185,129,0.15)",
                  color: p.sector === "Energy" ? "#f59e0b" : p.sector === "Defense" ? "#3b82f6" : "#10b981",
                }}>{p.sector}</span>
              </div>
            ))}
          </div>

          {/* --- Tab Navigation --- */}
          <div style={{ display: "flex", borderBottom: "1px solid #1e293b" }}>
            {[
              { id: "scanner", label: "SCANNER", count: MOCK_SCANNER.length },
              { id: "news", label: "NEWS FEED", count: MOCK_NEWS.length },
              { id: "trades", label: "TRADE LOG", count: MOCK_TRADES.length },
            ].map(tab => (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                flex: 1,
                padding: "8px",
                background: activeTab === tab.id ? "rgba(59,130,246,0.1)" : "transparent",
                border: "none",
                borderBottom: activeTab === tab.id ? "2px solid #3b82f6" : "2px solid transparent",
                color: activeTab === tab.id ? "#3b82f6" : "#64748b",
                fontFamily: "inherit",
                fontSize: "10px",
                fontWeight: 600,
                letterSpacing: "1px",
                cursor: "pointer",
                transition: "all 0.2s",
              }}>
                {tab.label} <span style={{ opacity: 0.6 }}>({tab.count})</span>
              </button>
            ))}
          </div>

          {/* --- Tab Content --- */}
          <div style={{ flex: 1, overflow: "auto", padding: "8px 16px" }}>
            {activeTab === "scanner" && (
              <div>
                {MOCK_SCANNER.map((s, i) => (
                  <div key={i} onClick={() => setSelectedCandidate(s)} style={{
                    display: "grid",
                    gridTemplateColumns: "60px 70px 50px 55px 1fr 50px",
                    gap: "8px",
                    padding: "8px 6px",
                    borderRadius: "4px",
                    background: selectedCandidate?.ticker === s.ticker ? "rgba(59,130,246,0.1)" : i % 2 === 0 ? "rgba(30,41,59,0.2)" : "transparent",
                    cursor: "pointer",
                    alignItems: "center",
                    borderLeft: selectedCandidate?.ticker === s.ticker ? "2px solid #3b82f6" : "2px solid transparent",
                    transition: "all 0.15s",
                  }}>
                    <div>
                      <div style={{ fontWeight: 700, color: "#f1f5f9" }}>{s.ticker}</div>
                      <div style={{ fontSize: "9px", color: "#64748b" }}>{s.sector}</div>
                    </div>
                    <span style={{ color: "#e2e8f0" }}>${s.price}</span>
                    <div>
                      <div style={{ fontSize: "9px", color: "#64748b" }}>STR</div>
                      <div style={{
                        fontWeight: 700,
                        color: s.strength >= 80 ? "#22c55e" : s.strength >= 60 ? "#f59e0b" : "#94a3b8",
                      }}>{s.strength}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "9px", color: "#64748b" }}>VOL</div>
                      <div style={{ fontWeight: 600, color: s.vol_ratio >= 4 ? "#f59e0b" : "#94a3b8" }}>{s.vol_ratio}x</div>
                    </div>
                    <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                      {s.signals.slice(0, 2).map((sig, j) => (
                        <span key={j} style={{
                          fontSize: "9px",
                          padding: "1px 5px",
                          borderRadius: "2px",
                          background: "rgba(148,163,184,0.1)",
                          color: "#94a3b8",
                        }}>{sig}</span>
                      ))}
                    </div>
                    <div style={{
                      textAlign: "center",
                      padding: "2px 8px",
                      borderRadius: "3px",
                      fontSize: "11px",
                      fontWeight: 700,
                      background: s.sentiment > 50 ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)",
                      color: s.sentiment > 50 ? "#22c55e" : "#ef4444",
                    }}>
                      {s.sentiment > 50 ? "▲" : "▼"}{s.sentiment}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {activeTab === "news" && (
              <div>
                {MOCK_NEWS.map((n, i) => (
                  <div key={i} style={{
                    padding: "8px 6px",
                    borderRadius: "4px",
                    background: i % 2 === 0 ? "rgba(30,41,59,0.2)" : "transparent",
                    marginBottom: "2px",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "3px" }}>
                      <span style={{ fontSize: "10px", color: "#64748b" }}>{n.time}</span>
                      <span style={{ fontWeight: 700, color: "#f1f5f9" }}>{n.ticker}</span>
                      <span style={{ color: sentimentColor(n.sentiment), fontWeight: 600, fontSize: "11px" }}>
                        {sentimentIcon(n.sentiment)} {n.sentiment.toUpperCase()}
                      </span>
                      <span style={{ fontSize: "9px", color: "#64748b" }}>SEV:{n.severity}/10</span>
                      <span style={{ fontSize: "9px", color: "#475569", marginLeft: "auto" }}>{n.source}</span>
                    </div>
                    <div style={{ color: "#cbd5e1", fontSize: "11px" }}>{n.headline}</div>
                  </div>
                ))}
              </div>
            )}

            {activeTab === "trades" && (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "50px 50px 90px 65px 65px 70px 55px", gap: "4px", fontSize: "10px", color: "#64748b", marginBottom: "4px", padding: "0 4px" }}>
                  <span>DATE</span><span>TICKER</span><span>ACTION</span><span>ENTRY</span><span>EXIT</span><span>P&L</span><span>%</span>
                </div>
                {MOCK_TRADES.map((t, i) => (
                  <div key={i} style={{
                    display: "grid",
                    gridTemplateColumns: "50px 50px 90px 65px 65px 70px 55px",
                    gap: "4px",
                    padding: "6px 4px",
                    borderRadius: "4px",
                    background: i % 2 === 0 ? "rgba(30,41,59,0.2)" : "transparent",
                    alignItems: "center",
                  }}>
                    <span style={{ color: "#64748b" }}>{t.date}</span>
                    <span style={{ fontWeight: 600, color: "#f1f5f9" }}>{t.ticker}</span>
                    <span style={{ fontSize: "10px" }}>{t.action}</span>
                    <span>${t.entry}</span>
                    <span>{t.exit ? `$${t.exit}` : "—"}</span>
                    <span style={{ color: pnlColor(t.pnl), fontWeight: 600 }}>{formatMoney(t.pnl)}</span>
                    <span style={{ color: pnlColor(t.pnl_pct) }}>{formatPct(t.pnl_pct)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ============ RIGHT: CHAT PANEL ============ */}
        <div style={{
          display: "flex",
          flexDirection: "column",
          background: "#0d1220",
          height: "100%",
          overflow: "hidden",
        }}>
          <div style={{
            padding: "10px 16px",
            borderBottom: "1px solid #1e293b",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}>
            <span style={{ fontSize: "14px" }}>🤖</span>
            <span style={{ fontWeight: 600, fontSize: "11px", letterSpacing: "1px", color: "#f1f5f9" }}>AI TRADING ASSISTANT</span>
            <span style={{
              marginLeft: "auto",
              fontSize: "9px",
              padding: "2px 6px",
              borderRadius: "3px",
              background: "rgba(16,185,129,0.15)",
              color: "#10b981",
              border: "1px solid rgba(16,185,129,0.3)",
            }}>LIVE CONTEXT</span>
          </div>

          {/* Chat Messages */}
          <div style={{
            flex: 1,
            overflow: "auto",
            padding: "12px 14px",
            display: "flex",
            flexDirection: "column",
            gap: "12px",
          }}>
            {chatMessages.map((msg, i) => (
              <div key={i} style={{
                display: "flex",
                flexDirection: "column",
                alignItems: msg.role === "user" ? "flex-end" : "flex-start",
              }}>
                <div style={{
                  maxWidth: "92%",
                  padding: "10px 14px",
                  borderRadius: msg.role === "user" ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
                  background: msg.role === "user"
                    ? "linear-gradient(135deg, #1e40af, #1d4ed8)"
                    : "rgba(30,41,59,0.6)",
                  color: "#e2e8f0",
                  fontSize: "12px",
                  lineHeight: 1.6,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  border: msg.role === "user" ? "none" : "1px solid #1e293b",
                }}>
                  {msg.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div style={{
                padding: "10px 14px",
                borderRadius: "12px 12px 12px 2px",
                background: "rgba(30,41,59,0.6)",
                border: "1px solid #1e293b",
                color: "#64748b",
                fontSize: "12px",
                maxWidth: "92%",
              }}>
                <span style={{ animation: "pulse 1.5s infinite" }}>Thinking...</span>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Suggestion Chips */}
          <div style={{ padding: "6px 14px", display: "flex", gap: "6px", flexWrap: "wrap" }}>
            {[
              "Any good opportunities today?",
              "Should I hold RTX?",
              "Analyze OXY",
              "How many PDT trades left?",
              "Oil price impact analysis",
            ].map((q, i) => (
              <button key={i} onClick={() => { setChatInput(q); setTimeout(() => inputRef.current?.focus(), 50); }} style={{
                padding: "4px 10px",
                borderRadius: "12px",
                background: "rgba(59,130,246,0.08)",
                border: "1px solid rgba(59,130,246,0.2)",
                color: "#60a5fa",
                fontSize: "10px",
                fontFamily: "inherit",
                cursor: "pointer",
                transition: "all 0.15s",
                whiteSpace: "nowrap",
              }}>
                {q}
              </button>
            ))}
          </div>

          {/* Chat Input */}
          <div style={{
            padding: "10px 14px",
            borderTop: "1px solid #1e293b",
            display: "flex",
            gap: "8px",
          }}>
            <input
              ref={inputRef}
              type="text"
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && sendMessage()}
              placeholder="Ask any trading question..."
              style={{
                flex: 1,
                padding: "10px 14px",
                borderRadius: "8px",
                border: "1px solid #1e293b",
                background: "#111827",
                color: "#e2e8f0",
                fontFamily: "inherit",
                fontSize: "12px",
                outline: "none",
                transition: "border-color 0.2s",
              }}
              onFocus={e => e.target.style.borderColor = "#3b82f6"}
              onBlur={e => e.target.style.borderColor = "#1e293b"}
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !chatInput.trim()}
              style={{
                padding: "10px 16px",
                borderRadius: "8px",
                border: "none",
                background: isLoading || !chatInput.trim() ? "#1e293b" : "#2563eb",
                color: isLoading || !chatInput.trim() ? "#475569" : "#fff",
                fontFamily: "inherit",
                fontSize: "12px",
                fontWeight: 600,
                cursor: isLoading || !chatInput.trim() ? "default" : "pointer",
                transition: "all 0.2s",
              }}
            >
              {isLoading ? "..." : "Send"}
            </button>
          </div>
        </div>
      </div>

      <style>{`
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #334155; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        input::placeholder { color: #475569; }
      `}</style>
    </div>
  );
}
