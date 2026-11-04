"""
Autonomous AI Trading Bot - Streamlit Dashboard.
Institutional-grade, dark-themed control center following UI/UX Pro Max guidelines.
Simple, focused, high-contrast 4-tab workflow:
1. 📊 Trading Console (Live Market, Chart & AI Hypothesis)
2. 💼 Portfolio & Execution (Forex & Alpaca Paper Holdings & Orders)
3. 📈 Risk & Performance (Equity Curve, Drawdown & Risk Firewall)
4. 🧠 AI Intelligence & Memory (Lessons Learned, Feature Importance & Logs)
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    DEFAULT_CONFIG,
    SUPPORTED_PAIRS,
    SAFETY_LIMITS,
    ForexPairConfig,
    SystemConfig,
    RiskSettings,
    LLMSettings,
)
from agents.orchestrator import AutonomousTradingAgent, CycleExecutionSummary
from database.memory import TradingMemoryDB
from core.indicators import TechnicalAnalyzer
from core.asset_classes import ALPACA_SUPPORTED_ASSETS, AssetClass, get_asset_class
from core.market_data_abstraction import get_data_source_badge, DataSourceType, DATA_LABELS
from core.alpaca_models import AlpacaOrderSide, AlpacaOrderType, AlpacaTimeInForce


# --- Page Configuration ---
st.set_page_config(
    page_title="AI Trading Bot | Control Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- UI/UX Pro Max Clinical & Institutional Dark CSS ---
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Backgrounds adhering to institutional dark mode standard */
    .stApp {
        background-color: #090e17;
        color: #f8fafc;
    }

    /* Standardized Card Container */
    .trade-card {
        background-color: #111c2e;
        border: 1px solid #23354d;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .trade-card:hover {
        border-color: #3b82f6;
    }

    /* High-Contrast Metric Box */
    .metric-container {
        background-color: #111c2e;
        border: 1px solid #23354d;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: left;
        min-height: 82px;
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-delta-pos {
        color: #22c55e;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #ef4444;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Clean Status Badges (WCAG AA Compliant) */
    .badge-buy {
        background: #16a34a;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-sell {
        background: #dc2626;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-hold {
        background: #d97706;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }

    /* Status Pill Chips */
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .chip-active {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .chip-shield {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .chip-paper {
        background-color: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    /* Streamlit Tabs Navigation Polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #23354d;
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        padding: 10px 16px;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.05);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Initialize Persistent Agent State ---
@st.cache_resource
def get_trading_agent() -> AutonomousTradingAgent:
    return AutonomousTradingAgent(config=DEFAULT_CONFIG)


agent = get_trading_agent()

if "selected_pair" not in st.session_state:
    st.session_state.selected_pair = "EUR/USD"
if "selected_timeframe" not in st.session_state:
    st.session_state.selected_timeframe = "1h"


# --- Sidebar Navigation & Controls ---
with st.sidebar:
    st.markdown("### 🤖 AUTONOMOUS AI BOT")
    st.caption("Quantitative Multi-Agent Trading System")

    # Unified Sleek System Status Banner
    st.markdown(
        """
        <div style="background:#111c2e; border:1px solid #23354d; border-radius:8px; padding:8px 10px; margin-bottom:12px;">
            <div style="display:flex; flex-wrap:wrap; gap:4px;">
                <span class="status-chip chip-active">🟢 Swarm Active</span>
                <span class="status-chip chip-shield">🛡️ Firewall Locked</span>
                <span class="status-chip chip-paper">🟡 Paper Mode</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### ⚙️ Market Selection")
    pair_options = list(SUPPORTED_PAIRS.keys())
    selected_pair = st.selectbox(
        "Forex Pair",
        options=pair_options,
        index=pair_options.index(st.session_state.selected_pair),
        label_visibility="collapsed",
    )
    st.session_state.selected_pair = selected_pair

    c_tf, c_src = st.columns(2)
    with c_tf:
        selected_tf = st.selectbox("Timeframe", options=["15m", "1h", "4h", "1d"], index=1)
        st.session_state.selected_timeframe = selected_tf
    with c_src:
        force_synthetic = st.checkbox("Offline Synth", value=False, help="Use deterministic offline simulation instead of Yahoo Finance")

    st.markdown("#### ⚡ Autonomous Execution")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        run_cycle_btn = st.button("⚡ Run Cycle", use_container_width=True, type="primary")
    with col_b2:
        run_5_cycles_btn = st.button("⏩ Run 5", use_container_width=True)

    # Collapsible Settings to avoid visual clutter
    with st.expander("🛡️ Safety Constraints (Immutable)"):
        st.markdown(
            f"""
            - **Max Risk/Trade**: `{SAFETY_LIMITS['ABSOLUTE_MAX_RISK_PER_TRADE_PCT']}%`
            - **Daily Drawdown Limit**: `{SAFETY_LIMITS['ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT']}%`
            - **Max Drawdown Ceiling**: `{SAFETY_LIMITS['ABSOLUTE_MAX_TOTAL_DRAWDOWN_PCT']}%`
            - **Min Risk:Reward**: `1:{SAFETY_LIMITS['ABSOLUTE_MIN_RR_RATIO']}`
            - **Max Open Positions**: `{SAFETY_LIMITS['MAX_OPEN_POSITIONS']}`
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🔑 AI Reasoning & Provider"):
        provider = st.selectbox("LLM Provider", options=["auto", "gemini", "openai", "ollama", "heuristic"], index=0)
        gemini_key = st.text_input("Gemini Key", type="password", value=agent.config.llm.gemini_api_key)
        openai_key = st.text_input("OpenAI Key", type="password", value=agent.config.llm.openai_api_key)
        if st.button("Apply AI Settings"):
            agent.config.llm.provider = provider
            agent.config.llm.gemini_api_key = gemini_key
            agent.config.llm.openai_api_key = openai_key
            agent.research_agent.settings = agent.config.llm
            st.success("AI configuration saved.")

    with st.expander("🦙 Alpaca Broker (Educational)"):
        st.caption("All broker network connectivity is disabled by safety policy. Operates locally in mock paper-trading mode.")
        st.text_input("API Key (env)", type="password", value=agent.config.alpaca.api_key, disabled=True)
        st.text_input("Secret Key (env)", type="password", value=agent.config.alpaca.secret_key, disabled=True)

    if st.button("🗑️ Reset All Paper Data", type="secondary"):
        agent.memory_agent.clear_database_if_requested()
        agent.alpaca_client.reset_mock_portfolio()
        agent.paper_trader = agent.paper_trader.__class__(
            initial_balance=agent.config.risk.initial_balance,
            db=agent.db,
            risk_settings=agent.config.risk,
        )
        st.rerun()


# --- Handle Execution Triggers ---
if run_cycle_btn:
    with st.spinner(f"Analyzing {selected_pair}..."):
        summary = agent.run_cycle(symbol=selected_pair, timeframe=selected_tf, force_synthetic=force_synthetic)
        st.session_state["latest_summary"] = summary
        st.toast(f"Autonomous cycle completed for {selected_pair}!", icon="✅")

if run_5_cycles_btn:
    progress = st.progress(0, text="Executing 5 continuous cycles...")
    for i in range(5):
        summary = agent.run_cycle(symbol=selected_pair, timeframe=selected_tf, force_synthetic=force_synthetic)
        progress.progress((i + 1) / 5, text=f"Cycle {i+1}/5: {summary.proposal.action}")
    st.session_state["latest_summary"] = summary
    st.toast("5 cycles completed successfully!", icon="🚀")


# Ensure baseline summary exists
if "latest_summary" not in st.session_state or st.session_state["latest_summary"] is None:
    with st.spinner(f"Scanning market for {selected_pair}..."):
        st.session_state["latest_summary"] = agent.run_cycle(symbol=selected_pair, timeframe=selected_tf, force_synthetic=force_synthetic)

latest_summary: CycleExecutionSummary = st.session_state["latest_summary"]
portfolio = agent.paper_trader.get_portfolio_summary()
alpaca_acct = agent.alpaca_client.get_account()
report = latest_summary.market_report
hyp = latest_summary.hypothesis
prop = latest_summary.proposal
risk = latest_summary.risk_result


# --- High-Level Top KPI Ribbon (5 Columns) ---
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    delta_class = "metric-delta-pos" if portfolio["net_pnl"] >= 0 else "metric-delta-neg"
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Forex Virtual Equity</div>
            <div class="metric-value">${portfolio['equity']:,.2f}</div>
            <div class="{delta_class}">{portfolio['return_pct']:+.2f}% (${portfolio['net_pnl']:+,.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Forex Cash & Margin</div>
            <div class="metric-value">${portfolio['balance']:,.2f}</div>
            <div style="color:#94a3b8;font-size:0.75rem;">Free Margin: ${portfolio['free_margin']:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Alpaca Paper Cash</div>
            <div class="metric-value">${alpaca_acct.portfolio_value:,.2f}</div>
            <div style="color:#94a3b8;font-size:0.75rem;">Buying Power: ${alpaca_acct.buying_power:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Win Rate / PF</div>
            <div class="metric-value">{portfolio['win_rate_pct']:.1f}%</div>
            <div style="color:#94a3b8;font-size:0.75rem;">Profit Factor: {portfolio['profit_factor']:.2f} | DD: {portfolio['max_drawdown_pct']:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi5:
    badge_cls = "badge-buy" if prop.action == "BUY" else ("badge-sell" if prop.action == "SELL" else "badge-hold")
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Latest AI Signal</div>
            <div style="margin-top:4px;"><span class="{badge_cls}">{prop.action}</span></div>
            <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px;">{report.symbol} | Conf: {prop.confidence:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# SIMPLIFIED 4-TAB WORKFLOW
# ==============================================================================
tab_console, tab_portfolio, tab_risk, tab_memory = st.tabs([
    "📊 Trading Console",
    "💼 Portfolio & Orders",
    "📈 Risk & Performance",
    "🧠 AI Intelligence & Memory",
])


# ==============================================================================
# TAB 1: TRADING CONSOLE (Live Market, Chart & AI Hypothesis)
# ==============================================================================
with tab_console:
    # Action Status Bar
    if risk.is_approved:
        st.success(
            f"🎯 **RISK APPROVED ORDER**: {risk.action} {risk.approved_lot_size} lots on {risk.symbol} @ {risk.entry_price:.5f} | "
            f"SL: {risk.stop_loss:.5f} | TP: {risk.take_profit:.5f} | R:R: 1:{risk.risk_reward_ratio:.2f} | Risk: ${risk.risk_amount_dollars:.2f}"
        )
    else:
        st.info(f"🛡️ **AGENT STATUS**: {prop.action} on {report.symbol} | Reason: {risk.reasons[0] if risk.reasons else prop.rationale}")

    # Fetch enriched bars for charting
    _, chart_df = agent.market_agent.analyze_pair(
        symbol=selected_pair,
        timeframe=selected_tf,
        force_synthetic=force_synthetic,
    )

    # Plotly Candlestick with Technical Overlays
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{selected_pair} ({selected_tf}) Price Action & EMAs", "Volume"],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=chart_df["timestamp"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    # EMAs
    if "ema_9" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["ema_9"], name="EMA 9", line=dict(color="#38bdf8", width=1.2)), row=1, col=1)
    if "ema_21" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["ema_21"], name="EMA 21", line=dict(color="#a855f7", width=1.2)), row=1, col=1)
    if "ema_50" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["ema_50"], name="EMA 50", line=dict(color="#f59e0b", width=1.4)), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in chart_df.columns and "bb_lower" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["bb_upper"], name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.35)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["bb_lower"], name="BB Lower", line=dict(color="rgba(148, 163, 184, 0.35)", dash="dot"), fill='tonexty', fillcolor="rgba(148, 163, 184, 0.05)"), row=1, col=1)

    # Open Position Markers
    for pos in agent.paper_trader.open_positions.values():
        if pos.symbol == selected_pair:
            m_color = "#22c55e" if pos.action == "BUY" else "#ef4444"
            fig.add_trace(
                go.Scatter(
                    x=[chart_df["timestamp"].iloc[-1]],
                    y=[pos.entry_price],
                    mode="markers+text",
                    marker=dict(size=13, color=m_color, symbol="triangle-up" if pos.action == "BUY" else "triangle-down"),
                    text=[f"{pos.action} ({pos.volume_lots}L)"],
                    textposition="top center",
                    name=f"Open {pos.action}",
                ),
                row=1,
                col=1,
            )

    # Volume
    vol_colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(chart_df["close"], chart_df["open"])]
    fig.add_trace(go.Bar(x=chart_df["timestamp"], y=chart_df["volume"], name="Volume", marker_color=vol_colors, opacity=0.7), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090e17",
        plot_bgcolor="#111c2e",
        height=500,
        margin=dict(l=25, r=25, t=30, b=25),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Technical Indicators Bar + Live Thesis Row
    col_tech, col_thesis = st.columns([1.1, 1.3])

    with col_tech:
        st.markdown("#### 🔍 Real-Time Technical Health")
        s = report.technical_state
        t1, t2, t3 = st.columns(3)
        with t1:
            st.metric("RSI (14)", f"{s.rsi_14:.1f}", "Overbought" if s.rsi_14 > 70 else ("Oversold" if s.rsi_14 < 30 else "Neutral"))
            st.metric("ATR (14)", f"{s.atr_14:.5f}")
        with t2:
            st.metric("Trend", s.trend_direction.replace("_", " "))
            st.metric("Regime", s.volatility_regime)
        with t3:
            st.metric("MACD Hist", f"{s.macd_hist:.5f}", "Bullish" if s.macd_hist > 0 else "Bearish")
            st.metric("BB Bandwidth", f"{s.bb_bandwidth:.3f}")

        if s.patterns:
            st.caption(f"Patterns Detected: {', '.join(s.patterns)}")

    with col_thesis:
        st.markdown("#### 🧠 Grounded AI Research Thesis")
        st.markdown(
            f"""
            <div class="trade-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <strong style="color:#f8fafc; font-size:1.0rem;">Active Thesis: {hyp.symbol}</strong>
                    <span class="{badge_cls}">{hyp.direction} ({hyp.confidence:.1%})</span>
                </div>
                <p style="color:#cbd5e1; font-size:0.88rem; line-height:1.5; margin-bottom:10px;">
                    {hyp.hypothesis_text}
                </p>
                <div style="background:rgba(239, 68, 68, 0.1); border-left:3px solid #ef4444; padding:8px 12px; border-radius:4px; font-size:0.82rem; color:#fca5a5;">
                    <strong>Invalidation:</strong> {hyp.invalidation_criteria}
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:12px; font-size:0.8rem; color:#94a3b8;">
                    <span>Entry: <strong style="color:#f8fafc;">{hyp.proposed_entry:.5f}</strong></span>
                    <span>SL: <strong style="color:#ef4444;">{hyp.proposed_stop_loss:.5f}</strong></span>
                    <span>TP: <strong style="color:#22c55e;">{hyp.proposed_take_profit:.5f}</strong></span>
                    <span>ML Up: <strong style="color:#38bdf8;">{prop.ml_ensemble_prob_up:.1%}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 2: PORTFOLIO & ORDERS (Forex & Alpaca Paper Trading)
# ==============================================================================
with tab_portfolio:
    p_sec1, p_sec2 = st.tabs(["💼 Active Holdings & Positions", "📝 Simulated Order Desk & History"])

    with p_sec1:
        # Forex Positions
        st.markdown("#### 💱 Forex Open Positions")
        fx_open = list(agent.paper_trader.open_positions.values())
        if fx_open:
            fx_data = []
            for p in fx_open:
                fx_data.append({
                    "Ticket": p.ticket_id,
                    "Pair": p.symbol,
                    "Action": p.action,
                    "Lots": p.volume_lots,
                    "Entry": f"{p.entry_price:.5f}",
                    "Mark Price": f"{p.current_price:.5f}",
                    "SL": f"{p.stop_loss:.5f}",
                    "TP": f"{p.take_profit:.5f}",
                    "Unrealized PnL ($)": f"${p.unrealized_pnl:+.2f}",
                    "Pips": f"{p.unrealized_pips:+.1f}",
                })
            st.dataframe(pd.DataFrame(fx_data), use_container_width=True)

            col_cfx1, col_cfx2 = st.columns([2, 1])
            with col_cfx1:
                fx_close_ticket = st.selectbox("Select Forex Position to Liquidate", options=[p.ticket_id for p in fx_open])
            with col_cfx2:
                if st.button("Close Forex Position", type="secondary"):
                    t_pos = agent.paper_trader.open_positions.get(fx_close_ticket)
                    if t_pos:
                        agent.paper_trader.close_trade(fx_close_ticket, t_pos.current_price, exit_reason="MANUAL_UI_EXIT")
                        st.toast(f"Closed {fx_close_ticket}!", icon="✅")
                        st.rerun()
        else:
            st.info("No active Forex positions.")

        st.markdown("---")

        # Alpaca Mock Positions
        st.markdown("#### 🦙 Alpaca Simulated Equity & Crypto Holdings")
        alpaca_positions = agent.alpaca_client.list_positions()
        if alpaca_positions:
            alp_data = []
            for ap in alpaca_positions:
                alp_data.append({
                    "Symbol": ap.symbol,
                    "Asset Class": ap.asset_class.upper(),
                    "Shares/Units": ap.qty,
                    "Avg Entry": f"${ap.avg_entry_price:,.2f}",
                    "Current Price": f"${ap.current_price:,.2f}",
                    "Market Value": f"${ap.market_value:,.2f}",
                    "Cost Basis": f"${ap.cost_basis:,.2f}",
                    "Unrealized PnL": f"${ap.unrealized_pl:+,.2f}",
                    "Return": f"{ap.unrealized_plpc * 100.0:+.2f}%",
                })
            st.dataframe(pd.DataFrame(alp_data), use_container_width=True)

            col_cap1, col_cap2 = st.columns([2, 1])
            with col_cap1:
                alp_close_sym = st.selectbox("Select Alpaca Position to Liquidate", options=[p.symbol for p in alpaca_positions])
            with col_cap2:
                if st.button("Liquidate Alpaca Holding", type="secondary"):
                    closed_ord = agent.alpaca_client.close_position(alp_close_sym)
                    if closed_ord:
                        st.toast(f"Liquidated {alp_close_sym} at market!", icon="✅")
                        st.rerun()
        else:
            st.info("No active Alpaca simulated positions. Place simulated orders below.")

    with p_sec2:
        col_desk, col_hist = st.columns([1, 1.2])

        with col_desk:
            st.markdown("#### 📝 Alpaca Simulated Order Desk")
            st.caption("Submit simulated Market, Limit, or Stop orders for US Equities & Crypto.")

            alpaca_syms = list(ALPACA_SUPPORTED_ASSETS.keys())
            order_sym = st.selectbox("Asset", options=alpaca_syms, index=0)
            meta = ALPACA_SUPPORTED_ASSETS[order_sym]

            mark_px, _, prov_badge = agent.alpaca_client.market_data.get_latest_price(order_sym)
            st.markdown(
                f"""
                <div style="background:#111c2e; border:1px solid #23354d; border-radius:6px; padding:6px 10px; margin-bottom:8px;">
                    <div style="font-size:0.75rem; color:#94a3b8;">{meta.name} | <span style="color:#facc15;">{prov_badge}</span></div>
                    <div style="font-size:1.2rem; font-weight:700; color:#f8fafc; font-family:'JetBrains Mono';">${mark_px:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c_od1, c_od2 = st.columns(2)
            with c_od1:
                o_side = st.selectbox("Side", options=["buy", "sell"], index=0)
            with c_od2:
                o_type = st.selectbox("Type", options=["market", "limit", "stop"], index=0)

            c_od3, c_od4 = st.columns(2)
            with c_od3:
                o_qty = st.number_input("Qty", min_value=0.01 if meta.is_fractionable else 1.0, value=10.0 if meta.asset_class == AssetClass.US_EQUITY else 0.25)
            with c_od4:
                o_tif = st.selectbox("TIF", options=["day", "gtc"], index=0)

            limit_val = None
            if o_type == "limit":
                limit_val = st.number_input("Limit Price", min_value=0.01, value=float(mark_px))
            elif o_type == "stop":
                limit_val = st.number_input("Stop Price", min_value=0.01, value=float(mark_px * 0.98))

            if st.button("🚀 Submit Simulated Order", use_container_width=True, type="primary"):
                sub_order = agent.alpaca_client.submit_order(
                    symbol=order_sym,
                    qty=o_qty,
                    side=o_side,
                    order_type=o_type,
                    time_in_force=o_tif,
                    limit_price=limit_val if o_type == "limit" else None,
                    stop_price=limit_val if o_type == "stop" else None,
                )
                if sub_order.status == "filled":
                    st.success(f"✅ Filled: {sub_order.side.upper()} {sub_order.filled_qty} {sub_order.symbol} @ ${sub_order.filled_avg_price:,.2f}")
                    st.rerun()
                elif sub_order.status in ("accepted", "new"):
                    st.info(f"⏳ Placed: {sub_order.side.upper()} {sub_order.qty} {sub_order.symbol} ({sub_order.status.upper()})")
                    st.rerun()
                else:
                    st.error(f"❌ Rejected: {sub_order.rejection_reason}")

        with col_hist:
            st.markdown("#### 📜 Order & Trade History")
            h_choice = st.radio("History View", options=["Forex Closed Trades", "Alpaca Mock Orders"], horizontal=True)

            if h_choice == "Forex Closed Trades":
                closed_trades = agent.paper_trader.closed_trades
                if closed_trades:
                    c_data = []
                    for t in reversed(closed_trades[-25:]):
                        c_data.append({
                            "Ticket": t.ticket_id,
                            "Pair": t.symbol,
                            "Action": t.action,
                            "Lots": t.volume_lots,
                            "Entry": f"{t.entry_price:.5f}",
                            "Exit": f"{t.exit_price:.5f}",
                            "PnL ($)": f"${t.pnl_dollars:+.2f}",
                            "Pips": f"{t.pips:+.1f}",
                            "Reason": t.exit_reason,
                        })
                    st.dataframe(pd.DataFrame(c_data), use_container_width=True)
                else:
                    st.caption("No closed Forex trades yet.")
            else:
                alp_orders = agent.alpaca_client.list_orders(limit=30)
                if alp_orders:
                    ao_data = []
                    for o in alp_orders:
                        ao_data.append({
                            "Symbol": o.symbol,
                            "Side": o.side.upper(),
                            "Type": o.order_type.upper(),
                            "Qty": o.qty,
                            "Filled Qty": o.filled_qty,
                            "Price": f"${o.filled_avg_price:,.2f}" if o.filled_avg_price else "—",
                            "Status": o.status.upper(),
                            "Time": o.created_at[:19].replace("T", " "),
                        })
                    st.dataframe(pd.DataFrame(ao_data), use_container_width=True)
                else:
                    st.caption("No mock Alpaca orders yet.")


# ==============================================================================
# TAB 3: RISK & PERFORMANCE
# ==============================================================================
with tab_risk:
    st.markdown("### 📈 Portfolio Growth & Risk Analytics")

    col_curve, col_audit = st.columns([1.4, 1.0])

    with col_curve:
        eq_history = agent.paper_trader.equity_history
        if len(eq_history) > 1:
            eq_df = pd.DataFrame(eq_history)
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], format="mixed", errors="coerce", utc=True)

            fig_eq = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                row_heights=[0.7, 0.3],
                subplot_titles=["Simulated Account Equity vs Balance ($)", "Drawdown Depth (%)"],
            )
            fig_eq.add_trace(go.Scatter(x=eq_df["timestamp"], y=eq_df["equity"], name="Equity", line=dict(color="#22c55e", width=2)), row=1, col=1)
            fig_eq.add_trace(go.Scatter(x=eq_df["timestamp"], y=eq_df["balance"], name="Balance", line=dict(color="#94a3b8", width=1.5, dash="dash")), row=1, col=1)
            fig_eq.add_trace(
                go.Scatter(
                    x=eq_df["timestamp"],
                    y=eq_df["drawdown_pct"],
                    name="Drawdown %",
                    fill='tozeroy',
                    line=dict(color="#ef4444", width=1),
                    fillcolor="rgba(239, 68, 68, 0.2)",
                ),
                row=2,
                col=1,
            )
            fig_eq.update_layout(
                template="plotly_dark",
                paper_bgcolor="#090e17",
                plot_bgcolor="#111c2e",
                height=450,
                margin=dict(l=25, r=25, t=30, b=25),
            )
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("Execute cycles to populate the real-time equity curve.")

    with col_audit:
        st.markdown("#### 🛡️ Deterministic Risk Firewall")
        st.markdown(
            f"""
            <div class="trade-card">
                <strong>Safety Verification Checklist:</strong>
                <ul style="margin-top:8px; line-height:1.7; font-size:0.85rem; color:#cbd5e1;">
                    <li>Risk Per Trade: <code>{agent.risk_manager.settings.risk_per_trade_pct}%</code> (Cap: {SAFETY_LIMITS['ABSOLUTE_MAX_RISK_PER_TRADE_PCT']}%)</li>
                    <li>Daily Drawdown Stop: <code>{agent.risk_manager.settings.max_daily_drawdown_pct}%</code></li>
                    <li>Total Drawdown Ceiling: <code>{agent.risk_manager.settings.max_total_drawdown_pct}%</code></li>
                    <li>Min Risk-Reward: <code>1:{agent.risk_manager.settings.min_risk_reward_ratio}</code></li>
                    <li>Forex Leverage: <code>1:{agent.risk_manager.settings.leverage}</code></li>
                </ul>
                <div style="background:rgba(59, 130, 246, 0.1); border-left:3px solid #3b82f6; padding:8px 10px; border-radius:4px; font-size:0.8rem; margin-top:10px;">
                    <strong>Latest Audit:</strong> {risk.action} on {risk.symbol} &rarr; 
                    <span style="color:{'#22c55e' if risk.is_approved else '#ef4444'}; font-weight:700;">
                        {'APPROVED' if risk.is_approved else 'BLOCKED'}
                    </span>
                    <br><span style="color:#94a3b8;">{risk.reasons[0] if risk.reasons else 'Safety boundary respected.'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 4: AI INTELLIGENCE & MEMORY
# ==============================================================================
with tab_memory:
    st.markdown("### 🧠 AI Experience Base & Explainability")

    col_m1, col_m2 = st.columns([1.2, 1.0])

    with col_m1:
        st.markdown("#### 📖 Lessons Learned from Retrospectives")
        lessons = agent.memory_agent.db.get_all_lessons(limit=25)
        if lessons:
            l_data = []
            for l in lessons:
                l_data.append({
                    "Pair": l["symbol"],
                    "Regime": l["regime"],
                    "Category": l["mistake_analysis"],
                    "Rule Learned": l["rule_learned"],
                    "Outcome PnL": f"${l['outcome_pnl']:+.2f}",
                })
            st.dataframe(pd.DataFrame(l_data), use_container_width=True)
        else:
            st.caption("No lessons recorded yet. Completed trades automatically generate structured post-mortems.")

        st.markdown("---")
        st.markdown("#### 📜 Live Agent Activity Stream")
        logs = agent.memory_agent.db.get_recent_logs(limit=30)
        if logs:
            st.dataframe(pd.DataFrame(logs)[["timestamp", "agent_name", "level", "message"]], use_container_width=True)
        else:
            st.caption("No logs recorded.")

    with col_m2:
        st.markdown("#### 🤖 Random Forest Feature Importance")
        rf_model = agent.predictor.rf_model
        if rf_model and hasattr(rf_model, "feature_importances_"):
            from core.prediction import MLFeaturePipeline
            feat_df = pd.DataFrame({
                "Feature": MLFeaturePipeline.FEATURE_NAMES,
                "Importance": rf_model.feature_importances_,
            }).sort_values("Importance", ascending=True)

            fig_feat = go.Figure(
                go.Bar(
                    x=feat_df["Importance"],
                    y=feat_df["Feature"],
                    orientation="h",
                    marker_color="#38bdf8",
                )
            )
            fig_feat.update_layout(
                template="plotly_dark",
                paper_bgcolor="#090e17",
                plot_bgcolor="#111c2e",
                height=320,
                margin=dict(l=20, r=20, t=25, b=20),
            )
            st.plotly_chart(fig_feat, use_container_width=True)

        st.markdown("#### 🎯 Empirical Prediction Accuracy")
        accuracy_stats = agent.memory_agent.db.get_prediction_accuracy()
        st.metric(
            "Model Accuracy",
            f"{accuracy_stats['accuracy_pct']:.1f}%",
            f"{accuracy_stats['correct']}/{accuracy_stats['total_evaluated']} trades evaluated",
        )
