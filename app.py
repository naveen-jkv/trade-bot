"""
Autonomous AI Trading Bot - Streamlit Dashboard.
Institutional-grade, dark-themed control center displaying real-time agent workflow,
technical analysis, AI hypotheses, ML predictions, simulated paper portfolio,
persistent memory, and risk management analytics.
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
    page_title="AUTONOMOUS AI TRADING BOT | Multi-Agent Forex System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium Dark Aesthetic CSS ---
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* Card styling */
    .trade-card {
        background: linear-gradient(135deg, rgba(26, 34, 52, 0.7) 0%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        margin-bottom: 16px;
    }

    /* Metric cards */
    .metric-container {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: left;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-delta-pos {
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #ef4444;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Agent status pill */
    .agent-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .pill-active {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-firewall {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .pill-learning {
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .pill-mock {
        background-color: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    /* Direction Badges */
    .badge-buy {
        background: #059669;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-sell {
        background: #dc2626;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-hold {
        background: #d97706;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Initialize State ---
@st.cache_resource
def get_trading_agent() -> AutonomousTradingAgent:
    """Returns persistent instance of the Autonomous Trading Agent."""
    return AutonomousTradingAgent(config=DEFAULT_CONFIG)


agent = get_trading_agent()

if "selected_pair" not in st.session_state:
    st.session_state.selected_pair = "EUR/USD"
if "selected_timeframe" not in st.session_state:
    st.session_state.selected_timeframe = "1h"
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# --- Sidebar Navigation & Controls ---
with st.sidebar:
    st.markdown("### 🤖 AUTONOMOUS AI AGENT")
    st.caption("Quantitative Multi-Agent Forex System")

    st.markdown("---")
    st.markdown("#### ⚡ Multi-Agent Swarm Status")
    st.markdown(
        """
        <div class="agent-pill pill-active">🟢 Market Agent: Active</div><br>
        <div class="agent-pill pill-active">🟢 AI Research Agent: Active</div><br>
        <div class="agent-pill pill-active">🟢 Strategy Agent (RF + LSTM): Active</div><br>
        <div class="agent-pill pill-firewall">🛡️ Risk Firewall: Immutable</div><br>
        <div class="agent-pill pill-active">🟢 Paper Trader: Simulating</div><br>
        <div class="agent-pill pill-learning">🧠 Memory & Eval Agent: Learning</div><br>
        <div class="agent-pill pill-mock">🦙 Alpaca Broker: Mock / Disconnected</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Market Configuration")

    pair_options = list(SUPPORTED_PAIRS.keys())
    selected_pair = st.selectbox(
        "Forex Currency Pair",
        options=pair_options,
        index=pair_options.index(st.session_state.selected_pair),
    )
    st.session_state.selected_pair = selected_pair

    timeframe_options = ["15m", "1h", "4h", "1d"]
    selected_tf = st.selectbox(
        "Candle Timeframe",
        options=timeframe_options,
        index=timeframe_options.index(st.session_state.selected_timeframe),
    )
    st.session_state.selected_timeframe = selected_tf

    data_source_mode = st.radio(
        "Data Feed",
        options=["Real Market (Yahoo Finance)", "Synthetic Offline (Deterministic)"],
        index=0,
    )
    force_synthetic = (data_source_mode == "Synthetic Offline (Deterministic)")

    st.markdown("---")
    st.markdown("#### 🚀 Autonomous Execution")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_cycle_btn = st.button("⚡ Run Cycle", use_container_width=True, type="primary")
    with col_btn2:
        run_5_cycles_btn = st.button("⏩ Run 5 Cycles", use_container_width=True)

    test_adaptation_btn = st.button("🧪 Run Strategy Adaptation Test", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🛡️ Immutable Risk Boundaries")
    st.markdown(
        f"""
        - **Max Risk per Trade**: `{SAFETY_LIMITS['ABSOLUTE_MAX_RISK_PER_TRADE_PCT']}%` equity
        - **Max Daily Drawdown**: `{SAFETY_LIMITS['ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT']}%`
        - **Max Total Drawdown**: `{SAFETY_LIMITS['ABSOLUTE_MAX_TOTAL_DRAWDOWN_PCT']}%`
        - **Min Risk:Reward**: `1:{SAFETY_LIMITS['ABSOLUTE_MIN_RR_RATIO']}`
        - **Max Open Positions**: `{SAFETY_LIMITS['MAX_OPEN_POSITIONS']}`
        <span style="color:#60a5fa;font-size:0.75rem;">🔒 Safety controls are mathematically locked.</span>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔑 LLM & Reasoning Settings"):
        provider = st.selectbox(
            "LLM Provider",
            options=["auto", "gemini", "openai", "ollama", "heuristic"],
            index=0,
        )
        gemini_key = st.text_input("Gemini API Key", type="password", value=agent.config.llm.gemini_api_key)
        openai_key = st.text_input("OpenAI API Key", type="password", value=agent.config.llm.openai_api_key)
        if st.button("Apply API Keys"):
            agent.config.llm.provider = provider
            agent.config.llm.gemini_api_key = gemini_key
            agent.config.llm.openai_api_key = openai_key
            agent.research_agent.settings = agent.config.llm
            st.success("API keys updated successfully.")

    with st.expander("🦙 Alpaca Architecture (Educational)"):
        st.markdown(
            """
            <div style="background:rgba(234,179,8,0.1);border:1px solid rgba(234,179,8,0.3);padding:8px 10px;border-radius:6px;margin-bottom:8px;">
                <span style="color:#facc15;font-size:0.78rem;font-weight:700;">🟡 EDUCATIONAL SIMULATOR ONLY</span><br>
                <span style="color:#94a3b8;font-size:0.72rem;">
                Live broker connectivity is disabled by safety policy. Keys are read only from environment variables and never stored in source code.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.text_input("Alpaca API Key (from env)", type="password", value=agent.config.alpaca.api_key, disabled=True)
        st.text_input("Alpaca Secret Key (from env)", type="password", value=agent.config.alpaca.secret_key, disabled=True)
        st.caption("🔒 Status: Disconnected / Local Simulation Engine")

    if st.button("🗑️ Reset Paper Portfolio", type="secondary"):
        agent.memory_agent.clear_database_if_requested()
        agent.paper_trader = agent.paper_trader.__class__(
            initial_balance=agent.config.risk.initial_balance,
            db=agent.db,
            risk_settings=agent.config.risk,
        )
        st.rerun()


# --- Handle Execution Buttons ---
if run_cycle_btn:
    with st.spinner(f"Agents executing autonomous cycle for {selected_pair}..."):
        summary = agent.run_cycle(
            symbol=selected_pair,
            timeframe=selected_tf,
            force_synthetic=force_synthetic,
        )
        st.session_state["latest_summary"] = summary
        st.toast(f"Autonomous cycle completed for {selected_pair}!", icon="✅")

if run_5_cycles_btn:
    progress_bar = st.progress(0, text="Executing 5 continuous cycles...")
    for step in range(5):
        summary = agent.run_cycle(
            symbol=selected_pair,
            timeframe=selected_tf,
            force_synthetic=force_synthetic,
        )
        progress_bar.progress((step + 1) / 5, text=f"Cycle {step + 1}/5 completed ({summary.proposal.action})")
    st.session_state["latest_summary"] = summary
    st.toast("5 Autonomous cycles executed successfully!", icon="🚀")

if test_adaptation_btn:
    with st.spinner("Testing experimental strategy parameters on out-of-sample data..."):
        report, df = agent.market_agent.analyze_pair(symbol=selected_pair, timeframe=selected_tf, force_synthetic=force_synthetic)
        adapt_res = agent.eval_agent.run_historical_adaptation_test(df, selected_pair)
        st.session_state["adaptation_result"] = adapt_res
        st.toast("Strategy adaptation backtest completed!", icon="📊")


# --- Ensure initial cycle runs once if none recorded ---
if "latest_summary" not in st.session_state or st.session_state["latest_summary"] is None:
    with st.spinner(f"Initializing initial market scan for {selected_pair}..."):
        st.session_state["latest_summary"] = agent.run_cycle(
            symbol=selected_pair,
            timeframe=selected_tf,
            force_synthetic=force_synthetic,
        )

latest_summary: CycleExecutionSummary = st.session_state["latest_summary"]
portfolio = agent.paper_trader.get_portfolio_summary()
report = latest_summary.market_report
hyp = latest_summary.hypothesis
prop = latest_summary.proposal
risk = latest_summary.risk_result


# --- Top Institutional Metric Cards ---
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    delta_class = "metric-delta-pos" if portfolio["net_pnl"] >= 0 else "metric-delta-neg"
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Virtual Equity</div>
            <div class="metric-value">${portfolio['equity']:,.2f}</div>
            <div class="{delta_class}">{portfolio['return_pct']:+.2f}% (${portfolio['net_pnl']:+,.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Account Balance</div>
            <div class="metric-value">${portfolio['balance']:,.2f}</div>
            <div style="color:#94a3b8;font-size:0.8rem;">Free Margin: ${portfolio['free_margin']:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Win Rate / PF</div>
            <div class="metric-value">{portfolio['win_rate_pct']:.1f}%</div>
            <div style="color:#94a3b8;font-size:0.8rem;">Profit Factor: {portfolio['profit_factor']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    dd_class = "metric-delta-neg" if portfolio["current_drawdown_pct"] > 3.0 else "metric-delta-pos"
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value">{portfolio['max_drawdown_pct']:.2f}%</div>
            <div class="{dd_class}">Current DD: {portfolio['current_drawdown_pct']:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    badge_style = "badge-buy" if prop.action == "BUY" else ("badge-sell" if prop.action == "SELL" else "badge-hold")
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">Latest Agent Action</div>
            <div style="margin-top:6px;"><span class="{badge_style}">{prop.action}</span></div>
            <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px;">Confidence: {prop.confidence:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# --- Main Dashboard Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Market & Live Execution",
    "🧠 AI Research & Hypotheses",
    "💼 Forex Paper Portfolio",
    "📈 Equity Curve & Risk Firewall",
    "🏛️ AI Memory & Experience Base",
    "📜 Activity Logs & Audits",
    "🦙 Alpaca Simulator (Educational)",
])


# ==============================================================================
# TAB 1: MARKET & LIVE EXECUTION
# ==============================================================================
with tab1:
    # Action Banner
    if risk.is_approved:
        st.success(
            f"🎯 **RISK APPROVED ORDER**: {risk.action} {risk.approved_lot_size} lots on {risk.symbol} @ {risk.entry_price:.5f} | "
            f"SL: {risk.stop_loss:.5f} | TP: {risk.take_profit:.5f} | R:R: {risk.risk_reward_ratio:.2f} | Risk: ${risk.risk_amount_dollars:.2f}"
        )
    else:
        st.info(f"🛡️ **AGENT STATUS**: {prop.action} on {report.symbol} | Reason: {risk.reasons[0] if risk.reasons else prop.rationale}")

    # Fetch enriched dataframe for charting
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
        subplot_titles=[f"{selected_pair} ({selected_tf}) Price Action & EMA Bands", "Volume & Volatility"],
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
            increasing_line_color="#10b981",
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
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["ema_50"], name="EMA 50", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in chart_df.columns and "bb_lower" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["bb_upper"], name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.4)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["bb_lower"], name="BB Lower", line=dict(color="rgba(148, 163, 184, 0.4)", dash="dot"), fill='tonexty', fillcolor="rgba(148, 163, 184, 0.05)"), row=1, col=1)

    # Markers for Open Positions
    for pos in agent.paper_trader.open_positions.values():
        if pos.symbol == selected_pair:
            marker_color = "#10b981" if pos.action == "BUY" else "#ef4444"
            fig.add_trace(
                go.Scatter(
                    x=[chart_df["timestamp"].iloc[-1]],
                    y=[pos.entry_price],
                    mode="markers+text",
                    marker=dict(size=14, color=marker_color, symbol="triangle-up" if pos.action == "BUY" else "triangle-down"),
                    text=[f"{pos.action} ({pos.volume_lots}L)"],
                    textposition="top center",
                    name=f"Open {pos.action}",
                ),
                row=1,
                col=1,
            )

    # Volume
    colors_vol = ["#10b981" if c >= o else "#ef4444" for c, o in zip(chart_df["close"], chart_df["open"])]
    fig.add_trace(
        go.Bar(x=chart_df["timestamp"], y=chart_df["volume"], name="Volume", marker_color=colors_vol, opacity=0.7),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=580,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Technical Indicators Dashboard Cards
    st.markdown("#### 🔍 Real-Time Technical State")
    s = report.technical_state

    c_t1, c_t2, c_t3, c_t4, c_t5, c_t6 = st.columns(6)
    with c_t1:
        st.metric("RSI (14)", f"{s.rsi_14:.1f}", delta="Oversold" if s.rsi_14 < 30 else ("Overbought" if s.rsi_14 > 70 else "Neutral"))
    with c_t2:
        st.metric("MACD Hist", f"{s.macd_hist:+.5f}", delta="Bullish" if s.macd_hist > 0 else "Bearish")
    with c_t3:
        st.metric("ATR (14)", f"{report.atr_pips:.1f} pips", delta=f"{s.atr_14:.5f}")
    with c_t4:
        st.metric("Trend Direction", s.trend_direction)
    with c_t5:
        st.metric("Volatility Regime", s.volatility_regime)
    with c_t6:
        st.metric("Support / Resistance", f"{s.support_level:.4f} / {s.resistance_level:.4f}")

    if s.patterns:
        st.markdown(f"**Price Action Patterns**: " + " ".join([f"`{p}`" for p in s.patterns]))


# ==============================================================================
# TAB 2: AI RESEARCH & HYPOTHESIS ENGINE
# ==============================================================================
with tab2:
    st.markdown("### 🧠 Autonomous AI Research Agent")
    st.caption("Hypothesis formulation strictly grounded in market facts. No price hallucination.")

    col_hyp, col_ml = st.columns([1.2, 1.0])

    with col_hyp:
        st.markdown(
            f"""
            <div class="trade-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span style="font-size:1.1rem; font-weight:700;">Active Thesis for {hyp.symbol}</span>
                    <span class="{badge_style}">{hyp.direction}</span>
                </div>
                <p style="color:#e2e8f0; font-size:0.95rem; line-height:1.6;">
                    {hyp.hypothesis_text}
                </p>
                <div style="background:rgba(0,0,0,0.3); border-radius:8px; padding:12px; margin-top:12px;">
                    <strong style="color:#f87171;">⚠️ Concrete Invalidation Criteria:</strong><br>
                    <span style="color:#cbd5e1; font-size:0.9rem;">{hyp.invalidation_criteria}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:16px; font-size:0.85rem; color:#94a3b8;">
                    <span>Proposed Entry: <strong style="color:white;">{hyp.proposed_entry:.5f}</strong></span>
                    <span>Stop Loss: <strong style="color:#f87171;">{hyp.proposed_stop_loss:.5f}</strong></span>
                    <span>Take Profit: <strong style="color:#34d399;">{hyp.proposed_take_profit:.5f}</strong></span>
                </div>
                <div style="margin-top:12px; font-size:0.75rem; color:#64748b;">
                    Model Provider: {hyp.model_provider_used} | Timestamp: {hyp.timestamp}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📚 Memory Lessons Applied to Current Thesis")
        if hyp.retrieved_lessons_applied:
            for l in hyp.retrieved_lessons_applied:
                st.info(l)
        else:
            st.caption("No historical error patterns in memory for this market regime. Proceeding with standard technical priors.")

    with col_ml:
        st.markdown("#### 🤖 Machine Learning Ensemble Forecast")
        st.markdown(
            f"""
            - **Combined Ensemble Probability (UP)**: `{prop.ml_ensemble_prob_up:.1%}`
            - **Random Forest Probability (UP)**: `{prop.rf_prob_up:.1%}`
            - **PyTorch LSTM Probability (UP)**: `{prop.lstm_prob_up:.1%}`
            - **Model Signal Alignment**: `{'ALIGNED' if (prop.action == hyp.direction and prop.action != 'HOLD') else 'NEUTRAL / CAUTION'}`
            """
        )

        # Feature Importance Chart from Random Forest
        rf_model = agent.predictor.rf_model
        if rf_model and hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            feature_names = agent.predictor.settings.__class__.__module__ # pipeline
            from core.prediction import MLFeaturePipeline
            feat_df = pd.DataFrame({
                "Feature": MLFeaturePipeline.FEATURE_NAMES,
                "Importance": importances,
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
                title="Random Forest Feature Importance",
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                height=320,
                margin=dict(l=20, r=20, t=35, b=20),
            )
            st.plotly_chart(fig_feat, use_container_width=True)


# ==============================================================================
# TAB 3: PAPER PORTFOLIO & TRADES
# ==============================================================================
with tab3:
    st.markdown("### 💼 Simulated Paper Trading Portfolio")
    st.caption("Realistic simulation including Forex spread, slippage, margin limits, and automatic SL/TP execution.")

    st.markdown("#### 🔓 Currently Open Positions")
    open_positions = list(agent.paper_trader.open_positions.values())

    if open_positions:
        open_data = []
        for p in open_positions:
            open_data.append({
                "Ticket": p.ticket_id,
                "Symbol": p.symbol,
                "Action": p.action,
                "Lots": p.volume_lots,
                "Entry Price": f"{p.entry_price:.5f}",
                "Current Price": f"{p.current_price:.5f}",
                "Stop Loss": f"{p.stop_loss:.5f}",
                "Take Profit": f"{p.take_profit:.5f}",
                "Unrealized PnL ($)": f"${p.unrealized_pnl:+.2f}",
                "Pips": f"{p.unrealized_pips:+.1f}",
                "Entry Time": p.entry_time[:19],
            })
        st.dataframe(pd.DataFrame(open_data), use_container_width=True)

        # Allow user to manually close position
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            ticket_to_close = st.selectbox("Select Position to Liquidate", options=[p.ticket_id for p in open_positions])
        with col_c2:
            if st.button("Close Selected Position Now", type="secondary"):
                target_pos = agent.paper_trader.open_positions.get(ticket_to_close)
                if target_pos:
                    agent.paper_trader.close_trade(ticket_to_close, target_pos.current_price, exit_reason="MANUAL_UI_EXIT")
                    st.toast(f"Closed position {ticket_to_close}!", icon="✅")
                    st.rerun()
    else:
        st.info("No active open positions. The agent is waiting for a high-conviction, risk-approved opportunity.")

    st.markdown("---")
    st.markdown("#### 📜 Completed Trade History")
    closed_trades = agent.paper_trader.closed_trades

    if closed_trades:
        closed_data = []
        for t in reversed(closed_trades):
            closed_data.append({
                "Ticket": t.ticket_id,
                "Symbol": t.symbol,
                "Action": t.action,
                "Lots": t.volume_lots,
                "Entry Price": f"{t.entry_price:.5f}",
                "Exit Price": f"{t.exit_price:.5f}",
                "PnL ($)": f"${t.pnl_dollars:+.2f}",
                "Pips": f"{t.pips:+.1f}",
                "Exit Reason": t.exit_reason,
                "Spread Cost": f"${t.spread_cost_dollars:.2f}",
                "Exit Time": t.exit_time[:19],
            })
        st.dataframe(pd.DataFrame(closed_data), use_container_width=True)
    else:
        st.caption("No closed trades yet. Run cycles to see execution results.")


# ==============================================================================
# TAB 4: EQUITY CURVE & RISK FIREWALL
# ==============================================================================
with tab4:
    st.markdown("### 📈 Portfolio Growth & Drawdown Analytics")

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

        fig_eq.add_trace(go.Scatter(x=eq_df["timestamp"], y=eq_df["equity"], name="Equity", line=dict(color="#10b981", width=2)), row=1, col=1)
        fig_eq.add_trace(go.Scatter(x=eq_df["timestamp"], y=eq_df["balance"], name="Balance", line=dict(color="#94a3b8", width=1.5, dash="dash")), row=1, col=1)

        # Drawdown fill
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
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=480,
            margin=dict(l=30, r=30, t=35, b=25),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # Risk Manager Audit Checklist
    st.markdown("#### 🛡️ Deterministic Risk Firewall Audit")
    c_r1, c_r2 = st.columns(2)

    with c_r1:
        st.markdown(
            f"""
            <div class="trade-card">
                <strong>Active Safety Boundary Status:</strong>
                <ul style="margin-top:8px; line-height:1.7;">
                    <li>Risk Per Trade Cap: <code>{agent.risk_manager.settings.risk_per_trade_pct}%</code> (Max: {SAFETY_LIMITS['ABSOLUTE_MAX_RISK_PER_TRADE_PCT']}%)</li>
                    <li>Daily Drawdown Threshold: <code>{agent.risk_manager.settings.max_daily_drawdown_pct}%</code></li>
                    <li>Total Drawdown Ceiling: <code>{agent.risk_manager.settings.max_total_drawdown_pct}%</code></li>
                    <li>Minimum Risk:Reward Ratio: <code>1:{agent.risk_manager.settings.min_risk_reward_ratio}</code></li>
                    <li>Account Leverage: <code>1:{agent.risk_manager.settings.leverage}</code></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_r2:
        st.markdown(
            f"""
            <div class="trade-card">
                <strong>Latest Order Audit Outcome:</strong><br>
                Status: <strong>{'APPROVED' if risk.is_approved else 'BLOCKED'}</strong><br>
                Action: <code>{risk.action}</code> on <code>{risk.symbol}</code><br>
                Calculated Lot Size: <code>{risk.approved_lot_size} lots</code><br>
                Risk Amount: <code>${risk.risk_amount_dollars:.2f}</code><br>
                Risk-to-Reward: <code>{risk.risk_reward_ratio:.2f}</code><br>
                Risk Score: <code>{risk.risk_score:.2f} / 1.0</code><br>
                Notes: <span style="color:#94a3b8;">{'; '.join(risk.reasons)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 5: AI MEMORY & EXPERIENCE BASE
# ==============================================================================
with tab5:
    st.markdown("### 🏛️ Persistent SQLite Memory & Retrospective Learnings")
    st.caption("The agent remembers every past thesis, trade outcome, and mistake. Experiences inform future decisions.")

    col_mem1, col_mem2 = st.columns([1.2, 1.0])

    with col_mem1:
        st.markdown("#### 📖 Accumulated Lessons Learned")
        lessons = agent.memory_agent.db.get_all_lessons(limit=30)
        if lessons:
            lessons_data = []
            for l in lessons:
                lessons_data.append({
                    "Timestamp": l["timestamp"][:19],
                    "Pair": l["symbol"],
                    "Regime": l["regime"],
                    "Mistake Category": l["mistake_analysis"],
                    "Rule Learned": l["rule_learned"],
                    "Outcome PnL": f"${l['outcome_pnl']:+.2f}",
                })
            st.dataframe(pd.DataFrame(lessons_data), use_container_width=True)
        else:
            st.info("No lessons recorded yet. Completed trades automatically generate structured post-mortems.")

    with col_mem2:
        st.markdown("#### 🎯 Empirical Prediction Accuracy")
        accuracy_stats = agent.memory_agent.db.get_prediction_accuracy()
        st.metric(
            "Overall Prediction Accuracy",
            f"{accuracy_stats['accuracy_pct']:.1f}%",
            f"{accuracy_stats['correct']}/{accuracy_stats['total_evaluated']} trades evaluated",
        )

        st.markdown("#### 🧪 Strategy Adaptation Out-of-Sample Results")
        if "adaptation_result" in st.session_state:
            ad = st.session_state["adaptation_result"]
            st.json(ad)
        else:
            st.caption("Click 'Run Strategy Adaptation Test' in the sidebar to simulate parameter mutations on out-of-sample data.")


# ==============================================================================
# TAB 6: ACTIVITY LOGS & DIAGNOSTICS
# ==============================================================================
with tab6:
    st.markdown("### 📜 Real-Time Agent Activity Stream")
    logs = agent.memory_agent.db.get_recent_logs(limit=100)

    if logs:
        log_df = pd.DataFrame(logs)[["timestamp", "agent_name", "level", "message"]]
        st.dataframe(log_df, use_container_width=True, height=450)
    else:
        st.info("No logs in database.")

    st.markdown("---")
    st.markdown("#### 🔬 System Architecture Diagnostics")
    st.code(
        f"""
Python Version: 3.13.14
Database Path: {agent.config.db_path}
Models Directory: {agent.config.models_dir}
Data Directory: {agent.config.data_dir}
Random Forest Trained: {agent.predictor.rf_model is not None}
PyTorch LSTM Initialized: {agent.predictor.lstm_model is not None}
Active Currency Pairs: {', '.join(SUPPORTED_PAIRS.keys())}
Alpaca Broker Status: Mock Simulation (Connectivity Disabled)
        """,
        language="text",
    )


# ==============================================================================
# TAB 7: ALPACA SIMULATED BROKER (EDUCATIONAL ARCHITECTURE)
# ==============================================================================
with tab7:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(234, 179, 8, 0.12) 0%, rgba(15, 23, 42, 0.95) 100%);
                    border: 1px solid rgba(234, 179, 8, 0.35); border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#facc15; font-size:1.05rem; font-weight:700;">
                    🛡️ EDUCATIONAL SIMULATOR — ALPACA-COMPATIBLE ARCHITECTURE
                </span>
                <span style="background:rgba(234,179,8,0.2); color:#fde047; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700;">
                    BROKER CONNECTIVITY: DISABLED
                </span>
            </div>
            <div style="color:#cbd5e1; font-size:0.82rem; margin-top:6px; line-height:1.4;">
                This module implements a modular Alpaca v2 REST-compatible architecture.
                <strong>No real funds or live broker endpoints are accessed.</strong>
                All orders are executed against local simulated matching logic with realistic slippage, fills, and portfolio tracking.
                Forex analysis is decoupled and strictly handled in the Forex Paper Trader tabs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alpaca_client = agent.alpaca_client
    acct = alpaca_client.get_account()
    clock = alpaca_client.get_clock()

    # Top Account Metrics Cards
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Simulated Portfolio Value</div>
                <div class="metric-value">${acct.portfolio_value:,.2f}</div>
                <div style="color:#38bdf8;font-size:0.8rem;">Equity: ${acct.equity:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Simulated Cash</div>
                <div class="metric-value">${acct.cash:,.2f}</div>
                <div style="color:#94a3b8;font-size:0.8rem;">Currency: {acct.currency}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Buying Power (2x Reg-T)</div>
                <div class="metric-value">${acct.buying_power:,.2f}</div>
                <div style="color:#94a3b8;font-size:0.8rem;">Day Trading BP: ${acct.daytrading_buying_power:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Account Status</div>
                <div style="margin-top:6px;"><span class="agent-pill pill-mock">SIMULATED ACTIVE</span></div>
                <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px;">ID: {acct.account_number}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol5:
        clock_status = "🟢 Market Open" if clock.is_open else "⚪ Market Closed"
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Market Clock</div>
                <div style="font-size:1.05rem;font-weight:700;color:#f8fafc;margin-top:4px;">{clock_status}</div>
                <div style="color:#94a3b8;font-size:0.75rem;">Next Open: {clock.next_open[:10]} {clock.next_open[11:16]}Z</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Order Placement and Asset Explorer Columns
    col_order, col_chart = st.columns([1, 1.4])

    with col_order:
        st.markdown("### 📝 Simulated Order Desk")
        st.caption("Submit simulated Market, Limit, or Stop orders for US Equities & Crypto.")

        alpaca_asset_options = list(ALPACA_SUPPORTED_ASSETS.keys())
        selected_alpaca_asset = st.selectbox(
            "Alpaca Asset / Ticker",
            options=alpaca_asset_options,
            index=0,
            key="alpaca_order_asset",
        )
        meta = ALPACA_SUPPORTED_ASSETS[selected_alpaca_asset]

        # Fetch live/mock mark price
        curr_mark_price, data_src, provenance_badge = alpaca_client.market_data.get_latest_price(selected_alpaca_asset)

        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #1e293b;padding:8px 12px;border-radius:8px;margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.8rem;color:#94a3b8;">{meta.name} ({meta.exchange})</span>
                    <span style="font-size:0.75rem;font-weight:700;color:#facc15;">{provenance_badge}</span>
                </div>
                <div style="font-size:1.3rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#f8fafc;margin-top:4px;">
                    ${curr_mark_price:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_side, c_type = st.columns(2)
        with c_side:
            order_side = st.selectbox("Order Side", options=["buy", "sell"], index=0)
        with c_type:
            order_type = st.selectbox("Order Type", options=["market", "limit", "stop"], index=0)

        c_qty, c_tif = st.columns(2)
        with c_qty:
            order_qty = st.number_input(
                "Quantity (Shares / Units)",
                min_value=0.01 if meta.is_fractionable else 1.0,
                max_value=100000.0,
                value=10.0 if meta.asset_class == AssetClass.US_EQUITY else 0.5,
                step=1.0 if meta.asset_class == AssetClass.US_EQUITY else 0.1,
            )
        with c_tif:
            order_tif = st.selectbox("Time In Force", options=["day", "gtc"], index=0)

        limit_px = None
        stop_px = None
        if order_type == "limit":
            limit_px = st.number_input("Limit Price ($)", min_value=0.01, value=float(curr_mark_price), step=0.5)
        elif order_type == "stop":
            stop_px = st.number_input("Stop Price ($)", min_value=0.01, value=float(curr_mark_price * 0.98), step=0.5)

        est_total = order_qty * (limit_px or curr_mark_price)
        st.markdown(f"**Estimated Order Value:** `${est_total:,.2f}`")

        if st.button("🚀 Submit Simulated Alpaca Order", use_container_width=True, type="primary"):
            sim_order = alpaca_client.submit_order(
                symbol=selected_alpaca_asset,
                qty=order_qty,
                side=order_side,
                order_type=order_type,
                time_in_force=order_tif,
                limit_price=limit_px,
                stop_price=stop_px,
            )
            if sim_order.status == "filled":
                st.success(f"✅ Order Executed: {sim_order.side.upper()} {sim_order.filled_qty} {sim_order.symbol} @ ${sim_order.filled_avg_price:,.2f}")
                st.rerun()
            elif sim_order.status in ("accepted", "new"):
                st.info(f"⏳ Order Placed: {sim_order.side.upper()} {sim_order.qty} {sim_order.symbol} ({sim_order.status.upper()})")
                st.rerun()
            else:
                st.error(f"❌ Order Rejected: {sim_order.rejection_reason}")

        if st.button("🔄 Reset Alpaca Mock Portfolio", type="secondary"):
            alpaca_client.reset_mock_portfolio()
            st.toast("Alpaca simulated portfolio reset to initial cash.", icon="🗑️")
            st.rerun()

    with col_chart:
        st.markdown(f"### 📈 Simulated Price Action: {selected_alpaca_asset}")
        bars_df, src_type, chart_badge = alpaca_client.market_data.fetch_bars(
            symbol=selected_alpaca_asset,
            timeframe="1h",
            period="30d",
        )
        st.caption(f"Data Provenance: {chart_badge}")

        if bars_df is not None and not bars_df.empty:
            fig_alpaca = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.75, 0.25],
                subplot_titles=[f"{selected_alpaca_asset} Simulated OHLC", "Volume"],
            )
            fig_alpaca.add_trace(
                go.Candlestick(
                    x=bars_df["timestamp"],
                    open=bars_df["open"],
                    high=bars_df["high"],
                    low=bars_df["low"],
                    close=bars_df["close"],
                    name="Price",
                    increasing_line_color="#10b981",
                    decreasing_line_color="#ef4444",
                ),
                row=1,
                col=1,
            )
            vol_colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(bars_df["close"], bars_df["open"])]
            fig_alpaca.add_trace(
                go.Bar(
                    x=bars_df["timestamp"],
                    y=bars_df["volume"],
                    name="Volume",
                    marker_color=vol_colors,
                    opacity=0.7,
                ),
                row=2,
                col=1,
            )
            fig_alpaca.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                height=420,
                margin=dict(l=25, r=25, t=35, b=25),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_alpaca, use_container_width=True)

    st.markdown("---")

    # Open Positions & Order History Tabs
    st.markdown("### 💼 Simulated Holdings & Order Audit Log")
    pos_tab, hist_tab = st.tabs(["📦 Open Positions", "📜 Order History"])

    with pos_tab:
        positions = alpaca_client.list_positions()
        if positions:
            pos_records = []
            for p in positions:
                pos_records.append({
                    "Symbol": p.symbol,
                    "Asset Class": p.asset_class.upper(),
                    "Shares / Units": p.qty,
                    "Avg Entry": f"${p.avg_entry_price:,.2f}",
                    "Mark Price": f"${p.current_price:,.2f}",
                    "Market Value": f"${p.market_value:,.2f}",
                    "Cost Basis": f"${p.cost_basis:,.2f}",
                    "Unrealized P&L": f"${p.unrealized_pl:+,.2f}",
                    "Return (%)": f"{p.unrealized_plpc * 100.0:+.2f}%",
                })
            st.dataframe(pd.DataFrame(pos_records), use_container_width=True)

            # Close Position Action
            c_pos_sel, c_pos_btn = st.columns([2, 1])
            with c_pos_sel:
                close_sym = st.selectbox("Select Position to Liquidate", options=[p.symbol for p in positions])
            with c_pos_btn:
                if st.button(f"Liquidate Position ({close_sym})", type="secondary"):
                    closed_ord = alpaca_client.close_position(close_sym)
                    if closed_ord:
                        st.success(f"Position in {close_sym} liquidated at market.")
                        st.rerun()
        else:
            st.info("No open Alpaca simulated positions. Use the order desk above to submit a simulated order.")

    with hist_tab:
        orders = alpaca_client.list_orders(limit=50)
        if orders:
            order_records = []
            for o in orders:
                order_records.append({
                    "Order ID": o.id[:8] + "...",
                    "Client Order ID": o.client_order_id,
                    "Timestamp": o.created_at[:19].replace("T", " "),
                    "Symbol": o.symbol,
                    "Side": o.side.upper(),
                    "Type": o.order_type.upper(),
                    "Requested Qty": o.qty,
                    "Filled Qty": o.filled_qty,
                    "Avg Fill Price": f"${o.filled_avg_price:,.2f}" if o.filled_avg_price else "—",
                    "Status": o.status.upper(),
                    "Rejection Note": o.rejection_reason or "—",
                })
            st.dataframe(pd.DataFrame(order_records), use_container_width=True)
        else:
            st.info("No simulated orders submitted yet.")
