"""
FastAPI Server & Real-Time Trading Terminal Backend.
Exposes institutional REST endpoints and WebSockets for the Modern Web Terminal.
Connects directly to the AutonomousTradingAgent, RiskManager, PaperTrader,
TradingMemoryDB, and AlpacaMockClient.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import uvicorn

from config import DEFAULT_CONFIG, SUPPORTED_PAIRS, SAFETY_LIMITS
from agents.orchestrator import AutonomousTradingAgent, CycleExecutionSummary
from core.asset_classes import ALPACA_SUPPORTED_ASSETS, get_asset_class, AssetClass
from core.market_data_abstraction import UnifiedMarketDataRouter, DataSourceType, get_data_source_badge


# Shared persistent agent instance
agent: Optional[AutonomousTradingAgent] = None
market_router: Optional[UnifiedMarketDataRouter] = None

# Active WebSocket connections
active_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, market_router
    agent = AutonomousTradingAgent(config=DEFAULT_CONFIG)
    market_router = UnifiedMarketDataRouter(cache_dir=DEFAULT_CONFIG.data_dir)
    print(">> Autonomous Trading Agent & Market Router initialized.")
    yield
    print(">> Shutting down trading server.")


app = FastAPI(
    title="Autonomous AI Trading Terminal API",
    description="Institutional-grade REST and WebSocket backend for Forex & Alpaca simulation.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Vite dev server (http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Models ---
class CycleRequest(BaseModel):
    symbol: str = "EUR/USD"
    timeframe: str = "1h"
    force_synthetic: bool = False


class ForexCloseRequest(BaseModel):
    ticket_id: str


class AlpacaOrderRequest(BaseModel):
    symbol: str
    qty: float
    side: str                          # 'buy' or 'sell'
    order_type: str = "market"         # 'market', 'limit', 'stop'
    time_in_force: str = "day"         # 'day', 'gtc'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


class AlpacaCloseRequest(BaseModel):
    symbol: str


# --- Helper Broadcast ---
async def broadcast_ws(event_type: str, data: Any):
    if not active_connections:
        return
    payload = json.dumps({"type": event_type, "timestamp": datetime.now(timezone.utc).isoformat(), "data": data})
    for ws in list(active_connections):
        try:
            await ws.send_text(payload)
        except Exception:
            if ws in active_connections:
                active_connections.remove(ws)


# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================

@app.get("/api/status")
async def get_system_status():
    """Returns system operational status, active swarm state, and supported instruments."""
    return {
        "status": "ONLINE",
        "swarm": {
            "market_agent": "ACTIVE",
            "research_agent": "ACTIVE",
            "strategy_agent": "ACTIVE",
            "risk_firewall": "IMMUTABLE_LOCKED",
            "paper_trader": "SIMULATING",
            "alpaca_mock_broker": "MOCK_ACTIVE",
        },
        "supported_pairs": list(SUPPORTED_PAIRS.keys()),
        "supported_alpaca_assets": [
            {
                "symbol": k,
                "name": v.name,
                "asset_class": v.asset_class.value,
                "exchange": v.exchange,
            }
            for k, v in ALPACA_SUPPORTED_ASSETS.items()
        ],
        "safety_limits": SAFETY_LIMITS,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/market")
async def get_market_data(
    symbol: str = Query("EUR/USD"),
    timeframe: str = Query("1h"),
    force_synthetic: bool = Query(False),
):
    """
    Fetches OHLCV chart bars with EMA overlays and current technical indicators.
    Formatted for direct ingestion into TradingView Lightweight Charts.
    """
    ac = get_asset_class(symbol)

    if ac == AssetClass.FOREX:
        rep, df = agent.market_agent.analyze_pair(symbol=symbol, timeframe=timeframe, force_synthetic=force_synthetic)
        mark_px = rep.current_price
        tech_state = rep.technical_state.to_dict()
        data_source = rep.data_source
        regime = rep.volatility_regime
    else:
        df, src_type, _ = market_router.fetch_bars(symbol=symbol, timeframe=timeframe, force_synthetic=force_synthetic)
        mark_px, _, _ = market_router.get_latest_price(symbol)
        data_source = src_type.value
        regime = "NORMAL"
        tech_state = {
            "close": mark_px,
            "rsi_14": 50.0,
            "trend_direction": "NEUTRAL",
            "volatility_regime": "NORMAL",
            "atr_14": 0.0,
            "bb_bandwidth": 0.0,
            "patterns": [],
        }

    # Format bars for Lightweight Charts: time (timestamp in seconds or 'YYYY-MM-DD')
    bars = []
    if df is not None and not df.empty:
        df_sorted = df.sort_values("timestamp")
        for _, row in df_sorted.iterrows():
            ts = row["timestamp"]
            # Convert to unix seconds for lightweight-charts
            if hasattr(ts, "timestamp"):
                t_sec = int(ts.timestamp())
            else:
                t_sec = int(pd.to_datetime(ts).timestamp())

            bar = {
                "time": t_sec,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 1000.0)),
            }
            if "ema_9" in row and not pd.isna(row["ema_9"]):
                bar["ema_9"] = float(row["ema_9"])
            if "ema_21" in row and not pd.isna(row["ema_21"]):
                bar["ema_21"] = float(row["ema_21"])
            if "ema_50" in row and not pd.isna(row["ema_50"]):
                bar["ema_50"] = float(row["ema_50"])
            if "bb_upper" in row and not pd.isna(row["bb_upper"]):
                bar["bb_upper"] = float(row["bb_upper"])
            if "bb_lower" in row and not pd.isna(row["bb_lower"]):
                bar["bb_lower"] = float(row["bb_lower"])
            bars.append(bar)

    badge = get_data_source_badge(DataSourceType(data_source) if data_source in DataSourceType._value2member_map_ else DataSourceType.HISTORICAL_REAL)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "mark_price": mark_px,
        "provenance": badge,
        "data_source": data_source,
        "volatility_regime": regime,
        "technical_state": tech_state,
        "bars": bars,
    }


@app.post("/api/cycle")
async def execute_cycle(req: CycleRequest):
    """Executes a single autonomous multi-agent cycle and persists results to SQLite."""
    summary = agent.run_cycle(
        symbol=req.symbol,
        timeframe=req.timeframe,
        force_synthetic=req.force_synthetic,
    )
    summary_dict = summary.to_dict()

    # Broadcast cycle event to connected WebSocket clients
    await broadcast_ws("CYCLE_COMPLETED", {
        "symbol": req.symbol,
        "action": summary.proposal.action,
        "price": summary.market_report.current_price,
        "risk_approved": summary.risk_result.is_approved,
    })

    return summary_dict


@app.get("/api/portfolio")
async def get_portfolio_overview():
    """Returns unified portfolio balances, open positions, and margin status for Forex and Alpaca."""
    fx_summary = agent.paper_trader.get_portfolio_summary()
    alpaca_acct = agent.alpaca_client.get_account().to_dict()

    # Open Forex Positions
    fx_positions = [
        {
            "ticket_id": p.ticket_id,
            "symbol": p.symbol,
            "action": p.action,
            "lots": p.volume_lots,
            "entry_price": p.entry_price,
            "current_price": p.current_price,
            "stop_loss": p.stop_loss,
            "take_profit": p.take_profit,
            "unrealized_pnl": p.unrealized_pnl,
            "unrealized_pips": p.unrealized_pips,
            "entry_time": p.entry_time,
        }
        for p in agent.paper_trader.open_positions.values()
    ]

    # Open Alpaca Positions
    alpaca_positions = [p.to_dict() for p in agent.alpaca_client.list_positions()]

    return {
        "forex": {
            **fx_summary,
            "open_positions": fx_positions,
        },
        "alpaca": {
            **alpaca_acct,
            "positions": alpaca_positions,
        },
    }


@app.post("/api/orders/forex/close")
async def close_forex_position(req: ForexCloseRequest):
    """Liquidates an active Forex position at current mark price."""
    pos = agent.paper_trader.open_positions.get(req.ticket_id)
    if not pos:
        raise HTTPException(status_code=404, detail="Forex position ticket not found.")
    closed = agent.paper_trader.close_trade(req.ticket_id, pos.current_price, exit_reason="MANUAL_WEB_TERMINAL")
    if not closed:
        raise HTTPException(status_code=400, detail="Failed to close position.")
    await broadcast_ws("POSITION_CLOSED", {"ticket_id": req.ticket_id, "pnl": closed.pnl_dollars})
    return {"message": "Position closed successfully.", "trade": closed.to_dict()}


@app.post("/api/orders/alpaca")
async def submit_alpaca_order(req: AlpacaOrderRequest):
    """Submits a simulated order to the Alpaca mock broker."""
    order = agent.alpaca_client.submit_order(
        symbol=req.symbol,
        qty=req.qty,
        side=req.side,
        order_type=req.order_type,
        time_in_force=req.time_in_force,
        limit_price=req.limit_price,
        stop_price=req.stop_price,
    )
    await broadcast_ws("ALPACA_ORDER", order.to_dict())
    return order.to_dict()


@app.post("/api/orders/alpaca/close")
async def close_alpaca_position(req: AlpacaCloseRequest):
    """Liquidates an active Alpaca holding at market."""
    order = agent.alpaca_client.close_position(req.symbol)
    if not order:
        raise HTTPException(status_code=404, detail=f"No open Alpaca position in {req.symbol}.")
    await broadcast_ws("ALPACA_POSITION_CLOSED", {"symbol": req.symbol})
    return {"message": f"Position in {req.symbol} liquidated.", "order": order.to_dict()}


@app.get("/api/history")
async def get_trade_history(limit: int = 50):
    """Returns closed trade history for both Forex and Alpaca mock orders."""
    fx_closed = [t.to_dict() for t in reversed(agent.paper_trader.closed_trades[-limit:])]
    alpaca_orders = [o.to_dict() for o in agent.alpaca_client.list_orders(limit=limit)]
    return {
        "forex_closed_trades": fx_closed,
        "alpaca_orders": alpaca_orders,
    }


@app.get("/api/memory")
async def get_memory_and_diagnostics():
    """Returns persistent memory lessons, ML model accuracy, and recent logs."""
    lessons = agent.memory_agent.db.get_all_lessons(limit=30)
    accuracy = agent.memory_agent.db.get_prediction_accuracy()
    logs = agent.memory_agent.db.get_recent_logs(limit=40)

    # Random forest feature importances
    rf_feat = []
    if agent.predictor.rf_model and hasattr(agent.predictor.rf_model, "feature_importances_"):
        from core.prediction import MLFeaturePipeline
        for name, imp in zip(MLFeaturePipeline.FEATURE_NAMES, agent.predictor.rf_model.feature_importances_):
            rf_feat.append({"feature": name, "importance": round(float(imp), 4)})
        rf_feat.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "lessons": lessons,
        "accuracy": accuracy,
        "recent_logs": logs,
        "feature_importances": rf_feat,
    }


@app.post("/api/reset")
async def reset_paper_portfolios():
    """Resets virtual accounts and clears test databases."""
    agent.memory_agent.clear_database_if_requested()
    agent.alpaca_client.reset_mock_portfolio()
    agent.paper_trader = agent.paper_trader.__class__(
        initial_balance=agent.config.risk.initial_balance,
        db=agent.db,
        risk_settings=agent.config.risk,
    )
    await broadcast_ws("PORTFOLIO_RESET", {})
    return {"message": "Paper portfolios reset successfully."}


# ==============================================================================
# WEBSOCKET STREAMING
# ==============================================================================

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Live streaming WebSocket for tick updates and asynchronous notifications."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        # Send initial handshake
        await websocket.send_text(json.dumps({
            "type": "CONNECTION_ESTABLISHED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Connected to Autonomous Trading Terminal stream.",
        }))
        while True:
            # Keep connection alive; accept pings
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)


# ==============================================================================
# STATIC FRONTEND SERVING
# ==============================================================================

# Mount frontend/dist if built, otherwise provide informative landing
dist_path = Path(__file__).resolve().parent / "frontend" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=dist_path, html=True), name="static_frontend")


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
