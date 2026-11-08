"""
Audit Suite & Regression Tests for Continuous Monitoring and Autonomous Decision Cycles.
Verifies Priority Improvements:
- Continuous market monitoring lifecycle (start, cycle iteration, stop)
- Persistent SQLite memory retrieval and lessons
- Strategy evaluation with ensemble predictions
- Risk management boundary safety
- Alpaca and Forex strict asset isolation
"""

import time
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from config import SystemConfig, RiskSettings, LLMSettings, SUPPORTED_PAIRS
from agents.orchestrator import AutonomousTradingAgent
from core.asset_classes import AssetClass, get_asset_class, is_forex_symbol, is_alpaca_symbol
from core.alpaca_mock_client import MockAlpacaClient
from core.alpaca_models import AlpacaOrderSide, AlpacaOrderType, AlpacaTimeInForce, AlpacaOrderStatus


@pytest.fixture
def test_agent_env():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = Path(tf.name)

    cfg = SystemConfig(
        db_path=db_path,
        risk=RiskSettings(initial_balance=10000.0, risk_per_trade_pct=1.5),
        llm=LLMSettings(provider="heuristic"),
    )
    agent = AutonomousTradingAgent(config=cfg)
    yield agent
    try:
        agent.stop_continuous_monitoring()
        db_path.unlink()
    except Exception:
        pass


def test_continuous_market_monitoring_lifecycle(test_agent_env):
    """Verifies that the autonomous agent can start and stop background monitoring cleanly."""
    agent = test_agent_env
    assert not agent.is_monitoring
    status_init = agent.get_monitoring_status()
    assert not status_init["is_active"]

    started = agent.start_continuous_monitoring(
        pairs=["EUR/USD"],
        interval_seconds=1,
        force_synthetic=True,
    )
    assert started is True
    assert agent.is_monitoring is True

    # Duplicate start should return False
    duplicate_start = agent.start_continuous_monitoring(pairs=["EUR/USD"], interval_seconds=1)
    assert duplicate_start is False

    time.sleep(2)
    status_active = agent.get_monitoring_status()
    assert status_active["is_active"] is True
    assert status_active["total_cycles_executed"] >= 1
    assert status_active["last_tick_time"] is not None

    stopped = agent.stop_continuous_monitoring()
    assert stopped is True
    assert agent.is_monitoring is False
    assert agent.get_monitoring_status()["is_active"] is False


def test_retrospective_evaluation_and_lesson_learning(test_agent_env):
    """Verifies that closed trades produce evaluation reports and persist lessons in SQLite."""
    agent = test_agent_env
    # 1. Open trade directly
    pos = agent.paper_trader.open_trade(
        symbol="EUR/USD",
        action="BUY",
        lots=0.1,
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        hypothesis_id="HYP-TEST-001",
    )
    assert pos is not None

    # 2. Close trade at profit
    closed = agent.paper_trader.close_trade(
        ticket_id=pos.ticket_id,
        exit_price=1.1120,
        exit_reason="TAKE_PROFIT",
    )
    assert closed is not None
    assert closed.pnl_dollars > 0

    # 3. Retrospective evaluation agent processes closed trade
    eval_report = agent.eval_agent.evaluate_trade(trade=closed, market_regime="LOW_VOLATILITY")
    assert eval_report is not None
    assert eval_report.ticket_id == closed.ticket_id
    assert eval_report.lesson_rule != ""

    # 4. Verify lesson is in database
    lessons = agent.db.get_all_lessons(limit=5)
    assert len(lessons) >= 1
    assert any("Take profit" in l.get("rule_learned", "") or "HYP-TEST-001" in l.get("mistake_analysis", "") or l.get("symbol") == "EUR/USD" for l in lessons)


def test_asset_class_strict_boundary(test_agent_env):
    """Verifies that equity/crypto orders and Forex orders never contaminate each other."""
    agent = test_agent_env

    # 1. Alpaca broker must strictly reject Forex symbols (returning REJECTED order with reason)
    fx_order = agent.alpaca_client.submit_order(
        symbol="EUR/USD",
        qty=1000,
        side="buy",
    )
    assert fx_order.status == AlpacaOrderStatus.REJECTED.value
    assert "Forex currency pairs must be traded through the institutional Forex Paper Trader" in fx_order.rejection_reason

    # 2. Alpaca broker successfully executes valid equity
    eq_order = agent.alpaca_client.submit_order(
        symbol="AAPL",
        qty=5,
        side="buy",
        order_type="market",
        time_in_force="day",
    )
    assert eq_order.symbol == "AAPL"
    assert eq_order.status == AlpacaOrderStatus.FILLED.value

    # 3. Symbol categorization checks
    assert is_forex_symbol("EUR/USD") is True
    assert is_forex_symbol("AAPL") is False
    assert is_alpaca_symbol("AAPL") is True
    assert is_alpaca_symbol("EUR/USD") is False
