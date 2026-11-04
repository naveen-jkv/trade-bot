"""
Unit Tests for Paper Trading Simulator.
"""

import pytest
import pandas as pd
from core.paper_trader import PaperTradingSimulator
from database.memory import TradingMemoryDB
from pathlib import Path
import tempfile


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = Path(tf.name)
    db = TradingMemoryDB(db_path)
    yield db
    try:
        db_path.unlink()
    except Exception:
        pass


def test_paper_trader_open_and_close(temp_db):
    sim = PaperTradingSimulator(initial_balance=10000.0, db=temp_db)
    assert sim.balance == 10000.0
    assert len(sim.open_positions) == 0

    pos = sim.open_trade(
        symbol="EUR/USD",
        action="BUY",
        lots=0.5,
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
    )
    assert pos is not None
    assert len(sim.open_positions) == 1
    assert pos.ticket_id in sim.open_positions

    # Close manually at profit (1.1050)
    closed = sim.close_trade(pos.ticket_id, exit_price=1.1050, exit_reason="MANUAL_PROFIT")
    assert closed is not None
    assert closed.pnl_dollars > 0
    assert len(sim.open_positions) == 0
    assert len(sim.closed_trades) == 1
    assert sim.balance > 10000.0


def test_candle_take_profit_hit(temp_db):
    sim = PaperTradingSimulator(initial_balance=10000.0, db=temp_db)
    pos = sim.open_trade(
        symbol="EUR/USD",
        action="BUY",
        lots=0.5,
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1080,
    )

    # Candle spikes up past take-profit
    candle = pd.Series({
        "timestamp": "2026-01-01T12:00:00Z",
        "open": 1.1010,
        "high": 1.1090,  # > 1.1080 TP
        "low": 1.1005,
        "close": 1.1085,
    })

    closed_trades = sim.process_candle("EUR/USD", candle)
    assert len(closed_trades) == 1
    assert closed_trades[0].exit_reason == "TAKE_PROFIT"
    assert closed_trades[0].pnl_dollars > 0
    assert len(sim.open_positions) == 0


def test_candle_stop_loss_hit(temp_db):
    sim = PaperTradingSimulator(initial_balance=10000.0, db=temp_db)
    pos = sim.open_trade(
        symbol="EUR/USD",
        action="BUY",
        lots=0.5,
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1080,
    )

    # Candle dips below stop-loss
    candle = pd.Series({
        "timestamp": "2026-01-01T13:00:00Z",
        "open": 1.0990,
        "high": 1.1000,
        "low": 1.0940,  # < 1.0950 SL
        "close": 1.0945,
    })

    closed_trades = sim.process_candle("EUR/USD", candle)
    assert len(closed_trades) == 1
    assert closed_trades[0].exit_reason == "STOP_LOSS"
    assert closed_trades[0].pnl_dollars < 0
    assert len(sim.open_positions) == 0
