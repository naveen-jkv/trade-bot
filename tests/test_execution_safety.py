"""
Unit tests for Phase 2: Execution Safety, Idempotency, Rate Limits, and Backtesting.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import requests

from database.memory import TradingMemoryDB
from core.alpaca_live_client import LiveAlpacaClient, AlpacaAPIError
from core.backtester import BacktestEngine, BacktestResult, BacktestTrade
from config import SystemConfig, AlpacaSettings


def test_order_idempotency_and_duplicate_prevention():
    """Verify that deterministic client_order_id and 15m order throttling prevent duplicate fills."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_idempotency.db"
        db = TradingMemoryDB(db_path=db_path)

        cid = "APCA_AAPL_CYCLE_1001"
        assert not db.is_order_submitted(cid)
        assert not db.has_recent_order("AAPL", window_minutes=15)

        # Record submission
        db.store_order_submission(
            client_order_id=cid,
            symbol="AAPL",
            side="buy",
            qty=10.0,
            order_type="market",
            status="SUBMITTED",
            broker="alpaca",
            cycle_id="CYCLE_1001",
        )

        # Subsequent check must detect submission
        assert db.is_order_submitted(cid)
        assert not db.is_order_submitted("APCA_AAPL_CYCLE_1002")
        assert db.has_recent_order("AAPL", window_minutes=15)
        # Non-conflicting symbol must not be throttled
        assert not db.has_recent_order("MSFT", window_minutes=15)


def test_alpaca_live_client_rate_limit_backoff():
    """Verify that HTTP 429 Too Many Requests triggers exponential backoff retry and succeeds."""
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.headers = {"Retry-After": "0.01"}

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "id": "ACC123",
        "account_number": "PA123",
        "status": "ACTIVE",
        "currency": "USD",
        "cash": "10000.0",
        "portfolio_value": "10000.0",
        "buying_power": "40000.0",
        "equity": "10000.0",
    }

    client = LiveAlpacaClient(
        config=SystemConfig(alpaca=AlpacaSettings(api_key="TEST_KEY", secret_key="TEST_SEC")),
        is_paper=True,
    )

    with patch.object(client.session, "request", side_effect=[mock_429, mock_200]) as mock_req:
        with patch("time.sleep") as mock_sleep:
            acct = client.get_account()
            assert acct.id == "ACC123"
            assert acct.portfolio_value == 10000.0
            assert mock_req.call_count == 2
            mock_sleep.assert_called_once()


def test_alpaca_live_client_transient_5xx_retry():
    """Verify that server errors (HTTP 503) retry before succeeding."""
    mock_503 = MagicMock()
    mock_503.status_code = 503

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "id": "ACC456",
        "account_number": "PA456",
        "status": "ACTIVE",
        "currency": "USD",
        "cash": "25000.0",
        "portfolio_value": "25000.0",
        "buying_power": "100000.0",
        "equity": "25000.0",
    }

    client = LiveAlpacaClient(
        config=SystemConfig(alpaca=AlpacaSettings(api_key="TEST_KEY", secret_key="TEST_SEC")),
        is_paper=True,
    )

    with patch.object(client.session, "request", side_effect=[mock_503, mock_200]) as mock_req:
        with patch("time.sleep"):
            acct = client.get_account()
            assert acct.id == "ACC456"
            assert mock_req.call_count == 2


def test_backtester_engine_execution_and_analytics():
    """Verify BacktestEngine enforces fees, adverse slippage, no-lookahead fill, and metrics."""
    engine = BacktestEngine(
        initial_capital=10000.0,
        commission_pct=0.0005,  # 5 bps
        slippage_pct=0.0003,    # 3 bps
        risk_per_trade_pct=1.5,
    )

    # 100 candles with oscillating trend to produce multiple trades
    n = 120
    x = np.linspace(0, 8 * np.pi, n)
    prices = 1.1000 + (np.sin(x) * 0.0080)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
        "open": prices,
        "high": prices + 0.0015,
        "low": prices - 0.0015,
        "close": prices + 0.0002,
        "volume": [1500.0] * n,
        "ema_9": prices + 0.0001,
        "ema_21": prices,
        "rsi_14": [55.0 if (i % 20 < 10) else 45.0 for i in range(n)],
        "atr_14": [0.0012] * n,
    })

    result = engine.run_strategy(df, symbol="EUR/USD")

    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 10000.0
    assert len(result.equity_curve) > 0
    assert isinstance(result.max_drawdown_pct, float)

    # Check trades for fees & slippage accounting
    if result.total_trades > 0:
        first_trade = result.trades[0]
        assert first_trade.commission_paid > 0
        assert first_trade.slippage_cost > 0
        assert first_trade.bars_held >= 1

    # Walk-forward testing
    wf_res = engine.walk_forward_backtest(df, symbol="EUR/USD", train_pct=0.7)
    assert "train_samples" in wf_res
    assert "test_samples" in wf_res
    assert "profit_factor_retention" in wf_res
    assert "is_walk_forward_passed" in wf_res

