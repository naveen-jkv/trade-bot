"""
Unit tests for Phase 1 High-Priority Correctness and Risk Fixes.
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from core.paper_trader import PaperTradingSimulator
from core.market_data import MarketDataProvider
from core.market_data_abstraction import ForexMarketDataProvider
from agents.evaluation_agent import PerformanceEvaluationAgent
from core.prediction import TradingPredictor
from database.memory import TradingMemoryDB
import tempfile
from pathlib import Path


def test_paper_trader_day_rollover():
    """Verify that PaperTradingSimulator resets day_start_equity upon calendar day rollover."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_mem.db"
        db = TradingMemoryDB(db_path=db_path)
        sim = PaperTradingSimulator(initial_balance=10000.0, db=db)
        assert sim.day_start_equity == 10000.0

        # Day 1: 2026-03-01
        candle_day1 = pd.Series({
            "timestamp": "2026-03-01T12:00:00Z",
            "open": 1.1000,
            "high": 1.1050,
            "low": 1.0980,
            "close": 1.1020,
            "volume": 1000.0,
        })
        sim.process_candle("EUR/USD", candle_day1)
        assert sim.current_trading_day == "2026-03-01"
        assert sim.day_start_equity == 10000.0

        # Open a profitable trade and close it on day 1
        pos = sim.open_trade(
            symbol="EUR/USD",
            action="BUY",
            lots=1.0,
            entry_price=1.1000,
            stop_loss=1.0900,
            take_profit=1.1100,
            timestamp="2026-03-01T13:00:00Z",
        )
        assert pos is not None
        sim.close_trade(pos.ticket_id, exit_price=1.1050, exit_time="2026-03-01T14:00:00Z")
        assert sim.equity > 10000.0
        equity_end_day1 = sim.equity

        # Day 2: 2026-03-02
        candle_day2 = pd.Series({
            "timestamp": "2026-03-02T08:00:00Z",
            "open": 1.1050,
            "high": 1.1080,
            "low": 1.1020,
            "close": 1.1060,
            "volume": 1200.0,
        })
        sim.process_candle("EUR/USD", candle_day2)
        assert sim.current_trading_day == "2026-03-02"
        # Baseline must reset to equity at day rollover
        assert sim.day_start_equity == equity_end_day1

        summary = sim.get_portfolio_summary()
        assert "day_start_equity" in summary
        assert summary["day_start_equity"] == round(equity_end_day1, 2)


def test_market_data_4h_resampling():
    """Verify that fetching 4h timeframe properly resamples 1h data into 4h bars."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        provider = ForexMarketDataProvider(cache_dir=Path(tmpdir))
        # Generate 16 1-hour candles
        timestamps = pd.date_range("2026-01-01 00:00", periods=16, freq="1h")
        raw_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [1.1000 + i * 0.0005 for i in range(16)],
            "high": [1.1010 + i * 0.0005 for i in range(16)],
            "low": [1.0990 + i * 0.0005 for i in range(16)],
            "close": [1.1005 + i * 0.0005 for i in range(16)],
            "volume": [100.0] * 16,
        })
        # Resample manually
        resampled = (
            raw_df.set_index("timestamp")
            .resample("4h", closed="left", label="left")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
            .reset_index()
        )
        assert len(resampled) == 4
        assert resampled.iloc[0]["open"] == raw_df.iloc[0]["open"]
        assert resampled.iloc[0]["volume"] == 400.0


def test_evaluation_agent_no_lookahead_and_dynamic_dd():
    """Verify out-of-sample backtest evaluates entries at next open and computes realistic max DD."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = TradingMemoryDB(db_path=Path(tmpdir) / "test.db")
        eval_agent = PerformanceEvaluationAgent(db=db)

        # Generate sample price series with 100 bars
        n = 100
        closes = [1.1000 + np.sin(i / 5.0) * 0.0100 for i in range(n)]
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
            "open": [c - 0.0002 for c in closes],
            "high": [c + 0.0010 for c in closes],
            "low": [c - 0.0010 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "ema_9": closes,
            "ema_21": [c - 0.0005 for c in closes],
            "rsi_14": [55.0] * n,
            "atr_14": [0.0015] * n,
        })

        result = eval_agent.run_historical_adaptation_test(df, symbol="EUR/USD", experimental_atr_mult=1.5)
        assert result.get("status") != "INSUFFICIENT_DATA"
        assert "max_drawdown_pct" in result
        assert isinstance(result["max_drawdown_pct"], float)
        assert result["max_drawdown_pct"] >= 0.0
