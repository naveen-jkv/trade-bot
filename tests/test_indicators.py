"""
Unit Tests for Technical Indicators and Price Action.
"""

import pytest
import numpy as np
import pandas as pd
from core.indicators import TechnicalAnalyzer, TechnicalState


@pytest.fixture
def sample_ohlcv_df():
    """Generates synthetic price data for indicator verification."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    base = 1.1000
    walk = np.cumsum(np.random.normal(0, 0.001, n))
    closes = base + walk
    opens = closes + np.random.normal(0, 0.0005, n)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.0008, n))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.0008, n))
    volumes = np.random.randint(1000, 5000, n).astype(float)

    return pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_rsi_bounds_and_values(sample_ohlcv_df):
    rsi = TechnicalAnalyzer.calculate_rsi(sample_ohlcv_df["close"], period=14)
    assert len(rsi) == len(sample_ohlcv_df)
    assert rsi.min() >= 0.0
    assert rsi.max() <= 100.0
    assert not rsi.isna().any()


def test_macd_calculation(sample_ohlcv_df):
    macd, signal, hist = TechnicalAnalyzer.calculate_macd(sample_ohlcv_df["close"])
    assert len(macd) == len(sample_ohlcv_df)
    assert len(signal) == len(sample_ohlcv_df)
    assert len(hist) == len(sample_ohlcv_df)
    # MACD hist should equal macd - signal
    np.testing.assert_allclose(hist.values, (macd - signal).values, rtol=1e-5)


def test_bollinger_bands(sample_ohlcv_df):
    upper, mid, lower, bw, pct_b = TechnicalAnalyzer.calculate_bollinger_bands(sample_ohlcv_df["close"], period=20)
    # Upper band should be >= Middle band >= Lower band
    valid = (upper.dropna() >= mid.dropna()) & (mid.dropna() >= lower.dropna())
    assert valid.all()


def test_atr_positive(sample_ohlcv_df):
    atr = TechnicalAnalyzer.calculate_atr(sample_ohlcv_df, period=14)
    assert (atr > 0).all()


def test_full_dataframe_analysis_and_state(sample_ohlcv_df):
    analyzed = TechnicalAnalyzer.analyze_dataframe(sample_ohlcv_df)
    assert "rsi_14" in analyzed.columns
    assert "macd" in analyzed.columns
    assert "bb_upper" in analyzed.columns
    assert "atr_14" in analyzed.columns

    state = TechnicalAnalyzer.get_latest_state(analyzed)
    assert isinstance(state, TechnicalState)
    assert state.close > 0
    assert state.trend_direction in ["STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"]
    assert state.volatility_regime in ["HIGH_VOLATILITY", "NORMAL", "COMPRESSED"]
    assert state.support_level <= state.resistance_level
