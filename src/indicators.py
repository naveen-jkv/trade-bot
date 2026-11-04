"""
Technical Indicators and Volatility Analysis Module for Forex Trading.

Calculates:
- EMA 10, EMA 20, EMA 50
- RSI (Relative Strength Index, 14-period standard)
- MACD (12, 26, 9) with Signal line and Histogram
- Bollinger Bands (20, 2) with Upper, Middle, Lower, %B, and Bandwidth
- Volatility Analysis (ATR 14, Rolling Return Volatility, Normalized ATR)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average for a given span."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) using Wilder's smoothing technique.
    RSI oscillates between 0 and 100.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    # Wilder's exponential moving average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    Returns DataFrame with columns: ['MACD', 'MACD_Signal', 'MACD_Hist'].
    """
    fast_ema = calculate_ema(series, span=fast_period)
    slow_ema = calculate_ema(series, span=slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_Signal": signal_line,
            "MACD_Hist": macd_hist,
        },
        index=series.index,
    )


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    Returns DataFrame with:
    - BB_Middle (SMA 20)
    - BB_Upper (Middle + 2*StdDev)
    - BB_Lower (Middle - 2*StdDev)
    - BB_Width ((Upper - Lower) / Middle)
    - BB_PctB ((Close - Lower) / (Upper - Lower))
    """
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    bandwidth = (upper - lower) / (middle + 1e-9)
    pct_b = (series - lower) / (upper - lower + 1e-9)

    return pd.DataFrame(
        {
            "BB_Middle": middle,
            "BB_Upper": upper,
            "BB_Lower": lower,
            "BB_Width": bandwidth,
            "BB_PctB": pct_b,
        },
        index=series.index,
    )


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    True Range = max(High - Low, |High - PrevClose|, |Low - PrevClose|).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def calculate_historical_volatility(
    close: pd.Series,
    window: int = 20,
    annualization_factor: float = np.sqrt(252),
) -> pd.Series:
    """
    Calculate Annualized Historical Volatility based on log returns.
    """
    log_ret = np.log(close / close.shift(1))
    rolling_std = log_ret.rolling(window=window, min_periods=window).std()
    return rolling_std * annualization_factor


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorially add all technical indicators and volatility metrics to the Forex DataFrame.
    Guarantees zero forward-looking bias.
    """
    data = df.copy()

    # 1. EMAs
    data["EMA_10"] = calculate_ema(data["Close"], span=10)
    data["EMA_20"] = calculate_ema(data["Close"], span=20)
    data["EMA_50"] = calculate_ema(data["Close"], span=50)

    # 2. RSI
    data["RSI_14"] = calculate_rsi(data["Close"], period=14)

    # 3. MACD
    macd_df = calculate_macd(data["Close"], fast_period=12, slow_period=26, signal_period=9)
    data["MACD"] = macd_df["MACD"]
    data["MACD_Signal"] = macd_df["MACD_Signal"]
    data["MACD_Hist"] = macd_df["MACD_Hist"]

    # 4. Bollinger Bands
    bb_df = calculate_bollinger_bands(data["Close"], period=20, num_std=2.0)
    data["BB_Middle"] = bb_df["BB_Middle"]
    data["BB_Upper"] = bb_df["BB_Upper"]
    data["BB_Lower"] = bb_df["BB_Lower"]
    data["BB_Width"] = bb_df["BB_Width"]
    data["BB_PctB"] = bb_df["BB_PctB"]

    # 5. Volatility Analysis
    data["ATR_14"] = calculate_atr(data["High"], data["Low"], data["Close"], period=14)
    data["Normalized_ATR"] = data["ATR_14"] / (data["Close"] + 1e-9)
    data["Hist_Volatility"] = calculate_historical_volatility(data["Close"], window=20)

    # 6. Bar Range Dynamics
    data["Bar_Range"] = data["High"] - data["Low"]
    data["Bar_Body"] = (data["Close"] - data["Open"]).abs()

    return data
