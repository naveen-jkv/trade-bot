"""
Forex Data Loader and Synthetic Demonstration Generator.

This module handles:
1. Historical Forex CSV file ingestion with rigorous validation.
2. Realistic synthetic Forex data generation for education and demonstration.
3. Market metadata for major currency pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD).
"""

from __future__ import annotations
import os
import io
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Supported Currency Pairs with realistic baseline parameters
PAIR_METADATA: Dict[str, Dict[str, Any]] = {
    "EUR/USD": {
        "base": "EUR",
        "quote": "USD",
        "base_price": 1.0850,
        "pip_unit": 0.0001,
        "typical_spread_pips": 1.2,
        "digits": 5,
        "daily_volatility": 0.0065,  # ~0.65% daily std
    },
    "GBP/USD": {
        "base": "GBP",
        "quote": "USD",
        "base_price": 1.2720,
        "pip_unit": 0.0001,
        "typical_spread_pips": 1.5,
        "digits": 5,
        "daily_volatility": 0.0080,
    },
    "USD/JPY": {
        "base": "USD",
        "quote": "JPY",
        "base_price": 153.20,
        "pip_unit": 0.01,
        "typical_spread_pips": 1.8,
        "digits": 3,
        "daily_volatility": 0.0075,
    },
    "AUD/USD": {
        "base": "AUD",
        "quote": "USD",
        "base_price": 0.6540,
        "pip_unit": 0.0001,
        "typical_spread_pips": 1.4,
        "digits": 5,
        "daily_volatility": 0.0085,
    },
    "USD/CAD": {
        "base": "USD",
        "quote": "CAD",
        "base_price": 1.3680,
        "pip_unit": 0.0001,
        "typical_spread_pips": 1.6,
        "digits": 5,
        "daily_volatility": 0.0060,
    },
}


def generate_synthetic_forex_data(
    pair: str = "EUR/USD",
    n_bars: int = 1200,
    timeframe: str = "1D",
    seed: int = 42,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate realistic synthetic Forex OHLCV data using Geometric Brownian Motion
    with stochastic volatility clustering and realistic candlestick physics.

    NOTE: This data is explicitly synthetic and intended for educational demonstration only.
    """
    if pair not in PAIR_METADATA:
        raise ValueError(f"Unsupported pair '{pair}'. Supported: {list(PAIR_METADATA.keys())}")

    meta = PAIR_METADATA[pair]
    rng = np.random.default_rng(seed)

    # Date range generation excluding weekends (forex market is closed on weekends)
    if end_date is None:
        end_dt = pd.Timestamp.now().normalize()
    else:
        end_dt = pd.Timestamp(end_date).normalize()

    if timeframe.upper() in ["1D", "DAILY"]:
        # Generate business days
        dates = pd.bdate_range(end=end_dt, periods=n_bars, freq="B")
    else:
        # Generate hourly bars excluding weekends
        full_hourly = pd.date_range(end=end_dt, periods=n_bars * 2, freq="1h")
        filtered = [d for d in full_hourly if d.weekday() < 5]
        dates = pd.DatetimeIndex(filtered[-n_bars:])

    # Stochastic volatility process (Heston-like clustering)
    base_vol = meta["daily_volatility"]
    vol = np.zeros(n_bars)
    vol[0] = base_vol
    alpha = 0.85  # persistence
    for t in range(1, n_bars):
        vol_shock = rng.normal(0, base_vol * 0.25)
        vol[t] = max(0.002, alpha * vol[t - 1] + (1 - alpha) * base_vol + vol_shock)

    # Returns simulation with mean reversion
    drift = 0.00005
    mean_reversion_strength = 0.002
    target_price = meta["base_price"]

    close_prices = np.zeros(n_bars)
    close_prices[0] = target_price

    for t in range(1, n_bars):
        pct_dev = (target_price - close_prices[t - 1]) / target_price
        ret = drift + mean_reversion_strength * pct_dev + vol[t] * rng.normal(0, 1)
        close_prices[t] = close_prices[t - 1] * (1.0 + ret)

    # Generate OHLC respecting candlestick physics
    opens = np.zeros(n_bars)
    highs = np.zeros(n_bars)
    lows = np.zeros(n_bars)
    volumes = np.zeros(n_bars, dtype=int)

    opens[0] = close_prices[0]
    highs[0] = close_prices[0] * 1.002
    lows[0] = close_prices[0] * 0.998
    volumes[0] = 50000

    digits = meta["digits"]

    for t in range(1, n_bars):
        # Open is close to previous close with small gap possibility
        gap = rng.normal(0, vol[t] * 0.15) * close_prices[t - 1]
        opens[t] = close_prices[t - 1] + gap

        # Intraday excursion
        candle_body_max = max(opens[t], close_prices[t])
        candle_body_min = min(opens[t], close_prices[t])

        upper_wick = abs(rng.exponential(scale=vol[t] * 0.45)) * close_prices[t]
        lower_wick = abs(rng.exponential(scale=vol[t] * 0.45)) * close_prices[t]

        highs[t] = candle_body_max + upper_wick
        lows[t] = candle_body_min - lower_wick

        # Volume proportional to bar range and volatility
        bar_range_pct = (highs[t] - lows[t]) / close_prices[t]
        base_volume = 45000 + int(bar_range_pct * 3_000_000)
        volumes[t] = int(base_volume * rng.uniform(0.7, 1.4))

    # Round to realistic forex digits
    df = pd.DataFrame(
        {
            "Timestamp": dates,
            "Open": np.round(opens, digits),
            "High": np.round(highs, digits),
            "Low": np.round(lows, digits),
            "Close": np.round(close_prices, digits),
            "Volume": volumes,
        }
    )

    # Sanity checks
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    df["Pair"] = pair
    df["Is_Synthetic"] = True

    return df


def validate_forex_dataframe(df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
    """
    Validate and clean an uploaded or loaded Forex DataFrame.
    Returns (is_valid, status_message, cleaned_df).
    """
    if df.empty:
        return False, "Dataset is empty.", df

    # Map column names case-insensitively
    col_mapping = {}
    for col in df.columns:
        norm = str(col).strip().lower().replace(" ", "_")
        if norm in ["timestamp", "date", "time", "datetime", "gmt_time", "dted"]:
            col_mapping[col] = "Timestamp"
        elif norm in ["open", "open_price", "o"]:
            col_mapping[col] = "Open"
        elif norm in ["high", "high_price", "h"]:
            col_mapping[col] = "High"
        elif norm in ["low", "low_price", "l"]:
            col_mapping[col] = "Low"
        elif norm in ["close", "close_price", "c"]:
            col_mapping[col] = "Close"
        elif norm in ["volume", "vol", "v", "tick_volume"]:
            col_mapping[col] = "Volume"

    df_renamed = df.rename(columns=col_mapping)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df_renamed.columns]
    if missing:
        return False, f"Missing required price columns: {missing}", df

    # Ensure Timestamp exists or create range
    if "Timestamp" in df_renamed.columns:
        df_renamed["Timestamp"] = pd.to_datetime(df_renamed["Timestamp"], errors="coerce")
        df_renamed = df_renamed.dropna(subset=["Timestamp"])
        df_renamed = df_renamed.sort_values("Timestamp").reset_index(drop=True)
    else:
        # Generate monotonic dummy timestamps
        df_renamed["Timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_renamed), freq="D")

    # If Volume missing, synthesize standard volume
    if "Volume" not in df_renamed.columns:
        df_renamed["Volume"] = 10000

    # Convert numeric types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df_renamed[col] = pd.to_numeric(df_renamed[col], errors="coerce")

    # Drop any row with null price values
    initial_len = len(df_renamed)
    df_renamed = df_renamed.dropna(subset=required).reset_index(drop=True)

    if len(df_renamed) < 60:
        return False, f"Insufficient bars ({len(df_renamed)} bars). Minimum 60 bars required for indicators.", df

    # Enforce candlestick validity: High >= max(Open, Close), Low <= min(Open, Close)
    df_renamed["High"] = df_renamed[["High", "Open", "Close"]].max(axis=1)
    df_renamed["Low"] = df_renamed[["Low", "Open", "Close"]].min(axis=1)

    if "Is_Synthetic" not in df_renamed.columns:
        df_renamed["Is_Synthetic"] = False

    cleaned_count = len(df_renamed)
    msg = f"Successfully validated {cleaned_count} bars (dropped {initial_len - cleaned_count} invalid rows)."
    return True, msg, df_renamed


def load_csv_data(filepath_or_buffer: Any) -> Tuple[bool, str, pd.DataFrame]:
    """
    Load CSV from a file path or file-like buffer (e.g. Streamlit UploadedFile).
    """
    try:
        if isinstance(filepath_or_buffer, str):
            if not os.path.exists(filepath_or_buffer):
                return False, f"File path does not exist: {filepath_or_buffer}", pd.DataFrame()
            df = pd.read_csv(filepath_or_buffer)
        else:
            # File-like buffer
            df = pd.read_csv(filepath_or_buffer)

        return validate_forex_dataframe(df)
    except Exception as e:
        return False, f"Failed to parse CSV: {str(e)}", pd.DataFrame()


def prepare_sample_datasets(data_dir: str = "data") -> Dict[str, str]:
    """
    Ensure sample datasets exist for all 5 pairs in data/ directory.
    Returns dictionary mapping pair -> filepath.
    """
    os.makedirs(data_dir, exist_ok=True)
    file_map = {}

    for pair in PAIR_METADATA.keys():
        safe_name = pair.replace("/", "_").lower()
        filepath = os.path.join(data_dir, f"{safe_name}_sample.csv")
        file_map[pair] = filepath

        if not os.path.exists(filepath):
            df = generate_synthetic_forex_data(pair=pair, n_bars=1000, timeframe="1D")
            df.to_csv(filepath, index=False)

    return file_map


if __name__ == "__main__":
    paths = prepare_sample_datasets()
    print("Pre-generated sample datasets:")
    for pair, p in paths.items():
        print(f"  {pair}: {p} (Exists: {os.path.exists(p)})")
