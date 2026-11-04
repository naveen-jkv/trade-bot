"""
Forex Market Data Engine.
Provides historical and live market feeds for EUR/USD, GBP/USD, USD/JPY, AUD/USD, and USD/CAD.
Explicitly tags data sources: HISTORICAL_REAL, DELAYED_LIVE, SYNTHETIC_OFFLINE.
Includes offline high-fidelity Forex price simulator with volatility clustering.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from config import SUPPORTED_PAIRS, DATA_DIR, ForexPairConfig


class DataSourceType(str, Enum):
    HISTORICAL_REAL = "HISTORICAL_REAL"
    DELAYED_LIVE = "DELAYED_LIVE"
    SYNTHETIC_OFFLINE = "SYNTHETIC_OFFLINE"
    ALPACA_MOCK = "ALPACA_MOCK"


class MarketDataProvider:
    """Manages acquisition, caching, and generation of multi-pair Forex market data."""

    def __init__(self, cache_dir: Path = DATA_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(
        self,
        symbol: str = "EUR/USD",
        timeframe: str = "1h",
        period: str = "60d",
        force_synthetic: bool = False
    ) -> Tuple[pd.DataFrame, DataSourceType]:
        """
        Fetches Forex OHLCV data. Attempts yfinance first; falls back cleanly to synthetic
        if offline or network unavailable, maintaining explicit data transparency.
        """
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS["EUR/USD"])

        if force_synthetic:
            df = self._generate_synthetic_forex_data(pair_cfg, num_candles=300, timeframe=timeframe)
            return df, DataSourceType.SYNTHETIC_OFFLINE

        # Attempt to load from yfinance
        try:
            ticker = yf.Ticker(pair_cfg.yahoo_symbol)
            # Map timeframe to yfinance intervals
            interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
            yf_interval = interval_map.get(timeframe, "1h")

            # Yfinance limits intraday period
            yf_period = period
            if yf_interval in ["1m", "5m"] and period not in ["1d", "5d", "7d"]:
                yf_period = "5d"
            elif yf_interval in ["15m", "1h"] and period in ["1y", "2y"]:
                yf_period = "60d"

            df = ticker.history(period=yf_period, interval=yf_interval)

            if df is not None and not df.empty and len(df) > 20:
                df = df.reset_index()
                # Rename columns uniformly
                rename_cols = {}
                for col in df.columns:
                    c_lower = str(col).lower()
                    if "date" in c_lower or "time" in c_lower:
                        rename_cols[col] = "timestamp"
                    elif "open" in c_lower:
                        rename_cols[col] = "open"
                    elif "high" in c_lower:
                        rename_cols[col] = "high"
                    elif "low" in c_lower:
                        rename_cols[col] = "low"
                    elif "close" in c_lower:
                        rename_cols[col] = "close"
                    elif "volume" in c_lower:
                        rename_cols[col] = "volume"

                df = df.rename(columns=rename_cols)
                req_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                for rc in req_cols:
                    if rc not in df.columns:
                        if rc == "volume":
                            df["volume"] = 1000.0
                        else:
                            raise ValueError(f"Missing price column {rc}")

                df = df[req_cols].dropna()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)

                # Cache locally
                cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
                try:
                    df.to_parquet(cache_file, index=False)
                except Exception:
                    df.to_csv(cache_file.with_suffix('.csv'), index=False)

                return df, DataSourceType.HISTORICAL_REAL

        except Exception as e:
            # Network issue or rate-limit: check local cache first
            cached_df = self._load_from_cache(symbol, timeframe)
            if cached_df is not None and len(cached_df) > 30:
                return cached_df, DataSourceType.HISTORICAL_REAL

        # Fallback to realistic synthetic Forex data
        df = self._generate_synthetic_forex_data(pair_cfg, num_candles=300, timeframe=timeframe)
        return df, DataSourceType.SYNTHETIC_OFFLINE

    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        cache_base = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}"
        if cache_base.with_suffix('.parquet').exists():
            try:
                return pd.read_parquet(cache_base.with_suffix('.parquet'))
            except Exception:
                pass
        if cache_base.with_suffix('.csv').exists():
            try:
                df = pd.read_csv(cache_base.with_suffix('.csv'))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception:
                pass
        return None

    def _generate_synthetic_forex_data(
        self,
        pair_cfg: ForexPairConfig,
        num_candles: int = 300,
        timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        Generates realistic Forex candles using an Ornstein-Uhlenbeck mean-reverting process
        with stochastic volatility clustering (mimicking GARCH) and realistic Forex wicks.
        """
        np.random.seed(42)  # Deterministic seed for reproducible testing

        # Base exchange rate anchor
        base_rates = {
            "EUR/USD": 1.0850,
            "GBP/USD": 1.2720,
            "USD/JPY": 154.50,
            "AUD/USD": 0.6550,
            "USD/CAD": 1.3780,
        }
        s0 = base_rates.get(pair_cfg.symbol, 1.1000)
        dt = 1.0 / 24.0 if timeframe == "1h" else 1.0 / (24.0 * 12.0)

        # Ornstein-Uhlenbeck parameters
        theta = 0.15          # Mean reversion speed
        mu = s0               # Long term mean
        sigma_base = 0.08 * s0  # Annualized volatility

        prices = [s0]
        vols = [sigma_base]

        for _ in range(num_candles):
            current_p = prices[-1]
            curr_vol = vols[-1]

            # GARCH-style volatility clustering
            shock = np.random.normal(0, 1)
            new_vol = max(sigma_base * 0.5, 0.9 * curr_vol + 0.1 * sigma_base + 0.02 * abs(shock) * s0)
            vols.append(new_vol)

            # Price jump with mean reversion
            dp = theta * (mu - current_p) * dt + new_vol * np.sqrt(dt) * shock
            new_p = max(0.001, current_p + dp)
            prices.append(new_p)

        prices = prices[1:]
        timestamps = [
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=(num_candles - i))
            for i in range(num_candles)
        ]

        opens, highs, lows, closes, volumes = [], [], [], [], []
        pip = pair_cfg.pip_size

        for i, close_p in enumerate(prices):
            prev_close = prices[i - 1] if i > 0 else close_p
            open_p = prev_close + np.random.uniform(-1, 1) * 0.5 * pip
            noise_high = abs(np.random.normal(0, 4.0)) * pip
            noise_low = abs(np.random.normal(0, 4.0)) * pip

            high_p = max(open_p, close_p) + noise_high
            low_p = min(open_p, close_p) - noise_low
            vol = float(np.random.randint(500, 8000))

            opens.append(round(open_p, 5))
            highs.append(round(high_p, 5))
            lows.append(round(low_p, 5))
            closes.append(round(close_p, 5))
            volumes.append(vol)

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })
