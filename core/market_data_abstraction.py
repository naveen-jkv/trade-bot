"""
Market Data Abstraction Layer.
Provides a unified, modular architecture for streaming and historical market feeds.
Decouples Forex currency pair analysis from Alpaca-supported US Equities and Cryptocurrencies.
Enforces clear, immutable data source attribution (Real vs Synthetic vs Mock Alpaca Data).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf

from config import DATA_DIR, SUPPORTED_PAIRS, ForexPairConfig
from core.asset_classes import AssetClass, get_asset_class, ALPACA_SUPPORTED_ASSETS

logger = logging.getLogger("MarketDataAbstraction")


class DataSourceType(str, Enum):
    """Explicit taxonomy of market data provenance."""
    HISTORICAL_REAL = "HISTORICAL_REAL"
    DELAYED_LIVE = "DELAYED_LIVE"
    SYNTHETIC_OFFLINE = "SYNTHETIC_OFFLINE"
    ALPACA_MOCK = "ALPACA_MOCK"


# Standardized UI Badges and Labels
DATA_LABELS: Dict[DataSourceType, str] = {
    DataSourceType.HISTORICAL_REAL: "🟢 [REAL HISTORICAL / DELAYED DATA]",
    DataSourceType.DELAYED_LIVE: "🟢 [DELAYED LIVE DATA]",
    DataSourceType.SYNTHETIC_OFFLINE: "🔵 [SYNTHETIC OFFLINE GARCH DATA]",
    DataSourceType.ALPACA_MOCK: "🟡 [MOCK ALPACA DATA - SIMULATED / EDUCATIONAL]",
}


def get_data_source_badge(source_type: DataSourceType) -> str:
    """Returns the user-facing provenance badge for data transparency."""
    return DATA_LABELS.get(source_type, "⚪ [UNKNOWN DATA SOURCE]")


class BaseMarketDataProvider(ABC):
    """Abstract base class for all market data ingestion pipelines."""

    @abstractmethod
    def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        period: str = "60d",
        force_synthetic: bool = False,
    ) -> Tuple[pd.DataFrame, DataSourceType, str]:
        """
        Fetches historical OHLCV bars.
        Returns:
            Tuple[DataFrame, DataSourceType, HumanReadableBadge]
        """
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Tuple[float, DataSourceType, str]:
        """Returns the current mark price along with provenance tagging."""
        pass


class ForexMarketDataProvider(BaseMarketDataProvider):
    """Specialized market data pipeline for Institutional Forex pairs."""

    def __init__(self, cache_dir: Path = DATA_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_bars(
        self,
        symbol: str = "EUR/USD",
        timeframe: str = "1h",
        period: str = "60d",
        force_synthetic: bool = False,
    ) -> Tuple[pd.DataFrame, DataSourceType, str]:
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS.get("EUR/USD"))

        if force_synthetic:
            df = self._generate_synthetic_forex_data(pair_cfg, num_candles=300, timeframe=timeframe)
            return df, DataSourceType.SYNTHETIC_OFFLINE, DATA_LABELS[DataSourceType.SYNTHETIC_OFFLINE]

        # Attempt to load from yfinance
        try:
            ticker = yf.Ticker(pair_cfg.yahoo_symbol)
            interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
            yf_interval = interval_map.get(timeframe, "1h")

            yf_period = period
            if yf_interval in ["1m", "5m"] and period not in ["1d", "5d", "7d"]:
                yf_period = "5d"
            elif yf_interval in ["15m", "1h"] and period in ["1y", "2y"]:
                yf_period = "60d"

            df = ticker.history(period=yf_period, interval=yf_interval)

            if df is not None and not df.empty and len(df) > 20:
                df = df.reset_index()
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

                # Resample 1h candles into true 4h candles if 4h timeframe requested
                if timeframe == "4h" and len(df) > 0:
                    df = (
                        df.set_index("timestamp")
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

                # Cache locally
                cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
                try:
                    df.to_parquet(cache_file, index=False)
                except Exception as ex:
                    logger.debug(f"Parquet cache save failed, falling back to CSV: {ex}")
                    df.to_csv(cache_file.with_suffix('.csv'), index=False)

                return df, DataSourceType.HISTORICAL_REAL, DATA_LABELS[DataSourceType.HISTORICAL_REAL]

        except Exception as e:
            logger.warning(f"Failed to fetch Forex data via yfinance for {symbol}: {e}")
            cached_df = self._load_from_cache(symbol, timeframe)
            if cached_df is not None and len(cached_df) > 30:
                return cached_df, DataSourceType.HISTORICAL_REAL, DATA_LABELS[DataSourceType.HISTORICAL_REAL]

        df = self._generate_synthetic_forex_data(pair_cfg, num_candles=300, timeframe=timeframe)
        return df, DataSourceType.SYNTHETIC_OFFLINE, DATA_LABELS[DataSourceType.SYNTHETIC_OFFLINE]

    def get_latest_price(self, symbol: str) -> Tuple[float, DataSourceType, str]:
        df, src_type, badge = self.fetch_bars(symbol=symbol, timeframe="1h", period="5d")
        latest = float(df["close"].iloc[-1])
        return latest, src_type, badge

    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        cache_base = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}"
        if cache_base.with_suffix('.parquet').exists():
            try:
                return pd.read_parquet(cache_base.with_suffix('.parquet'))
            except Exception as e:
                logger.debug(f"Failed to load parquet cache {cache_base}: {e}")
        if cache_base.with_suffix('.csv').exists():
            try:
                df = pd.read_csv(cache_base.with_suffix('.csv'))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                logger.debug(f"Failed to load csv cache {cache_base}: {e}")
        return None

    def _generate_synthetic_forex_data(
        self,
        pair_cfg: ForexPairConfig,
        num_candles: int = 300,
        timeframe: str = "1h"
    ) -> pd.DataFrame:
        np.random.seed(42)
        base_rates = {
            "EUR/USD": 1.0850,
            "GBP/USD": 1.2720,
            "USD/JPY": 154.50,
            "AUD/USD": 0.6550,
            "USD/CAD": 1.3780,
        }
        s0 = base_rates.get(pair_cfg.symbol if pair_cfg else "EUR/USD", 1.1000)
        dt = 1.0 / 24.0 if timeframe == "1h" else 1.0 / (24.0 * 12.0)
        theta = 0.15
        mu = s0
        sigma_base = 0.08 * s0
        prices = [s0]
        vols = [sigma_base]

        for _ in range(num_candles):
            current_p = prices[-1]
            curr_vol = vols[-1]
            shock = np.random.normal(0, 1)
            new_vol = max(sigma_base * 0.5, 0.9 * curr_vol + 0.1 * sigma_base + 0.02 * abs(shock) * s0)
            vols.append(new_vol)
            dp = theta * (mu - current_p) * dt + new_vol * np.sqrt(dt) * shock
            new_p = max(0.001, current_p + dp)
            prices.append(new_p)

        prices = prices[1:]
        timestamps = [
            datetime.now(timezone.utc) - timedelta(hours=(num_candles - i))
            for i in range(num_candles)
        ]
        opens, highs, lows, closes, volumes = [], [], [], [], []
        pip = pair_cfg.pip_size if pair_cfg else 0.0001

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


class AlpacaMarketDataProvider(BaseMarketDataProvider):
    """
    Market data provider for Alpaca-supported asset classes (US Equities & Crypto).
    Provides simulated high-fidelity mock data or delayed public quotes for educational study.
    Labels every bar with ALPACA_MOCK or DELAYED_LIVE.
    """

    # Baseline seed prices for educational mock simulation
    STOCK_BASELINES: Dict[str, float] = {
        "AAPL": 224.50,
        "MSFT": 448.20,
        "NVDA": 132.80,
        "TSLA": 254.10,
        "SPY": 562.30,
        "QQQ": 483.75,
        "BTC/USD": 64500.0,
        "ETH/USD": 3480.0,
    }

    def __init__(self, cache_dir: Path = DATA_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_bars(
        self,
        symbol: str = "AAPL",
        timeframe: str = "1h",
        period: str = "30d",
        force_synthetic: bool = False,
    ) -> Tuple[pd.DataFrame, DataSourceType, str]:
        sym = symbol.strip().upper()
        # Map crypto symbols for yfinance fallback if requested
        yf_sym = sym
        if "/" in sym:
            yf_sym = sym.replace("/", "-")

        if not force_synthetic:
            try:
                ticker = yf.Ticker(yf_sym)
                df = ticker.history(period="30d", interval="1h")
                if df is not None and not df.empty and len(df) > 10:
                    df = df.reset_index()
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
                    for c in req_cols:
                        if c not in df.columns:
                            df[c] = 100.0
                    df = df[req_cols].dropna()
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp").reset_index(drop=True)

                    if timeframe == "4h" and len(df) > 0:
                        df = (
                            df.set_index("timestamp")
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

                    return df, DataSourceType.DELAYED_LIVE, DATA_LABELS[DataSourceType.DELAYED_LIVE]
            except Exception as ex:
                logger.debug(f"Failed to fetch live bars for {sym} via yfinance: {ex}")

        # Generate educational mock data
        df = self._generate_mock_equity_bars(sym, num_candles=150)
        return df, DataSourceType.ALPACA_MOCK, DATA_LABELS[DataSourceType.ALPACA_MOCK]

    def get_latest_price(self, symbol: str) -> Tuple[float, DataSourceType, str]:
        sym = symbol.strip().upper()
        df, src_type, badge = self.fetch_bars(symbol=sym, force_synthetic=False)
        latest_px = float(df["close"].iloc[-1])
        return latest_px, src_type, badge

    def _generate_mock_equity_bars(self, symbol: str, num_candles: int = 150) -> pd.DataFrame:
        s0 = self.STOCK_BASELINES.get(symbol, 150.0)
        np.random.seed(abs(hash(symbol)) % (2**31 - 1))

        # Geometric Brownian Motion
        mu = 0.0002
        sigma = 0.015 if "/" not in symbol else 0.035
        returns = np.random.normal(mu, sigma, num_candles)
        price_series = s0 * np.exp(np.cumsum(returns))

        timestamps = [
            datetime.now(timezone.utc) - timedelta(hours=(num_candles - i))
            for i in range(num_candles)
        ]
        opens, highs, lows, closes, volumes = [], [], [], [], []

        for i, p in enumerate(price_series):
            prev_p = price_series[i - 1] if i > 0 else p
            open_p = prev_p + np.random.uniform(-0.002, 0.002) * prev_p
            spread = abs(np.random.normal(0, sigma * 0.5)) * p
            high_p = max(open_p, p) + spread
            low_p = min(open_p, p) - spread
            vol = float(np.random.randint(10000, 500000)) if "/" not in symbol else float(np.random.uniform(5, 120))

            opens.append(round(open_p, 2 if "/" not in symbol else 2))
            highs.append(round(high_p, 2 if "/" not in symbol else 2))
            lows.append(round(low_p, 2 if "/" not in symbol else 2))
            closes.append(round(p, 2 if "/" not in symbol else 2))
            volumes.append(vol)

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })


class UnifiedMarketDataRouter(BaseMarketDataProvider):
    """
    Central Router dispatching requests cleanly across asset classes.
    Preserves strict separation between Forex and Alpaca assets.
    """

    def __init__(self, cache_dir: Path = DATA_DIR):
        self.forex_provider = ForexMarketDataProvider(cache_dir=cache_dir)
        self.alpaca_provider = AlpacaMarketDataProvider(cache_dir=cache_dir)

    def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        period: str = "60d",
        force_synthetic: bool = False,
    ) -> Tuple[pd.DataFrame, DataSourceType, str]:
        ac = get_asset_class(symbol)
        if ac == AssetClass.FOREX:
            return self.forex_provider.fetch_bars(symbol, timeframe, period, force_synthetic)
        else:
            return self.alpaca_provider.fetch_bars(symbol, timeframe, period, force_synthetic)

    def get_latest_price(self, symbol: str) -> Tuple[float, DataSourceType, str]:
        ac = get_asset_class(symbol)
        if ac == AssetClass.FOREX:
            return self.forex_provider.get_latest_price(symbol)
        else:
            return self.alpaca_provider.get_latest_price(symbol)

