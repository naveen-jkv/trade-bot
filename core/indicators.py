"""
Technical Indicators & Price Action Engine.
Calculates RSI, MACD, EMAs, SMAs, Bollinger Bands, ATR, Volatility,
Candlestick patterns, Support/Resistance levels, and Market Regimes.
Strictly non-anticipative (no lookahead bias).
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class TechnicalState:
    """Snapshot of technical indicators at a specific candle."""
    close: float
    open: float
    high: float
    low: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: Optional[float]
    sma_20: float
    sma_50: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_pct_b: float
    bb_bandwidth: float
    atr_14: float
    volatility_annualized: float
    trend_direction: str          # 'STRONG_BULLISH', 'BULLISH', 'NEUTRAL', 'BEARISH', 'STRONG_BEARISH'
    volatility_regime: str        # 'HIGH_VOLATILITY', 'NORMAL', 'COMPRESSED'
    patterns: List[str]           # e.g. ['BULLISH_ENGULFING', 'PIN_BAR']
    support_level: float
    resistance_level: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TechnicalAnalyzer:
    """Vectorized, robust calculation of quantitative indicators."""

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index with Wilder's smoothing."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Exponential moving average with com = period - 1
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def calculate_macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Moving Average Convergence Divergence."""
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist

    @staticmethod
    def calculate_bollinger_bands(
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands, Bandwidth, and %B."""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        bandwidth = (upper - lower) / sma.replace(0, np.nan)
        pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
        return upper, sma, lower, bandwidth, pct_b

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range using Wilder's smoothing."""
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(com=period - 1, min_periods=period).mean()
        return atr.bfill()

    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame, idx: int = -1) -> List[str]:
        """Identifies key price-action patterns for the candle at index idx."""
        if len(df) < 3:
            return []

        curr = df.iloc[idx]
        prev = df.iloc[idx - 1]
        patterns = []

        body = abs(curr['close'] - curr['open'])
        candle_range = curr['high'] - curr['low']
        is_bullish = curr['close'] > curr['open']
        is_bearish = curr['close'] < curr['open']

        if candle_range > 0:
            upper_wick = curr['high'] - max(curr['close'], curr['open'])
            lower_wick = min(curr['close'], curr['open']) - curr['low']

            # Pin Bar / Hammer (lower wick >= 2x body, tiny upper wick)
            if lower_wick >= 2 * body and upper_wick <= 0.3 * candle_range:
                patterns.append("HAMMER_PIN_BAR_BULLISH")

            # Shooting Star / Inverted Pin Bar (upper wick >= 2x body)
            if upper_wick >= 2 * body and lower_wick <= 0.3 * candle_range:
                patterns.append("SHOOTING_STAR_BEARISH")

            # Doji (indecision candle)
            if body <= 0.1 * candle_range:
                patterns.append("DOJI_INDECISION")

            # Bullish Engulfing
            if (is_bullish and prev['close'] < prev['open'] and
                curr['open'] <= prev['close'] and curr['close'] >= prev['open']):
                patterns.append("BULLISH_ENGULFING")

            # Bearish Engulfing
            if (is_bearish and prev['close'] > prev['open'] and
                curr['open'] >= prev['close'] and curr['close'] <= prev['open']):
                patterns.append("BEARISH_ENGULFING")

            # Inside Bar
            if curr['high'] <= prev['high'] and curr['low'] >= prev['low']:
                patterns.append("INSIDE_BAR_CONSOLIDATION")

        return patterns

    @staticmethod
    def find_support_resistance(df: pd.DataFrame, window: int = 30) -> Tuple[float, float]:
        """Computes local key swing support and resistance levels."""
        if len(df) < window:
            window = len(df)
        recent = df.tail(window)
        support = float(recent['low'].min())
        resistance = float(recent['high'].max())
        return support, resistance

    @classmethod
    def analyze_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends all computed technical indicator columns to the DataFrame.
        Ensures lowercase column names and datetime indexing.
        """
        data = df.copy()
        data.columns = [c.lower() for c in data.columns]

        # EMAs
        data['ema_9'] = data['close'].ewm(span=9, adjust=False).mean()
        data['ema_21'] = data['close'].ewm(span=21, adjust=False).mean()
        data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
        if len(data) >= 200:
            data['ema_200'] = data['close'].ewm(span=200, adjust=False).mean()
        else:
            data['ema_200'] = np.nan

        # SMAs
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()

        # RSI
        data['rsi_14'] = cls.calculate_rsi(data['close'], 14)

        # MACD
        macd, signal, hist = cls.calculate_macd(data['close'])
        data['macd'] = macd
        data['macd_signal'] = signal
        data['macd_hist'] = hist

        # Bollinger Bands
        upper, middle, lower, bandwidth, pct_b = cls.calculate_bollinger_bands(data['close'], 20, 2.0)
        data['bb_upper'] = upper
        data['bb_middle'] = middle
        data['bb_lower'] = lower
        data['bb_bandwidth'] = bandwidth
        data['bb_pct_b'] = pct_b

        # ATR
        data['atr_14'] = cls.calculate_atr(data, 14)

        # Volatility (Annualized standard deviation of returns assuming 24h forex)
        log_ret = np.log(data['close'] / data['close'].shift(1))
        data['volatility_annualized'] = log_ret.rolling(window=20).std() * np.sqrt(252 * 24)

        return data

    @classmethod
    def get_latest_state(cls, df: pd.DataFrame) -> TechnicalState:
        """Extracts the rich TechnicalState from the most recent bar."""
        if 'rsi_14' not in df.columns:
            df = cls.analyze_dataframe(df)

        last = df.iloc[-1]
        patterns = cls.detect_candlestick_patterns(df, -1)
        support, resistance = cls.find_support_resistance(df, 30)

        # Determine Trend Direction
        close = float(last['close'])
        ema_9 = float(last['ema_9'])
        ema_21 = float(last['ema_21'])
        ema_50 = float(last['ema_50'])
        ema_200 = float(last['ema_200']) if not np.isnan(last.get('ema_200', np.nan)) else None

        if ema_200:
            if close > ema_9 > ema_21 > ema_50 > ema_200:
                trend = "STRONG_BULLISH"
            elif close > ema_50 and close > ema_200:
                trend = "BULLISH"
            elif close < ema_9 < ema_21 < ema_50 < ema_200:
                trend = "STRONG_BEARISH"
            elif close < ema_50 and close < ema_200:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
        else:
            if close > ema_9 > ema_21 > ema_50:
                trend = "STRONG_BULLISH"
            elif close > ema_21:
                trend = "BULLISH"
            elif close < ema_9 < ema_21 < ema_50:
                trend = "STRONG_BEARISH"
            elif close < ema_21:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

        # Determine Volatility Regime
        bw = float(last.get('bb_bandwidth', 0.02))
        if bw > 0.035:
            vol_regime = "HIGH_VOLATILITY"
        elif bw < 0.015:
            vol_regime = "COMPRESSED"
        else:
            vol_regime = "NORMAL"

        return TechnicalState(
            close=round(close, 5),
            open=round(float(last['open']), 5),
            high=round(float(last['high']), 5),
            low=round(float(last['low']), 5),
            rsi_14=round(float(last['rsi_14']), 2),
            macd=round(float(last['macd']), 5),
            macd_signal=round(float(last['macd_signal']), 5),
            macd_hist=round(float(last['macd_hist']), 5),
            ema_9=round(ema_9, 5),
            ema_21=round(ema_21, 5),
            ema_50=round(ema_50, 5),
            ema_200=round(ema_200, 5) if ema_200 else None,
            sma_20=round(float(last['sma_20']), 5),
            sma_50=round(float(last['sma_50']), 5),
            bb_upper=round(float(last['bb_upper']), 5),
            bb_middle=round(float(last['bb_middle']), 5),
            bb_lower=round(float(last['bb_lower']), 5),
            bb_pct_b=round(float(last['bb_pct_b']), 3),
            bb_bandwidth=round(bw, 4),
            atr_14=round(float(last['atr_14']), 5),
            volatility_annualized=round(float(last.get('volatility_annualized', 0.1)), 4),
            trend_direction=trend,
            volatility_regime=vol_regime,
            patterns=patterns,
            support_level=round(support, 5),
            resistance_level=round(resistance, 5),
        )
