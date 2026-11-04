"""
Market Analysis Agent.
Ingests market data, calculates indicators, determines multi-timeframe trends,
volatility regimes, price-action patterns, and support/resistance zones.
Provides structured quantitative intelligence to the AI Research Agent.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from core.indicators import TechnicalAnalyzer, TechnicalState
from core.market_data import MarketDataProvider, DataSourceType
from database.memory import TradingMemoryDB
from config import DEFAULT_CONFIG, SUPPORTED_PAIRS


@dataclass
class MarketAnalysisReport:
    """Quantitative synthesis produced by the Market Analysis Agent."""
    symbol: str
    timeframe: str
    timestamp: str
    data_source: str
    current_price: float
    technical_state: TechnicalState
    trend_summary: str              # e.g. "STRONG_BULLISH with EMA alignment"
    volatility_regime: str          # "HIGH_VOLATILITY", "NORMAL", "COMPRESSED"
    atr_pips: float
    support_level: float
    resistance_level: float
    candle_patterns: List[str]
    stats_summary: Dict[str, float] # mean_return, skew, kurtosis, volatility
    data_bars_count: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["technical_state"] = self.technical_state.to_dict()
        return d


class MarketAnalysisAgent:
    """
    Autonomous agent responsible for deep technical and statistical market parsing.
    Operates strictly without lookahead bias on available historical / real-time bars.
    """

    def __init__(
        self,
        data_provider: Optional[MarketDataProvider] = None,
        db: Optional[TradingMemoryDB] = None,
    ):
        self.data_provider = data_provider or MarketDataProvider()
        self.db = db or TradingMemoryDB(DEFAULT_CONFIG.db_path)

    def analyze_pair(
        self,
        symbol: str = "EUR/USD",
        timeframe: str = "1h",
        force_synthetic: bool = False,
    ) -> Tuple[MarketAnalysisReport, pd.DataFrame]:
        """
        Executes full market analysis cycle for a given Forex currency pair.
        Returns: (MarketAnalysisReport, enriched_dataframe)
        """
        # 1. Ingest Market Data
        df, data_source = self.data_provider.fetch_data(
            symbol=symbol,
            timeframe=timeframe,
            period="60d",
            force_synthetic=force_synthetic,
        )

        # 2. Enrich with Indicators
        enriched_df = TechnicalAnalyzer.analyze_dataframe(df)

        # 3. Extract Latest State
        state = TechnicalAnalyzer.get_latest_state(enriched_df)

        # 4. Compute Statistical Properties of Returns
        returns = enriched_df["close"].pct_change().dropna()
        if len(returns) > 10:
            mean_ret = float(returns.mean())
            skew_val = float(stats.skew(returns))
            kurt_val = float(stats.kurtosis(returns))
            vol_val = float(returns.std() * np.sqrt(24 * 252))
        else:
            mean_ret, skew_val, kurt_val, vol_val = 0.0, 0.0, 0.0, 0.0

        stats_dict = {
            "mean_return_hourly": round(mean_ret, 6),
            "skewness": round(skew_val, 3),
            "kurtosis": round(kurt_val, 3),
            "annualized_volatility": round(vol_val, 4),
        }

        # 5. Calculate ATR in Pips
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS["EUR/USD"])
        atr_pips = round(state.atr_14 / pair_cfg.pip_size, 1)

        # 6. Synthesize Trend Narrative
        trend_summary = f"{state.trend_direction} (RSI: {state.rsi_14:.1f}, MACD Hist: {state.macd_hist:+.5f})"

        ts = str(enriched_df.iloc[-1].get("timestamp", datetime.now(timezone.utc).isoformat()))

        report = MarketAnalysisReport(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts,
            data_source=data_source.value,
            current_price=state.close,
            technical_state=state,
            trend_summary=trend_summary,
            volatility_regime=state.volatility_regime,
            atr_pips=atr_pips,
            support_level=state.support_level,
            resistance_level=state.resistance_level,
            candle_patterns=state.patterns,
            stats_summary=stats_dict,
            data_bars_count=len(enriched_df),
        )

        # Log agent execution
        try:
            self.db.store_log(
                agent_name="MarketAnalysisAgent",
                level="INFO",
                message=f"Analyzed {symbol} ({timeframe}) - Trend: {state.trend_direction}, Regime: {state.volatility_regime}, RSI: {state.rsi_14:.1f}, ATR: {atr_pips:.1f} pips",
                metadata={"symbol": symbol, "data_source": data_source.value, "price": state.close},
            )
        except Exception:
            pass

        return report, enriched_df
