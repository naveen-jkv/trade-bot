"""
Trading System Configuration & Global Settings.
Encapsulates market parameters, risk controls, ML parameters, and agent options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "database" / "trading_agent.db"

# Ensure directories exist
for p in [DATA_DIR, MODELS_DIR, LOGS_DIR, DB_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ForexPairConfig:
    symbol: str             # e.g. "EUR/USD"
    yahoo_symbol: str       # e.g. "EURUSD=X"
    pip_size: float         # 0.0001 or 0.01 for JPY
    typical_spread_pips: float
    lot_size: int = 100000  # Standard Forex lot is 100,000 units
    min_lot: float = 0.01   # Micro lot
    max_lot: float = 10.0


# Supported Forex Currency Pairs
SUPPORTED_PAIRS: Dict[str, ForexPairConfig] = {
    "EUR/USD": ForexPairConfig(
        symbol="EUR/USD",
        yahoo_symbol="EURUSD=X",
        pip_size=0.0001,
        typical_spread_pips=1.2,
    ),
    "GBP/USD": ForexPairConfig(
        symbol="GBP/USD",
        yahoo_symbol="GBPUSD=X",
        pip_size=0.0001,
        typical_spread_pips=1.8,
    ),
    "USD/JPY": ForexPairConfig(
        symbol="USD/JPY",
        yahoo_symbol="JPY=X",
        pip_size=0.01,
        typical_spread_pips=1.5,
    ),
    "AUD/USD": ForexPairConfig(
        symbol="AUD/USD",
        yahoo_symbol="AUDUSD=X",
        pip_size=0.0001,
        typical_spread_pips=1.6,
    ),
    "USD/CAD": ForexPairConfig(
        symbol="USD/CAD",
        yahoo_symbol="CAD=X",
        pip_size=0.0001,
        typical_spread_pips=1.8,
    ),
}

# Immutable Safety Rules (Hard limits the AI cannot override)
SAFETY_LIMITS = {
    "ABSOLUTE_MAX_RISK_PER_TRADE_PCT": 2.5,   # % of equity
    "ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT": 5.0,   # %
    "ABSOLUTE_MAX_TOTAL_DRAWDOWN_PCT": 12.0,  # %
    "ABSOLUTE_MIN_RR_RATIO": 1.2,             # Minimum risk-reward
    "MAX_OPEN_POSITIONS": 5,                  # Portfolio cap
}


@dataclass
class RiskSettings:
    """Configurable risk parameters within safety boundaries."""
    initial_balance: float = 10000.0
    risk_per_trade_pct: float = 1.5           # Default 1.5% per trade
    max_open_positions: int = 3               # Max concurrent positions
    max_daily_drawdown_pct: float = 3.5       # Daily stop-trading ceiling
    max_total_drawdown_pct: float = 8.0       # Max drawdown pause
    min_risk_reward_ratio: float = 1.5        # Target at least 1:1.5 RR
    leverage: int = 30                        # 1:30 leverage for Forex
    atr_sl_multiplier: float = 1.8            # ATR multiplier for Stop Loss
    atr_tp_multiplier: float = 2.7            # ATR multiplier for Take Profit (1:1.5 RR)
    allow_counter_trend: bool = False         # Trend filter preference

    def validate_against_safety(self):
        """Ensures user or AI modifications do not breach hard safety rules."""
        if self.risk_per_trade_pct > SAFETY_LIMITS["ABSOLUTE_MAX_RISK_PER_TRADE_PCT"]:
            self.risk_per_trade_pct = SAFETY_LIMITS["ABSOLUTE_MAX_RISK_PER_TRADE_PCT"]
        if self.max_daily_drawdown_pct > SAFETY_LIMITS["ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT"]:
            self.max_daily_drawdown_pct = SAFETY_LIMITS["ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT"]
        if self.max_total_drawdown_pct > SAFETY_LIMITS["ABSOLUTE_MAX_TOTAL_DRAWDOWN_PCT"]:
            self.max_total_drawdown_pct = SAFETY_LIMITS["ABSOLUTE_MAX_TOTAL_DRAWDOWN_PCT"]
        if self.min_risk_reward_ratio < SAFETY_LIMITS["ABSOLUTE_MIN_RR_RATIO"]:
            self.min_risk_reward_ratio = SAFETY_LIMITS["ABSOLUTE_MIN_RR_RATIO"]
        if self.max_open_positions > SAFETY_LIMITS["MAX_OPEN_POSITIONS"]:
            self.max_open_positions = SAFETY_LIMITS["MAX_OPEN_POSITIONS"]


@dataclass
class MLSettings:
    """Machine learning hyperparameters."""
    rf_n_estimators: int = 100
    rf_max_depth: int = 8
    rf_min_samples_split: int = 10
    lstm_sequence_length: int = 20
    lstm_hidden_dim: int = 48
    lstm_num_layers: int = 2
    lstm_epochs: int = 25
    lstm_batch_size: int = 32
    prediction_horizon: int = 3               # Bars ahead to forecast
    confidence_threshold: float = 0.55        # Minimum model probability to act


@dataclass
class LLMSettings:
    """Configurable LLM provider settings."""
    provider: str = "auto"                    # 'auto', 'gemini', 'openai', 'ollama', 'heuristic'
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    gemini_model: str = "gemini-1.5-flash"
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    temperature: float = 0.2                  # Low temperature for deterministic analysis


@dataclass
class AlpacaSettings:
    """
    Alpaca architecture configuration for educational development.
    SAFETY RULE: Broker connectivity is strictly disabled by default.
    Credentials are read ONLY from environment variables and never hardcoded in source.
    """
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    base_url: str = "https://paper-api.alpaca.markets"
    live_connectivity_enabled: bool = False   # Hardcoded immutable safety guard
    mock_initial_cash: float = 100000.0       # Standard paper trading starter balance
    is_paper: bool = True
    account_id: str = "MOCK-ALPACA-EDU-001"


@dataclass
class SystemConfig:
    default_pair: str = "EUR/USD"
    default_timeframe: str = "1h"
    risk: RiskSettings = field(default_factory=RiskSettings)
    ml: MLSettings = field(default_factory=MLSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    alpaca: AlpacaSettings = field(default_factory=AlpacaSettings)
    db_path: Path = DB_PATH
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR


# Global default configuration instance
DEFAULT_CONFIG = SystemConfig()
