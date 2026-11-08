"""
Core quantitative trading, technical analysis, risk, and execution modules.
Includes modular Alpaca architecture for educational development.
"""

from core.indicators import TechnicalAnalyzer, TechnicalState
from core.market_data import MarketDataProvider, DataSourceType
# Lazy load heavy ML modules (torch, sklearn) on demand to avoid unnecessary memory pressure
def __getattr__(name):
    if name in ("TradingPredictor", "MLFeaturePipeline", "ForexLSTMNetwork"):
        import core.prediction as _pred
        return getattr(_pred, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from core.risk_manager import RiskManager, RiskValidationResult
from core.paper_trader import PaperTradingSimulator, Position, ClosedTrade

# Asset classification
from core.asset_classes import (
    AssetClass,
    AssetMetadata,
    ALPACA_SUPPORTED_ASSETS,
    get_asset_class,
    is_forex_symbol,
    is_alpaca_symbol,
)

# Market Data Abstraction
from core.market_data_abstraction import (
    BaseMarketDataProvider,
    ForexMarketDataProvider,
    AlpacaMarketDataProvider,
    UnifiedMarketDataRouter,
    DATA_LABELS,
    get_data_source_badge,
)

# Alpaca Architecture (Simulated / Educational / Live Real-Money)
from core.alpaca_models import (
    AlpacaAccount,
    AlpacaOrder,
    AlpacaPosition,
    AlpacaClock,
    AlpacaAsset,
    AlpacaOrderStatus,
    AlpacaOrderSide,
    AlpacaOrderType,
    AlpacaTimeInForce,
)
from core.alpaca_interface import AlpacaClientInterface
from core.alpaca_mock_client import MockAlpacaClient
from core.alpaca_live_client import LiveAlpacaClient
from core.alpaca_factory import create_alpaca_client

__all__ = [
    "TechnicalAnalyzer",
    "TechnicalState",
    "MarketDataProvider",
    "DataSourceType",
    "TradingPredictor",
    "MLFeaturePipeline",
    "ForexLSTMNetwork",
    "RiskManager",
    "RiskValidationResult",
    "PaperTradingSimulator",
    "Position",
    "ClosedTrade",
    "AssetClass",
    "AssetMetadata",
    "ALPACA_SUPPORTED_ASSETS",
    "get_asset_class",
    "is_forex_symbol",
    "is_alpaca_symbol",
    "BaseMarketDataProvider",
    "ForexMarketDataProvider",
    "AlpacaMarketDataProvider",
    "UnifiedMarketDataRouter",
    "DATA_LABELS",
    "get_data_source_badge",
    "AlpacaAccount",
    "AlpacaOrder",
    "AlpacaPosition",
    "AlpacaClock",
    "AlpacaAsset",
    "AlpacaOrderStatus",
    "AlpacaOrderSide",
    "AlpacaOrderType",
    "AlpacaTimeInForce",
    "AlpacaClientInterface",
    "MockAlpacaClient",
    "LiveAlpacaClient",
    "create_alpaca_client",
]
