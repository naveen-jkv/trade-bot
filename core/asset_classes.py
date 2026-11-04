"""
Asset Class Architecture & Classification.
Maintains clear structural boundaries between Forex currency pairs and Alpaca-supported
asset classes (US Equities and Cryptocurrencies).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class AssetClass(str, Enum):
    """Categorization of supported asset classes."""
    FOREX = "FOREX"
    US_EQUITY = "US_EQUITY"
    CRYPTO = "CRYPTO"


@dataclass(frozen=True)
class AssetMetadata:
    """Institutional specifications for a financial instrument."""
    symbol: str
    name: str
    asset_class: AssetClass
    exchange: str
    price_precision: int
    min_order_size: float
    is_fractionable: bool
    base_currency: str = "USD"
    quote_currency: str = "USD"


# Pre-configured catalog of Alpaca-supported educational assets
ALPACA_SUPPORTED_ASSETS: Dict[str, AssetMetadata] = {
    # US Equities & ETFs
    "AAPL": AssetMetadata(
        symbol="AAPL",
        name="Apple Inc.",
        asset_class=AssetClass.US_EQUITY,
        exchange="NASDAQ",
        price_precision=2,
        min_order_size=0.001,
        is_fractionable=True,
    ),
    "MSFT": AssetMetadata(
        symbol="MSFT",
        name="Microsoft Corporation",
        asset_class=AssetClass.US_EQUITY,
        exchange="NASDAQ",
        price_precision=2,
        min_order_size=0.001,
        is_fractionable=True,
    ),
    "NVDA": AssetMetadata(
        symbol="NVDA",
        name="NVIDIA Corporation",
        asset_class=AssetClass.US_EQUITY,
        exchange="NASDAQ",
        price_precision=2,
        min_order_size=0.001,
        is_fractionable=True,
    ),
    "TSLA": AssetMetadata(
        symbol="TSLA",
        name="Tesla, Inc.",
        asset_class=AssetClass.US_EQUITY,
        exchange="NASDAQ",
        price_precision=2,
        min_order_size=0.001,
        is_fractionable=True,
    ),
    "SPY": AssetMetadata(
        symbol="SPY",
        name="SPDR S&P 500 ETF Trust",
        asset_class=AssetClass.US_EQUITY,
        exchange="NYSE Arca",
        price_precision=2,
        min_order_size=0.001,
        is_fractionable=True,
    ),
    "QQQ": AssetMetadata(
        symbol="QQQ",
        name="Invesco QQQ Trust (Nasdaq 100)",
        asset_class=AssetClass.US_EQUITY,
        exchange="NASDAQ",
        price_precision=2,
        min_order_size=0.001,
        is_fractionable=True,
    ),
    # Cryptocurrencies (Alpaca Crypto)
    "BTC/USD": AssetMetadata(
        symbol="BTC/USD",
        name="Bitcoin",
        asset_class=AssetClass.CRYPTO,
        exchange="FTX/CBSE",
        price_precision=2,
        min_order_size=0.0001,
        is_fractionable=True,
        base_currency="BTC",
        quote_currency="USD",
    ),
    "ETH/USD": AssetMetadata(
        symbol="ETH/USD",
        name="Ethereum",
        asset_class=AssetClass.CRYPTO,
        exchange="FTX/CBSE",
        price_precision=2,
        min_order_size=0.001,
        is_fractionable=True,
        base_currency="ETH",
        quote_currency="USD",
    ),
}


def get_asset_class(symbol: str) -> AssetClass:
    """Identifies the asset class for any symbol."""
    sym_clean = symbol.strip().upper()
    if sym_clean in ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "CAD=X"]:
        return AssetClass.FOREX
    if sym_clean in ["BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD", "BTC", "ETH"]:
        return AssetClass.CRYPTO
    if sym_clean in ALPACA_SUPPORTED_ASSETS:
        return ALPACA_SUPPORTED_ASSETS[sym_clean].asset_class
    # Default US Equity for standard stock tickers (e.g. AMZN, META)
    if "/" not in sym_clean and "=" not in sym_clean:
        return AssetClass.US_EQUITY
    return AssetClass.FOREX


def is_forex_symbol(symbol: str) -> bool:
    """Checks whether the symbol belongs to the Forex universe."""
    return get_asset_class(symbol) == AssetClass.FOREX


def is_alpaca_symbol(symbol: str) -> bool:
    """Checks whether the symbol is supported in the Alpaca module."""
    ac = get_asset_class(symbol)
    return ac in (AssetClass.US_EQUITY, AssetClass.CRYPTO)

