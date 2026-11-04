"""
Alpaca Data Models & Enums.
Strictly mirrors official Alpaca REST v2 schemas for accounts, orders, positions,
assets, and clock representations for seamless drop-in educational compatibility.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class AlpacaOrderStatus(str, Enum):
    NEW = "new"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class AlpacaOrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class AlpacaOrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class AlpacaTimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class AlpacaAccount:
    """Mirrors Alpaca GET /v2/account."""
    id: str
    account_number: str
    status: str                        # 'ACTIVE', 'ONBOARDING', etc.
    currency: str = "USD"
    cash: float = 100000.0
    portfolio_value: float = 100000.0
    buying_power: float = 200000.0     # 2x margin standard
    regt_buying_power: float = 200000.0
    daytrading_buying_power: float = 400000.0
    non_marginable_buying_power: float = 100000.0
    pattern_day_trader: bool = False
    trade_suspended_by_user: bool = False
    trading_blocked: bool = False
    transfers_blocked: bool = False
    account_blocked: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    shorting_enabled: bool = True
    multiplier: str = "2"
    equity: float = 100000.0
    last_equity: float = 100000.0
    long_market_value: float = 0.0
    short_market_value: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    last_maintenance_margin: float = 0.0
    daytrade_count: int = 0
    is_live_broker: bool = False       # Invariant: False for educational simulation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "account_number": self.account_number,
            "status": self.status,
            "currency": self.currency,
            "cash": round(self.cash, 2),
            "portfolio_value": round(self.portfolio_value, 2),
            "buying_power": round(self.buying_power, 2),
            "equity": round(self.equity, 2),
            "pattern_day_trader": self.pattern_day_trader,
            "daytrade_count": self.daytrade_count,
            "is_live_broker": self.is_live_broker,
            "data_mode": "MOCK_EDUCATIONAL_SIMULATION",
        }


@dataclass
class AlpacaOrder:
    """Mirrors Alpaca GET /v2/orders and POST /v2/orders response."""
    id: str
    client_order_id: str
    created_at: str
    updated_at: str
    submitted_at: str
    filled_at: Optional[str]
    expired_at: Optional[str]
    canceled_at: Optional[str]
    failed_at: Optional[str]
    replaced_at: Optional[str]
    replaced_by: Optional[str]
    replaces: Optional[str]
    asset_id: str
    symbol: str
    asset_class: str                   # 'us_equity' or 'crypto'
    notional: Optional[float]
    qty: float
    filled_qty: float
    filled_avg_price: Optional[float]
    order_class: str = "simple"
    order_type: str = "market"
    type: str = "market"
    side: str = "buy"
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "new"
    extended_hours: bool = False
    legs: Optional[List[Any]] = None
    trail_percent: Optional[float] = None
    trail_price: Optional[float] = None
    hwm: Optional[float] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "client_order_id": self.client_order_id,
            "created_at": self.created_at,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "qty": self.qty,
            "filled_qty": self.filled_qty,
            "filled_avg_price": round(self.filled_avg_price, 4) if self.filled_avg_price else None,
            "order_type": self.order_type,
            "side": self.side,
            "time_in_force": self.time_in_force,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class AlpacaPosition:
    """Mirrors Alpaca GET /v2/positions."""
    asset_id: str
    symbol: str
    exchange: str
    asset_class: str
    avg_entry_price: float
    qty: float
    side: str                          # 'long' or 'short'
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    unrealized_intraday_pl: float
    unrealized_intraday_plpc: float
    current_price: float
    lastday_price: float
    change_today: float
    qty_available: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "asset_class": self.asset_class,
            "qty": round(self.qty, 4),
            "avg_entry_price": round(self.avg_entry_price, 4),
            "current_price": round(self.current_price, 4),
            "market_value": round(self.market_value, 2),
            "cost_basis": round(self.cost_basis, 2),
            "unrealized_pl": round(self.unrealized_pl, 2),
            "unrealized_plpc": round(self.unrealized_plpc, 4),
            "side": self.side,
        }


@dataclass
class AlpacaClock:
    """Mirrors Alpaca GET /v2/clock."""
    timestamp: str
    is_open: bool
    next_open: str
    next_close: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "is_open": self.is_open,
            "next_open": self.next_open,
            "next_close": self.next_close,
        }


@dataclass
class AlpacaAsset:
    """Mirrors Alpaca GET /v2/assets/{symbol}."""
    id: str
    symbol: str
    name: str
    asset_class: str
    exchange: str
    status: str = "active"
    tradable: bool = True
    marginable: bool = True
    shortable: bool = True
    fractionable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "name": self.name,
            "asset_class": self.asset_class,
            "exchange": self.exchange,
            "status": self.status,
            "tradable": self.tradable,
            "fractionable": self.fractionable,
        }

