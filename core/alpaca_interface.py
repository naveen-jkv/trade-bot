"""
Alpaca Client Modular Interface Definition.
Defines the institutional abstract contract for Alpaca brokerage interaction.
Educational Guarantee: All implementations must respect broker connectivity disabling.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

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


class AlpacaClientInterface(ABC):
    """
    Abstract interface for Alpaca v2 REST API interactions.
    Enables modular swapping between simulated local execution engines and future architectures.
    """

    @property
    @abstractmethod
    def is_live_connected(self) -> bool:
        """Returns True only if connected to live broker (must be False in mock/educational mode)."""
        pass

    @property
    @abstractmethod
    def broker_connectivity_enabled(self) -> bool:
        """Safety indicator: indicates whether real broker network calls are permitted."""
        pass

    @abstractmethod
    def get_account(self) -> AlpacaAccount:
        """Fetches account state, buying power, equity, and balances."""
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,                             # 'buy' or 'sell'
        order_type: str = "market",            # 'market', 'limit', 'stop'
        time_in_force: str = "day",            # 'day', 'gtc'
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
    ) -> AlpacaOrder:
        """Submits an order according to the Alpaca v2 order specification."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[AlpacaOrder]:
        """Retrieves a specific order by order_id or client_order_id."""
        pass

    @abstractmethod
    def list_orders(
        self,
        status: str = "all",
        limit: int = 50,
        nested: bool = True,
    ) -> List[AlpacaOrder]:
        """Lists submitted orders filtered by status ('open', 'closed', 'all')."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancels an active or pending order."""
        pass

    def cancel_all_orders(self) -> bool:
        """Cancels all active open orders."""
        open_orders = self.list_orders(status="open")
        all_ok = True
        for o in open_orders:
            if not self.cancel_order(o.id):
                all_ok = False
        return all_ok

    @abstractmethod
    def list_positions(self) -> List[AlpacaPosition]:
        """Lists all open positions."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[AlpacaPosition]:
        """Fetches an open position for a given symbol."""
        pass

    @abstractmethod
    def close_position(
        self,
        symbol: str,
        qty: Optional[float] = None,
    ) -> Optional[AlpacaOrder]:
        """Liquidates an open position in part or full."""
        pass

    def close_all_positions(self, cancel_orders: bool = True) -> List[AlpacaOrder]:
        """Liquidates all open positions and optionally cancels open orders."""
        if cancel_orders:
            self.cancel_all_orders()
        closed_orders = []
        for pos in self.list_positions():
            order = self.close_position(pos.symbol)
            if order:
                closed_orders.append(order)
        return closed_orders

    @abstractmethod
    def get_clock(self) -> AlpacaClock:
        """Retrieves market open/close clock and status."""
        pass

    @abstractmethod
    def get_asset(self, symbol: str) -> Optional[AlpacaAsset]:
        """Retrieves asset metadata and trade permissions."""
        pass

    @abstractmethod
    def list_assets(self, asset_class: Optional[str] = None) -> List[AlpacaAsset]:
        """Lists supported tradable assets."""
        pass

