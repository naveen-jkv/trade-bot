"""
Modular Alpaca Mock Client (Simulated Broker Engine).
Designed strictly for educational development and algorithmic architecture research.

MANDATORY SAFETY BOUNDARIES:
1. Live broker connectivity is permanently DISABLED (broker_connectivity_enabled = False).
2. Never submits real orders or communicates with live broker endpoints.
3. Operates exclusively against local simulated matching engine and SQLite memory.
4. API keys are read purely from environment variables and never hardcoded in source.
5. Strict separation of concerns: Forex pairs are rejected here and directed to the Forex engine.
"""

from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

from config import DEFAULT_CONFIG, SystemConfig
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
from core.asset_classes import (
    AssetClass,
    get_asset_class,
    is_forex_symbol,
    ALPACA_SUPPORTED_ASSETS,
)
from core.market_data_abstraction import (
    UnifiedMarketDataRouter,
    DataSourceType,
    get_data_source_badge,
)
from database.memory import TradingMemoryDB


class MockAlpacaClient(AlpacaClientInterface):
    """
    High-fidelity simulated Alpaca broker conforming to REST v2 API specification.
    Operates offline without live broker accounts.
    """

    # IMMUTABLE ARCHITECTURAL SAFETY GUARDS
    _LIVE_CONNECTIVITY_ENABLED: bool = False
    _ALLOW_LIVE_BROKER_ENDPOINTS: bool = False

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        db: Optional[TradingMemoryDB] = None,
        market_data: Optional[UnifiedMarketDataRouter] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.db = db or TradingMemoryDB(self.config.db_path)
        self.market_data = market_data or UnifiedMarketDataRouter(cache_dir=self.config.data_dir)

        # Account parameters (Initial educational paper balance: $100,000)
        self.account_id = getattr(self.config.alpaca, "account_id", "MOCK-ALPACA-EDU-001")
        self.cash: float = getattr(self.config.alpaca, "mock_initial_cash", 100000.0)
        self.initial_cash: float = self.cash

        # In-memory order books and open positions
        self.orders: Dict[str, AlpacaOrder] = {}
        self.positions: Dict[str, AlpacaPosition] = {}

        # Load persisted mock orders and positions from SQLite if available
        self._load_from_db()

    # --- Safety Properties ---

    @property
    def is_live_connected(self) -> bool:
        """Always False. Broker connectivity is permanently disabled."""
        return False

    @property
    def broker_connectivity_enabled(self) -> bool:
        """Always False. Prevents any external network access to real accounts."""
        return False

    def attempt_live_connection(self) -> None:
        """Hard safety tripwire: attempts to establish live broker connection will raise immediately."""
        raise RuntimeError(
            "Live broker connectivity is disabled by safety policy. "
            "The system is operating in simulated educational mock mode only."
        )

    # --- Account Information ---

    def get_account(self) -> AlpacaAccount:
        """Calculates current simulated account balance, portfolio value, and buying power."""
        self._update_all_positions_mark_to_market()

        long_mkt_val = sum(p.market_value for p in self.positions.values() if p.side == "long")
        short_mkt_val = sum(p.market_value for p in self.positions.values() if p.side == "short")
        portfolio_val = self.cash + long_mkt_val - short_mkt_val
        equity = portfolio_val

        # 2x standard Reg-T margin buying power
        buying_power = max(0.0, equity * 2.0)
        daytrade_bp = max(0.0, equity * 4.0)

        return AlpacaAccount(
            id=self.account_id,
            account_number=self.account_id,
            status="ACTIVE",
            currency="USD",
            cash=round(self.cash, 2),
            portfolio_value=round(portfolio_val, 2),
            buying_power=round(buying_power, 2),
            regt_buying_power=round(buying_power, 2),
            daytrading_buying_power=round(daytrade_bp, 2),
            non_marginable_buying_power=round(self.cash, 2),
            pattern_day_trader=False,
            created_at="2026-01-01T00:00:00Z",
            equity=round(equity, 2),
            last_equity=round(self.initial_cash, 2),
            long_market_value=round(long_mkt_val, 2),
            short_market_value=round(short_mkt_val, 2),
            daytrade_count=0,
            is_live_broker=False,
        )

    # --- Order Submission & Execution Engine ---

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> AlpacaOrder:
        """
        Executes a simulated order against the mock matching engine.
        Enforces asset class separation: Forex pairs are blocked here and directed to the Forex engine.
        """
        sym_clean = symbol.strip().upper()
        side_clean = side.strip().lower()
        type_clean = order_type.strip().lower()
        tif_clean = time_in_force.strip().lower()

        order_id = str(uuid.uuid4())
        cid = client_order_id or f"mock_{uuid.uuid4().hex[:10]}"
        now_ts = datetime.now(timezone.utc).isoformat()

        # Requirement 5: Keep Forex separate from Alpaca
        if is_forex_symbol(sym_clean):
            rejected_order = AlpacaOrder(
                id=order_id,
                client_order_id=cid,
                created_at=now_ts,
                updated_at=now_ts,
                submitted_at=now_ts,
                filled_at=None,
                expired_at=None,
                canceled_at=None,
                failed_at=now_ts,
                replaced_at=None,
                replaced_by=None,
                replaces=None,
                asset_id=sym_clean,
                symbol=sym_clean,
                asset_class="forex",
                notional=None,
                qty=qty,
                filled_qty=0.0,
                filled_avg_price=None,
                order_type=type_clean,
                type=type_clean,
                side=side_clean,
                time_in_force=tif_clean,
                status=AlpacaOrderStatus.REJECTED.value,
                rejection_reason=(
                    "Forex currency pairs must be traded through the institutional Forex Paper Trader "
                    "with lot sizes and pip spreads, not the Alpaca Equities/Crypto interface."
                ),
            )
            self.orders[order_id] = rejected_order
            self._save_order_to_db(rejected_order)
            return rejected_order

        # Validate order size
        if qty <= 0:
            return self._create_rejected_order(order_id, cid, sym_clean, qty, side_clean, type_clean, tif_clean, "Quantity must be greater than 0.")

        # Determine asset class
        ac = get_asset_class(sym_clean)
        asset_class_str = "us_equity" if ac == AssetClass.US_EQUITY else "crypto"

        # Fetch current mark price from market data abstraction layer
        curr_price, src_type, _ = self.market_data.get_latest_price(sym_clean)

        # Build order object
        order = AlpacaOrder(
            id=order_id,
            client_order_id=cid,
            created_at=now_ts,
            updated_at=now_ts,
            submitted_at=now_ts,
            filled_at=None,
            expired_at=None,
            canceled_at=None,
            failed_at=None,
            replaced_at=None,
            replaced_by=None,
            replaces=None,
            asset_id=sym_clean,
            symbol=sym_clean,
            asset_class=asset_class_str,
            notional=round(qty * curr_price, 2),
            qty=qty,
            filled_qty=0.0,
            filled_avg_price=None,
            order_type=type_clean,
            type=type_clean,
            side=side_clean,
            time_in_force=tif_clean,
            limit_price=limit_price,
            stop_price=stop_price,
            status=AlpacaOrderStatus.NEW.value,
        )

        # Simulated Execution Logic
        if type_clean == AlpacaOrderType.MARKET.value:
            # Simulate realistic slippage (0.02% for liquid equities, 0.05% for crypto)
            slippage_pct = 0.0005 if asset_class_str == "crypto" else 0.0002
            slippage = curr_price * slippage_pct * (1 if side_clean == "buy" else -1)
            exec_price = round(curr_price + slippage, 4)

            order_cost = exec_price * qty

            if side_clean == AlpacaOrderSide.BUY.value:
                # Check buying power
                acct = self.get_account()
                if order_cost > acct.buying_power:
                    order.status = AlpacaOrderStatus.REJECTED.value
                    order.rejection_reason = f"Insufficient buying power. Required: ${order_cost:,.2f}, Available: ${acct.buying_power:,.2f}"
                else:
                    self._fill_buy_order(order, exec_price, qty)
            else:  # SELL
                # Check if position exists
                pos = self.positions.get(sym_clean)
                if not pos or pos.qty < qty:
                    available = pos.qty if pos else 0.0
                    order.status = AlpacaOrderStatus.REJECTED.value
                    order.rejection_reason = f"Insufficient position to sell. Requested: {qty}, Available: {available}"
                else:
                    self._fill_sell_order(order, exec_price, qty, pos)

        elif type_clean == AlpacaOrderType.LIMIT.value:
            # Check if limit is immediately marketable
            if limit_price is None:
                order.status = AlpacaOrderStatus.REJECTED.value
                order.rejection_reason = "Limit price is required for limit orders."
            elif side_clean == AlpacaOrderSide.BUY.value and limit_price >= curr_price:
                self._fill_buy_order(order, curr_price, qty)
            elif side_clean == AlpacaOrderSide.SELL.value and limit_price <= curr_price:
                pos = self.positions.get(sym_clean)
                if not pos or pos.qty < qty:
                    order.status = AlpacaOrderStatus.REJECTED.value
                    order.rejection_reason = "Insufficient position to sell."
                else:
                    self._fill_sell_order(order, curr_price, qty, pos)
            else:
                # Place in order book as accepted pending limit order
                order.status = AlpacaOrderStatus.ACCEPTED.value

        elif type_clean == AlpacaOrderType.STOP.value:
            if stop_price is None:
                order.status = AlpacaOrderStatus.REJECTED.value
                order.rejection_reason = "Stop price is required for stop orders."
            else:
                order.status = AlpacaOrderStatus.ACCEPTED.value

        else:
            order.status = AlpacaOrderStatus.REJECTED.value
            order.rejection_reason = f"Unsupported order type: {type_clean}"

        self.orders[order_id] = order
        self._save_order_to_db(order)
        return order

    def _fill_buy_order(self, order: AlpacaOrder, price: float, qty: float) -> None:
        """Executes a simulated BUY order, updates cash, and tracks position."""
        cost = price * qty
        self.cash -= cost

        now_ts = datetime.now(timezone.utc).isoformat()
        order.status = AlpacaOrderStatus.FILLED.value
        order.filled_at = now_ts
        order.updated_at = now_ts
        order.filled_qty = qty
        order.filled_avg_price = price

        sym = order.symbol
        if sym in self.positions:
            pos = self.positions[sym]
            new_qty = pos.qty + qty
            total_cost = pos.cost_basis + cost
            avg_price = total_cost / new_qty
            pos.qty = new_qty
            pos.avg_entry_price = round(avg_price, 4)
            pos.cost_basis = round(total_cost, 2)
            pos.current_price = price
            pos.market_value = round(new_qty * price, 2)
            pos.unrealized_pl = round(pos.market_value - pos.cost_basis, 2)
            pos.unrealized_plpc = round(pos.unrealized_pl / pos.cost_basis, 4) if pos.cost_basis > 0 else 0.0
            self._save_position_to_db(pos)
        else:
            pos = AlpacaPosition(
                asset_id=sym,
                symbol=sym,
                exchange="SIMULATED",
                asset_class=order.asset_class,
                avg_entry_price=price,
                qty=qty,
                side="long",
                market_value=round(cost, 2),
                cost_basis=round(cost, 2),
                unrealized_pl=0.0,
                unrealized_plpc=0.0,
                unrealized_intraday_pl=0.0,
                unrealized_intraday_plpc=0.0,
                current_price=price,
                lastday_price=price,
                change_today=0.0,
                qty_available=qty,
            )
            self.positions[sym] = pos
            self._save_position_to_db(pos)

    def _fill_sell_order(self, order: AlpacaOrder, price: float, qty: float, pos: AlpacaPosition) -> None:
        """Executes a simulated SELL order, credits proceeds, and adjusts position."""
        proceeds = price * qty
        self.cash += proceeds

        now_ts = datetime.now(timezone.utc).isoformat()
        order.status = AlpacaOrderStatus.FILLED.value
        order.filled_at = now_ts
        order.updated_at = now_ts
        order.filled_qty = qty
        order.filled_avg_price = price

        sym = order.symbol
        if pos.qty <= qty:
            # Fully closed
            del self.positions[sym]
            self._delete_position_from_db(sym)
        else:
            # Partially closed
            rem_qty = pos.qty - qty
            pos.qty = rem_qty
            pos.cost_basis = round(pos.avg_entry_price * rem_qty, 2)
            pos.current_price = price
            pos.market_value = round(rem_qty * price, 2)
            pos.unrealized_pl = round(pos.market_value - pos.cost_basis, 2)
            pos.unrealized_plpc = round(pos.unrealized_pl / pos.cost_basis, 4) if pos.cost_basis > 0 else 0.0
            self._save_position_to_db(pos)

    def _create_rejected_order(
        self, order_id: str, cid: str, symbol: str, qty: float, side: str, otype: str, tif: str, reason: str
    ) -> AlpacaOrder:
        now_ts = datetime.now(timezone.utc).isoformat()
        order = AlpacaOrder(
            id=order_id,
            client_order_id=cid,
            created_at=now_ts,
            updated_at=now_ts,
            submitted_at=now_ts,
            filled_at=None,
            expired_at=None,
            canceled_at=None,
            failed_at=now_ts,
            replaced_at=None,
            replaced_by=None,
            replaces=None,
            asset_id=symbol,
            symbol=symbol,
            asset_class="us_equity",
            notional=None,
            qty=qty,
            filled_qty=0.0,
            filled_avg_price=None,
            order_type=otype,
            type=otype,
            side=side,
            time_in_force=tif,
            status=AlpacaOrderStatus.REJECTED.value,
            rejection_reason=reason,
        )
        self.orders[order_id] = order
        self._save_order_to_db(order)
        return order

    # --- Position & Order Management ---

    def get_order(self, order_id: str) -> Optional[AlpacaOrder]:
        return self.orders.get(order_id)

    def list_orders(self, status: str = "all", limit: int = 50, nested: bool = True) -> List[AlpacaOrder]:
        order_list = list(self.orders.values())
        if status == "open":
            order_list = [o for o in order_list if o.status in (AlpacaOrderStatus.NEW.value, AlpacaOrderStatus.ACCEPTED.value)]
        elif status == "closed":
            order_list = [o for o in order_list if o.status in (AlpacaOrderStatus.FILLED.value, AlpacaOrderStatus.CANCELED.value, AlpacaOrderStatus.REJECTED.value)]
        # Sort descending by created_at
        order_list.sort(key=lambda o: o.created_at, reverse=True)
        return order_list[:limit]

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if not order:
            return False
        if order.status in (AlpacaOrderStatus.NEW.value, AlpacaOrderStatus.ACCEPTED.value):
            now_ts = datetime.now(timezone.utc).isoformat()
            order.status = AlpacaOrderStatus.CANCELED.value
            order.canceled_at = now_ts
            order.updated_at = now_ts
            self._save_order_to_db(order)
            return True
        return False

    def list_positions(self) -> List[AlpacaPosition]:
        self._update_all_positions_mark_to_market()
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[AlpacaPosition]:
        self._update_all_positions_mark_to_market()
        return self.positions.get(symbol.strip().upper())

    def close_position(self, symbol: str, qty: Optional[float] = None) -> Optional[AlpacaOrder]:
        sym = symbol.strip().upper()
        pos = self.positions.get(sym)
        if not pos:
            return None
        close_qty = qty if qty is not None and qty <= pos.qty else pos.qty
        side = "sell" if pos.side == "long" else "buy"
        return self.submit_order(symbol=sym, qty=close_qty, side=side, order_type="market")

    def _update_all_positions_mark_to_market(self) -> None:
        """Refreshes positions against current mark prices."""
        for sym, pos in list(self.positions.items()):
            try:
                curr_px, _, _ = self.market_data.get_latest_price(sym)
                pos.current_price = curr_px
                pos.market_value = round(pos.qty * curr_px, 2)
                pos.unrealized_pl = round(pos.market_value - pos.cost_basis, 2)
                pos.unrealized_plpc = round(pos.unrealized_pl / pos.cost_basis, 4) if pos.cost_basis > 0 else 0.0
            except Exception:
                pass

    # --- Assets & Clock ---

    def get_clock(self) -> AlpacaClock:
        """Returns mock market clock (Equities open 09:30-16:00 ET weekdays, Crypto 24/7)."""
        now = datetime.now(timezone.utc)
        # Check standard US market hours approximation (13:30 to 20:00 UTC, Mon-Fri)
        is_weekday = now.weekday() < 5
        hour = now.hour + now.minute / 60.0
        is_market_open = is_weekday and (13.5 <= hour <= 20.0)

        next_open = (now + timedelta(days=1)).replace(hour=13, minute=30, second=0).isoformat()
        next_close = now.replace(hour=20, minute=0, second=0).isoformat()

        return AlpacaClock(
            timestamp=now.isoformat(),
            is_open=is_market_open,
            next_open=next_open,
            next_close=next_close,
        )

    def get_asset(self, symbol: str) -> Optional[AlpacaAsset]:
        sym = symbol.strip().upper()
        meta = ALPACA_SUPPORTED_ASSETS.get(sym)
        if meta:
            return AlpacaAsset(
                id=sym,
                symbol=meta.symbol,
                name=meta.name,
                asset_class="us_equity" if meta.asset_class == AssetClass.US_EQUITY else "crypto",
                exchange=meta.exchange,
                fractionable=meta.is_fractionable,
            )
        return None

    def list_assets(self, asset_class: Optional[str] = None) -> List[AlpacaAsset]:
        assets = []
        for meta in ALPACA_SUPPORTED_ASSETS.values():
            ac_str = "us_equity" if meta.asset_class == AssetClass.US_EQUITY else "crypto"
            if asset_class is None or asset_class == ac_str:
                assets.append(
                    AlpacaAsset(
                        id=meta.symbol,
                        symbol=meta.symbol,
                        name=meta.name,
                        asset_class=ac_str,
                        exchange=meta.exchange,
                        fractionable=meta.is_fractionable,
                    )
                )
        return assets

    def reset_mock_portfolio(self) -> None:
        """Resets the mock Alpaca portfolio back to initial cash."""
        self.cash = self.initial_cash
        self.orders.clear()
        self.positions.clear()
        try:
            self.db.clear_alpaca_mock_data()
        except Exception:
            pass

    # --- Database Persistence ---

    def _save_order_to_db(self, order: AlpacaOrder) -> None:
        try:
            self.db.store_alpaca_order(order.to_dict())
        except Exception:
            pass

    def _save_position_to_db(self, pos: AlpacaPosition) -> None:
        try:
            self.db.store_alpaca_position(pos.to_dict())
        except Exception:
            pass

    def _delete_position_from_db(self, symbol: str) -> None:
        try:
            self.db.delete_alpaca_position(symbol)
        except Exception:
            pass

    def _load_from_db(self) -> None:
        try:
            db_orders = self.db.get_alpaca_orders(limit=100)
            for o in db_orders:
                alpaca_order = AlpacaOrder(
                    id=o["id"],
                    client_order_id=o.get("client_order_id", ""),
                    created_at=o.get("created_at", ""),
                    updated_at=o.get("created_at", ""),
                    submitted_at=o.get("created_at", ""),
                    filled_at=o.get("filled_at"),
                    expired_at=None,
                    canceled_at=None,
                    failed_at=None,
                    replaced_at=None,
                    replaced_by=None,
                    replaces=None,
                    asset_id=o.get("symbol", ""),
                    symbol=o.get("symbol", ""),
                    asset_class=o.get("asset_class", "us_equity"),
                    notional=None,
                    qty=float(o.get("qty", 0.0)),
                    filled_qty=float(o.get("filled_qty", 0.0)),
                    filled_avg_price=float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
                    order_type=o.get("order_type", "market"),
                    type=o.get("order_type", "market"),
                    side=o.get("side", "buy"),
                    time_in_force=o.get("time_in_force", "day"),
                    limit_price=float(o["limit_price"]) if o.get("limit_price") else None,
                    stop_price=float(o["stop_price"]) if o.get("stop_price") else None,
                    status=o.get("status", "new"),
                    rejection_reason=o.get("rejection_reason"),
                )
                self.orders[alpaca_order.id] = alpaca_order

            db_positions = self.db.get_alpaca_positions()
            for p in db_positions:
                sym = p["symbol"]
                qty = float(p.get("qty", 0.0))
                avg_entry = float(p.get("avg_entry_price", 0.0))
                cost_basis = float(p.get("cost_basis", avg_entry * qty))
                curr_px = avg_entry
                mkt_val = qty * curr_px
                pos = AlpacaPosition(
                    asset_id=sym,
                    symbol=sym,
                    exchange="SIMULATED",
                    asset_class=p.get("asset_class", "us_equity"),
                    avg_entry_price=avg_entry,
                    qty=qty,
                    side=p.get("side", "long"),
                    market_value=round(mkt_val, 2),
                    cost_basis=round(cost_basis, 2),
                    unrealized_pl=0.0,
                    unrealized_plpc=0.0,
                    unrealized_intraday_pl=0.0,
                    unrealized_intraday_plpc=0.0,
                    current_price=curr_px,
                    lastday_price=curr_px,
                    change_today=0.0,
                    qty_available=qty,
                )
                self.positions[sym] = pos
        except Exception:
            pass

