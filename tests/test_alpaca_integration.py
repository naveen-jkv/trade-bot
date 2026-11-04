"""
Comprehensive Tests for Alpaca Modular Architecture (Educational Simulator).
Verifies:
1. Modular interface compliance (AlpacaClientInterface).
2. Broker connectivity disabled safeguards (strict prohibition of live broker calls).
3. Mock account data and simulated execution.
4. Prohibition of real orders / live brokerage access.
5. Strict separation between Forex and Alpaca asset classes.
6. Market-data abstraction layer and provenance tagging.
7. Simulated portfolio and order history tracking.
8. Clear labels distinguishing mock from real/delayed data.
9. Secret hygiene (no hardcoded keys).
10. SQLite persistence of mock orders and positions.
"""

import pytest
from pathlib import Path
import tempfile
import os

from config import SystemConfig, AlpacaSettings
from database.memory import TradingMemoryDB
from core.asset_classes import (
    AssetClass,
    get_asset_class,
    is_forex_symbol,
    is_alpaca_symbol,
    ALPACA_SUPPORTED_ASSETS,
)
from core.market_data_abstraction import (
    BaseMarketDataProvider,
    ForexMarketDataProvider,
    AlpacaMarketDataProvider,
    UnifiedMarketDataRouter,
    DataSourceType,
    get_data_source_badge,
    DATA_LABELS,
)
from core.alpaca_models import (
    AlpacaAccount,
    AlpacaOrder,
    AlpacaPosition,
    AlpacaOrderStatus,
    AlpacaOrderSide,
    AlpacaOrderType,
)
from core.alpaca_interface import AlpacaClientInterface
from core.alpaca_mock_client import MockAlpacaClient


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = Path(tf.name)
    db = TradingMemoryDB(db_path)
    yield db
    try:
        db_path.unlink()
    except Exception:
        pass


@pytest.fixture
def mock_alpaca(temp_db):
    cfg = SystemConfig(
        db_path=temp_db.db_path,
        alpaca=AlpacaSettings(
            mock_initial_cash=100000.0,
            live_connectivity_enabled=False,
        ),
    )
    client = MockAlpacaClient(config=cfg, db=temp_db)
    return client


def test_alpaca_modular_interface_compliance(mock_alpaca):
    """Ensures MockAlpacaClient fulfills the abstract AlpacaClientInterface."""
    assert isinstance(mock_alpaca, AlpacaClientInterface)


def test_broker_connectivity_disabled_safeguards(mock_alpaca):
    """Verifies that live broker connectivity is permanently disabled and guarded."""
    assert mock_alpaca.is_live_connected is False
    assert mock_alpaca.broker_connectivity_enabled is False

    # Hard tripwire: live connection attempt must raise RuntimeError
    with pytest.raises(RuntimeError, match="Live broker connectivity is disabled"):
        mock_alpaca.attempt_live_connection()


def test_mock_account_data(mock_alpaca):
    """Verifies mock account properties and buying power calculations."""
    acct = mock_alpaca.get_account()
    assert isinstance(acct, AlpacaAccount)
    assert acct.cash == 100000.0
    assert acct.portfolio_value == 100000.0
    assert acct.buying_power == 200000.0  # 2x Reg-T margin
    assert acct.is_live_broker is False
    assert acct.status == "ACTIVE"


def test_asset_class_separation_forex_rejection(mock_alpaca):
    """
    Requirement 5: Strict separation of Forex and Alpaca.
    Submitting a Forex currency pair to Alpaca client must be rejected.
    """
    assert is_forex_symbol("EUR/USD") is True
    assert is_alpaca_symbol("AAPL") is True
    assert is_alpaca_symbol("BTC/USD") is True

    # Attempt to submit EUR/USD order to Alpaca interface
    order = mock_alpaca.submit_order(
        symbol="EUR/USD",
        qty=1000.0,
        side="buy",
        order_type="market",
    )
    assert order.status == AlpacaOrderStatus.REJECTED.value
    assert "Forex" in order.rejection_reason


def test_simulated_market_order_and_positions(mock_alpaca):
    """
    Requirement 3 & 4: Simulated execution and position management without real orders.
    """
    # 1. Market BUY order for AAPL
    order = mock_alpaca.submit_order(
        symbol="AAPL",
        qty=10.0,
        side="buy",
        order_type="market",
    )
    assert order.status == AlpacaOrderStatus.FILLED.value
    assert order.filled_qty == 10.0
    assert order.filled_avg_price is not None
    assert order.filled_avg_price > 0

    # Check cash deduction
    acct = mock_alpaca.get_account()
    expected_cost = order.filled_avg_price * 10.0
    assert acct.cash == pytest.approx(100000.0 - expected_cost, rel=1e-3)

    # Check position created
    positions = mock_alpaca.list_positions()
    assert len(positions) == 1
    pos = mock_alpaca.get_position("AAPL")
    assert pos is not None
    assert pos.symbol == "AAPL"
    assert pos.qty == 10.0
    assert pos.cost_basis == pytest.approx(expected_cost, rel=1e-3)

    # 2. Market SELL order to liquidate half
    sell_order = mock_alpaca.submit_order(
        symbol="AAPL",
        qty=5.0,
        side="sell",
        order_type="market",
    )
    assert sell_order.status == AlpacaOrderStatus.FILLED.value
    pos_after = mock_alpaca.get_position("AAPL")
    assert pos_after.qty == 5.0

    # 3. Liquidate remainder via close_position
    close_order = mock_alpaca.close_position("AAPL")
    assert close_order.status == AlpacaOrderStatus.FILLED.value
    assert mock_alpaca.get_position("AAPL") is None
    assert len(mock_alpaca.list_positions()) == 0


def test_limit_order_workflow_and_cancellation(mock_alpaca):
    """Tests limit order queuing and cancellation."""
    # Place limit order far below market price
    order = mock_alpaca.submit_order(
        symbol="MSFT",
        qty=5.0,
        side="buy",
        order_type="limit",
        limit_price=10.0,  # Well below current price
    )
    assert order.status == AlpacaOrderStatus.ACCEPTED.value

    # Check order in list_orders
    open_orders = mock_alpaca.list_orders(status="open")
    assert any(o.id == order.id for o in open_orders)

    # Cancel order
    cancelled = mock_alpaca.cancel_order(order.id)
    assert cancelled is True
    cancelled_order = mock_alpaca.get_order(order.id)
    assert cancelled_order.status == AlpacaOrderStatus.CANCELED.value


def test_market_data_abstraction_layer():
    """
    Requirement 6 & 8: Market data abstraction and explicit provenance badges.
    """
    router = UnifiedMarketDataRouter()

    # Forex query
    fx_bars, fx_src, fx_badge = router.fetch_bars("EUR/USD", force_synthetic=True)
    assert len(fx_bars) > 50
    assert fx_src in (DataSourceType.SYNTHETIC_OFFLINE, DataSourceType.HISTORICAL_REAL)
    assert "SYNTHETIC" in fx_badge or "REAL" in fx_badge

    # Alpaca Equity query
    eq_bars, eq_src, eq_badge = router.fetch_bars("AAPL", force_synthetic=True)
    assert len(eq_bars) > 50
    assert eq_src in (DataSourceType.ALPACA_MOCK, DataSourceType.DELAYED_LIVE)
    assert "MOCK ALPACA" in eq_badge or "DELAYED" in eq_badge


def test_secret_hygiene_no_hardcoded_keys():
    """
    Requirement 9: API secrets are never stored in source code.
    """
    settings = AlpacaSettings()
    # By default, without environment variables, keys are empty strings
    if "ALPACA_API_KEY" not in os.environ:
        assert settings.api_key == ""
    if "ALPACA_SECRET_KEY" not in os.environ:
        assert settings.secret_key == ""


def test_sqlite_persistence_for_mock_data(temp_db):
    """
    Requirement 10: Mock orders and positions persist across instances.
    """
    client = MockAlpacaClient(db=temp_db)
    client.submit_order(symbol="NVDA", qty=15.0, side="buy", order_type="market")

    # Verify directly from DB
    orders_in_db = temp_db.get_alpaca_orders()
    assert len(orders_in_db) >= 1
    assert orders_in_db[0]["symbol"] == "NVDA"

    positions_in_db = temp_db.get_alpaca_positions()
    assert len(positions_in_db) == 1
    assert positions_in_db[0]["symbol"] == "NVDA"
    assert positions_in_db[0]["qty"] == 15.0

