"""
Comprehensive Tests for Live & Real-Money Alpaca Trading Architecture.
Verifies:
1. Live & Paper endpoint routing (https://api.alpaca.markets vs https://paper-api.alpaca.markets).
2. Immutable safety locks: Live real-money trading is blocked unless explicitly confirmed.
3. Emergency Kill Switch and dollar cap protections.
4. Asset class strict segregation (rejects Forex).
5. REST v2 schema mapping (account, orders, positions, clock).
6. Factory instantiation across MOCK, PAPER, and LIVE modes.
7. Panic Emergency Flatten actions.
"""

from unittest.mock import MagicMock, patch
import os
from pathlib import Path
import tempfile
import pytest

from config import SystemConfig, AlpacaSettings
from database.memory import TradingMemoryDB
from core.alpaca_interface import AlpacaClientInterface
from core.alpaca_live_client import LiveAlpacaClient, AlpacaAPIError
from core.alpaca_mock_client import MockAlpacaClient
from core.alpaca_factory import create_alpaca_client
from core.alpaca_models import AlpacaOrderStatus, AlpacaAccount, AlpacaOrder, AlpacaPosition
from core.risk_manager import RiskManager


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
def live_client_paper(temp_db):
    cfg = SystemConfig(
        db_path=temp_db.db_path,
        alpaca=AlpacaSettings(
            api_key="PK_TEST_KEY_123",
            secret_key="SK_TEST_SECRET_456",
            trading_mode="PAPER",
            confirm_real_money=False,
            max_order_usd=1000.0,
        ),
    )
    return LiveAlpacaClient(config=cfg, db=temp_db, is_paper=True)


@pytest.fixture
def live_client_real_money(temp_db):
    cfg = SystemConfig(
        db_path=temp_db.db_path,
        alpaca=AlpacaSettings(
            api_key="AK_REAL_KEY_123",
            secret_key="SK_REAL_SECRET_456",
            trading_mode="LIVE",
            confirm_real_money=True,
            max_order_usd=500.0,
        ),
    )
    return LiveAlpacaClient(config=cfg, db=temp_db, is_paper=False)


def test_live_client_paper_endpoint_and_headers(live_client_paper):
    """Verifies that Paper mode targets the paper endpoint and sets auth headers."""
    assert live_client_paper.is_paper is True
    assert live_client_paper.base_url == "https://paper-api.alpaca.markets"
    assert live_client_paper.is_live_connected is True
    assert live_client_paper.broker_connectivity_enabled is True
    assert live_client_paper.session.headers["APCA-API-KEY-ID"] == "PK_TEST_KEY_123"
    assert live_client_paper.session.headers["APCA-API-SECRET-KEY"] == "SK_TEST_SECRET_456"


def test_live_client_real_money_endpoint(live_client_real_money):
    """Verifies that Real Money mode targets the live endpoint."""
    assert live_client_real_money.is_paper is False
    assert live_client_real_money.base_url == "https://api.alpaca.markets"
    assert live_client_real_money.confirm_real_money is True
    assert live_client_real_money.max_order_usd == 500.0


def test_real_money_safety_lock_unconfirmed(temp_db):
    """
    CRITICAL SAFETY TRIPWIRE:
    If trading_mode is LIVE (Real Money) but confirm_real_money is False,
    order submission MUST be rejected and never transmitted to the broker.
    """
    cfg = SystemConfig(
        db_path=temp_db.db_path,
        alpaca=AlpacaSettings(
            api_key="AK_REAL_KEY_123",
            secret_key="SK_REAL_SECRET_456",
            trading_mode="LIVE",
            confirm_real_money=False,  # Unconfirmed!
        ),
    )
    client = LiveAlpacaClient(config=cfg, db=temp_db, is_paper=False)

    order = client.submit_order(symbol="AAPL", qty=1.0, side="buy")
    assert order.status == AlpacaOrderStatus.REJECTED.value
    assert "REAL MONEY SAFETY LOCK ACTIVATED" in order.rejection_reason


def test_emergency_kill_switch_blocks_orders(live_client_real_money):
    """Verifies that enabling the kill switch blocks all order submissions."""
    live_client_real_money.set_kill_switch(True)
    order = live_client_real_money.submit_order(symbol="NVDA", qty=2.0, side="buy")
    assert order.status == AlpacaOrderStatus.REJECTED.value
    assert "KILL SWITCH" in order.rejection_reason


def test_forex_symbol_rejected_by_alpaca(live_client_paper):
    """Verifies asset segregation: Forex pairs submitted to Alpaca are rejected."""
    order = live_client_paper.submit_order(symbol="EUR/USD", qty=1000.0, side="buy")
    assert order.status == AlpacaOrderStatus.REJECTED.value
    assert "Forex" in order.rejection_reason


def test_mocked_get_account(live_client_real_money):
    """Tests fetching live account details and verifying data model mapping."""
    mock_payload = {
        "id": "ACC-REAL-001",
        "account_number": "123456789",
        "status": "ACTIVE",
        "currency": "USD",
        "cash": "25000.50",
        "portfolio_value": "32000.75",
        "buying_power": "50000.00",
        "regt_buying_power": "50000.00",
        "daytrading_buying_power": "100000.00",
        "non_marginable_buying_power": "25000.50",
        "pattern_day_trader": False,
        "equity": "32000.75",
        "last_equity": "31500.00",
        "long_market_value": "7000.25",
        "short_market_value": "0.00",
        "daytrade_count": 1,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch.object(live_client_real_money.session, "request", return_value=mock_resp):
        acct = live_client_real_money.get_account()
        assert isinstance(acct, AlpacaAccount)
        assert acct.cash == 25000.50
        assert acct.equity == 32000.75
        assert acct.buying_power == 50000.00
        assert acct.is_live_broker is True


def test_mocked_submit_order_success(live_client_real_money, temp_db):
    """Tests submitting a live order with successful broker response."""
    mock_order_payload = {
        "id": "ORD-LIVE-777",
        "client_order_id": "alpaca_test_cid",
        "created_at": "2026-11-08T12:00:00Z",
        "updated_at": "2026-11-08T12:00:00Z",
        "submitted_at": "2026-11-08T12:00:00Z",
        "filled_at": None,
        "symbol": "AAPL",
        "asset_class": "us_equity",
        "qty": "5.0",
        "filled_qty": "0.0",
        "filled_avg_price": None,
        "order_type": "market",
        "type": "market",
        "side": "buy",
        "time_in_force": "day",
        "status": "new",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = mock_order_payload

    with patch.object(live_client_real_money.session, "request", return_value=mock_resp):
        order = live_client_real_money.submit_order(symbol="AAPL", qty=5.0, side="buy")
        assert isinstance(order, AlpacaOrder)
        assert order.id == "ORD-LIVE-777"
        assert order.symbol == "AAPL"
        assert order.status == "new"

        # Verify persisted in DB
        db_orders = temp_db.get_alpaca_orders()
        assert any(o["id"] == "ORD-LIVE-777" for o in db_orders)


def test_mocked_list_and_close_positions(live_client_paper):
    """Tests listing positions and closing an active position."""
    mock_pos_payload = [
        {
            "asset_id": "TSLA",
            "symbol": "TSLA",
            "exchange": "NASDAQ",
            "asset_class": "us_equity",
            "avg_entry_price": "250.00",
            "qty": "4.0",
            "side": "long",
            "market_value": "1040.00",
            "cost_basis": "1000.00",
            "unrealized_pl": "40.00",
            "unrealized_plpc": "0.04",
            "current_price": "260.00",
            "lastday_price": "248.00",
            "change_today": "12.00",
            "qty_available": "4.0",
        }
    ]

    mock_resp_list = MagicMock()
    mock_resp_list.status_code = 200
    mock_resp_list.json.return_value = mock_pos_payload

    mock_resp_close = MagicMock()
    mock_resp_close.status_code = 200
    mock_resp_close.json.return_value = {
        "id": "ORD-CLOSE-999",
        "symbol": "TSLA",
        "qty": "4.0",
        "side": "sell",
        "order_type": "market",
        "status": "filled",
    }

    def mocked_requests(method, url, **kwargs):
        if method == "GET" and "/v2/positions" in url:
            return mock_resp_list
        elif method == "DELETE" and "/v2/positions/TSLA" in url:
            return mock_resp_close
        return MagicMock(status_code=404)

    with patch.object(live_client_paper.session, "request", side_effect=mocked_requests):
        positions = live_client_paper.list_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "TSLA"
        assert positions[0].qty == 4.0

        closed_order = live_client_paper.close_position("TSLA")
        assert closed_order is not None
        assert closed_order.id == "ORD-CLOSE-999"


def test_emergency_panic_close_all_positions(live_client_real_money):
    """Tests the panic button action liquidating all positions and cancelling orders."""
    mock_resp = MagicMock()
    mock_resp.status_code = 207
    mock_resp.json.return_value = [
        {"status": 200, "body": {"id": "LIQ-1", "symbol": "AAPL", "qty": "10", "side": "sell", "status": "filled"}},
        {"status": 200, "body": {"id": "LIQ-2", "symbol": "MSFT", "qty": "5", "side": "sell", "status": "filled"}},
    ]

    with patch.object(live_client_real_money.session, "request", return_value=mock_resp):
        orders = live_client_real_money.close_all_positions(cancel_orders=True)
        assert len(orders) == 2
        assert orders[0].id == "LIQ-1"
        assert orders[1].id == "LIQ-2"


def test_alpaca_factory_selection(temp_db):
    """Verifies that create_alpaca_client produces the correct implementation."""
    # 1. Mode MOCK -> MockAlpacaClient
    cfg_mock = SystemConfig(alpaca=AlpacaSettings(trading_mode="MOCK"))
    client_mock = create_alpaca_client(config=cfg_mock, db=temp_db)
    assert isinstance(client_mock, MockAlpacaClient)

    # 2. Mode PAPER without keys -> MockAlpacaClient (safe fallback)
    cfg_paper_no_keys = SystemConfig(alpaca=AlpacaSettings(trading_mode="PAPER", api_key="", secret_key=""))
    client_fallback = create_alpaca_client(config=cfg_paper_no_keys, db=temp_db)
    assert isinstance(client_fallback, MockAlpacaClient)

    # 3. Mode PAPER with keys -> LiveAlpacaClient (is_paper=True)
    cfg_paper = SystemConfig(alpaca=AlpacaSettings(trading_mode="PAPER", api_key="PK_1", secret_key="SK_1"))
    client_paper = create_alpaca_client(config=cfg_paper, db=temp_db)
    assert isinstance(client_paper, LiveAlpacaClient)
    assert client_paper.is_paper is True

    # 4. Mode LIVE with keys -> LiveAlpacaClient (is_paper=False)
    cfg_live = SystemConfig(alpaca=AlpacaSettings(trading_mode="LIVE", api_key="AK_1", secret_key="SK_1"))
    client_live = create_alpaca_client(config=cfg_live, db=temp_db)
    assert isinstance(client_live, LiveAlpacaClient)
    assert client_live.is_paper is False


def test_risk_manager_validate_alpaca_order():
    """Verifies that the Risk Manager audits real-money Alpaca orders properly."""
    rm = RiskManager()

    # 1. Normal order within budget
    ok, reasons, notional = rm.validate_alpaca_order(
        symbol="AAPL",
        side="buy",
        qty=2.0,
        price=220.0,
        buying_power=5000.0,
        max_order_usd=1000.0,
    )
    assert ok is True
    assert notional == 440.0

    # 2. Order exceeding max_order_usd
    ok_exceed, reasons_exceed, _ = rm.validate_alpaca_order(
        symbol="AAPL",
        side="buy",
        qty=10.0,
        price=220.0,
        buying_power=5000.0,
        max_order_usd=1000.0,  # 10 * 220 = $2,200 > $1,000
    )
    assert ok_exceed is False
    assert any("MAX_ORDER_USD" in r for r in reasons_exceed)

    # 3. Order exceeding buying power
    ok_bp, reasons_bp, _ = rm.validate_alpaca_order(
        symbol="TSLA",
        side="buy",
        qty=3.0,
        price=250.0,
        buying_power=500.0,  # 750 > 500
        max_order_usd=2000.0,
    )
    assert ok_bp is False
    assert any("buying power" in r.lower() for r in reasons_bp)

    # 4. Order with kill switch active
    rm.activate_kill_switch()
    ok_ks, reasons_ks, _ = rm.validate_alpaca_order(
        symbol="NVDA",
        side="buy",
        qty=1.0,
        price=130.0,
        buying_power=10000.0,
    )
    assert ok_ks is False
    assert any("KILL SWITCH" in r for r in reasons_ks)

