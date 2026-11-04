"""
Unit Tests for Risk Validation Firewall.
"""

import pytest
from core.risk_manager import RiskManager, RiskValidationResult
from config import RiskSettings, SAFETY_LIMITS


def test_lot_size_calculation():
    rm = RiskManager()
    lots, risk_dollars = rm.calculate_lot_size(
        symbol="EUR/USD",
        entry_price=1.1000,
        stop_loss=1.0950,  # 50 pips SL
        current_equity=10000.0,
    )
    # 1.5% of $10,000 = $150
    # 50 pips on EUR/USD: 1 lot = $500, so $150 / $500 = 0.30 lots
    assert lots > 0
    assert 0.25 <= lots <= 0.35
    assert risk_dollars <= 10000.0 * (SAFETY_LIMITS["ABSOLUTE_MAX_RISK_PER_TRADE_PCT"] / 100.0)


def test_reject_low_risk_reward_ratio():
    rm = RiskManager()
    res = rm.validate_proposal(
        symbol="EUR/USD",
        action="BUY",
        entry_price=1.1000,
        stop_loss=1.0900,   # 100 pips risk
        take_profit=1.1050, # 50 pips reward (RR = 0.5, below min 1.5)
        current_equity=10000.0,
        initial_balance=10000.0,
        peak_equity=10000.0,
        open_positions=[],
        trend_direction="BULLISH",
    )
    assert not res.is_approved
    assert any("Risk-to-Reward ratio too low" in r for r in res.reasons)


def test_reject_max_open_positions():
    rm = RiskManager()
    existing_positions = [
        {"symbol": "GBP/USD"},
        {"symbol": "USD/JPY"},
        {"symbol": "AUD/USD"},
    ]
    res = rm.validate_proposal(
        symbol="EUR/USD",
        action="BUY",
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100, # RR = 2.0
        current_equity=10000.0,
        initial_balance=10000.0,
        peak_equity=10000.0,
        open_positions=existing_positions,
        trend_direction="BULLISH",
    )
    assert not res.is_approved
    assert any("Max open positions reached" in r for r in res.reasons)


def test_reject_when_drawdown_limit_breached():
    rm = RiskManager()
    # Current equity $9,000 against peak $10,000 = 10% drawdown (exceeds default 8% max total drawdown)
    res = rm.validate_proposal(
        symbol="EUR/USD",
        action="BUY",
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        current_equity=9000.0,
        initial_balance=10000.0,
        peak_equity=10000.0,
        open_positions=[],
        trend_direction="BULLISH",
    )
    assert not res.is_approved
    assert any("Drawdown" in r for r in res.reasons)


def test_approve_valid_proposal():
    rm = RiskManager()
    res = rm.validate_proposal(
        symbol="EUR/USD",
        action="BUY",
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,  # RR = 2.0
        current_equity=10000.0,
        initial_balance=10000.0,
        peak_equity=10000.0,
        open_positions=[],
        trend_direction="BULLISH",
    )
    assert res.is_approved
    assert res.approved_lot_size > 0
    assert any("PASSED" in r for r in res.reasons)
