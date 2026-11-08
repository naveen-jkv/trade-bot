"""
Risk Validation Engine & Money Management.
The non-negotiable firewall of the autonomous trading system.
Enforces hard safety rules, position sizing, drawdown ceilings,
and risk-to-reward ratio thresholds before any order touches the simulator.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from config import (
    RiskSettings,
    SAFETY_LIMITS,
    SUPPORTED_PAIRS,
    ForexPairConfig,
    DEFAULT_CONFIG,
)
from core.asset_classes import is_alpaca_symbol, is_forex_symbol
from core.alerting import default_alert_manager, AlertEvent, AlertSeverity


@dataclass
class RiskValidationResult:
    """Outcome of risk assessment on a proposed trade."""
    is_approved: bool
    symbol: str
    action: str                        # 'BUY', 'SELL', 'HOLD'
    approved_lot_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount_dollars: float
    risk_reward_ratio: float
    reasons: List[str] = field(default_factory=list)
    risk_score: float = 0.0            # 0.0 (safe) to 1.0 (extreme risk)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_approved": self.is_approved,
            "symbol": self.symbol,
            "action": self.action,
            "approved_lot_size": round(self.approved_lot_size, 2),
            "entry_price": round(self.entry_price, 5),
            "stop_loss": round(self.stop_loss, 5),
            "take_profit": round(self.take_profit, 5),
            "risk_amount_dollars": round(self.risk_amount_dollars, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "reasons": self.reasons,
            "risk_score": round(self.risk_score, 2),
        }


class RiskManager:
    """
    Mathematical risk control engine.
    The AI agent can suggest trades, but ONLY the RiskManager can approve them.
    No prompt injection or LLM reasoning can override these deterministic limits.
    """

    def __init__(self, settings: Optional[RiskSettings] = None):
        self.settings = settings or DEFAULT_CONFIG.risk
        self.settings.validate_against_safety()
        self.kill_switch_active: bool = False

    def activate_kill_switch(self) -> None:
        """Immediately halts all order approvals."""
        self.kill_switch_active = True
        default_alert_manager.send_alert(
            event=AlertEvent.KILL_SWITCH_ENGAGED,
            severity=AlertSeverity.CRITICAL,
            title="Trading Kill Switch Engaged",
            message="Kill switch has been activated. All new order submissions and approvals are locked.",
        )

    def deactivate_kill_switch(self) -> None:
        """Resumes normal risk evaluation."""
        self.kill_switch_active = False
        default_alert_manager.send_alert(
            event=AlertEvent.KILL_SWITCH_DISENGAGED,
            severity=AlertSeverity.INFO,
            title="Trading Kill Switch Disengaged",
            message="Kill switch has been deactivated. Normal risk assessment resumed.",
        )

    def update_settings(self, new_settings: RiskSettings) -> None:
        """Updates configurable parameters after validating against hard safety limits."""
        new_settings.validate_against_safety()
        self.settings = new_settings

    def calculate_lot_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        current_equity: float,
    ) -> Tuple[float, float]:
        """
        Calculates position size in standard lots based on fixed risk percentage of equity.
        Returns: (lots, risk_in_dollars)
        """
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS["EUR/USD"])
        risk_pct = min(self.settings.risk_per_trade_pct, SAFETY_LIMITS["ABSOLUTE_MAX_RISK_PER_TRADE_PCT"])
        risk_dollars = current_equity * (risk_pct / 100.0)

        price_diff = abs(entry_price - stop_loss)
        if price_diff <= 0:
            return 0.0, 0.0

        pips_at_risk = price_diff / pair_cfg.pip_size
        if pips_at_risk <= 0:
            return 0.0, 0.0

        # Pip value per standard lot (100,000 units)
        # For XXX/USD pairs, 1 pip per lot = $10.00
        # For USD/JPY, 1 pip per lot = $1,000 / entry_price ~= $6.50
        # For USD/CAD, 1 pip per lot = $10 / entry_price
        if symbol.endswith("/USD"):
            pip_value_per_lot = 10.0
        elif symbol.startswith("USD/"):
            pip_value_per_lot = 10.0 / entry_price if entry_price > 0 else 10.0
        else:
            pip_value_per_lot = 10.0

        dollars_per_pip_needed = risk_dollars / pips_at_risk
        raw_lots = dollars_per_pip_needed / pip_value_per_lot

        # Clamp between min_lot and max_lot
        clamped_lots = max(pair_cfg.min_lot, min(raw_lots, pair_cfg.max_lot))
        # Round to 2 decimal places (standard micro-lot resolution)
        final_lots = round(clamped_lots, 2)

        actual_risk_dollars = final_lots * pip_value_per_lot * pips_at_risk
        return final_lots, actual_risk_dollars

    def validate_proposal(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        current_equity: float,
        initial_balance: float,
        peak_equity: float,
        open_positions: List[Dict[str, Any]],
        current_spread_pips: float = 1.5,
        trend_direction: str = "NEUTRAL",
        day_start_equity: Optional[float] = None,
    ) -> RiskValidationResult:
        """
        Strictly audits a proposed trading action.
        Verifies:
        1. Action validity (BUY/SELL)
        2. Maximum open positions cap
        3. Existing exposure on same symbol
        4. Currency-level directional exposure limit
        5. Current drawdown vs daily / total drawdown ceilings
        6. Spread reasonableness
        7. Stop-loss placement & validity
        8. Minimum Risk-to-Reward ratio
        9. Counter-trend filters
        """
        reasons: List[str] = []
        is_approved = True
        risk_score = 0.1

        if self.kill_switch_active:
            return RiskValidationResult(
                is_approved=False,
                symbol=symbol,
                action=action,
                approved_lot_size=0.0,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_amount_dollars=0.0,
                risk_reward_ratio=0.0,
                reasons=["Emergency Kill Switch is ACTIVE. All trading halted."],
                risk_score=1.0,
            )

        if action not in ["BUY", "SELL"]:
            return RiskValidationResult(
                is_approved=False,
                symbol=symbol,
                action=action,
                approved_lot_size=0.0,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_amount_dollars=0.0,
                risk_reward_ratio=0.0,
                reasons=["Action must be BUY or SELL."],
                risk_score=0.0,
            )

        # 1. Check Maximum Open Positions
        max_positions = min(self.settings.max_open_positions, SAFETY_LIMITS["MAX_OPEN_POSITIONS"])
        if len(open_positions) >= max_positions:
            is_approved = False
            reasons.append(f"Max open positions reached ({len(open_positions)}/{max_positions}).")
            risk_score += 0.3

        # 2. Check for Duplicate Orders on the same symbol
        for pos in open_positions:
            if pos.get("symbol") == symbol:
                is_approved = False
                reasons.append(f"Already holding an active position on {symbol}.")
                risk_score += 0.2
                break

        # 3. Check Currency-Level Exposure Limits (Prevents correlated cluster risk)
        currencies_in_pair = symbol.split("/") if "/" in symbol else []
        if len(currencies_in_pair) == 2:
            base_curr, quote_curr = currencies_in_pair[0], currencies_in_pair[1]
            proposed_long = base_curr if action == "BUY" else quote_curr
            proposed_short = quote_curr if action == "BUY" else base_curr

            long_count = 1
            short_count = 1
            for pos in open_positions:
                pos_sym = pos.get("symbol", "")
                pos_act = pos.get("action", "")
                if "/" in pos_sym:
                    b, q = pos_sym.split("/")
                    p_long = b if pos_act == "BUY" else q
                    p_short = q if pos_act == "BUY" else b
                    if p_long == proposed_long:
                        long_count += 1
                    if p_short == proposed_short:
                        short_count += 1

            max_curr = getattr(self.settings, "max_currency_exposure", SAFETY_LIMITS.get("MAX_CURRENCY_EXPOSURE", 2))
            if long_count > max_curr:
                is_approved = False
                reasons.append(
                    f"Currency exposure limit breached: too many Long {proposed_long} exposures ({long_count} > max {max_curr})."
                )
                risk_score += 0.25
            if short_count > max_curr:
                is_approved = False
                reasons.append(
                    f"Currency exposure limit breached: too many Short {proposed_short} exposures ({short_count} > max {max_curr})."
                )
                risk_score += 0.25

        # 4. Check Peak Drawdown (Total Drawdown)
        if peak_equity > 0:
            total_drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100.0
            max_allowed_dd = min(self.settings.max_total_drawdown_pct, SAFETY_LIMITS["ABSOLUTE_MAX_TOTAL_DRAWDOWN_PCT"])
            if total_drawdown_pct >= max_allowed_dd:
                is_approved = False
                reasons.append(
                    f"Total Drawdown limit breached: {total_drawdown_pct:.1f}% >= {max_allowed_dd:.1f}%. Trading halted."
                )
                risk_score = 1.0

        # 5. Check Daily Drawdown from Day-Start Baseline (resets each calendar day)
        daily_baseline = day_start_equity if (day_start_equity is not None and day_start_equity > 0) else initial_balance
        if daily_baseline > 0:
            daily_loss_pct = ((daily_baseline - current_equity) / daily_baseline) * 100.0
            max_daily_dd = min(self.settings.max_daily_drawdown_pct, SAFETY_LIMITS["ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT"])
            if daily_loss_pct >= max_daily_dd:
                is_approved = False
                reasons.append(
                    f"Daily Drawdown ceiling reached: {daily_loss_pct:.1f}% >= {max_daily_dd:.1f}% (Baseline Equity: ${daily_baseline:,.2f}). Cooldown active."
                )
                risk_score = 1.0
                default_alert_manager.send_alert(
                    event=AlertEvent.DAILY_DRAWDOWN_BREACH,
                    severity=AlertSeverity.CRITICAL,
                    title="Daily Drawdown Ceiling Breached",
                    message=f"Current daily loss {daily_loss_pct:.1f}% exceeded max limit {max_daily_dd:.1f}% against baseline ${daily_baseline:,.2f}.",
                    metadata={"current_equity": current_equity, "daily_baseline": daily_baseline, "daily_loss_pct": daily_loss_pct},
                )

        # 5. Check Spread Ceiling
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS["EUR/USD"])
        max_allowed_spread = pair_cfg.typical_spread_pips * 2.5
        if current_spread_pips > max_allowed_spread:
            is_approved = False
            reasons.append(
                f"Market spread too wide: {current_spread_pips:.1f} pips > max {max_allowed_spread:.1f} pips."
            )
            risk_score += 0.25

        # 6. Validate Stop Loss & Take Profit Direction
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = abs(take_profit - entry_price)

        if action == "BUY":
            if stop_loss >= entry_price:
                is_approved = False
                reasons.append(f"Invalid BUY Stop Loss ({stop_loss:.5f} >= Entry {entry_price:.5f}).")
            if take_profit <= entry_price:
                is_approved = False
                reasons.append(f"Invalid BUY Take Profit ({take_profit:.5f} <= Entry {entry_price:.5f}).")
        elif action == "SELL":
            if stop_loss <= entry_price:
                is_approved = False
                reasons.append(f"Invalid SELL Stop Loss ({stop_loss:.5f} <= Entry {entry_price:.5f}).")
            if take_profit >= entry_price:
                is_approved = False
                reasons.append(f"Invalid SELL Take Profit ({take_profit:.5f} >= Entry {entry_price:.5f}).")

        # 7. Validate Risk-to-Reward Ratio
        rr_ratio = (tp_distance / sl_distance) if sl_distance > 0 else 0.0
        min_rr = max(self.settings.min_risk_reward_ratio, SAFETY_LIMITS["ABSOLUTE_MIN_RR_RATIO"])
        if rr_ratio < min_rr:
            is_approved = False
            reasons.append(
                f"Risk-to-Reward ratio too low: {rr_ratio:.2f} < minimum required {min_rr:.2f}."
            )
            risk_score += 0.2

        # 8. Trend Filter Verification
        if not self.settings.allow_counter_trend:
            if action == "BUY" and trend_direction in ["BEARISH", "STRONG_BEARISH"]:
                is_approved = False
                reasons.append(f"Counter-trend BUY rejected: Market trend is {trend_direction}.")
                risk_score += 0.2
            elif action == "SELL" and trend_direction in ["BULLISH", "STRONG_BULLISH"]:
                is_approved = False
                reasons.append(f"Counter-trend SELL rejected: Market trend is {trend_direction}.")
                risk_score += 0.2

        # 9. Calculate Approved Lot Size
        lot_size, risk_dollars = self.calculate_lot_size(symbol, entry_price, stop_loss, current_equity)
        if lot_size <= 0:
            is_approved = False
            reasons.append("Calculated lot size is zero.")

        if is_approved:
            reasons.append("Risk validation PASSED. All safety constraints satisfied.")

        return RiskValidationResult(
            is_approved=is_approved,
            symbol=symbol,
            action=action,
            approved_lot_size=lot_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount_dollars=risk_dollars,
            risk_reward_ratio=rr_ratio,
            reasons=reasons,
            risk_score=min(1.0, risk_score),
        )

    def validate_alpaca_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        buying_power: float,
        max_order_usd: float = 1000.0,
    ) -> Tuple[bool, List[str], float]:
        """
        Audits an Alpaca order (US Equity or Crypto) against real-money risk rules.
        Returns:
            Tuple[is_approved, reasons, notional_value]
        """
        reasons: List[str] = []
        is_approved = True

        if self.kill_switch_active:
            return False, ["EMERGENCY KILL SWITCH IS ACTIVE. Order rejected."], 0.0

        if is_forex_symbol(symbol):
            return False, [f"{symbol} is a Forex pair. Cannot route to Alpaca."], 0.0

        if not is_alpaca_symbol(symbol):
            reasons.append(f"Warning: {symbol} is not in standard Alpaca asset catalog.")

        if side.lower() not in ("buy", "sell"):
            return False, ["Side must be 'buy' or 'sell'."], 0.0

        if qty <= 0:
            return False, ["Order quantity must be positive."], 0.0

        if price <= 0:
            return False, ["Estimated price must be greater than zero."], 0.0

        notional_value = qty * price

        # Hard dollar cap per order
        if notional_value > max_order_usd:
            is_approved = False
            reasons.append(
                f"Order notional value (${notional_value:,.2f}) exceeds MAX_ORDER_USD limit (${max_order_usd:,.2f})."
            )

        # Buying power check for BUY orders
        if side.lower() == "buy" and notional_value > buying_power:
            is_approved = False
            reasons.append(
                f"Insufficient buying power: Required ${notional_value:,.2f} > Available ${buying_power:,.2f}."
            )

        if is_approved:
            reasons.append("Risk validation PASSED for Alpaca order.")

        return is_approved, reasons, round(notional_value, 2)
