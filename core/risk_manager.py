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
    ) -> RiskValidationResult:
        """
        Strictly audits a proposed trading action.
        Verifies:
        1. Action validity (BUY/SELL)
        2. Maximum open positions cap
        3. Existing exposure on same symbol
        4. Current drawdown vs daily / total drawdown ceilings
        5. Spread reasonableness
        6. Stop-loss placement & validity
        7. Minimum Risk-to-Reward ratio
        8. Counter-trend filters
        """
        reasons: List[str] = []
        is_approved = True
        risk_score = 0.1

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

        # 3. Check Peak Drawdown (Total Drawdown)
        if peak_equity > 0:
            total_drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100.0
            max_allowed_dd = min(self.settings.max_total_drawdown_pct, SAFETY_LIMITS["ABSOLUTE_MAX_TOTAL_DRAWDOWN_PCT"])
            if total_drawdown_pct >= max_allowed_dd:
                is_approved = False
                reasons.append(
                    f"Total Drawdown limit breached: {total_drawdown_pct:.1f}% >= {max_allowed_dd:.1f}%. Trading halted."
                )
                risk_score = 1.0

        # 4. Check Daily Drawdown from initial balance reference
        if initial_balance > 0:
            daily_loss_pct = ((initial_balance - current_equity) / initial_balance) * 100.0
            max_daily_dd = min(self.settings.max_daily_drawdown_pct, SAFETY_LIMITS["ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT"])
            if daily_loss_pct >= max_daily_dd:
                is_approved = False
                reasons.append(
                    f"Daily Drawdown ceiling reached: {daily_loss_pct:.1f}% >= {max_daily_dd:.1f}%. Cooldown active."
                )
                risk_score = 1.0

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
