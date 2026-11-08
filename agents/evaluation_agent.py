"""
Performance Evaluation Agent.
Conducts rigorous retrospective self-evaluations after trades and cycles.
Evaluates prediction accuracy against subsequent price action, diagnoses mistakes,
formulates persistent lessons, and executes chronological out-of-sample strategy backtests.
Strictly prohibited from modifying hard safety rules.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from core.paper_trader import ClosedTrade
from database.memory import TradingMemoryDB
from config import DEFAULT_CONFIG, SAFETY_LIMITS

logger = logging.getLogger("PerformanceEvaluationAgent")


@dataclass
class EvaluationReport:
    """Retrospective analysis of a completed trade or evaluation cycle."""
    ticket_id: str
    symbol: str
    action: str
    pnl: float
    pips: float
    exit_reason: str
    prediction_accuracy: bool
    mistake_category: str
    lesson_rule: str
    strategy_adaptation_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceEvaluationAgent:
    """
    Self-evaluating agent that enforces continuous empirical learning.
    Translates every win and loss into structured, permanent knowledge.
    """

    def __init__(self, db: Optional[TradingMemoryDB] = None):
        self.db = db or TradingMemoryDB(DEFAULT_CONFIG.db_path)

    def evaluate_trade(
        self,
        trade: ClosedTrade,
        market_regime: str = "NORMAL",
        prediction_id: Optional[int] = None,
    ) -> EvaluationReport:
        """
        Retrospective post-mortem on a closed paper trade:
        1. Compares prediction with realized market outcome.
        2. Categorizes outcome and identifies errors.
        3. Formulates a concrete rule for persistent memory.
        4. Updates prediction database record.
        """
        is_win = trade.pnl_dollars > 0
        actual_dir = "UP" if (trade.exit_price > trade.entry_price) else "DOWN"

        # 1. Update ML prediction evaluation in DB if linked
        if prediction_id:
            try:
                self.db.evaluate_prediction(
                    prediction_id=prediction_id,
                    actual_direction=actual_dir,
                    is_correct=is_win,
                )
            except Exception as e:
                logger.warning(f"Failed to evaluate prediction {prediction_id}: {e}")

        # 2. Diagnose mistakes and formulate rules
        mistake_cat = "NONE"
        lesson_rule = ""
        tags = []

        if is_win:
            mistake_cat = "DISCIPLINED_EXECUTION"
            tags.append("profitable_trade")
            if trade.exit_reason == "TAKE_PROFIT":
                lesson_rule = f"Patience rewarded: {trade.action} on {trade.symbol} achieved full target at {trade.take_profit:.5f}."
            else:
                lesson_rule = f"Favorable exit secured for {trade.action} on {trade.symbol} yielding +${trade.pnl_dollars:.2f}."
        else:
            tags.append("loss_analysis")
            if trade.exit_reason == "STOP_LOSS":
                tags.append("stop_loss_hit")
                if market_regime == "HIGH_VOLATILITY":
                    mistake_cat = "VOLATILITY_WHIPSAW"
                    tags.append("high_volatility")
                    lesson_rule = (
                        f"In HIGH_VOLATILITY regime for {trade.symbol}, standard stop-loss was vulnerable to noise. "
                        "Require wider ATR padding (>= 2.2x ATR) and smaller lot sizing during volatile expansions."
                    )
                elif market_regime == "COMPRESSED":
                    mistake_cat = "FALSE_BREAKOUT"
                    tags.append("compressed_regime")
                    lesson_rule = (
                        f"In COMPRESSED volatility regime, breakout on {trade.symbol} lacked follow-through. "
                        "Wait for confirmed second bar close above consolidation before entering."
                    )
                else:
                    mistake_cat = "COUNTER_MOMENTUM"
                    tags.append("trend_reversal")
                    lesson_rule = (
                        f"Adverse trend continuation hit Stop Loss on {trade.symbol}. "
                        "Strictly adhere to 50 EMA alignment and verify MACD histogram momentum before entering."
                    )
            else:
                mistake_cat = "EARLY_EXIT_OR_SLIPPAGE"
                lesson_rule = f"Manual or time-based liquidation on {trade.symbol} realized a loss of ${trade.pnl_dollars:.2f}."

        # 3. Store Lesson in Persistent Memory
        ts = datetime.now(timezone.utc).isoformat()
        try:
            self.db.store_lesson(
                symbol=trade.symbol,
                trigger_event="WIN_ANALYSIS" if is_win else "LOSS_ANALYSIS",
                regime=market_regime,
                mistake_analysis=mistake_cat,
                rule_learned=lesson_rule,
                outcome_pnl=trade.pnl_dollars,
                tags=",".join(tags),
                timestamp=ts,
            )
            self.db.store_log(
                agent_name="PerformanceEvaluationAgent",
                level="DECISION",
                message=f"Evaluated Trade {trade.ticket_id} ({trade.symbol} {trade.action}) -> Outcome: {'WIN' if is_win else 'LOSS'} (${trade.pnl_dollars:+.2f}) | Rule: {lesson_rule[:80]}...",
                metadata={"ticket_id": trade.ticket_id, "is_win": is_win, "pnl": trade.pnl_dollars},
            )
        except Exception as e:
            logger.warning(f"Failed to persist trade evaluation log for {trade.ticket_id}: {e}")

        return EvaluationReport(
            ticket_id=trade.ticket_id,
            symbol=trade.symbol,
            action=trade.action,
            pnl=trade.pnl_dollars,
            pips=trade.pips,
            exit_reason=trade.exit_reason,
            prediction_accuracy=is_win,
            mistake_category=mistake_cat,
            lesson_rule=lesson_rule,
            strategy_adaptation_status="PERSISTED_TO_MEMORY",
        )

    def run_historical_adaptation_test(
        self,
        df: pd.DataFrame,
        symbol: str,
        test_period_name: str = "Out-Of-Sample Chronological Slice",
        experimental_atr_mult: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Tests an experimental strategy parameter variation on a separate out-of-sample historical slice.
        Validates whether adjustments improve Profit Factor without breaching safety limits.
        The AI cannot change safety controls!
        """
        if len(df) < 50:
            return {"status": "INSUFFICIENT_DATA"}

        # Use the latter 30% of bars as out-of-sample testing
        split_idx = int(len(df) * 0.7)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        # Simple simulated walk-forward backtest
        trades = []
        in_pos = False
        pos_dir = ""
        entry_px = 0.0
        sl_px = 0.0
        tp_px = 0.0

        for i in range(25, len(test_df) - 1):
            row = test_df.iloc[i]
            c = row["close"]
            atr = row.get("atr_14", 0.0015)

            if in_pos:
                # Check exit
                if pos_dir == "BUY":
                    if row["low"] <= sl_px:
                        trades.append({"pnl": sl_px - entry_px, "win": False})
                        in_pos = False
                    elif row["high"] >= tp_px:
                        trades.append({"pnl": tp_px - entry_px, "win": True})
                        in_pos = False
                elif pos_dir == "SELL":
                    if row["high"] >= sl_px:
                        trades.append({"pnl": entry_px - sl_px, "win": False})
                        in_pos = False
                    elif row["low"] <= tp_px:
                        trades.append({"pnl": entry_px - tp_px, "win": True})
                        in_pos = False
            else:
                # Signal: EMA crossover + RSI confirmation evaluated on bar i close
                ema_9 = row.get("ema_9", c)
                ema_21 = row.get("ema_21", c)
                rsi = row.get("rsi_14", 50.0)
                next_bar = test_df.iloc[i + 1]
                next_open = next_bar["open"]

                if ema_9 > ema_21 and rsi > 52.0:
                    in_pos = True
                    pos_dir = "BUY"
                    entry_px = next_open  # Enter at next bar's open (avoids lookahead bias)
                    sl_px = entry_px - (atr * experimental_atr_mult)
                    tp_px = entry_px + (atr * experimental_atr_mult * 1.5)
                elif ema_9 < ema_21 and rsi < 48.0:
                    in_pos = True
                    pos_dir = "SELL"
                    entry_px = next_open  # Enter at next bar's open (avoids lookahead bias)
                    sl_px = entry_px + (atr * experimental_atr_mult)
                    tp_px = entry_px - (atr * experimental_atr_mult * 1.5)

        total_sim = len(trades)
        wins = [t for t in trades if t["win"]]
        losses = [t for t in trades if not t["win"]]
        win_rate = (len(wins) / total_sim * 100.0) if total_sim > 0 else 0.0

        gross_profit = sum(t["pnl"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)
        
        # Calculate dynamic peak-to-trough max drawdown on simulated equity curve
        if trades:
            eq_curve = [10000.0]
            for t in trades:
                dollar_pnl = t["pnl"] * 100000.0  # Standard lot notional
                eq_curve.append(eq_curve[-1] + dollar_pnl)
            eq_arr = np.array(eq_curve)
            running_peak = np.maximum.accumulate(eq_arr)
            drawdowns = np.where(running_peak > 0, (running_peak - eq_arr) / running_peak * 100.0, 0.0)
            max_dd = float(np.max(drawdowns))
        else:
            max_dd = 0.0

        # Validation criterion: Win rate >= 50% and Profit Factor >= 1.25 and Max DD <= 5.0%
        is_validated = (win_rate >= 50.0 and pf >= 1.25 and max_dd <= SAFETY_LIMITS["ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT"])

        params = {"experimental_atr_mult": experimental_atr_mult, "strategy": "EMA_Cross_RSI"}

        # Store strategy evaluation in database
        try:
            self.db.store_strategy_evaluation(
                strategy_name="EMA_Cross_Adaptive_ATR",
                test_period=test_period_name,
                win_rate=win_rate,
                profit_factor=pf,
                max_drawdown=max_dd,
                parameters=params,
                validated=is_validated,
            )
        except Exception as e:
            logger.warning(f"Failed to persist strategy evaluation: {e}")

        return {
            "strategy_name": "EMA_Cross_Adaptive_ATR",
            "test_period": test_period_name,
            "total_trades": total_sim,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(pf, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "is_validated": is_validated,
            "parameters": params,
        }
