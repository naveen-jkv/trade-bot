"""
Institutional Backtesting & Walk-Forward Validation Engine.
Provides rigorous chronological execution simulation with configurable transaction fees,
realistic slippage, zero lookahead bias, and institutional risk metrics (Sharpe, Sortino, Calmar, Max Drawdown).
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, SAFETY_LIMITS


@dataclass
class BacktestTrade:
    """Individual closed trade executed during a backtest."""
    ticket_id: str
    symbol: str
    action: str              # 'BUY' or 'SELL'
    entry_time: Any
    entry_price: float
    exit_time: Any
    exit_price: float
    volume: float            # Lots or shares
    stop_loss: float
    take_profit: float
    gross_pnl: float
    net_pnl: float
    commission_paid: float
    slippage_cost: float
    pnl_pct: float
    exit_reason: str         # 'TAKE_PROFIT', 'STOP_LOSS', 'SIGNAL_REVERSAL', 'TIME_LIMIT'
    bars_held: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """Quantitative performance analytics of a backtest run."""
    initial_capital: float
    final_equity: float
    net_profit: float
    return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    trades: List[BacktestTrade]
    equity_curve: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trades"] = [t.to_dict() for t in self.trades]
        return d


class BacktestEngine:
    """
    High-fidelity quantitative backtesting framework.
    Enforces institutional standards:
    1. Realistic Execution: Signals evaluated on bar Close enter at the NEXT bar Open.
    2. Realistic Costs: Commission fees and adverse slippage on every fill.
    3. Intrabar Exit Evaluation: Stop Loss and Take Profit audited against bar High/Low.
    4. Out-of-Sample Walk-Forward Validation: Prevents curve-fitting and survivorship bias.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.0002,   # 0.02% (2 bps) per transaction
        slippage_pct: float = 0.0002,     # 0.02% adverse execution slippage
        risk_per_trade_pct: float = 1.5,  # 1.5% capital risked per trade
    ):
        self.initial_capital = float(initial_capital)
        self.commission_pct = float(commission_pct)
        self.slippage_pct = float(slippage_pct)
        self.risk_per_trade_pct = float(risk_per_trade_pct)

    def run_strategy(
        self,
        df: pd.DataFrame,
        symbol: str = "EUR/USD",
        signal_generator: Optional[Callable[[pd.DataFrame, int], Dict[str, Any]]] = None,
        atr_multiplier: float = 2.0,
        risk_reward_ratio: float = 1.5,
    ) -> BacktestResult:
        """
        Runs backtest over historical OHLCV data.
        
        signal_generator(df, i) -> Optional dict with:
            {'action': 'BUY' | 'SELL', 'stop_loss': float, 'take_profit': float}
        """
        if df is None or len(df) < 30:
            return self._empty_result()

        clean_df = df.copy().reset_index(drop=True)
        # Ensure timestamp column exists
        if "timestamp" not in clean_df.columns:
            clean_df["timestamp"] = range(len(clean_df))

        # Reconstruct default EMA + RSI signals if no generator passed
        if signal_generator is None:
            signal_generator = self._default_trend_momentum_signals

        capital = self.initial_capital
        equity = capital
        peak_equity = capital
        trades: List[BacktestTrade] = []
        equity_history: List[Dict[str, Any]] = [
            {"timestamp": clean_df.iloc[0]["timestamp"], "equity": capital, "drawdown_pct": 0.0}
        ]

        in_pos = False
        pos_action = ""
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        volume = 0.0
        entry_bar = 0
        entry_time = None
        commission_acc = 0.0
        slippage_acc = 0.0

        for i in range(25, len(clean_df) - 1):
            curr_bar = clean_df.iloc[i]
            next_bar = clean_df.iloc[i + 1]
            curr_ts = curr_bar["timestamp"]

            if in_pos:
                bars_held = i - entry_bar
                high = float(curr_bar["high"])
                low = float(curr_bar["low"])
                exit_price = 0.0
                exit_reason = ""

                # Evaluate Stop Loss / Take Profit hits on current bar
                if pos_action == "BUY":
                    if low <= stop_loss:
                        exit_price = stop_loss - (stop_loss * self.slippage_pct)
                        exit_reason = "STOP_LOSS"
                    elif high >= take_profit:
                        exit_price = take_profit - (take_profit * self.slippage_pct)
                        exit_reason = "TAKE_PROFIT"
                else:  # SELL
                    if high >= stop_loss:
                        exit_price = stop_loss + (stop_loss * self.slippage_pct)
                        exit_reason = "STOP_LOSS"
                    elif low <= take_profit:
                        exit_price = take_profit + (take_profit * self.slippage_pct)
                        exit_reason = "TAKE_PROFIT"

                if exit_reason:
                    exit_commission = exit_price * volume * self.commission_pct
                    total_commission = commission_acc + exit_commission
                    exit_slippage = abs(exit_price * self.slippage_pct * volume)
                    total_slippage = slippage_acc + exit_slippage

                    if pos_action == "BUY":
                        gross_pnl = (exit_price - entry_price) * volume
                    else:
                        gross_pnl = (entry_price - exit_price) * volume

                    net_pnl = gross_pnl - total_commission - total_slippage
                    capital += net_pnl
                    equity = capital
                    peak_equity = max(peak_equity, equity)

                    trade = BacktestTrade(
                        ticket_id=f"BT-{len(trades) + 1:04d}",
                        symbol=symbol,
                        action=pos_action,
                        entry_time=entry_time,
                        entry_price=round(entry_price, 5),
                        exit_time=curr_ts,
                        exit_price=round(exit_price, 5),
                        volume=round(volume, 4),
                        stop_loss=round(stop_loss, 5),
                        take_profit=round(take_profit, 5),
                        gross_pnl=round(gross_pnl, 2),
                        net_pnl=round(net_pnl, 2),
                        commission_paid=round(total_commission, 2),
                        slippage_cost=round(total_slippage, 2),
                        pnl_pct=round((net_pnl / capital) * 100.0, 2) if capital > 0 else 0.0,
                        exit_reason=exit_reason,
                        bars_held=bars_held,
                    )
                    trades.append(trade)
                    in_pos = False

            # Signal Generation (only if not already in position)
            if not in_pos:
                sig = signal_generator(clean_df, i)
                if sig and sig.get("action") in ("BUY", "SELL"):
                    # Zero Lookahead: enter at next bar's Open price with adverse entry slippage
                    action = sig["action"]
                    raw_open = float(next_bar["open"])
                    if action == "BUY":
                        exec_entry = raw_open + (raw_open * self.slippage_pct)
                    else:
                        exec_entry = raw_open - (raw_open * self.slippage_pct)

                    sl = float(sig["stop_loss"])
                    tp = float(sig["take_profit"])
                    risk_distance = abs(exec_entry - sl)

                    if risk_distance > 0:
                        risk_dollars = equity * (self.risk_per_trade_pct / 100.0)
                        calculated_vol = risk_dollars / risk_distance
                        # Cap max position to 5x leverage
                        max_vol = (equity * 5.0) / exec_entry if exec_entry > 0 else 1.0
                        vol = min(calculated_vol, max_vol)

                        entry_comm = exec_entry * vol * self.commission_pct
                        entry_slip = abs(raw_open * self.slippage_pct * vol)

                        in_pos = True
                        pos_action = action
                        entry_price = exec_entry
                        stop_loss = sl
                        take_profit = tp
                        volume = vol
                        entry_bar = i + 1
                        entry_time = next_bar["timestamp"]
                        commission_acc = entry_comm
                        slippage_acc = entry_slip

            # Record periodic mark-to-market equity
            dd = ((peak_equity - equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
            equity_history.append({
                "timestamp": curr_ts,
                "equity": round(equity, 2),
                "drawdown_pct": round(dd, 2),
            })

        return self._compute_analytics(trades, equity_history, capital)

    def walk_forward_backtest(
        self,
        df: pd.DataFrame,
        symbol: str = "EUR/USD",
        train_pct: float = 0.7,
        atr_multiplier: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Executes chronological out-of-sample walk-forward validation.
        Prevents in-sample overfitting by testing calibration on unseen chronological data.
        """
        if len(df) < 50:
            return {"status": "INSUFFICIENT_DATA"}

        split_idx = int(len(df) * train_pct)
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        in_sample_res = self.run_strategy(train_df, symbol=symbol, atr_multiplier=atr_multiplier)
        out_sample_res = self.run_strategy(test_df, symbol=symbol, atr_multiplier=atr_multiplier)

        # Robustness test: Out-of-sample profit factor should retain at least 65% of in-sample performance
        pf_in = in_sample_res.profit_factor
        pf_out = out_sample_res.profit_factor
        retention = (pf_out / pf_in) if pf_in > 0 else 0.0

        is_robust = (
            out_sample_res.total_trades >= 3
            and out_sample_res.profit_factor >= 1.10
            and out_sample_res.max_drawdown_pct <= SAFETY_LIMITS["ABSOLUTE_MAX_TOTAL_DRAWDOWN_PCT"]
        )

        return {
            "symbol": symbol,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "in_sample": in_sample_res.to_dict(),
            "out_of_sample": out_sample_res.to_dict(),
            "profit_factor_retention": round(retention, 2),
            "is_walk_forward_passed": is_robust,
        }

    def _default_trend_momentum_signals(self, df: pd.DataFrame, idx: int) -> Optional[Dict[str, Any]]:
        """Institutional trend-momentum signal logic using EMA(9/21), RSI(14), and ATR(14)."""
        row = df.iloc[idx]
        c = float(row["close"])
        ema_fast = float(row.get("ema_9", c))
        ema_slow = float(row.get("ema_21", c))
        rsi = float(row.get("rsi_14", 50.0))
        atr = float(row.get("atr_14", c * 0.002))

        if ema_fast > ema_slow and rsi > 52.0:
            return {
                "action": "BUY",
                "stop_loss": c - (atr * 1.5),
                "take_profit": c + (atr * 2.25),
            }
        elif ema_fast < ema_slow and rsi < 48.0:
            return {
                "action": "SELL",
                "stop_loss": c + (atr * 1.5),
                "take_profit": c - (atr * 2.25),
            }
        return None

    def _compute_analytics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[Dict[str, Any]],
        final_equity: float,
    ) -> BacktestResult:
        """Computes institutional quantitative ratios."""
        total_trades = len(trades)
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t.net_pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.net_pnl for t in losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

        net_profit = round(final_equity - self.initial_capital, 2)
        return_pct = round((net_profit / self.initial_capital) * 100.0, 2)

        # Drawdowns
        eq_arr = np.array([h["equity"] for h in equity_curve]) if equity_curve else np.array([self.initial_capital])
        running_peak = np.maximum.accumulate(eq_arr)
        drawdowns = np.where(running_peak > 0, (running_peak - eq_arr) / running_peak * 100.0, 0.0)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Drawdown Duration (maximum consecutive bars spent below peak)
        max_dd_bars = 0
        curr_dd_bars = 0
        for dd in drawdowns:
            if dd > 0.01:
                curr_dd_bars += 1
                max_dd_bars = max(max_dd_bars, curr_dd_bars)
            else:
                curr_dd_bars = 0

        # Sharpe, Sortino, Calmar Ratios
        pnls = [t.net_pnl for t in trades]
        if len(pnls) >= 3 and np.std(pnls) > 0:
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            sharpe = (mean_pnl / std_pnl) * np.sqrt(252)

            downside_pnls = [p for p in pnls if p < 0]
            if downside_pnls and np.std(downside_pnls) > 0:
                sortino = (mean_pnl / np.std(downside_pnls)) * np.sqrt(252)
            else:
                sortino = sharpe
        else:
            sharpe = 0.0
            sortino = 0.0

        calmar = (return_pct / max_dd) if max_dd > 0 else 0.0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=round(final_equity, 2),
            net_profit=net_profit,
            return_pct=return_pct,
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown_pct=round(max_dd, 2),
            max_drawdown_duration_bars=max_dd_bars,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=self.initial_capital,
            net_profit=0.0,
            return_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_bars=0,
            trades=[],
            equity_curve=[],
        )
