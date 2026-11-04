"""
Paper Trading Simulation Engine.
Realistic Forex paper-trading environment with virtual account balance,
position tracking, spread/slippage costs, stop-loss and take-profit execution,
margin calculations, and institutional-grade performance analytics.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import uuid
import numpy as np
import pandas as pd

from config import (
    RiskSettings,
    SUPPORTED_PAIRS,
    ForexPairConfig,
    DEFAULT_CONFIG,
    DB_PATH,
)
from database.memory import TradingMemoryDB


@dataclass
class Position:
    """Represents an active paper trading position."""
    ticket_id: str
    symbol: str
    action: str              # 'BUY' or 'SELL'
    volume_lots: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: str
    spread_cost_dollars: float
    hypothesis_id: Optional[int] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pips: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedTrade:
    """Represents a finalized, closed paper trade."""
    ticket_id: str
    symbol: str
    action: str
    volume_lots: float
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: str
    exit_price: float
    pnl_dollars: float
    pips: float
    exit_reason: str         # 'TAKE_PROFIT', 'STOP_LOSS', 'MANUAL_EXIT', 'TIME_LIMIT'
    spread_cost_dollars: float
    hypothesis_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PaperTradingSimulator:
    """
    High-fidelity Forex paper trading simulator.
    Simulates real-world execution conditions including spread, slippage,
    and automatic stop-loss / take-profit fills on incoming price bars.
    """

    def __init__(
        self,
        initial_balance: Optional[float] = None,
        db: Optional[TradingMemoryDB] = None,
        risk_settings: Optional[RiskSettings] = None,
    ):
        self.settings = risk_settings or DEFAULT_CONFIG.risk
        self.initial_balance = initial_balance or self.settings.initial_balance
        self.balance = float(self.initial_balance)
        self.equity = float(self.initial_balance)
        self.peak_equity = float(self.initial_balance)
        self.margin_used = 0.0

        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.equity_history: List[Dict[str, Any]] = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "balance": self.balance,
                "equity": self.equity,
                "drawdown_pct": 0.0,
            }
        ]

        self.db = db or TradingMemoryDB(DEFAULT_CONFIG.db_path)
        self._load_existing_state_from_db()

    def _load_existing_state_from_db(self) -> None:
        """Restores closed trades and historical statistics from the SQLite persistent memory."""
        try:
            db_closed = self.db.get_closed_trades(limit=500)
            for r in reversed(db_closed):
                trade = ClosedTrade(
                    ticket_id=r["ticket_id"],
                    symbol=r["symbol"],
                    action=r["action"],
                    volume_lots=float(r["volume_lots"]),
                    entry_time=r["entry_time"],
                    entry_price=float(r["entry_price"]),
                    stop_loss=float(r["stop_loss"]),
                    take_profit=float(r["take_profit"]),
                    exit_time=r["exit_time"],
                    exit_price=float(r["exit_price"]),
                    pnl_dollars=float(r["pnl_dollars"]),
                    pips=float(r["pips"]),
                    exit_reason=r["exit_reason"],
                    spread_cost_dollars=float(r.get("spread_cost_dollars", 0.0) or 0.0),
                    hypothesis_id=r.get("hypothesis_id"),
                )
                self.closed_trades.append(trade)

            # Reconstruct balance
            net_closed_pnl = sum(t.pnl_dollars for t in self.closed_trades)
            self.balance = round(self.initial_balance + net_closed_pnl, 2)
            self.equity = self.balance
            self.peak_equity = max(self.peak_equity, self.balance)
        except Exception:
            pass

    def get_pip_value(self, symbol: str, current_price: float, lots: float) -> float:
        """Computes pip value in account currency (USD)."""
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS["EUR/USD"])
        if symbol.endswith("/USD"):
            return 10.0 * lots
        elif symbol.startswith("USD/"):
            return (10.0 / current_price * lots) if current_price > 0 else (10.0 * lots)
        return 10.0 * lots

    def calculate_margin(self, symbol: str, price: float, lots: float) -> float:
        """Calculates margin required given account leverage (default 1:30)."""
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS["EUR/USD"])
        notional_value = lots * pair_cfg.lot_size * (price if not symbol.startswith("USD/") else 1.0)
        leverage = self.settings.leverage
        return notional_value / leverage if leverage > 0 else notional_value

    def open_trade(
        self,
        symbol: str,
        action: str,
        lots: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        hypothesis_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[Position]:
        """
        Executes a simulated market entry.
        Applies spread cost, checks margin availability, registers position,
        and logs into persistent database.
        """
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS["EUR/USD"])
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        # Check required margin
        req_margin = self.calculate_margin(symbol, entry_price, lots)
        free_margin = self.equity - self.margin_used
        if req_margin > free_margin:
            return None  # Insufficient margin

        # Model spread cost (deducted as transaction cost)
        spread_pips = pair_cfg.typical_spread_pips
        pip_val = self.get_pip_value(symbol, entry_price, lots)
        spread_cost = spread_pips * pip_val

        # Simulate slight slippage (0.1 to 0.3 pip adverse execution)
        slippage_pips = np.random.uniform(0.05, 0.25)
        if action == "BUY":
            adjusted_entry = entry_price + (slippage_pips * pair_cfg.pip_size)
        else:
            adjusted_entry = entry_price - (slippage_pips * pair_cfg.pip_size)

        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

        pos = Position(
            ticket_id=ticket_id,
            symbol=symbol,
            action=action,
            volume_lots=lots,
            entry_price=round(adjusted_entry, 5),
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            entry_time=ts,
            spread_cost_dollars=round(spread_cost, 2),
            hypothesis_id=hypothesis_id,
            current_price=round(adjusted_entry, 5),
            unrealized_pnl=-round(spread_cost, 2),
            unrealized_pips=-round(spread_pips, 1),
        )

        self.open_positions[ticket_id] = pos
        self.margin_used += req_margin

        # Persist to SQLite
        try:
            self.db.store_trade_open(
                ticket_id=ticket_id,
                symbol=symbol,
                action=action,
                volume_lots=lots,
                entry_price=pos.entry_price,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                spread_cost=pos.spread_cost_dollars,
                hypothesis_id=hypothesis_id,
                entry_time=ts,
            )
            self.db.store_log(
                agent_name="PaperTrader",
                level="INFO",
                message=f"Opened {action} {lots} lots on {symbol} @ {pos.entry_price:.5f} (SL: {stop_loss:.5f}, TP: {take_profit:.5f})",
                metadata={"ticket_id": ticket_id, "hypothesis_id": hypothesis_id},
            )
        except Exception:
            pass

        return pos

    def close_trade(
        self,
        ticket_id: str,
        exit_price: float,
        exit_reason: str = "MANUAL_EXIT",
        exit_time: Optional[str] = None,
    ) -> Optional[ClosedTrade]:
        """
        Closes an existing position, calculates final realized PnL,
        updates account balance, releases margin, and persists to DB.
        """
        if ticket_id not in self.open_positions:
            return None

        pos = self.open_positions.pop(ticket_id)
        pair_cfg = SUPPORTED_PAIRS.get(pos.symbol, SUPPORTED_PAIRS["EUR/USD"])
        ts = exit_time or datetime.now(timezone.utc).isoformat()

        # Calculate gross price difference
        if pos.action == "BUY":
            price_delta = exit_price - pos.entry_price
        else:
            price_delta = pos.entry_price - exit_price

        pips = price_delta / pair_cfg.pip_size
        pip_val = self.get_pip_value(pos.symbol, exit_price, pos.volume_lots)
        gross_pnl = pips * pip_val
        net_pnl = gross_pnl - pos.spread_cost_dollars

        # Update Account State
        self.balance = round(self.balance + net_pnl, 2)
        freed_margin = self.calculate_margin(pos.symbol, pos.entry_price, pos.volume_lots)
        self.margin_used = max(0.0, self.margin_used - freed_margin)
        self.equity = round(self.balance + sum(p.unrealized_pnl for p in self.open_positions.values()), 2)
        self.peak_equity = max(self.peak_equity, self.equity)

        closed = ClosedTrade(
            ticket_id=ticket_id,
            symbol=pos.symbol,
            action=pos.action,
            volume_lots=pos.volume_lots,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            exit_time=ts,
            exit_price=round(exit_price, 5),
            pnl_dollars=round(net_pnl, 2),
            pips=round(pips, 1),
            exit_reason=exit_reason,
            spread_cost_dollars=pos.spread_cost_dollars,
            hypothesis_id=pos.hypothesis_id,
        )

        self.closed_trades.append(closed)

        # Persist to DB
        try:
            self.db.store_trade_close(
                ticket_id=ticket_id,
                exit_price=closed.exit_price,
                pnl_dollars=closed.pnl_dollars,
                pips=closed.pips,
                exit_reason=closed.exit_reason,
                exit_time=ts,
            )
            # Update associated hypothesis status
            if pos.hypothesis_id:
                status = "WIN" if closed.pnl_dollars > 0 else "LOSS"
                self.db.update_hypothesis_status(pos.hypothesis_id, status)

            self.db.store_log(
                agent_name="PaperTrader",
                level="INFO",
                message=f"Closed {pos.action} on {pos.symbol} @ {exit_price:.5f} ({exit_reason}) -> PnL: ${closed.pnl_dollars:+.2f} ({closed.pips:+.1f} pips)",
                metadata={"ticket_id": ticket_id, "pnl": closed.pnl_dollars},
            )
        except Exception:
            pass

        return closed

    def process_candle(
        self,
        symbol: str,
        candle: pd.Series,
    ) -> List[ClosedTrade]:
        """
        Audits all open positions for this symbol against candle high, low, and close.
        Automatically executes Take Profit or Stop Loss hits.
        Updates unrealized PnL for continuing positions.
        """
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS["EUR/USD"])
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        ts_raw = candle.get("timestamp")
        if hasattr(ts_raw, "isoformat"):
            ts = ts_raw.isoformat()
        elif ts_raw is not None:
            ts = str(ts_raw)
        else:
            ts = datetime.now(timezone.utc).isoformat()

        closed_this_bar: List[ClosedTrade] = []
        tickets_to_close: List[Tuple[str, float, str]] = []

        for ticket_id, pos in list(self.open_positions.items()):
            if pos.symbol != symbol:
                continue

            pos.current_price = close
            # Update unrealized pips & PnL
            if pos.action == "BUY":
                delta = close - pos.entry_price
                # Check Stop Loss hit
                if low <= pos.stop_loss:
                    tickets_to_close.append((ticket_id, pos.stop_loss, "STOP_LOSS"))
                    continue
                # Check Take Profit hit
                elif high >= pos.take_profit:
                    tickets_to_close.append((ticket_id, pos.take_profit, "TAKE_PROFIT"))
                    continue
            else:  # SELL
                delta = pos.entry_price - close
                # Check Stop Loss hit
                if high >= pos.stop_loss:
                    tickets_to_close.append((ticket_id, pos.stop_loss, "STOP_LOSS"))
                    continue
                # Check Take Profit hit
                elif low <= pos.take_profit:
                    tickets_to_close.append((ticket_id, pos.take_profit, "TAKE_PROFIT"))
                    continue

            pips = delta / pair_cfg.pip_size
            pip_val = self.get_pip_value(symbol, close, pos.volume_lots)
            pos.unrealized_pips = round(pips, 1)
            pos.unrealized_pnl = round((pips * pip_val) - pos.spread_cost_dollars, 2)

        # Execute any triggered exits
        for tid, px, reason in tickets_to_close:
            c = self.close_trade(tid, px, reason, ts)
            if c:
                closed_this_bar.append(c)

        # Recompute total equity and drawdown
        unrealized_sum = sum(p.unrealized_pnl for p in self.open_positions.values())
        self.equity = round(self.balance + unrealized_sum, 2)
        self.peak_equity = max(self.peak_equity, self.equity)

        dd_pct = ((self.peak_equity - self.equity) / self.peak_equity * 100.0) if self.peak_equity > 0 else 0.0

        self.equity_history.append({
            "timestamp": ts,
            "balance": self.balance,
            "equity": self.equity,
            "drawdown_pct": round(dd_pct, 2),
        })

        return closed_this_bar

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Calculates institutional risk & performance statistics."""
        total_trades = len(self.closed_trades)
        wins = [t for t in self.closed_trades if t.pnl_dollars > 0]
        losses = [t for t in self.closed_trades if t.pnl_dollars <= 0]

        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
        gross_profit = sum(t.pnl_dollars for t in wins)
        gross_loss = abs(sum(t.pnl_dollars for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

        net_pnl = round(self.equity - self.initial_balance, 2)
        return_pct = round((net_pnl / self.initial_balance) * 100.0, 2)

        # Drawdowns
        current_dd = round(((self.peak_equity - self.equity) / self.peak_equity) * 100.0, 2) if self.peak_equity > 0 else 0.0
        max_dd = max([h["drawdown_pct"] for h in self.equity_history], default=0.0)

        # Sharpe Ratio approximation
        pnls = [t.pnl_dollars for t in self.closed_trades]
        if len(pnls) > 5 and np.std(pnls) > 0:
            sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)
        else:
            sharpe = 0.0

        total_pips = sum(t.pips for t in self.closed_trades)

        return {
            "initial_balance": self.initial_balance,
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "free_margin": round(self.equity - self.margin_used, 2),
            "margin_used": round(self.margin_used, 2),
            "net_pnl": net_pnl,
            "return_pct": return_pct,
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "current_drawdown_pct": current_dd,
            "sharpe_ratio": round(sharpe, 2),
            "total_pips": round(total_pips, 1),
            "open_positions_count": len(self.open_positions),
        }
