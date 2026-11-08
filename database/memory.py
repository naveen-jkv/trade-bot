"""
SQLite Persistent Memory Engine.
Stores hypotheses, predictions, trades, market regimes, lessons learned, and agent logs.
Allows semantic/contextual retrieval of historical decisions to inform future actions.
"""

import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


class TradingMemoryDB:
    """Manages all persistent trading memory, outcomes, and retrospective evaluations."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")  # High concurrency
        return conn

    def _init_database(self) -> None:
        """Create tables and indices if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Trading Hypotheses Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,            -- 'BUY', 'SELL', 'HOLD'
                confidence REAL NOT NULL,           -- 0.0 to 1.0
                hypothesis_text TEXT NOT NULL,
                invalidation_criteria TEXT,
                indicators_snapshot TEXT,          -- JSON string of indicators
                market_regime TEXT,
                outcome_status TEXT DEFAULT 'PENDING' -- 'PENDING', 'WIN', 'LOSS', 'EXPIRED'
            );
            """)

            # 2. Model Predictions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,           -- 'RANDOM_FOREST', 'LSTM', 'ENSEMBLE'
                predicted_direction TEXT NOT NULL,  -- 'UP', 'DOWN', 'FLAT'
                predicted_prob REAL NOT NULL,
                actual_direction TEXT,              -- Filled upon retrospective evaluation
                is_correct INTEGER,                 -- 1 for correct, 0 for incorrect, NULL if pending
                evaluated_at TEXT
            );
            """)

            # 3. Simulated Trades Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                ticket_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,               -- 'BUY' or 'SELL'
                volume_lots REAL NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                pnl_dollars REAL,
                pips REAL,
                exit_reason TEXT,                   -- 'TAKE_PROFIT', 'STOP_LOSS', 'MANUAL', 'TIME_LIMIT'
                spread_cost_dollars REAL,
                hypothesis_id INTEGER,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses (id)
            );
            """)

            # 4. Retrospective Lessons Learned Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS lessons_learned (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trigger_event TEXT NOT NULL,        -- 'LOSS_ANALYSIS', 'WIN_ANALYSIS', 'FALSE_BREAKOUT'
                regime TEXT NOT NULL,
                mistake_analysis TEXT NOT NULL,
                rule_learned TEXT NOT NULL,
                outcome_pnl REAL,
                tags TEXT                           -- Comma-separated tags (e.g. 'counter_trend,high_vol')
            );
            """)

            # 5. Strategy Adaptations / Evaluations Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                test_period TEXT NOT NULL,
                win_rate REAL NOT NULL,
                profit_factor REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                parameters_json TEXT NOT NULL,
                validated INTEGER NOT NULL          -- 1 if passed OOS criteria, 0 otherwise
            );
            """)

            # 6. Agent Activity Logs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                level TEXT NOT NULL,                -- 'INFO', 'WARNING', 'ERROR', 'DECISION'
                message TEXT NOT NULL,
                metadata_json TEXT
            );
            """)

            # 7. Alpaca Mock Orders Table (Educational Architecture)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS alpaca_mock_orders (
                id TEXT PRIMARY KEY,
                client_order_id TEXT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                asset_class TEXT NOT NULL,
                qty REAL NOT NULL,
                filled_qty REAL NOT NULL,
                filled_avg_price REAL,
                order_type TEXT NOT NULL,
                side TEXT NOT NULL,
                time_in_force TEXT NOT NULL,
                limit_price REAL,
                stop_price REAL,
                status TEXT NOT NULL,
                rejection_reason TEXT
            );
            """)

            # 8. Alpaca Mock Positions Table (Educational Architecture)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS alpaca_mock_positions (
                symbol TEXT PRIMARY KEY,
                asset_class TEXT NOT NULL,
                qty REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                cost_basis REAL NOT NULL,
                side TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 9. Order Submissions & Idempotency Audit Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_submissions (
                client_order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                broker TEXT NOT NULL,
                cycle_id TEXT,
                submitted_at TEXT NOT NULL
            );
            """)

            # Indices for rapid querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hyp_sym ON hypotheses(symbol, timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_sym ON trades(symbol, entry_time);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_sym ON predictions(symbol, timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lessons_regime ON lessons_learned(symbol, regime);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_agent ON agent_logs(agent_name, timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alpaca_orders ON alpaca_mock_orders(symbol, created_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_sub_sym ON order_submissions(symbol, submitted_at);")

            conn.commit()

    # --- Hypotheses ---

    def store_hypothesis(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        confidence: float,
        hypothesis_text: str,
        invalidation_criteria: str,
        indicators_snapshot: Dict[str, Any],
        market_regime: str,
        timestamp: Optional[str] = None
    ) -> int:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO hypotheses (
                    timestamp, symbol, timeframe, direction, confidence,
                    hypothesis_text, invalidation_criteria, indicators_snapshot,
                    market_regime, outcome_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """, (
                ts, symbol, timeframe, direction, confidence,
                hypothesis_text, invalidation_criteria, json.dumps(indicators_snapshot),
                market_regime
            ))
            conn.commit()
            return cursor.lastrowid

    def update_hypothesis_status(self, hypothesis_id: int, status: str) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE hypotheses SET outcome_status = ? WHERE id = ?", (status, hypothesis_id))
            conn.commit()

    def get_recent_hypotheses(self, symbol: Optional[str] = None, limit: int = 15) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute("""
                    SELECT * FROM hypotheses WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM hypotheses ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # --- Predictions ---

    def store_prediction(
        self,
        symbol: str,
        model_type: str,
        predicted_direction: str,
        predicted_prob: float,
        timestamp: Optional[str] = None
    ) -> int:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, symbol, model_type, predicted_direction, predicted_prob
                ) VALUES (?, ?, ?, ?, ?)
            """, (ts, symbol, model_type, predicted_direction, predicted_prob))
            conn.commit()
            return cursor.lastrowid

    def evaluate_prediction(
        self,
        prediction_id: int,
        actual_direction: str,
        is_correct: bool,
        evaluated_at: Optional[str] = None
    ) -> None:
        ts = evaluated_at or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE predictions
                SET actual_direction = ?, is_correct = ?, evaluated_at = ?
                WHERE id = ?
            """, (actual_direction, 1 if is_correct else 0, ts, prediction_id))
            conn.commit()

    def get_prediction_accuracy(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if model_type:
                cursor.execute("""
                    SELECT COUNT(*) as total, SUM(is_correct) as correct
                    FROM predictions
                    WHERE model_type = ? AND is_correct IS NOT NULL
                """, (model_type,))
            else:
                cursor.execute("""
                    SELECT COUNT(*) as total, SUM(is_correct) as correct
                    FROM predictions
                    WHERE is_correct IS NOT NULL
                """)
            row = cursor.fetchone()
            total = row["total"] or 0
            correct = row["correct"] or 0
            accuracy = (correct / total * 100.0) if total > 0 else 0.0
            return {"total_evaluated": total, "correct": correct, "accuracy_pct": round(accuracy, 2)}

    # --- Trades ---

    def store_trade_open(
        self,
        ticket_id: str,
        symbol: str,
        action: str,
        volume_lots: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        spread_cost: float,
        hypothesis_id: Optional[int] = None,
        entry_time: Optional[str] = None
    ) -> None:
        ts = entry_time or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO trades (
                    ticket_id, symbol, action, volume_lots, entry_time,
                    entry_price, stop_loss, take_profit, spread_cost_dollars, hypothesis_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticket_id, symbol, action, volume_lots, ts,
                entry_price, stop_loss, take_profit, spread_cost, hypothesis_id
            ))
            conn.commit()

    def store_trade_close(
        self,
        ticket_id: str,
        exit_price: float,
        pnl_dollars: float,
        pips: float,
        exit_reason: str,
        exit_time: Optional[str] = None
    ) -> None:
        ts = exit_time or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE trades
                SET exit_time = ?, exit_price = ?, pnl_dollars = ?, pips = ?, exit_reason = ?
                WHERE ticket_id = ?
            """, (ts, exit_price, pnl_dollars, pips, exit_reason, ticket_id))
            conn.commit()

    def get_closed_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute("""
                    SELECT * FROM trades
                    WHERE exit_time IS NOT NULL AND symbol = ?
                    ORDER BY exit_time DESC LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM trades
                    WHERE exit_time IS NOT NULL
                    ORDER BY exit_time DESC LIMIT ?
                """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- Lessons Learned ---

    def store_lesson(
        self,
        symbol: str,
        trigger_event: str,
        regime: str,
        mistake_analysis: str,
        rule_learned: str,
        outcome_pnl: float,
        tags: str = "",
        timestamp: Optional[str] = None
    ) -> int:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO lessons_learned (
                    timestamp, symbol, trigger_event, regime,
                    mistake_analysis, rule_learned, outcome_pnl, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, symbol, trigger_event, regime, mistake_analysis, rule_learned, outcome_pnl, tags))
            conn.commit()
            return cursor.lastrowid

    def retrieve_relevant_lessons(
        self,
        symbol: str,
        regime: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieves historical lessons matching symbol or market regime to prevent repeating mistakes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if regime:
                cursor.execute("""
                    SELECT * FROM lessons_learned
                    WHERE symbol = ? OR regime = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, regime, limit))
            else:
                cursor.execute("""
                    SELECT * FROM lessons_learned
                    WHERE symbol = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, limit))
            return [dict(r) for r in cursor.fetchall()]

    def get_all_lessons(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM lessons_learned ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- Strategy Evaluations ---

    def store_strategy_evaluation(
        self,
        strategy_name: str,
        test_period: str,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
        parameters: Dict[str, Any],
        validated: bool,
        timestamp: Optional[str] = None
    ) -> int:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO strategy_evaluations (
                    timestamp, strategy_name, test_period, win_rate,
                    profit_factor, max_drawdown, parameters_json, validated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts, strategy_name, test_period, win_rate,
                profit_factor, max_drawdown, json.dumps(parameters),
                1 if validated else 0
            ))
            conn.commit()
            return cursor.lastrowid

    def get_latest_strategy_evaluations(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM strategy_evaluations ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- Agent Logs ---

    def store_log(
        self,
        agent_name: str,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata) if metadata else None
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO agent_logs (timestamp, agent_name, level, message, metadata_json)
                VALUES (?, ?, ?, ?, ?)
            """, (ts, agent_name, level, message, meta_str))
            conn.commit()

    def get_recent_logs(self, limit: int = 100, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if agent_name:
                cursor.execute("""
                    SELECT * FROM agent_logs WHERE agent_name = ? ORDER BY timestamp DESC LIMIT ?
                """, (agent_name, limit))
            else:
                cursor.execute("""
                    SELECT * FROM agent_logs ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- Aggregate Analytics ---

    def get_overall_performance(self) -> Dict[str, Any]:
        """Calculates win rate, net PnL, profit factor, total trades from persistent storage."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as win_count,
                    SUM(CASE WHEN pnl_dollars <= 0 THEN 1 ELSE 0 END) as loss_count,
                    COALESCE(SUM(pnl_dollars), 0.0) as net_pnl,
                    COALESCE(SUM(CASE WHEN pnl_dollars > 0 THEN pnl_dollars ELSE 0 END), 0.0) as gross_profit,
                    COALESCE(SUM(CASE WHEN pnl_dollars < 0 THEN ABS(pnl_dollars) ELSE 0 END), 0.0) as gross_loss,
                    COALESCE(SUM(pips), 0.0) as total_pips
                FROM trades
                WHERE exit_time IS NOT NULL
            """)
            row = cursor.fetchone()
            total = row["total_trades"] or 0
            wins = row["win_count"] or 0
            losses = row["loss_count"] or 0
            net_pnl = row["net_pnl"] or 0.0
            gross_profit = row["gross_profit"] or 0.0
            gross_loss = row["gross_loss"] or 0.0
            total_pips = row["total_pips"] or 0.0

            win_rate = (wins / total * 100.0) if total > 0 else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

            return {
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round(win_rate, 2),
                "net_pnl": round(net_pnl, 2),
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "total_pips": round(total_pips, 1),
            }

    # --- Alpaca Mock Persistence (Educational Architecture) ---

    def store_alpaca_order(self, order_dict: Dict[str, Any]) -> None:
        """Stores or updates a mock Alpaca order."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alpaca_mock_orders (
                    id, client_order_id, created_at, symbol, asset_class,
                    qty, filled_qty, filled_avg_price, order_type, side,
                    time_in_force, limit_price, stop_price, status, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_dict["id"],
                order_dict.get("client_order_id", ""),
                order_dict["created_at"],
                order_dict["symbol"],
                order_dict.get("asset_class", "us_equity"),
                order_dict["qty"],
                order_dict.get("filled_qty", 0.0),
                order_dict.get("filled_avg_price"),
                order_dict.get("order_type", "market"),
                order_dict["side"],
                order_dict.get("time_in_force", "day"),
                order_dict.get("limit_price"),
                order_dict.get("stop_price"),
                order_dict["status"],
                order_dict.get("rejection_reason"),
            ))
            conn.commit()

    def get_alpaca_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves mock Alpaca orders."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM alpaca_mock_orders WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM alpaca_mock_orders ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            return [dict(r) for r in cursor.fetchall()]

    def store_alpaca_position(self, pos_dict: Dict[str, Any]) -> None:
        """Stores or updates an open mock Alpaca position."""
        now_ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alpaca_mock_positions (
                    symbol, asset_class, qty, avg_entry_price, cost_basis, side, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pos_dict["symbol"],
                pos_dict.get("asset_class", "us_equity"),
                pos_dict["qty"],
                pos_dict["avg_entry_price"],
                pos_dict["cost_basis"],
                pos_dict.get("side", "long"),
                now_ts,
            ))
            conn.commit()

    def get_alpaca_positions(self) -> List[Dict[str, Any]]:
        """Retrieves all open mock Alpaca positions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alpaca_mock_positions WHERE qty > 0 ORDER BY symbol ASC")
            return [dict(r) for r in cursor.fetchall()]

    def delete_alpaca_position(self, symbol: str) -> None:
        """Deletes a closed mock position."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM alpaca_mock_positions WHERE symbol = ?", (symbol,))
            conn.commit()

    def clear_alpaca_mock_data(self) -> None:
        """Clears all mock Alpaca orders and positions."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM alpaca_mock_orders")
            conn.execute("DELETE FROM alpaca_mock_positions")
            conn.commit()

    # --- Order Idempotency & Deduplication ---

    def store_order_submission(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        status: str = "SUBMITTED",
        broker: str = "alpaca",
        cycle_id: Optional[str] = None,
        submitted_at: Optional[str] = None,
    ) -> None:
        """Records an order submission attempt for deterministic idempotency audits."""
        ts = submitted_at or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO order_submissions (
                    client_order_id, symbol, side, qty, order_type,
                    status, broker, cycle_id, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                client_order_id,
                symbol,
                side.lower(),
                float(qty),
                order_type.lower(),
                status.upper(),
                broker.lower(),
                cycle_id,
                ts,
            ))
            conn.commit()

    def is_order_submitted(self, client_order_id: str) -> bool:
        """Checks if an order with the given deterministic client_order_id was already processed."""
        if not client_order_id:
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM order_submissions WHERE client_order_id = ? LIMIT 1", (client_order_id,))
            return cursor.fetchone() is not None

    def has_recent_order(self, symbol: str, window_minutes: int = 15) -> bool:
        """Checks if an order was already submitted for this symbol within the lookback window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM order_submissions WHERE symbol = ? AND submitted_at >= ? LIMIT 1",
                (symbol, cutoff)
            )
            return cursor.fetchone() is not None

    def get_submitted_orders(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves historical submitted orders for auditing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM order_submissions WHERE symbol = ? ORDER BY submitted_at DESC LIMIT ?",
                    (symbol, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM order_submissions ORDER BY submitted_at DESC LIMIT ?",
                    (limit,)
                )
            return [dict(r) for r in cursor.fetchall()]

