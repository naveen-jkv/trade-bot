"""
AI Memory Agent.
Provides semantic and contextual memory management over the persistent SQLite database.
Coordinates retrieval of historical lessons, hypotheses, predictions, and strategy evaluations.
Maintains institutional knowledge across trading sessions.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

from database.memory import TradingMemoryDB
from config import DEFAULT_CONFIG


class AIMemoryAgent:
    """
    Custodian of the agent's persistent memory and experience base.
    Allows the AI agent to look back in time, recall how similar market conditions played out,
    and prevent repeated errors.
    """

    def __init__(self, db: Optional[TradingMemoryDB] = None):
        self.db = db or TradingMemoryDB(DEFAULT_CONFIG.db_path)

    def retrieve_context_for_pair(
        self,
        symbol: str,
        regime: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieves relevant experiences, past lessons, and recent hypotheses for this pair & regime.
        """
        lessons = self.db.retrieve_relevant_lessons(symbol=symbol, regime=regime, limit=limit)
        recent_hypotheses = self.db.get_recent_hypotheses(symbol=symbol, limit=limit)
        recent_trades = self.db.get_closed_trades(symbol=symbol, limit=limit)
        acc_stats = self.db.get_prediction_accuracy()

        return {
            "symbol": symbol,
            "regime": regime,
            "retrieved_lessons": lessons,
            "recent_hypotheses": recent_hypotheses,
            "recent_trades": recent_trades,
            "prediction_accuracy": acc_stats,
        }

    def record_agent_decision(
        self,
        agent_name: str,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Logs an internal agent decision or thought process."""
        self.db.store_log(agent_name=agent_name, level=level, message=message, metadata=metadata)

    def get_full_memory_summary(self) -> Dict[str, Any]:
        """
        Generates comprehensive statistics on accumulated experiences,
        lessons learned, prediction accuracies, and trade performance.
        """
        performance = self.db.get_overall_performance()
        lessons = self.db.get_all_lessons(limit=50)
        recent_logs = self.db.get_recent_logs(limit=50)
        recent_evals = self.db.get_latest_strategy_evaluations(limit=10)
        accuracy = self.db.get_prediction_accuracy()

        return {
            "overall_performance": performance,
            "lessons_count": len(lessons),
            "recent_lessons": lessons[:10],
            "recent_strategy_evaluations": recent_evals,
            "prediction_accuracy": accuracy,
            "recent_logs": recent_logs[:25],
        }

    def clear_database_if_requested(self) -> None:
        """Utility method to reset paper-trading simulation state if user explicitly commands it."""
        with self.db._get_connection() as conn:
            conn.execute("DELETE FROM trades;")
            conn.execute("DELETE FROM hypotheses;")
            conn.execute("DELETE FROM predictions;")
            conn.execute("DELETE FROM lessons_learned;")
            conn.execute("DELETE FROM strategy_evaluations;")
            conn.execute("DELETE FROM agent_logs;")
            conn.commit()
