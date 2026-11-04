"""
Integration Test for End-to-End Autonomous Agent Workflow.
Verifies the complete cycle:
Market Data -> Market Analysis -> Research Hypothesis -> Strategy Proposal
-> Risk Validation -> Paper Trading -> Evaluation -> SQLite Memory Persistence.
"""

import pytest
from pathlib import Path
import tempfile

from config import SystemConfig, RiskSettings, LLMSettings
from agents.orchestrator import AutonomousTradingAgent


@pytest.fixture
def test_agent():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = Path(tf.name)

    cfg = SystemConfig(
        db_path=db_path,
        risk=RiskSettings(initial_balance=10000.0, risk_per_trade_pct=1.5),
        llm=LLMSettings(provider="heuristic"),  # Deterministic testable mode
    )
    agent = AutonomousTradingAgent(config=cfg)
    yield agent
    try:
        db_path.unlink()
    except Exception:
        pass


def test_full_autonomous_cycle(test_agent):
    # Run cycle on EUR/USD with synthetic offline data for reproducibility
    summary = test_agent.run_cycle(
        symbol="EUR/USD",
        timeframe="1h",
        force_synthetic=True,
    )

    assert summary is not None
    assert summary.market_report.symbol == "EUR/USD"
    assert summary.market_report.current_price > 0
    assert summary.hypothesis.direction in ["BUY", "SELL", "HOLD"]
    assert summary.proposal.action in ["BUY", "SELL", "HOLD"]
    assert summary.risk_result is not None

    # Check persistence in database
    hypotheses = test_agent.db.get_recent_hypotheses(limit=5)
    assert len(hypotheses) >= 1

    logs = test_agent.db.get_recent_logs(limit=10)
    assert len(logs) >= 1

    summary_dict = test_agent.paper_trader.get_portfolio_summary()
    assert summary_dict["balance"] > 0
    assert summary_dict["equity"] > 0
