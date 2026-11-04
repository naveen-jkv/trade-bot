"""
Multi-Agent Trading Intelligence System.
Encapsulates specialized agents:
- MarketAnalysisAgent: Technical and statistical market parsing
- AIResearchAgent: Grounded hypothesis generation with LLM and persistent memory retrieval
- StrategyEvaluationAgent: ML ensemble integration and signal arbitration
- PerformanceEvaluationAgent: Retrospective evaluation, mistake analysis, and OOS backtesting
- AIMemoryAgent: Persistent SQLite knowledge manager
- AutonomousTradingAgent: High-level system orchestrator
"""

from agents.market_agent import MarketAnalysisAgent, MarketAnalysisReport
from agents.research_agent import AIResearchAgent, TradingHypothesis
from agents.strategy_agent import StrategyEvaluationAgent, TradeProposal
from agents.evaluation_agent import PerformanceEvaluationAgent, EvaluationReport
from agents.memory_agent import AIMemoryAgent
from agents.orchestrator import AutonomousTradingAgent, CycleExecutionSummary

__all__ = [
    "MarketAnalysisAgent",
    "MarketAnalysisReport",
    "AIResearchAgent",
    "TradingHypothesis",
    "StrategyEvaluationAgent",
    "TradeProposal",
    "PerformanceEvaluationAgent",
    "EvaluationReport",
    "AIMemoryAgent",
    "AutonomousTradingAgent",
    "CycleExecutionSummary",
]
