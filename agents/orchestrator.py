"""
Autonomous Trading Agent Orchestrator.
Coordinates the end-to-end multi-agent lifecycle:
Market Data -> Market Analysis -> AI Research -> Strategy Evaluation
-> Risk Validation -> Paper Trading -> Performance Evaluation -> Persistent Memory.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from config import DEFAULT_CONFIG, SystemConfig, SUPPORTED_PAIRS
from database.memory import TradingMemoryDB
from core.market_data import MarketDataProvider
from core.prediction import TradingPredictor
from core.risk_manager import RiskManager, RiskValidationResult
from core.paper_trader import PaperTradingSimulator, Position, ClosedTrade
from agents.market_agent import MarketAnalysisAgent, MarketAnalysisReport
from agents.research_agent import AIResearchAgent, TradingHypothesis
from agents.strategy_agent import StrategyEvaluationAgent, TradeProposal
from agents.evaluation_agent import PerformanceEvaluationAgent, EvaluationReport
from agents.memory_agent import AIMemoryAgent
from core.alpaca_mock_client import MockAlpacaClient


@dataclass
class CycleExecutionSummary:
    """Consolidated results of a single autonomous agent execution loop."""
    cycle_id: str
    timestamp: str
    symbol: str
    timeframe: str
    market_report: MarketAnalysisReport
    hypothesis: TradingHypothesis
    proposal: TradeProposal
    risk_result: RiskValidationResult
    executed_position: Optional[Dict[str, Any]]
    closed_trades: List[Dict[str, Any]]
    evaluations: List[Dict[str, Any]]
    portfolio_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market_report": self.market_report.to_dict(),
            "hypothesis": self.hypothesis.to_dict(),
            "proposal": self.proposal.to_dict(),
            "risk_result": self.risk_result.to_dict(),
            "executed_position": self.executed_position,
            "closed_trades": self.closed_trades,
            "evaluations": self.evaluations,
            "portfolio_summary": self.portfolio_summary,
        }


class AutonomousTradingAgent:
    """
    High-level Autonomous Trading Agent.
    Orchestrates the entire team of specialized sub-agents and core quantitative engines.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.db = TradingMemoryDB(self.config.db_path)

        # Core Engines
        self.data_provider = MarketDataProvider(cache_dir=self.config.data_dir)
        self.predictor = TradingPredictor(models_dir=self.config.models_dir, settings=self.config.ml)
        self.risk_manager = RiskManager(settings=self.config.risk)
        self.paper_trader = PaperTradingSimulator(
            initial_balance=self.config.risk.initial_balance,
            db=self.db,
            risk_settings=self.config.risk
        )

        # Specialized Agents
        self.market_agent = MarketAnalysisAgent(data_provider=self.data_provider, db=self.db)
        self.research_agent = AIResearchAgent(llm_settings=self.config.llm, db=self.db)
        self.strategy_agent = StrategyEvaluationAgent(predictor=self.predictor, db=self.db, ml_settings=self.config.ml)
        self.eval_agent = PerformanceEvaluationAgent(db=self.db)
        self.memory_agent = AIMemoryAgent(db=self.db)

        # Modular Alpaca Architecture (Simulated / Educational)
        self.alpaca_client = MockAlpacaClient(config=self.config, db=self.db)

        self.last_cycle_summary: Optional[CycleExecutionSummary] = None

    def run_cycle(
        self,
        symbol: str = "EUR/USD",
        timeframe: str = "1h",
        force_synthetic: bool = False,
    ) -> CycleExecutionSummary:
        """
        Executes one full autonomous cycle for the given currency pair:
        1. Fetch & analyze market data
        2. Process incoming bar on active positions (triggering SL/TP if hit)
        3. Retrospectively evaluate any closed trades
        4. Retrieve historical memory & formulate hypothesis
        5. Evaluate strategy with ML ensemble
        6. Validate proposal through deterministic Risk Firewall
        7. If approved and action in [BUY, SELL], execute paper trade
        8. Store full trail in persistent SQLite memory
        """
        ts = datetime.now(timezone.utc).isoformat()
        cycle_id = f"CYC-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"

        # Step 1: Market Analysis Agent
        report, enriched_df = self.market_agent.analyze_pair(
            symbol=symbol,
            timeframe=timeframe,
            force_synthetic=force_synthetic,
        )

        # Step 2: Paper Trading Simulator audits open positions against the latest candle
        latest_candle = enriched_df.iloc[-1]
        closed_trades = self.paper_trader.process_candle(symbol, latest_candle)

        # Step 3: Performance Evaluation Agent inspects closed trades
        eval_reports: List[EvaluationReport] = []
        for ct in closed_trades:
            er = self.eval_agent.evaluate_trade(
                trade=ct,
                market_regime=report.volatility_regime,
                prediction_id=None,
            )
            eval_reports.append(er)

        # Step 4: AI Research Agent retrieves memory & formulates hypothesis
        hypothesis = self.research_agent.generate_hypothesis(report)

        # Step 5: Strategy Evaluation Agent synthesizes Research + Random Forest + PyTorch LSTM
        proposal = self.strategy_agent.evaluate_strategy(report, hypothesis, enriched_df)

        # Step 6: Risk Validation Engine audits the proposed action
        open_pos_list = [p.to_dict() for p in self.paper_trader.open_positions.values()]
        pair_cfg = SUPPORTED_PAIRS.get(symbol, SUPPORTED_PAIRS["EUR/USD"])

        if proposal.action in ["BUY", "SELL"]:
            risk_res = self.risk_manager.validate_proposal(
                symbol=symbol,
                action=proposal.action,
                entry_price=proposal.entry_price,
                stop_loss=proposal.stop_loss,
                take_profit=proposal.take_profit,
                current_equity=self.paper_trader.equity,
                initial_balance=self.paper_trader.initial_balance,
                peak_equity=self.paper_trader.peak_equity,
                open_positions=open_pos_list,
                current_spread_pips=pair_cfg.typical_spread_pips,
                trend_direction=report.technical_state.trend_direction,
            )
        else:
            risk_res = RiskValidationResult(
                is_approved=False,
                symbol=symbol,
                action="HOLD",
                approved_lot_size=0.0,
                entry_price=proposal.entry_price,
                stop_loss=proposal.stop_loss,
                take_profit=proposal.take_profit,
                risk_amount_dollars=0.0,
                risk_reward_ratio=proposal.risk_reward_ratio,
                reasons=["Proposal action is HOLD (no order initiated)."],
                risk_score=0.0,
            )

        # Step 7: Paper Trading Simulator executes approved trade
        executed_pos_dict = None
        if risk_res.is_approved and risk_res.approved_lot_size > 0:
            new_pos = self.paper_trader.open_trade(
                symbol=symbol,
                action=risk_res.action,
                lots=risk_res.approved_lot_size,
                entry_price=risk_res.entry_price,
                stop_loss=risk_res.stop_loss,
                take_profit=risk_res.take_profit,
                hypothesis_id=hypothesis.id,
                timestamp=ts,
            )
            if new_pos:
                executed_pos_dict = new_pos.to_dict()

        # Step 8: Get updated portfolio snapshot
        portfolio_summary = self.paper_trader.get_portfolio_summary()

        summary = CycleExecutionSummary(
            cycle_id=cycle_id,
            timestamp=ts,
            symbol=symbol,
            timeframe=timeframe,
            market_report=report,
            hypothesis=hypothesis,
            proposal=proposal,
            risk_result=risk_res,
            executed_position=executed_pos_dict,
            closed_trades=[ct.to_dict() for ct in closed_trades],
            evaluations=[er.to_dict() for er in eval_reports],
            portfolio_summary=portfolio_summary,
        )

        self.last_cycle_summary = summary
        return summary
