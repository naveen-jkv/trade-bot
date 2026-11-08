"""
Strategy Evaluation Agent.
Synthesizes technical analysis from MarketAnalysisAgent, qualitative reasoning from AIResearchAgent,
and quantitative forecasts from TradingPredictor (Random Forest & PyTorch LSTM).
Resolves cross-model contradictions and formulates disciplined TradeProposals for the Risk Engine.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from agents.market_agent import MarketAnalysisReport
from agents.research_agent import TradingHypothesis
from core.prediction import TradingPredictor
from database.memory import TradingMemoryDB
from config import DEFAULT_CONFIG, MLSettings

logger = logging.getLogger("StrategyEvaluationAgent")


@dataclass
class TradeProposal:
    """Formal trading action proposal ready for risk validation."""
    symbol: str
    action: str                     # 'BUY', 'SELL', 'HOLD'
    confidence: float               # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    hypothesis_id: Optional[int]
    rationale: str
    ml_ensemble_prob_up: float
    rf_prob_up: float
    lstm_prob_up: float
    ml_prediction_id: Optional[int]
    risk_reward_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyEvaluationAgent:
    """
    Arbitrates and aligns independent signals from:
    1. AI Research Agent (macro/micro hypothesis & invalidation)
    2. Random Forest Classifier (non-linear feature importance)
    3. PyTorch LSTM (sequential time-series structure)
    Produces a single cohesive, risk-conscious proposal.
    """

    def __init__(
        self,
        predictor: Optional[TradingPredictor] = None,
        db: Optional[TradingMemoryDB] = None,
        ml_settings: Optional[MLSettings] = None,
    ):
        self.predictor = predictor or TradingPredictor(settings=ml_settings)
        self.db = db or TradingMemoryDB(DEFAULT_CONFIG.db_path)
        self.settings = ml_settings or DEFAULT_CONFIG.ml

    def evaluate_strategy(
        self,
        report: MarketAnalysisReport,
        hypothesis: TradingHypothesis,
        data_df: pd.DataFrame,
    ) -> TradeProposal:
        """
        Executes ML inference, evaluates alignment with Research Agent hypothesis,
        stores prediction in persistent memory, and outputs a refined TradeProposal.
        """
        # 1. Execute ML Inference (Random Forest + LSTM)
        ml_res = self.predictor.predict_latest(data_df, report.symbol)
        rf_up = ml_res.get("rf_prob_up", 0.50)
        lstm_up = ml_res.get("lstm_prob_up", 0.50)
        ensemble_up = ml_res.get("ensemble_prob_up", 0.50)

        # 2. Store ML Prediction in SQLite Memory for Retrospective Evaluation
        pred_id = None
        ts = datetime.now(timezone.utc).isoformat()
        try:
            pred_id = self.db.store_prediction(
                symbol=report.symbol,
                model_type="ENSEMBLE_RF_LSTM",
                predicted_direction="UP" if ensemble_up > 0.52 else ("DOWN" if ensemble_up < 0.48 else "FLAT"),
                predicted_prob=max(ensemble_up, 1.0 - ensemble_up),
                timestamp=ts,
            )
        except Exception as e:
            logger.warning(f"Failed to persist prediction to DB: {e}")

        # 3. Arbitrate Signals (Research vs ML vs Technical)
        research_dir = hypothesis.direction
        research_conf = hypothesis.confidence

        final_action = "HOLD"
        composite_conf = 0.50
        rationale_items = []

        if research_dir == "BUY":
            if ensemble_up >= 0.52:
                # Strong Agreement
                final_action = "BUY"
                composite_conf = min(0.92, (research_conf * 0.55) + (ensemble_up * 0.45) + 0.05)
                rationale_items.append(f"AI Research and ML Ensemble align bullishly (ML Up: {ensemble_up:.1%})")
            elif ensemble_up < 0.45:
                # Direct Contradiction: ML predicts Down while Research says Buy
                final_action = "HOLD"
                composite_conf = 0.45
                rationale_items.append(f"Disagreement: Research suggests BUY, but ML Ensemble predicts DOWN ({ensemble_up:.1%} UP prob). Reverting to HOLD.")
            else:
                # Moderate bullish
                final_action = "BUY"
                composite_conf = (research_conf * 0.6) + (ensemble_up * 0.4)
                rationale_items.append("Research indicates BUY with neutral/mild ML support")

        elif research_dir == "SELL":
            ml_down_prob = 1.0 - ensemble_up
            if ml_down_prob >= 0.52:
                # Strong Agreement
                final_action = "SELL"
                composite_conf = min(0.92, (research_conf * 0.55) + (ml_down_prob * 0.45) + 0.05)
                rationale_items.append(f"AI Research and ML Ensemble align bearishly (ML Down: {ml_down_prob:.1%})")
            elif ml_down_prob < 0.45:
                # Direct Contradiction
                final_action = "HOLD"
                composite_conf = 0.45
                rationale_items.append(f"Disagreement: Research suggests SELL, but ML Ensemble leans UP ({ensemble_up:.1%} UP prob). Reverting to HOLD.")
            else:
                final_action = "SELL"
                composite_conf = (research_conf * 0.6) + (ml_down_prob * 0.4)
                rationale_items.append("Research indicates SELL with neutral/mild ML support")

        else:
            final_action = "HOLD"
            composite_conf = 0.50
            rationale_items.append("Market is in consolidation / no clear directional edge")

        # 4. Check Risk-to-Reward ratio
        sl_dist = abs(hypothesis.proposed_entry - hypothesis.proposed_stop_loss)
        tp_dist = abs(hypothesis.proposed_take_profit - hypothesis.proposed_entry)
        rr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0

        proposal = TradeProposal(
            symbol=report.symbol,
            action=final_action,
            confidence=round(composite_conf, 2),
            entry_price=hypothesis.proposed_entry,
            stop_loss=hypothesis.proposed_stop_loss,
            take_profit=hypothesis.proposed_take_profit,
            hypothesis_id=hypothesis.id,
            rationale=" | ".join(rationale_items) + f" | Rationale: {hypothesis.hypothesis_text}",
            ml_ensemble_prob_up=round(ensemble_up, 3),
            rf_prob_up=round(rf_up, 3),
            lstm_prob_up=round(lstm_up, 3),
            ml_prediction_id=pred_id,
            risk_reward_ratio=round(rr, 2),
        )

        try:
            self.db.store_log(
                agent_name="StrategyEvaluationAgent",
                level="DECISION",
                message=f"Proposal for {report.symbol}: {final_action} (Conf: {composite_conf:.2f}, RR: {rr:.2f}) - {rationale_items[0]}",
                metadata={"action": final_action, "confidence": composite_conf, "prediction_id": pred_id},
            )
        except Exception as e:
            logger.warning(f"Failed to persist strategy proposal log: {e}")

        return proposal
