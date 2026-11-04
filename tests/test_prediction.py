"""
Unit Tests for Machine Learning Predictor (Random Forest & PyTorch LSTM).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from core.prediction import TradingPredictor, MLFeaturePipeline
from core.indicators import TechnicalAnalyzer
from config import MLSettings


@pytest.fixture
def sample_market_df():
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    base = 1.0850
    walk = np.cumsum(np.random.normal(0, 0.0008, n))
    closes = base + walk
    opens = closes + np.random.normal(0, 0.0003, n)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.0005, n))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.0005, n))
    volumes = np.random.randint(1000, 5000, n).astype(float)

    raw_df = pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return TechnicalAnalyzer.analyze_dataframe(raw_df)


def test_feature_pipeline_generation(sample_market_df):
    X, y = MLFeaturePipeline.generate_features(sample_market_df, horizon=3)
    assert len(X) == len(y)
    assert len(X) > 50
    for name in MLFeaturePipeline.FEATURE_NAMES:
        assert name in X.columns
    # Binary target
    assert set(y.unique()).issubset({0, 1})


def test_predictor_training_and_inference(sample_market_df):
    with tempfile.TemporaryDirectory() as tmp_dir:
        settings = MLSettings(
            rf_n_estimators=15,
            rf_max_depth=4,
            lstm_sequence_length=10,
            lstm_hidden_dim=16,
            lstm_epochs=5,
            lstm_batch_size=16,
        )
        predictor = TradingPredictor(models_dir=Path(tmp_dir), settings=settings)

        # Train models
        train_res = predictor.train_models(sample_market_df, symbol="EUR/USD")
        assert "rf_test_accuracy" in train_res
        assert "lstm_test_accuracy" in train_res
        assert predictor.is_trained

        # Inference on latest bar
        inf_res = predictor.predict_latest(sample_market_df, symbol="EUR/USD")
        assert "direction" in inf_res
        assert inf_res["direction"] in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= inf_res["confidence"] <= 1.0
        assert 0.0 <= inf_res["rf_prob_up"] <= 1.0
        assert 0.0 <= inf_res["lstm_prob_up"] <= 1.0
        assert "feature_importances" in inf_res
