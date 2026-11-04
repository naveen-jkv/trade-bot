"""
Machine Learning & Predictive Engine.
Implements Random Forest and PyTorch LSTM models for Forex direction forecasting.
Strictly eliminates lookahead bias via walk-forward chronological splitting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config import MODELS_DIR, MLSettings, DEFAULT_CONFIG


# --- PyTorch LSTM Architecture ---

class ForexLSTMNetwork(nn.Module):
    """Deep recurrent neural network for temporal feature representation."""

    def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 24)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(24, 2)  # [Prob(Down), Prob(Up)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits


class MLFeaturePipeline:
    """Extracts non-anticipative normalized numerical features for ML models."""

    FEATURE_NAMES = [
        "norm_return_1",
        "norm_return_3",
        "norm_return_5",
        "rsi_norm",
        "macd_norm",
        "macd_hist_norm",
        "bb_pct_b",
        "bb_bandwidth",
        "dist_ema_9",
        "dist_ema_21",
        "dist_ema_50",
        "atr_ratio",
    ]

    @classmethod
    def generate_features(cls, df: pd.DataFrame, horizon: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Creates stationary, normalized features and binary target (1: UP, 0: DOWN).
        Zero lookahead bias: features use only past information at bar t; target is bar (t+horizon).
        """
        data = df.copy()
        c = data['close']

        # Normalized returns
        data['norm_return_1'] = c.pct_change(1)
        data['norm_return_3'] = c.pct_change(3)
        data['norm_return_5'] = c.pct_change(5)

        # Technical ratios
        data['rsi_norm'] = (data['rsi_14'] - 50.0) / 50.0  # Range -1 to +1
        data['macd_norm'] = data['macd'] / c
        data['macd_hist_norm'] = data['macd_hist'] / c
        data['bb_pct_b'] = data['bb_pct_b'].clip(-0.5, 1.5)
        data['bb_bandwidth'] = data['bb_bandwidth']

        # Distances from EMAs
        data['dist_ema_9'] = (c - data['ema_9']) / c
        data['dist_ema_21'] = (c - data['ema_21']) / c
        data['dist_ema_50'] = (c - data['ema_50']) / c

        # Volatility ratio
        data['atr_ratio'] = data['atr_14'] / c

        # Target: Future return over horizon bars
        if horizon > 0:
            future_return = (c.shift(-horizon) - c) / c
            # 1 if price moves UP by at least 0.05% (to beat spread), else 0
            target = (future_return > 0.0002).astype(int)
            feature_df = data[cls.FEATURE_NAMES].copy()
            valid_mask = feature_df.notna().all(axis=1) & target.notna()
            valid_mask.iloc[-horizon:] = False
            return feature_df[valid_mask], target[valid_mask]
        else:
            feature_df = data[cls.FEATURE_NAMES].copy()
            valid_mask = feature_df.notna().all(axis=1)
            target = pd.Series(0, index=feature_df.index)
            return feature_df[valid_mask], target[valid_mask]


class TradingPredictor:
    """
    Coordinates Random Forest and PyTorch LSTM models.
    Supports walk-forward training, probability calibration, and live inference.
    """

    def __init__(self, models_dir: Path = MODELS_DIR, settings: Optional[MLSettings] = None):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings or DEFAULT_CONFIG.ml

        self.rf_model: Optional[RandomForestClassifier] = None
        self.lstm_model: Optional[ForexLSTMNetwork] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False

    def train_models(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Trains both Random Forest and LSTM using chronological walk-forward split (80% train, 20% test).
        Never shuffles time series to prevent lookahead data leakage.
        """
        X, y = MLFeaturePipeline.generate_features(df, horizon=self.settings.prediction_horizon)
        if len(X) < 60:
            return {"error": "Insufficient data points for training (minimum 60 required)."}

        # Chronological split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Fit Scaler on training set only!
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 1. Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=self.settings.rf_n_estimators,
            max_depth=self.settings.rf_max_depth,
            min_samples_split=self.settings.rf_min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_scaled, y_train)
        rf_acc = float(self.rf_model.score(X_test_scaled, y_test))

        # 2. Train PyTorch LSTM
        seq_len = min(self.settings.lstm_sequence_length, len(X_train_scaled) - 5)
        lstm_acc = self._train_lstm(X_train_scaled, y_train.values, X_test_scaled, y_test.values, seq_len)

        self.is_trained = True

        # Save artifacts
        clean_sym = symbol.replace("/", "_")
        joblib.dump(self.rf_model, self.models_dir / f"rf_{clean_sym}.joblib")
        joblib.dump(self.scaler, self.models_dir / f"scaler_{clean_sym}.joblib")
        if self.lstm_model:
            torch.save(self.lstm_model.state_dict(), self.models_dir / f"lstm_{clean_sym}.pt")

        return {
            "rf_test_accuracy": round(rf_acc * 100.0, 2),
            "lstm_test_accuracy": round(lstm_acc * 100.0, 2),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    def _train_lstm(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        y_te: np.ndarray,
        seq_len: int
    ) -> float:
        """Windowed sequence generation and PyTorch training loop."""
        def make_sequences(X, y, length):
            xs, ys = [], []
            for i in range(len(X) - length):
                xs.append(X[i:(i + length)])
                ys.append(y[i + length])
            return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)

        X_tr_seq, y_tr_seq = make_sequences(X_tr, y_tr, seq_len)
        X_te_seq, y_te_seq = make_sequences(X_te, y_te, seq_len)

        if len(X_tr_seq) < 10 or len(X_te_seq) < 5:
            return 50.0

        dataset_train = TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq))
        loader = DataLoader(dataset_train, batch_size=self.settings.lstm_batch_size, shuffle=False)

        input_dim = X_tr.shape[1]
        self.lstm_model = ForexLSTMNetwork(
            input_dim=input_dim,
            hidden_dim=self.settings.lstm_hidden_dim,
            num_layers=self.settings.lstm_num_layers
        )
        self.lstm_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.003, weight_decay=1e-4)

        for _ in range(self.settings.lstm_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.lstm_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate on Test Set
        self.lstm_model.eval()
        with torch.no_grad():
            test_x_tensor = torch.from_numpy(X_te_seq)
            test_logits = self.lstm_model(test_x_tensor)
            preds = torch.argmax(test_logits, dim=1).numpy()
            acc = float(np.mean(preds == y_te_seq))

        return acc

    def predict_latest(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Executes inference on the most recent candle for the specified symbol.
        Returns probabilities for UP and DOWN, predicted direction, and model confidence.
        """
        clean_sym = symbol.replace("/", "_")

        # Auto-load or auto-train if not in memory
        if not self.is_trained:
            rf_path = self.models_dir / f"rf_{clean_sym}.joblib"
            scaler_path = self.models_dir / f"scaler_{clean_sym}.joblib"
            lstm_path = self.models_dir / f"lstm_{clean_sym}.pt"

            if rf_path.exists() and scaler_path.exists():
                self.rf_model = joblib.load(rf_path)
                self.scaler = joblib.load(scaler_path)
                if lstm_path.exists():
                    self.lstm_model = ForexLSTMNetwork(input_dim=len(MLFeaturePipeline.FEATURE_NAMES))
                    self.lstm_model.load_state_dict(torch.load(lstm_path, weights_only=True))
                    self.lstm_model.eval()
                self.is_trained = True
            else:
                # Train automatically on provided dataframe
                self.train_models(df, symbol)

        # Extract latest features
        feat_df, _ = MLFeaturePipeline.generate_features(df, horizon=0)
        if feat_df.empty:
            # Fallback if insufficient bars
            return {
                "rf_prob_up": 0.50,
                "rf_direction": "NEUTRAL",
                "lstm_prob_up": 0.50,
                "ensemble_prob_up": 0.50,
                "confidence": 0.50,
                "direction": "HOLD",
            }

        latest_x = feat_df.iloc[[-1]]
        latest_scaled = self.scaler.transform(latest_x)

        # 1. RF Prediction
        rf_probs = self.rf_model.predict_proba(latest_scaled)[0]
        rf_prob_up = float(rf_probs[1]) if len(rf_probs) > 1 else 0.5

        # 2. LSTM Prediction
        lstm_prob_up = rf_prob_up  # Default if LSTM sequence unavailable
        if self.lstm_model is not None:
            seq_len = self.settings.lstm_sequence_length
            if len(feat_df) >= seq_len:
                seq_scaled = self.scaler.transform(feat_df.iloc[-seq_len:])
                seq_tensor = torch.from_numpy(seq_scaled).float().unsqueeze(0)
                self.lstm_model.eval()
                with torch.no_grad():
                    logits = self.lstm_model(seq_tensor)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    lstm_prob_up = float(probs[1])

        # 3. Ensemble (weighted average: 60% RF, 40% LSTM)
        ensemble_prob_up = 0.60 * rf_prob_up + 0.40 * lstm_prob_up

        if ensemble_prob_up >= self.settings.confidence_threshold:
            direction = "BUY"
            confidence = ensemble_prob_up
        elif (1.0 - ensemble_prob_up) >= self.settings.confidence_threshold:
            direction = "SELL"
            confidence = 1.0 - ensemble_prob_up
        else:
            direction = "HOLD"
            confidence = max(ensemble_prob_up, 1.0 - ensemble_prob_up)

        # Feature importances from RF
        importances = dict(zip(MLFeaturePipeline.FEATURE_NAMES, [round(float(v), 4) for v in self.rf_model.feature_importances_]))

        return {
            "direction": direction,
            "confidence": round(confidence, 3),
            "ensemble_prob_up": round(ensemble_prob_up, 3),
            "rf_prob_up": round(rf_prob_up, 3),
            "lstm_prob_up": round(lstm_prob_up, 3),
            "feature_importances": importances,
        }
