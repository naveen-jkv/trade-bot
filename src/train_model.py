"""
Model Training Module for AI Forex Simulator.

Provides:
1. Random Forest Classifier with balanced class weights and regularized tree depth.
2. Experimental LSTM Recurrent Neural Network for temporal sequence modeling.
3. Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
4. Model serialization and persistence via joblib.
"""

from __future__ import annotations
import os
import json
import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.preprocessing import LABEL_MAPPING, FEATURE_COLUMNS, create_lstm_sequences

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =====================================================================
# 1. Random Forest Architecture
# =====================================================================

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 120,
    max_depth: int = 7,
    min_samples_split: int = 10,
    min_samples_leaf: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a regularized Random Forest classifier for Forex direction forecasting.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    # Predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    y_prob_test = rf.predict_proba(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    # Classification report dictionary
    target_names = [LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING.keys())]
    present_classes = sorted(list(set(np.unique(y_test)).union(set(np.unique(y_pred_test)))))
    filtered_names = [LABEL_MAPPING[i] for i in present_classes]

    report = classification_report(
        y_test,
        y_pred_test,
        labels=present_classes,
        target_names=filtered_names,
        output_dict=True,
        zero_division=0,
    )

    conf_mat = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])

    # Feature importances
    feature_importance_df = pd.DataFrame(
        {"Feature": FEATURE_COLUMNS, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Time-series cross-validation on train split
    tscv = TimeSeriesSplit(n_splits=4)
    cv_scores = []
    for tr_idx, val_idx in tscv.split(X_train):
        rf_cv = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf_cv.fit(X_train[tr_idx], y_train[tr_idx])
        cv_scores.append(accuracy_score(y_train[val_idx], rf_cv.predict(X_train[val_idx])))

    return {
        "model": rf,
        "model_type": "RandomForest",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "feature_importance": feature_importance_df,
        "cv_scores": [float(s) for s in cv_scores],
        "cv_mean": float(np.mean(cv_scores)) if cv_scores else 0.0,
        "y_prob_test": y_prob_test,
        "y_pred_test": y_pred_test,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        },
    }


# =====================================================================
# 2. PyTorch LSTM Architecture (Optional Experimental Model)
# =====================================================================

if HAS_TORCH:
    class ForexLSTMNet(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            num_classes: int = 3,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_dim, 32)
            self.relu = nn.ReLU()
            self.out = nn.Linear(32, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            last_hidden = out[:, -1, :]
            dense = self.dropout(self.relu(self.fc1(last_hidden)))
            logits = self.out(dense)
            return logits


class LSTMModelWrapper:
    """Scikit-learn compatible wrapper for PyTorch Forex LSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        seq_length: int = 15,
        epochs: int = 30,
        lr: float = 0.001,
        batch_size: int = 32,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.model = ForexLSTMNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=3,
        ).to(self.device)
        self.train_losses: List[float] = []
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray) -> "LSTMModelWrapper":
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        x_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.long)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.train_losses = []
        for epoch in range(self.epochs):
            total_loss = 0.0
            count = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_y)
                count += len(batch_y)
            epoch_loss = total_loss / max(1, count)
            self.train_losses.append(float(epoch_loss))

        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_seq, dtype=torch.float32)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_seq)
        return np.argmax(probs, axis=1)


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seq_length: int = 15,
    hidden_dim: int = 64,
    epochs: int = 35,
    lr: float = 0.001,
) -> Dict[str, Any]:
    """
    Train an experimental PyTorch LSTM on multi-step sequential Forex data.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available for LSTM training.")

    X_train_seq, y_train_seq = create_lstm_sequences(X_train, y_train, seq_length=seq_length)
    X_test_seq, y_test_seq = create_lstm_sequences(X_test, y_test, seq_length=seq_length)

    if len(X_train_seq) < 50 or len(X_test_seq) < 20:
        raise ValueError("Insufficient sequence count for LSTM. Need more historical bars.")

    wrapper = LSTMModelWrapper(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        seq_length=seq_length,
        epochs=epochs,
        lr=lr,
    )

    wrapper.fit(X_train_seq, y_train_seq)

    y_pred_train = wrapper.predict(X_train_seq)
    y_pred_test = wrapper.predict(X_test_seq)
    y_prob_test = wrapper.predict_proba(X_test_seq)

    train_acc = accuracy_score(y_train_seq, y_pred_train)
    test_acc = accuracy_score(y_test_seq, y_pred_test)
    macro_f1 = f1_score(y_test_seq, y_pred_test, average="macro", zero_division=0)

    present_classes = sorted(list(set(np.unique(y_test_seq)).union(set(np.unique(y_pred_test)))))
    filtered_names = [LABEL_MAPPING[i] for i in present_classes]

    report = classification_report(
        y_test_seq,
        y_pred_test,
        labels=present_classes,
        target_names=filtered_names,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_test_seq, y_pred_test, labels=[0, 1, 2])

    return {
        "model": wrapper,
        "model_type": "LSTM",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "train_losses": wrapper.train_losses,
        "y_prob_test": y_prob_test,
        "y_pred_test": y_pred_test,
        "seq_length": seq_length,
        "params": {
            "seq_length": seq_length,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "lr": lr,
        },
    }


# =====================================================================
# 3. Model Persistence and Serialization
# =====================================================================

def save_model_bundle(
    filepath: str,
    model: Any,
    scaler: Any,
    feature_names: List[str],
    metadata: Dict[str, Any],
) -> str:
    """Save model bundle and associated metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "metadata": metadata,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    joblib.dump(bundle, filepath)
    return filepath


def load_model_bundle(filepath: str) -> Dict[str, Any]:
    """Load model bundle from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)
