"""
Data Preprocessing, Target Engineering, and Leakage-Free Dataset Preparation.

Strictly adheres to ML Best Practices:
1. Purely backward-looking stationary features.
2. Chronological train/test split (no random shuffle or temporal contamination).
3. Scalers are fit exclusively on the training split.
4. Tri-class labeling: UP, DOWN, UNCERTAIN.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.indicators import add_all_indicators

# Explicit class labels
LABEL_MAPPING = {
    0: "UNCERTAIN",
    1: "UP",
    2: "DOWN",
}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

FEATURE_COLUMNS = [
    "Norm_EMA_10",
    "Norm_EMA_20",
    "Norm_EMA_50",
    "EMA_Spread_10_20",
    "EMA_Spread_20_50",
    "RSI_Norm",
    "Norm_MACD",
    "Norm_MACD_Signal",
    "Norm_MACD_Hist",
    "BB_PctB",
    "BB_Width",
    "Normalized_ATR",
    "Hist_Volatility",
    "Return_1",
    "Return_3",
    "Return_5",
    "Bar_Body_Ratio",
]


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate stationary, scale-independent features from raw prices and indicators.
    All calculations are strictly backward-looking.
    """
    if "EMA_10" not in df.columns:
        df = add_all_indicators(df)

    data = df.copy()

    close = data["Close"]
    eps = 1e-9

    # Stationary moving average price differentials
    data["Norm_EMA_10"] = (close - data["EMA_10"]) / (close + eps)
    data["Norm_EMA_20"] = (close - data["EMA_20"]) / (close + eps)
    data["Norm_EMA_50"] = (close - data["EMA_50"]) / (close + eps)

    data["EMA_Spread_10_20"] = (data["EMA_10"] - data["EMA_20"]) / (close + eps)
    data["EMA_Spread_20_50"] = (data["EMA_20"] - data["EMA_50"]) / (close + eps)

    # Scaled RSI [0, 1]
    data["RSI_Norm"] = data["RSI_14"] / 100.0

    # Stationary MACD metrics
    data["Norm_MACD"] = data["MACD"] / (close + eps)
    data["Norm_MACD_Signal"] = data["MACD_Signal"] / (close + eps)
    data["Norm_MACD_Hist"] = data["MACD_Hist"] / (close + eps)

    # Bollinger Bands features
    data["BB_PctB"] = data["BB_PctB"].clip(-0.5, 1.5)
    data["BB_Width"] = data["BB_Width"]

    # Volatility and momentum returns
    data["Normalized_ATR"] = data["Normalized_ATR"]
    data["Hist_Volatility"] = data["Hist_Volatility"].fillna(0.0)

    data["Return_1"] = close.pct_change(1)
    data["Return_3"] = close.pct_change(3)
    data["Return_5"] = close.pct_change(5)

    data["Bar_Body_Ratio"] = data["Bar_Body"] / (data["Bar_Range"] + eps)

    return data


def create_targets(
    df: pd.DataFrame,
    horizon_bars: int = 1,
    threshold_pips: Optional[float] = None,
    pip_unit: float = 0.0001,
    threshold_atr_fraction: float = 0.25,
) -> pd.DataFrame:
    """
    Generate forward classification target (UP, DOWN, UNCERTAIN).

    Classification rules:
    - UP: Forward Return > +threshold
    - DOWN: Forward Return < -threshold
    - UNCERTAIN: |Forward Return| <= threshold (market noise/neutral consolidation)
    """
    data = df.copy()
    forward_close = data["Close"].shift(-horizon_bars)
    data["Forward_Return"] = (forward_close - data["Close"]) / (data["Close"] + 1e-9)

    # Determine dynamic threshold
    if threshold_pips is not None and threshold_pips > 0:
        threshold_price = threshold_pips * pip_unit
        threshold_ratio = threshold_price / (data["Close"] + 1e-9)
    else:
        # Volatility-based adaptive threshold (fractions of ATR)
        if "Normalized_ATR" not in data.columns:
            threshold_ratio = 0.002
        else:
            threshold_ratio = data["Normalized_ATR"] * threshold_atr_fraction

    data["Target_Class"] = 0  # Default UNCERTAIN
    is_up = data["Forward_Return"] > threshold_ratio
    is_down = data["Forward_Return"] < -threshold_ratio

    data.loc[is_up, "Target_Class"] = 1  # UP
    data.loc[is_down, "Target_Class"] = 2  # DOWN
    data["Target_Name"] = data["Target_Class"].map(LABEL_MAPPING)

    return data


def prepare_training_data(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
    horizon_bars: int = 1,
    threshold_pips: Optional[float] = None,
    pip_unit: float = 0.0001,
) -> Dict[str, Any]:
    """
    Full pipeline to prepare leak-free train and test datasets.

    Enforces:
    - Chronological split without shuffling.
    - Dropping indicator warm-up period.
    - Scaler fitted ONLY on train data.
    """
    # 1. Feature Engineering
    engineered = create_engineered_features(df)

    # 2. Target Creation
    labeled = create_targets(
        engineered,
        horizon_bars=horizon_bars,
        threshold_pips=threshold_pips,
        pip_unit=pip_unit,
    )

    # 3. Clean NaN rows from indicator warm-up and future horizon shift
    warmup_period = 55  # EMA 50 needs at least 50 bars
    valid_data = labeled.iloc[warmup_period:-horizon_bars].copy()
    valid_data = valid_data.dropna(subset=FEATURE_COLUMNS + ["Target_Class"]).reset_index(drop=True)

    if len(valid_data) < 100:
        raise ValueError(f"Insufficient samples ({len(valid_data)}) after indicator warm-up.")

    # 4. Chronological Train / Test Split
    split_idx = int(len(valid_data) * train_ratio)
    train_df = valid_data.iloc[:split_idx].copy()
    test_df = valid_data.iloc[split_idx:].copy()

    X_train_raw = train_df[FEATURE_COLUMNS].values
    y_train = train_df["Target_Class"].values

    X_test_raw = test_df[FEATURE_COLUMNS].values
    y_test = test_df["Target_Class"].values

    # 5. Fit Scaler ONLY on Training Split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
        "split_index": split_idx,
        "full_data": valid_data,
        "label_mapping": LABEL_MAPPING,
    }


def create_lstm_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build temporal 3D sequences for LSTM training: [Samples, Sequence_Length, Features].
    Strictly causal (no look-ahead).
    """
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i - sequence_length : i])
        y_seq.append(y[i])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)
