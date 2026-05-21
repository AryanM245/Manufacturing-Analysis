"""Train XGBoost model for AI4I predictive maintenance use case."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

DATA_PATH = Path("data/ai4i2020_clean.csv")
MODEL_PATH = Path("models/xgb_failure_model.pkl")
SCALER_PATH = Path("models/feature_scaler.pkl")


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Create engineered features and return X, y, feature_names."""
    data = df.copy()

    if "machine_failure" not in data.columns:
        raise ValueError("Target column 'machine_failure' not found in dataset.")

    data = data.sort_values(by="udi") if "udi" in data.columns else data

    # Moving averages and gradients
    data["process_temp_ma_5"] = data["process_temperature_k"].rolling(window=5, min_periods=1).mean()
    data["air_temp_ma_5"] = data["air_temperature_k"].rolling(window=5, min_periods=1).mean()
    data["rot_speed_ma_5"] = data["rotational_speed_rpm"].rolling(window=5, min_periods=1).mean()

    data["temp_gradient"] = data["process_temperature_k"] - data["air_temperature_k"]
    data["temp_gradient_delta"] = data["temp_gradient"].diff().fillna(0.0)

    # Health score from normalized stress components
    scaler_for_score = MinMaxScaler()
    score_components = data[["tool_wear_min", "torque_nm", "process_temperature_k"]].copy()
    normalized = scaler_for_score.fit_transform(score_components)
    data[["wear_norm", "torque_norm", "proc_temp_norm"]] = normalized

    data["machine_health_score"] = (
        0.45 * (1 - data["wear_norm"]) +
        0.30 * (1 - data["torque_norm"]) +
        0.25 * (1 - data["proc_temp_norm"])
    ) * 100

    drop_cols = ["machine_failure", "product_id", "type"]
    feature_cols = [c for c in data.columns if c not in drop_cols]

    X = data[feature_cols]
    y = data["machine_failure"].astype(int)
    return X, y, feature_cols


def train() -> None:
    """Train model, print key metrics, and persist artifacts."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X, y, feature_names = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.35).astype(int)  # tuned threshold to prioritize recall

    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("=== Predictive Maintenance Model Metrics ===")
    print(f"Recall:      {recall:.4f}")
    print(f"False Negatives: {fn}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"Confusion Matrix -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Saved model to: {MODEL_PATH}")
    print(f"✅ Saved scaler to: {SCALER_PATH}")


if __name__ == "__main__":
    train()
