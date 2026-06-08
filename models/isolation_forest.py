"""
isolation_forest.py
-------------------
Isolation Forest anomaly detector for battery cell manufacturing sensor data.
Trains on a clean subset, scores the full dataset, and returns results.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

FEATURES = ["voltage_v", "temperature_c", "internal_res_mohm", "fill_level_ml"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "isolation_forest.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "if_scaler.pkl")


def train(df: pd.DataFrame, contamination: float = 0.04) -> tuple:
    """Train Isolation Forest on normal data."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_features=len(FEATURES),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler


def predict(df: pd.DataFrame, model=None, scaler=None) -> pd.DataFrame:
    """Score dataframe and return anomaly predictions + scores."""
    if model is None:
        model = joblib.load(MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)

    X = scaler.transform(df[FEATURES])
    raw_scores = model.decision_function(X)   # higher = more normal
    preds = model.predict(X)                  # -1 = anomaly, 1 = normal

    result = df.copy()
    result["if_score"] = raw_scores
    # Normalize score to [0,1] where 1 = most anomalous
    result["if_anomaly_score"] = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    result["if_pred"] = (preds == -1).astype(int)

    return result


def evaluate(result: pd.DataFrame) -> dict:
    """Compute evaluation metrics."""
    y_true = result["label"]
    y_pred = result["if_pred"]
    y_score = result["if_anomaly_score"]

    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_score)

    return {
        "model": "Isolation Forest",
        "auc_roc": round(auc, 4),
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1": round(report["1"]["f1-score"], 4),
        "report": report,
    }


if __name__ == "__main__":
    from data.simulate import generate_sensor_data
    df = generate_sensor_data()
    model, scaler = train(df)
    result = predict(df, model, scaler)
    metrics = evaluate(result)
    print(f"\nIsolation Forest Results:")
    for k, v in metrics.items():
        if k != "report":
            print(f"  {k}: {v}")
