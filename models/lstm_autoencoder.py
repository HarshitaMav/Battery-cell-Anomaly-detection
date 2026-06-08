"""
lstm_autoencoder.py
-------------------
LSTM Autoencoder for time-series anomaly detection on battery cell
manufacturing sensor data. Anomalies are identified by high reconstruction error.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

FEATURES = ["voltage_v", "temperature_c", "internal_res_mohm", "fill_level_ml"]
SEQ_LEN = 20           # lookback window (20 samples = ~3.3 min at 10s intervals)
SCALER_PATH = os.path.join(os.path.dirname(__file__), "lstm_scaler.pkl")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lstm_ae.keras")


def _build_model(seq_len: int, n_features: int):
    """Build LSTM Autoencoder architecture."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
    except ImportError:
        raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")

    inp = Input(shape=(seq_len, n_features))

    # Encoder
    x = LSTM(64, activation="tanh", return_sequences=False)(inp)

    # Bottleneck
    encoded = RepeatVector(seq_len)(x)

    # Decoder
    x = LSTM(64, activation="tanh", return_sequences=True)(encoded)
    decoded = TimeDistributed(Dense(n_features))(x)

    model = Model(inp, decoded)
    model.compile(optimizer="adam", loss="mse")
    return model


def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Slide a window across rows to produce (n_sequences, seq_len, n_features)."""
    seqs = []
    for i in range(len(X) - seq_len + 1):
        seqs.append(X[i : i + seq_len])
    return np.array(seqs)


def train(
    df: pd.DataFrame,
    epochs: int = 30,
    batch_size: int = 64,
    validation_split: float = 0.1,
) -> tuple:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, SCALER_PATH)

    X_seq = _make_sequences(X, SEQ_LEN)

    model = _build_model(SEQ_LEN, len(FEATURES))
    model.fit(
        X_seq, X_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        verbose=1,
    )
    model.save(MODEL_PATH)

    return model, scaler


def _reconstruction_errors(model, X_seq: np.ndarray) -> np.ndarray:
    """Per-sequence mean squared reconstruction error."""
    X_pred = model.predict(X_seq, verbose=0)
    mse = np.mean(np.power(X_seq - X_pred, 2), axis=(1, 2))
    return mse


def predict(df: pd.DataFrame, model=None, scaler=None, threshold: float = None) -> pd.DataFrame:
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)

    if model is None:
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Could not load LSTM model: {e}")

    X = scaler.transform(df[FEATURES])
    X_seq = _make_sequences(X, SEQ_LEN)
    errors = _reconstruction_errors(model, X_seq)

    # Pad beginning (no sequence for first SEQ_LEN - 1 rows)
    pad = np.full(SEQ_LEN - 1, errors[0])
    errors_padded = np.concatenate([pad, errors])

    if threshold is None:
        threshold = np.percentile(errors_padded, 96)   # flag top 4%

    result = df.copy()
    result["lstm_recon_error"] = errors_padded
    # Normalize to [0,1]
    result["lstm_anomaly_score"] = (errors_padded - errors_padded.min()) / (
        errors_padded.max() - errors_padded.min()
    )
    result["lstm_pred"] = (errors_padded > threshold).astype(int)
    result["lstm_threshold"] = threshold

    return result


def evaluate(result: pd.DataFrame) -> dict:
    y_true = result["label"]
    y_pred = result["lstm_pred"]
    y_score = result["lstm_anomaly_score"]

    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_score)

    return {
        "model": "LSTM Autoencoder",
        "auc_roc": round(auc, 4),
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1": round(report["1"]["f1-score"], 4),
        "report": report,
    }


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.simulate import generate_sensor_data
    df = generate_sensor_data()
    model, scaler = train(df, epochs=20)
    result = predict(df, model, scaler)
    metrics = evaluate(result)
    print(f"\nLSTM Autoencoder Results:")
    for k, v in metrics.items():
        if k != "report":
            print(f"  {k}: {v}")
