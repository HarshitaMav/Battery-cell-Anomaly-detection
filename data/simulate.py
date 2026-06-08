"""
simulate.py
-----------
Generates synthetic 4680 battery cell manufacturing sensor data.
Simulates 4 key process parameters with realistic normal distributions
and injects labeled anomalies (point, contextual, collective).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_sensor_data(
    n_samples: int = 5000,
    anomaly_fraction: float = 0.04,
    start_time: datetime = datetime(2025, 1, 1, 6, 0, 0),
    freq_seconds: int = 10,
) -> pd.DataFrame:
    """
    Simulate sensor readings from a battery cell production line.

    Sensors:
        - voltage_v       : Cell voltage during formation cycling (V)
        - temperature_c   : Electrolyte fill temperature (°C)
        - internal_res_mohm: Internal resistance post-formation (mΩ)
        - fill_level_ml   : Electrolyte fill volume (mL)

    Anomaly types injected:
        - Point anomaly    : Single extreme spike
        - Contextual       : Drift over a short window
        - Collective       : Correlated multi-sensor deviation
    """

    timestamps = [start_time + timedelta(seconds=i * freq_seconds) for i in range(n_samples)]

    # --- Normal process distributions (based on realistic 4680 cell specs) ---
    voltage       = np.random.normal(loc=3.65, scale=0.03, size=n_samples)
    temperature   = np.random.normal(loc=25.0, scale=1.2,  size=n_samples)
    internal_res  = np.random.normal(loc=2.50, scale=0.15, size=n_samples)
    fill_level    = np.random.normal(loc=5.80, scale=0.10, size=n_samples)

    # Inject slow sinusoidal drift to mimic real process variation
    t = np.linspace(0, 4 * np.pi, n_samples)
    temperature  += 0.5 * np.sin(t)
    internal_res += 0.05 * np.cos(t * 0.7)

    labels = np.zeros(n_samples, dtype=int)
    anomaly_type = ["normal"] * n_samples

    n_anomalies = int(n_samples * anomaly_fraction)

    # -- Point anomalies (random spikes) ---
    n_point = n_anomalies // 3
    point_idx = np.random.choice(n_samples, n_point, replace=False)
    for idx in point_idx:
        voltage[idx]      += np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.5)
        temperature[idx]  += np.random.choice([-1, 1]) * np.random.uniform(8, 15)
        labels[idx] = 1
        anomaly_type[idx] = "point"

    # -- Contextual anomalies (short drift windows) ---
    n_context = n_anomalies // 3
    for _ in range(n_context):
        start = np.random.randint(0, n_samples - 30)
        window = slice(start, start + np.random.randint(5, 30))
        internal_res[window] += np.random.uniform(0.8, 1.5)
        fill_level[window]   -= np.random.uniform(0.3, 0.7)
        labels[window] = 1
        for i in range(*window.indices(n_samples)):
            anomaly_type[i] = "contextual"

    # -- Collective anomalies (correlated multi-sensor shift) ---
    n_collective = n_anomalies - n_point - n_context
    for _ in range(n_collective):
        start = np.random.randint(0, n_samples - 20)
        window = slice(start, start + np.random.randint(5, 20))
        voltage[window]      += 0.25
        temperature[window]  += 6.0
        internal_res[window] += 0.6
        labels[window] = 1
        for i in range(*window.indices(n_samples)):
            anomaly_type[i] = "collective"

    df = pd.DataFrame({
        "timestamp":          timestamps,
        "voltage_v":          np.round(voltage, 4),
        "temperature_c":      np.round(temperature, 3),
        "internal_res_mohm":  np.round(internal_res, 4),
        "fill_level_ml":      np.round(fill_level, 4),
        "label":              labels,
        "anomaly_type":       anomaly_type,
    })

    return df


if __name__ == "__main__":
    df = generate_sensor_data()
    df.to_csv("sensor_data.csv", index=False)
    print(f"Generated {len(df)} samples | Anomalies: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(df.head())
