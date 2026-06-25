# ⚡ Battery Cell Manufacturing — Anomaly Detection

**Isolation Forest vs LSTM Autoencoder · Signal Processing · SPC · Pass/Fail · on simulated Tesla 4680 cell sensor data**

## Overview

This project detects anomalies in battery cell manufacturing sensor data using two complementary ML approaches. It simulates a 4680 cell production line with 4 key sensors and compares unsupervised anomaly detection models against labeled ground truth.

The project is aligned with **cell manufacturing system engineering** workflows — covering equipment health monitoring, signal processing, statistical process control (SPC), pass/fail criteria, and test documentation.

**Why this matters for cell manufacturing:**  
Undetected sensor anomalies in formation cycling, electrolyte filling, or end-of-line testing lead to defective cells, warranty failures, or safety issues. Early detection = fewer bad cells reaching pack assembly.

---

## Project Structure

```
battery-cell-anomaly-detection/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt
├── data/
│   └── simulate.py                 # 4680 cell sensor data generator
├── models/
│   ├── isolation_forest.py         # Isolation Forest detector
│   └── lstm_autoencoder.py         # LSTM Autoencoder detector
└── notebooks/
    └── analysis.ipynb              # Full analysis walkthrough
```

---

## Sensors Simulated

| Sensor | Instrument Type | Description | Normal Distribution |
|--------|----------------|-------------|---------------------|
| `voltage_v` | Voltage Probe | Cell voltage during formation cycling | N(3.65, 0.03) V |
| `temperature_c` | Thermocouple | Electrolyte fill temperature | N(25.0, 1.2) °C |
| `internal_res_mohm` | Load Cell / Impedance | Internal resistance post-formation | N(2.50, 0.15) mΩ |
| `fill_level_ml` | Flow / Pressure Sensor | Electrolyte fill volume | N(5.80, 0.10) mL |

**5,000 samples** at 10-second intervals (~14 hours of production data)  
**~4% anomaly rate** across 3 types: point, contextual, collective

---

## Models

### Isolation Forest
- Unsupervised, ensemble of random isolation trees
- Excellent for **point anomalies** (instantaneous spikes)
- Low latency — suitable for real-time deployment
- No temporal context

### LSTM Autoencoder
- Sequence model with 20-step lookback window
- Better for **contextual and collective anomalies** (gradual drift)
- Detects patterns that span multiple timesteps
- Higher reconstruction error = anomaly

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/harshitamav/battery-cell-anomaly-detection
cd battery-cell-anomaly-detection
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py

# Or explore the notebook
jupyter notebook notebooks/analysis.ipynb
```

> **Note:** TensorFlow is optional. The dashboard falls back gracefully to a demo LSTM simulation if TF is not installed. For full LSTM training: `pip install tensorflow`

---

## Dashboard Sections

### Existing
- **01 · Sensor time series** — real-time sensor timeline with ground truth anomaly overlay
- **02 · Anomaly score comparison** — IF vs LSTM scores with adjustable threshold lines
- **03 · Model performance** — AUC-ROC, Precision, Recall, F1 side by side
- **04 · Confusion matrices** — for both models

### New — System Engineering Features
- **05 · Signal processing** — raw vs Butterworth low-pass filtered signal per sensor; configurable cutoff frequency and filter order
- **06 · Statistical Process Control (SPC)** — X-bar control chart with UCL/LCL at ±3σ, Cpk process capability index per sensor
- **07 · Equipment operation window** — configurable spec limits (LSL/USL) per sensor, pass/fail classification, yield %, operation window summary table
- **08 · Test report export** — downloadable CSV reports: anomaly events log and full test summary (conditions + model metrics + SPC metrics)

---

## Signal Processing — Butterworth Filter

Raw sensor signals contain high-frequency noise from electrical interference and mechanical vibration. A **Butterworth low-pass filter** smooths the signal before ML models run, preserving real manufacturing trends while attenuating noise.

```python
from scipy.signal import butter, filtfilt

def butterworth_filter(signal, cutoff=0.08, order=4):
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)   # zero-phase: no time lag
```

- **Cutoff frequency** — configurable in sidebar (0.01–0.30 normalized)
- **Filter order** — configurable (2 / 4 / 6 / 8)
- **filtfilt** — zero-phase implementation; applies filter forward + backward to avoid phase shift

---

## Statistical Process Control (SPC)

X-bar control charts monitor each sensor against statistically derived control limits. **Cpk** quantifies whether the equipment's natural variation fits within engineering spec limits.

```
Cpk = min( (USL − mean) / 3σ,  (mean − LSL) / 3σ )
```

| Cpk | Interpretation |
|-----|----------------|
| ≥ 1.67 | Excellent — world-class manufacturing |
| ≥ 1.33 | Good — Six Sigma production standard ✓ |
| 1.00 – 1.33 | Marginal — needs improvement |
| < 1.00 | Poor — regularly produces out-of-spec parts |

- **UCL/LCL** (±3σ) detect statistical anomalies — unusual variation relative to process mean
- **USL/LSL** detect engineering violations — values outside the required operating range
- Both limits are independently configurable per sensor in the sidebar

---

## Model Comparison Summary

| Dimension | Isolation Forest | LSTM Autoencoder |
|-----------|-----------------|------------------|
| Best for | Point anomalies | Contextual/collective |
| Training speed | Seconds | Minutes |
| Temporal awareness | None | 20-step window |
| Production fit | Real-time baseline | Batch re-analysis |
| Explainability | Path length (feature-level) | Reconstruction error per feature |

**Recommended production architecture:** Ensemble both — IF for always-on real-time detection at every quality gate, LSTM for deeper analysis at end-of-line testing.

---

## Tech Stack

`Python` · `scikit-learn` · `TensorFlow/Keras` · `SciPy` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
scipy
tensorflow        # optional — dashboard falls back to demo mode if not installed
```

---

*Harshita Mav · [linkedin.com/in/harshita-mav](https://linkedin.com/in/harshita-mav) · [github.com/harshitamav](https://github.com/harshitamav)*
