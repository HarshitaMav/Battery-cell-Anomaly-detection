# ⚡ Battery Cell Manufacturing — Anomaly Detection

**Isolation Forest vs LSTM Autoencoder on simulated Tesla 4680 cell sensor data**

> Built by Harshita Mav · MS Computer Science, Syracuse University  
> Targeting: Tesla Cell Manufacturing Internship, Fall 2026

---

## Overview

This project detects anomalies in battery cell manufacturing sensor data using two complementary ML approaches. It simulates a 4680 cell production line with 4 key sensors and compares unsupervised anomaly detection models against labeled ground truth.

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

| Sensor | Description | Normal Distribution |
|--------|-------------|---------------------|
| `voltage_v` | Cell voltage during formation cycling | N(3.65, 0.03) V |
| `temperature_c` | Electrolyte fill temperature | N(25.0, 1.2) °C |
| `internal_res_mohm` | Internal resistance post-formation | N(2.50, 0.15) mΩ |
| `fill_level_ml` | Electrolyte fill volume | N(5.80, 0.10) mL |

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

## Dashboard Features

- **Real-time sensor timeline** with anomaly overlay
- **Adjustable detection threshold** per model (sidebar sliders)
- **Sensor selection** — toggle individual sensors
- **Anomaly score comparison** — IF vs LSTM side by side
- **Model metrics** — AUC-ROC, Precision, Recall, F1
- **Confusion matrices** for both models
- **Raw anomaly event table** with export

---

## Model Comparison Summary

| Dimension | Isolation Forest | LSTM Autoencoder |
|-----------|-----------------|------------------|
| Best for | Point anomalies | Contextual/collective |
| Training speed | Seconds | Minutes |
| Temporal awareness | None | 20-step window |
| Production fit | Real-time baseline | Batch re-analysis |
| Explainability | Path length (feature-level) | Reconstruction error per feature |

**Recommended production architecture:** Ensemble both — IF for always-on real-time detection, LSTM for deeper analysis at quality gates.

---

## Tech Stack

`Python` · `scikit-learn` · `TensorFlow/Keras` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`

---

*Harshita Mav · [linkedin.com/in/harshita-mav](https://linkedin.com/in/harshita-mav) · [github.com/harshitamav](https://github.com/harshitamav)*
