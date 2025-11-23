# OPSD PowerDesk: Day-Ahead Forecasting & Anomaly Detection

## Overview
This project implements a comprehensive day-ahead (24-hour horizon) electric load forecasting system for three European countries: **Germany (DE)**, **France (FR)**, and **Spain (ES)**. 

The system leverages Open Power System Data (OPSD) and includes:
1.  **Time Series Analysis**: STL decomposition, Stationarity tests, and ACF/PACF analysis.
2.  **Forecasting**: Classical **SARIMA** models (optimized via AIC/BIC grid search) and optional **GRU** neural networks.
3.  **Anomaly Detection**: 
    *   Unsupervised: Rolling Z-score (Window 336h) and CUSUM.
    *   Supervised: Machine Learning classifier (Logistic Regression) trained on "silver" labels.
4.  **Live Simulation**: A simulated real-time environment with **Online Adaptation** (Rolling SARIMA refit) triggered by schedule or concept drift.
5.  **Dashboard**: An interactive Streamlit dashboard for monitoring the live feed.

## Repository Structure
The repository follows the required layout:

| Path | Purpose |
| :--- | :--- |
| `README.md` | Project documentation and usage instructions. |
| `requirements.txt` | Python dependencies. |
| `config.yaml` | Configuration for countries, thresholds, and horizons. |
| `data/` | Contains the OPSD time series CSV. |
| `src/load_opsd.py` | Data ingestion and cleaning. |
| `src/decompose_acf_pacf.py` | STL analysis and SARIMA order selection. |
| `src/forecast.py` | Backtesting framework (Expanding origin). |
| `src/anomaly.py` | Unsupervised anomaly detection (Z-score, CUSUM). |
| `src/anomaly_ml.py` | ML-based anomaly classification. |
| `src/live_loop.py` | Live simulation with online adaptation. |
| `src/dashboard_app.py` | Streamlit visualization dashboard. |
| `src/metrics.py` | Metric calculation helpers (MASE, sMAPE, Coverage). |
| `outputs/` | Generated artifacts (CSVs, Plots, JSONs). |

## Setup & Usage

### 1. Environment Setup
Ensure you have Python 3.x installed.
```bash
pip install -r requirements.txt
```

### 2. Execution Pipeline
Run the scripts in the following order to reproduce the full analysis:

**Step 1: Data Preparation**
Loads raw OPSD data and creates tidy per-country files.
```bash
python src/load_opsd.py
```

**Step 2: Analysis & Model Selection**
Performs STL decomposition and determines optimal SARIMA orders.
```bash
python src/decompose_acf_pacf.py
```

**Step 3: Forecasting (Backtest)**
Runs expanding-window backtests for SARIMA and GRU models.
```bash
python src/forecast.py
```

**Step 4: Anomaly Detection**
Detects anomalies using statistical methods and trains the ML classifier.
```bash
python src/anomaly.py
python src/anomaly_ml.py
```

**Step 5: Live Simulation**
Simulates a live data stream for **DE** (Germany) with online model adaptation.
```bash
python src/live_loop.py
```

### 3. Dashboard
Launch the interactive dashboard to view the live simulation results.
```bash
streamlit run src/dashboard_app.py
```

## Key Parameters
*   **Seasonality**: 24 hours
*   **Forecast Horizon**: 24 hours
*   **Backtest Warm-up**: > 60 days
*   **Anomaly Threshold**: Z-score > 3.0
*   **Live History**: 120 days start
*   **Adaptation Strategy**: Rolling SARIMA (90-day window)
