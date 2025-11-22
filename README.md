# OPSD PowerDesk

## Overview
This project implements a day-ahead load forecasting system for Germany (DE), France (FR), and Spain (ES). It includes data processing, SARIMA/GRU forecasting, anomaly detection (Z-score, CUSUM, ML), and a live simulation with online adaptation.

## Environment
- Python 3.x
- Install dependencies: `pip install -r requirements.txt`

## Usage

1. **Data Loading**:
   ```bash
   python src/load_opsd.py
   ```

2. **Analysis**:
   ```bash
   python src/decompose_acf_pacf.py
   ```

3. **Forecasting**:
   ```bash
   python src/forecast.py
   ```

4. **Anomaly Detection**:
   ```bash
   python src/anomaly.py
   python src/anomaly_ml.py
   ```

5. **Online Simulation**:
   ```bash
   python src/live_loop.py
   ```

6. **Dashboard**:
   ```bash
   streamlit run src/dashboard_app.py
   ```

## Structure
- `src/`: Source code.
- `outputs/`: Generated CSVs, plots, and metrics.
- `data/`: Input data (OPSD).
