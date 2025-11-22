import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Input
import warnings
import json

warnings.filterwarnings("ignore")

OUTPUT_DIR = 'outputs/'
COUNTRIES = ['DE', 'FR', 'ES']
MODEL_ORDERS = {
    'DE': {'order': (2, 0, 1), 'seasonal_order': (1, 1, 1, 24)},
    'FR': {'order': (2, 0, 0), 'seasonal_order': (1, 1, 1, 24)},
    'ES': {'order': (2, 0, 1), 'seasonal_order': (1, 1, 1, 24)}
}

def load_data(country_code):
    file_path = os.path.join(OUTPUT_DIR, f'{country_code}_cleaned.csv')
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('H')
    df['load'] = df['load'].ffill()
    
    # Add exogenous features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    # One-hot encoding for hour and dayofweek
    # We can use pandas get_dummies or just keep them as is if using tree models, 
    # but for SARIMAX/NN, one-hot or cyclic features are better. 
    # For SARIMAX, too many exogenous vars can be slow. 
    # Let's stick to wind/solar if available and maybe simple cyclic time features or just raw for now if not strictly required to be one-hot.
    # Prompt says: "Optional exogenous: hour‑of‑day and day‑of‑week one‑hots; optional wind/solar series."
    # I will add one-hots.
    
    # Hour one-hots (drop first to avoid dummy trap if intercept is used, but SARIMAX handles it)
    hour_dummies = pd.get_dummies(df['hour'], prefix='h', drop_first=True).astype(float)
    dow_dummies = pd.get_dummies(df['dayofweek'], prefix='d', drop_first=True).astype(float)
    
    exog = pd.concat([hour_dummies, dow_dummies], axis=1)
    
    if 'wind' in df.columns:
        df['wind'] = df['wind'].fillna(0).astype(float)
        exog['wind'] = df['wind']
    if 'solar' in df.columns:
        df['solar'] = df['solar'].fillna(0).astype(float)
        exog['solar'] = df['solar']
        
    # Ensure exog index matches df
    exog.index = df.index
    
    # Ensure load is float
    df['load'] = df['load'].astype(float)
    
    return df, exog

def split_data(df):
    n = len(df)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)
    
    train = df.iloc[:train_end]
    dev = df.iloc[train_end:dev_end]
    test = df.iloc[dev_end:]
    
    return train, dev, test

def calculate_metrics(y_true, y_pred, y_train_history, seasonality=24):
    # y_true and y_pred are numpy arrays or series
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # MSE, RMSE, MAPE, sMAPE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE: mean(|(y_true - y_pred) / y_true|) * 100
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # sMAPE: 2 * mean(|y_true - y_pred| / (|y_true| + |y_pred|)) * 100
    smape = 200 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    
    # MASE
    # Denominator: Mean Absolute Error of seasonal naive forecast on training set
    # Naive seasonal forecast: y_t = y_{t-m}
    y_train_vals = y_train_history.values if hasattr(y_train_history, 'values') else y_train_history
    naive_errors = np.abs(y_train_vals[seasonality:] - y_train_vals[:-seasonality])
    d = np.mean(naive_errors)
    
    mae = mean_absolute_error(y_true, y_pred)
    mase = mae / d if d != 0 else np.nan
    
    return {
        'MASE': mase,
        'sMAPE': smape,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def run_sarima_backtest(train_df, eval_df, train_exog, eval_exog, order, seasonal_order):
    # Fit on Train
    print(f"Fitting SARIMAX{order}x{seasonal_order} on training data...")
    # Using a subset of exog to avoid collinearity issues or too many vars if needed, 
    # but let's try with all constructed exog.
    # Note: One-hot encoding 24 hours + 7 days = ~30 vars. Plus wind/solar. 
    # This might be heavy for SARIMAX. 
    # Let's use only wind/solar and maybe cyclic time features to reduce dimensionality if it's too slow.
    # For now, I'll stick to the prompt's "Optional exogenous: hour‑of‑day and day‑of‑week one‑hots".
    # If it fails or is too slow, I will fallback to just wind/solar.
    
    # To speed up, I will use a smaller history for fitting if train is huge (50k rows).
    # Fitting SARIMA on 40k rows with 30 exog vars is very slow.
    # I will use the last 2000 hours of train for fitting the initial model parameters.
    
    fit_start_idx = max(0, len(train_df) - 2000)
    train_subset = train_df.iloc[fit_start_idx:]
    exog_subset = train_exog.iloc[fit_start_idx:]
    
    model = SARIMAX(train_subset['load'], 
                    exog=exog_subset,
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # Backtest Loop
    # Expanding origin: stride=24, horizon=24
    # We need to predict 24h ahead, then advance 24h.
    
    forecasts = []
    history_load = train_df['load'].tolist()
    # We need to keep track of exog history as well if we were refitting, 
    # but for 'append'/'extend', we just need the new data.
    
    # Actually, statsmodels 'apply' or 'extend' is good.
    # But 'apply' creates a new results object.
    
    # Strategy:
    # 1. Predict next 24h using current results.
    # 2. Append the *actual* 24h data to the results object (updating state).
    # 3. Repeat.
    
    # We need the full eval_df to iterate over.
    n_steps = len(eval_df)
    stride = 24
    
    # We need to ensure we have exog for the forecast horizon
    
    print("Starting SARIMA backtest...")
    
    # Current model state is at end of train_subset.
    # If we skipped part of train, we need to filter through the rest of train first?
    # Yes, if we only fit on last 2000, the state is at the end of those 2000.
    # Which is the end of train. So we are good.
    
    current_results = results
    
    predictions = []
    
    # Iterate through eval set in chunks of 24
    for i in range(0, n_steps, stride):
        if i + stride > n_steps:
            break # Drop incomplete last batch if any
            
        # Forecast horizon indices
        horizon_start = i
        horizon_end = i + stride
        
        # Exog for forecast
        exog_forecast = eval_exog.iloc[horizon_start:horizon_end]
        
        # Get forecast
        # get_forecast returns prediction results
        pred_res = current_results.get_forecast(steps=stride, exog=exog_forecast)
        yhat = pred_res.predicted_mean
        conf_int = pred_res.conf_int(alpha=0.2) # 80% PI -> alpha=0.2
        
        # Store results
        chunk_dates = eval_df.index[horizon_start:horizon_end]
        chunk_actuals = eval_df['load'].iloc[horizon_start:horizon_end]
        
        for j in range(stride):
            predictions.append({
                'timestamp': chunk_dates[j],
                'y_true': chunk_actuals.iloc[j],
                'yhat': yhat.iloc[j],
                'lo': conf_int.iloc[j, 0],
                'hi': conf_int.iloc[j, 1],
                'horizon': j + 1,
                'train_end': str(train_df.index[-1]) # Static train end for reference
            })
        
        # Update model with actuals
        # We append the observed data from this step to update the state for the next step
        new_obs = chunk_actuals
        new_exog = eval_exog.iloc[horizon_start:horizon_end]
        
        current_results = current_results.append(new_obs, exog=new_exog, refit=False)
        
    return pd.DataFrame(predictions)

def create_sequences(data, target, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length : i + seq_length + horizon])
    return np.array(X), np.array(y)

def run_neural_backtest(train_df, eval_df, train_exog, eval_exog):
    # GRU/LSTM direct multi-horizon (last 168h -> next 24 steps)
    seq_length = 168
    horizon = 24
    
    # Prepare data scaling
    # We scale load and exog features
    # Combine load and exog
    
    # Feature columns: load + exog columns
    feature_cols = ['load'] + list(train_exog.columns)
    
    train_data = pd.concat([train_df[['load']], train_exog], axis=1)
    eval_data = pd.concat([eval_df[['load']], eval_exog], axis=1)
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    # Note: In a real scenario, we should be careful about scaling exog if they are dummies. 
    # But MinMax on 0/1 is fine (stays 0/1).
    
    # Create sequences for training
    # Target is just 'load' (column 0)
    X_train, y_train = create_sequences(train_scaled, train_scaled[:, 0], seq_length, horizon)
    
    # Build Model
    print("Training Neural Network (GRU)...")
    model = Sequential([
        Input(shape=(seq_length, X_train.shape[2])),
        GRU(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(horizon) # Output 24 steps
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Backtest
    # We need to slide the window across eval_df
    # For the first prediction in eval, we need the last 168h of train.
    
    # Combine train tail and eval for sliding window
    full_seq = np.concatenate([train_scaled[-seq_length:], scaler.transform(eval_data)], axis=0)
    
    predictions = []
    n_eval_steps = len(eval_df)
    stride = 24
    
    # We iterate with stride 24
    # The input for the first eval batch (indices 0 to 23) starts at full_seq index 0 (length 168)
    # full_seq structure: [168h of train] [eval data...]
    
    print("Starting Neural Network backtest...")
    
    for i in range(0, n_eval_steps, stride):
        if i + stride > n_eval_steps:
            break
            
        # Input window index in full_seq
        # i=0 -> input is full_seq[0:168] -> predicts eval[0:24]
        # i=24 -> input is full_seq[24:168+24] -> predicts eval[24:48]
        
        current_X = full_seq[i : i + seq_length]
        current_X = current_X.reshape(1, seq_length, -1)
        
        pred_scaled = model.predict(current_X, verbose=0) # Shape (1, 24)
        
        # Inverse transform
        # We only predicted 'load'. We need to inverse transform it.
        # The scaler was fitted on [load, exog...]. 
        # We can create a dummy array to inverse transform.
        
        dummy = np.zeros((horizon, train_scaled.shape[1]))
        dummy[:, 0] = pred_scaled[0]
        pred_inv = scaler.inverse_transform(dummy)[:, 0]
        
        # Store results
        chunk_dates = eval_df.index[i : i + stride]
        chunk_actuals = eval_df['load'].iloc[i : i + stride]
        
        for j in range(stride):
            predictions.append({
                'timestamp': chunk_dates[j],
                'y_true': chunk_actuals.iloc[j],
                'yhat': pred_inv[j],
                'lo': np.nan, # NN doesn't give PI easily
                'hi': np.nan,
                'horizon': j + 1,
                'train_end': str(train_df.index[-1])
            })
            
    return pd.DataFrame(predictions)

def main():
    summary_metrics = []
    
    for cc in COUNTRIES:
        print(f"\nProcessing {cc}...")
        df, exog = load_data(cc)
        train_df, dev_df, test_df = split_data(df)
        train_exog, dev_exog, test_exog = split_data(exog)
        
        # SARIMA
        order = MODEL_ORDERS[cc]['order']
        seasonal_order = MODEL_ORDERS[cc]['seasonal_order']
        
        # Dev Backtest
        print(f"Running SARIMA Dev Backtest for {cc}...")
        sarima_dev_res = run_sarima_backtest(train_df, dev_df, train_exog, dev_exog, order, seasonal_order)
        sarima_dev_res.to_csv(os.path.join(OUTPUT_DIR, f'{cc}_forecasts_dev_sarima.csv'), index=False)
        
        # Test Backtest
        # For test, we should technically include dev in training or just continue expanding from dev?
        # "Train = first 80%, Dev = next 10%, Test = final 10%"
        # Usually Test evaluation assumes model trained on Train+Dev.
        # But the prompt says "Train = first 80%".
        # And "Backtest: expanding origin".
        # So we can just continue the expansion from the end of Dev.
        # The state of the SARIMA model after running on Dev is exactly what we need (it has seen Train + Dev).
        
        # We need to re-run the dev backtest to get the state object? 
        # Or we can just run one long backtest on concat(dev, test) and split the results?
        # Yes, running on concat(dev, test) is cleaner for state continuity.
        
        print(f"Running SARIMA Combined Backtest (Dev+Test) for {cc}...")
        combined_eval_df = pd.concat([dev_df, test_df])
        combined_eval_exog = pd.concat([dev_exog, test_exog])
        
        sarima_full_res = run_sarima_backtest(train_df, combined_eval_df, train_exog, combined_eval_exog, order, seasonal_order)
        
        # Split results back to Dev and Test
        # We can split by timestamp
        dev_start = dev_df.index[0]
        test_start = test_df.index[0]
        
        sarima_dev_res = sarima_full_res[sarima_full_res['timestamp'] < test_start]
        sarima_test_res = sarima_full_res[sarima_full_res['timestamp'] >= test_start]
        
        sarima_dev_res.to_csv(os.path.join(OUTPUT_DIR, f'{cc}_forecasts_dev.csv'), index=False)
        sarima_test_res.to_csv(os.path.join(OUTPUT_DIR, f'{cc}_forecasts_test.csv'), index=False)
        
        # Metrics
        metrics_dev = calculate_metrics(sarima_dev_res['y_true'], sarima_dev_res['yhat'], train_df['load'])
        metrics_test = calculate_metrics(sarima_test_res['y_true'], sarima_test_res['yhat'], train_df['load']) # MASE uses train history
        
        # PI Coverage
        # 80% PI
        in_pi_dev = ((sarima_dev_res['y_true'] >= sarima_dev_res['lo']) & (sarima_dev_res['y_true'] <= sarima_dev_res['hi'])).mean() * 100
        in_pi_test = ((sarima_test_res['y_true'] >= sarima_test_res['lo']) & (sarima_test_res['y_true'] <= sarima_test_res['hi'])).mean() * 100
        
        metrics_dev['PI_Coverage_80'] = in_pi_dev
        metrics_test['PI_Coverage_80'] = in_pi_test
        
        print(f"{cc} SARIMA Test Metrics: {metrics_test}")
        
        summary_metrics.append({
            'Country': cc,
            'Set': 'Test',
            'Model': 'SARIMA',
            **metrics_test
        })
        
        # Neural Network (Optional)
        # Run on Dev+Test
        print(f"Running Neural Network Backtest for {cc}...")
        nn_full_res = run_neural_backtest(train_df, combined_eval_df, train_exog, combined_eval_exog)
        
        nn_test_res = nn_full_res[nn_full_res['timestamp'] >= test_start]
        nn_metrics_test = calculate_metrics(nn_test_res['y_true'], nn_test_res['yhat'], train_df['load'])
        
        summary_metrics.append({
            'Country': cc,
            'Set': 'Test',
            'Model': 'GRU',
            **nn_metrics_test
        })
        
        # Save NN results too (optional but good for comparison)
        nn_test_res.to_csv(os.path.join(OUTPUT_DIR, f'{cc}_forecasts_test_gru.csv'), index=False)

    # Save Summary Table
    summary_df = pd.DataFrame(summary_metrics)
    print("\nTest Comparison Table:")
    print(summary_df)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_summary.csv'), index=False)

if __name__ == "__main__":
    main()
