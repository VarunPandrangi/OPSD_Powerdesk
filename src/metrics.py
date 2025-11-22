import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def calculate_mase(y_true, y_pred, y_train_history, seasonality=24):
    if len(y_true) == 0: return np.nan
    mae = mean_absolute_error(y_true, y_pred)
    
    y_train_vals = y_train_history.values if hasattr(y_train_history, 'values') else y_train_history
    naive_errors = np.abs(y_train_vals[seasonality:] - y_train_vals[:-seasonality])
    d = np.mean(naive_errors)
    
    return mae / d if d != 0 else np.nan

def calculate_coverage(y_true, lo, hi):
    if len(y_true) == 0: return np.nan
    inside = (y_true >= lo) & (y_true <= hi)
    return inside.mean() * 100
