import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings

warnings.filterwarnings("ignore")

OUTPUT_DIR = 'outputs/'
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
COUNTRIES = ['DE', 'FR', 'ES']

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(country_code):
    file_path = os.path.join(OUTPUT_DIR, f'{country_code}_cleaned.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please run load_opsd.py first.")
    
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Ensure frequency is set (hourly)
    df = df.asfreq('H')
    # Fill missing if any (forward fill for simplicity in this demo)
    df['load'] = df['load'].ffill()
    return df

def plot_sanity_check(df, country_code):
    # 1.3 Basic sanity plot (last 14 days)
    last_14_days = df.last('14D')
    plt.figure(figsize=(12, 6))
    plt.plot(last_14_days.index, last_14_days['load'], label='Load')
    plt.title(f'{country_code} - Last 14 Days Load')
    plt.xlabel('Timestamp')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(PLOT_DIR, f'{country_code}_sanity_check.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved sanity check plot to {output_path}")

def perform_stl(df, country_code):
    # 1.4.i STL Decomposition
    # period=24 for daily seasonality in hourly data
    # Using a subset (e.g., last year) for clearer visualization if data is huge, 
    # but STL handles large data okay. Plotting might be messy for years.
    # Let's plot the last 30 days for the decomposition visualization to be readable.
    
    subset = df.last('30D')
    stl = STL(subset['load'], period=24)
    res = stl.fit()
    
    fig = res.plot()
    fig.set_size_inches(12, 10)
    plt.suptitle(f'{country_code} - STL Decomposition (Last 30 Days)', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, f'{country_code}_stl_decomposition.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved STL decomposition plot to {output_path}")
    return res

def check_stationarity_and_difference(series):
    # 1.4.ii Stationarity and differencing
    # ADF test on original
    # We assume D=1 because of strong daily seasonality (s=24)
    D = 1
    diff_series = series.diff(24).dropna()
    
    # Check stationarity on seasonally differenced data
    result = adfuller(diff_series)
    p_value = result[1]
    
    d = 0
    if p_value > 0.05:
        # If not stationary, apply regular differencing
        d = 1
        diff_series = diff_series.diff(1).dropna()
        
    return d, D, diff_series

def plot_acf_pacf_plots(series, country_code, d, D):
    # 1.4.iii ACF/PACF
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, ax=ax[0], lags=48, title=f'{country_code} ACF (d={d}, D={D})')
    plot_pacf(series, ax=ax[1], lags=48, title=f'{country_code} PACF (d={d}, D={D})')
    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, f'{country_code}_acf_pacf.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ACF/PACF plot to {output_path}")

def find_best_sarima(series, d, D):
    # 1.4.iv Information criteria (AIC/BIC)
    # Small grid search
    # p: 1, 2
    # q: 0, 1
    # P: 0, 1
    # Q: 0, 1
    
    ps = [1, 2]
    qs = [0, 1]
    Ps = [0, 1]
    Qs = [0, 1]
    
    param_grid = list(itertools.product(ps, qs, Ps, Qs))
    
    best_bic = float('inf')
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    # Use last 4 weeks (approx 672 hours) for model selection to be fast
    train_subset = series.tail(24 * 28) 
    
    print(f"Grid searching {len(param_grid)} combinations on last 4 weeks of data...")
    
    for p, q, P, Q in param_grid:
        try:
            model = SARIMAX(train_subset, 
                            order=(p, d, q), 
                            seasonal_order=(P, D, Q, 24),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit(disp=False)
            
            if results.bic < best_bic:
                best_bic = results.bic
                best_aic = results.aic
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, 24)
        except:
            continue
            
    return best_order, best_seasonal_order, best_bic, best_aic

def main():
    ensure_dir(PLOT_DIR)
    
    results_summary = []

    for cc in COUNTRIES:
        print(f"\nAnalyzing {cc}...")
        try:
            df = load_data(cc)
        except FileNotFoundError as e:
            print(e)
            continue
        
        # 1.3 Sanity Plot
        plot_sanity_check(df, cc)
        
        # 1.4.i STL
        perform_stl(df, cc)
        
        # 1.4.ii Stationarity
        d, D, diff_series = check_stationarity_and_difference(df['load'])
        print(f"Determined differencing: d={d}, D={D} (s=24)")
        
        # 1.4.iii ACF/PACF
        plot_acf_pacf_plots(diff_series, cc, d, D)
        
        # 1.4.iv AIC/BIC Grid Search
        best_order, best_seasonal_order, bic, aic = find_best_sarima(df['load'], d, D)
        
        summary = f"Country: {cc} | Best Model: SARIMA{best_order}x{best_seasonal_order} | BIC: {bic:.2f} | AIC: {aic:.2f}"
        print(summary)
        results_summary.append(summary)
        
    # 1.4.v Document chosen order
    output_file = os.path.join(OUTPUT_DIR, 'model_orders.txt')
    with open(output_file, 'w') as f:
        for line in results_summary:
            f.write(line + '\n')
    
    print(f"\nAnalysis Complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
