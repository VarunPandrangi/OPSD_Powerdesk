import pandas as pd
import os

# Configuration: We select Germany (DE), France (FR), and Spain (ES) [cite: 9]
COUNTRIES = ['DE', 'FR', 'ES']
DATA_PATH = 'data/time_series_60min_singleindex.csv'
OUTPUT_DIR = 'outputs/'

def load_and_process_data():
    print("Loading dataset... this might take a minute.")
    # Load data, parsing timestamps 
    df = pd.read_csv(DATA_PATH, parse_dates=['utc_timestamp'])
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cc in COUNTRIES:
        print(f"Processing country: {cc}...")
        
        # 1. Select relevant columns
        # We need: utc_timestamp, load, wind (optional), solar (optional) [cite: 10-17]
        # The column names in OPSD usually look like: DE_load_actual_entsoe_transparency
        cols_to_keep = {
            'utc_timestamp': 'timestamp',
            f'{cc}_load_actual_entsoe_transparency': 'load',
            f'{cc}_solar_generation_actual': 'solar',
            f'{cc}_wind_generation_actual': 'wind'
        }
        
        # Check which columns actually exist in the CSV (some might be named slightly differently)
        available_cols = [c for c in cols_to_keep.keys() if c in df.columns]
        
        # Filter the dataframe
        country_df = df[available_cols].copy()
        
        # 2. Rename columns [cite: 12, 13]
        country_df.rename(columns=cols_to_keep, inplace=True)
        
        # 3. Drop rows with missing load data [cite: 18]
        if 'load' in country_df.columns:
            country_df.dropna(subset=['load'], inplace=True)
        else:
            print(f"Warning: No load data found for {cc}")
            continue

        # 4. Sort by timestamp [cite: 18]
        country_df.sort_values('timestamp', inplace=True)
        
        # 5. Save to CSV [cite: 19]
        output_file = os.path.join(OUTPUT_DIR, f'{cc}_cleaned.csv')
        country_df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")
        
        # 6. Basic Sanity Check (Print first few rows)
        print(country_df.head())
        print("-" * 30)

if __name__ == "__main__":
    load_and_process_data()