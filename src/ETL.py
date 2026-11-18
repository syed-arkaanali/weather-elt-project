from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import pandas as pd
import os

# --- PATH LOGIC ---
# Get the directory where this script (ETL.py) is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


os.makedirs(PROCESSED_DIR, exist_ok=True) # Creates processed folder
os.makedirs(MODELS_DIR, exist_ok=True)
# ------------------

def process_weather_data(file_name: str, output_name: str):
    """Cleans and transforms raw weather CSV data."""
    input_path = os.path.join(RAW_DIR, file_name)
    output_path = os.path.join(PROCESSED_DIR, output_name)
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return None

    print(f"Processing {file_name}...")
    df = pd.read_csv(input_path)
    
    # 1. Convert time to datetime and sort
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # 2. Handle missing values
    df = df.ffill().bfill()
    
    # 3. Feature Engineering: Extract hour and month
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    
    # 4. Save to processed folder
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}, shape={df.shape}\n")
    return df

def prepare_for_lstm(file_name: str, city_name: str ,seq_length=48, forecast_horizon=120):

    input_path = os.path.join(PROCESSED_DIR, file_name)
    df = pd.read_csv(input_path)

    features = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m', 'hour', 'month']
    target_col = 'temperature_2m'

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    scaler_path = os.path.join(MODELS_DIR, f"scaler_{city_name}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    target_idx = features.index(target_col)
    X, y = [], []

    for i in range(len(scaled_data) - seq_length - forecast_horizon):
        X.append(scaled_data[i : i + seq_length])
        y.append(scaled_data[i + seq_length + forecast_horizon, target_idx])

    return np.array(X), np.array(y)