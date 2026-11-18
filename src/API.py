import requests
import pandas as pd
import os

# --- PATH LOGIC ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
# ------------------

def fetch_historical(lat: float, lon: float, start_date: str, end_date: str, city_name: str):
    """Call Historical Weather API to get historical data"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    
    # Save with city name to avoid overwriting
    out_path = os.path.join(RAW_DIR, f"historical_{city_name}_{start_date}_{end_date}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved historical data to {out_path}, shape={df.shape}")
    return out_path

def fetch_forecast(lat: float, lon: float, city_name: str):
    """Call Forecast API to get the latest forecast"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    
    # Save with city name to avoid overwriting
    out_path = os.path.join(RAW_DIR, f"forecast_{city_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved forecast data to {out_path}, shape={df.shape}")
    return out_path