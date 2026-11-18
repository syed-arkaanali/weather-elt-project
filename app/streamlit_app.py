import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
from tensorflow.keras.models import load_model
from datetime import timedelta
import plotly.graph_objects as go

# --- PATH SETUP ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --- 1. UI SETUP ---
st.set_page_config(page_title="Bhopal Weather Predictor", page_icon="🌤️")
st.title("🌤️ Bhopal 5-Day Weather Predictor")
st.write("Deep Learning LSTM Model vs Official Open-Meteo Forecast")

# Bhopal Coordinates (Hardcoded since we only have Bhopal)
LAT, LON = 23.2599, 77.4126

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = load_model(os.path.join(MODELS_DIR, "weather_lstm_bhopal_multistep.keras"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_bhopal.pkl"))
    return model, scaler

model, scaler = load_assets()

# --- 3. FETCH LIVE FORECAST DATA ---
@st.cache_data(ttl=3600)
def get_latest_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
        "past_days": 2,
        "forecast_days": 1
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    return df

# --- 4. PREDICTION LOGIC ---
if st.button("Generate Forecast", type="primary"):
    with st.spinner("Fetching latest data and predicting 5 days..."):
        df = get_latest_forecast()
        
        features = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m', 'hour', 'month']
        
        # Get the past 48 hours
        seq_length = 48
        input_data = df[features].iloc[:seq_length].values
        scaled_input = scaler.transform(input_data)
        
        # Predict 120 hours at once
        X_pred = np.expand_dims(scaled_input, axis=0)
        pred_scaled = model.predict(X_pred)[0]
        
        # Reverse the scaling
        dummy = np.zeros((120, len(features)))
        dummy[:, 0] = pred_scaled
        pred_temps = scaler.inverse_transform(dummy)[:, 0]
        
        # --- DISPLAY METRICS ---
        st.subheader("Temperature Forecast for Bhopal")
        
        last_known_time = df['time'].iloc[47]
        day_1_name = (last_known_time + timedelta(hours=24)).strftime('%A')
        day_2_name = (last_known_time + timedelta(hours=48)).strftime('%A')
        
        col1, col2 = st.columns(2)
        col1.metric(label=f"{day_1_name} (Tomorrow)", value=f"{int(round(pred_temps[23]))} °C")
        col2.metric(label=f"{day_2_name} (Day After)", value=f"{int(round(pred_temps[47]))} °C")
        
        # --- CHART (Plotly) ---
        st.subheader("Temperature Graph: Past vs Prediction")
        
        past_df = df[['time', 'temperature_2m']].iloc[:48].copy()
        future_times = [df['time'].iloc[47] + pd.Timedelta(hours=i+1) for i in range(120)]
        future_df = pd.DataFrame({'time': future_times, 'prediction': pred_temps})
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=past_df['time'], y=past_df['temperature_2m'],
            mode='lines', name='Actual (Past)',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>Date:</b> %{x|%b %d, %H:%M}<br><b>Temp:</b> %{y:.1f} °C<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_df['time'], y=future_df['prediction'],
            mode='lines', name='Prediction (Future)',
            line=dict(color='#ff4b4b', width=3),
            hovertemplate='<b>Date:</b> %{x|%b %d, %H:%M}<br><b>Predicted Temp:</b> %{y:.1f} °C<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Date & Time",
            yaxis_title="Temperature (°C)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)