
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import plotly.graph_objects as go

from datetime import timedelta
from tensorflow.keras.models import load_model


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bhopal Weather Predictor",
    page_icon="🌤️",
    layout="wide"
)

LAT, LON = 23.2599, 77.4126

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ==========================================
# 2. BLACK THEME CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #111111;
    color: white;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background-color: #111111;
}

h1, h2, h3, h4, p, label, span {
    color: white;
}

div.stButton > button {
    background-color: #303030;
    color: white !important;
    border: 1px solid #555555;
    border-radius: 12px;
    font-weight: 600;
    padding: 10px 22px;
}

div.stButton > button:hover {
    background-color: #424242;
    border-color: #888888;
    color: white !important;
}

div[data-testid="stMetric"] {
    background-color: #242424;
    border: 1px solid #383838;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: white !important;
}

hr {
    border-color: #383838;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL AND SCALER
# ==========================================

@st.cache_resource
def load_assets():

    model = load_model(
        os.path.join(
            MODELS_DIR,
            "weather_lstm_bhopal_multistep.keras"
        )
    )

    scaler = joblib.load(
        os.path.join(
            MODELS_DIR,
            "scaler_bhopal.pkl"
        )
    )

    return model, scaler


# ==========================================
# 4. FETCH WEATHER DATA
# ==========================================

@st.cache_data(ttl=1800)
def get_weather_data():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LAT,
        "longitude": LON,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "weather_code,"
            "wind_speed_10m"
        ),

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),

        "timezone": "auto",
        "past_days": 2,
        "forecast_days": 1
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    current = data["current"]

    df = pd.DataFrame(data["hourly"])

    df["time"] = pd.to_datetime(df["time"])

    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month

    return current, df


# ==========================================
# 5. GENERATE LSTM PREDICTIONS
# ==========================================

def generate_predictions(df, current_time, model, scaler):

    features = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "hour",
        "month"
    ]

    history_df = df[
        df["time"] <= current_time
    ].tail(48).copy()

    if len(history_df) != 48:
        raise ValueError(
            "48 historical hourly observations are required."
        )

    input_data = history_df[features].values

    scaled_input = scaler.transform(input_data)

    X_pred = np.expand_dims(
        scaled_input,
        axis=0
    )

    pred_scaled = model.predict(
        X_pred,
        verbose=0
    )[0]

    dummy = np.zeros(
        (120, len(features))
    )

    dummy[:, 0] = pred_scaled

    pred_temps = scaler.inverse_transform(
        dummy
    )[:, 0]

    future_times = [
        current_time + timedelta(hours=i)
        for i in range(1, 121)
    ]

    future_df = pd.DataFrame({
        "time": future_times,
        "predicted_temperature": pred_temps
    })

    return history_df, future_df


# ==========================================
# 6. HEADER
# ==========================================

st.markdown("# 🌤️ Weather")
st.markdown("📍 Bhopal, Madhya Pradesh")

st.divider()


# ==========================================
# 7. SESSION STATE
# ==========================================

if "weather_data" not in st.session_state:
    st.session_state.weather_data = None


# ==========================================
# 8. GENERATE FORECAST
# ==========================================

if st.button("Get Weather Forecast", type="primary"):

    with st.spinner("Generating 5-day temperature predictions..."):

        try:

            model, scaler = load_assets()

            current, df = get_weather_data()

            current_time = pd.to_datetime(
                current["time"]
            )

            history_df, future_df = generate_predictions(
                df,
                current_time,
                model,
                scaler
            )

            st.session_state.weather_data = {
                "history": history_df,
                "future": future_df
            }

        except Exception as e:

            st.error(f"Forecast error: {e}")


# ==========================================
# 9. DISPLAY 5-DAY PREDICTIONS
# ==========================================

if st.session_state.weather_data is not None:

    history_df = st.session_state.weather_data["history"]

    future_df = st.session_state.weather_data["future"]

    # --------------------------------------
    # FIVE DAY FORECAST
    # --------------------------------------

    st.markdown("## 5-Day Temperature Prediction")

    st.caption("Daily predicted high and low temperatures")

    daily_cols = st.columns(5)

    for day in range(5):

        day_data = future_df.iloc[
            day * 24:(day + 1) * 24
        ]

        max_temp = day_data[
            "predicted_temperature"
        ].max()

        min_temp = day_data[
            "predicted_temperature"
        ].min()

        avg_temp = day_data[
            "predicted_temperature"
        ].mean()

        day_name = day_data["time"].iloc[0].strftime("%A")

        day_date = day_data["time"].iloc[0].strftime("%d %b")

        with daily_cols[day]:

            with st.container(border=True):

                st.markdown(f"**{day_name}**")

                st.caption(day_date)

                st.metric(
                    "Predicted High",
                    f"{max_temp:.1f}°C"
                )

                st.metric(
                    "Predicted Low",
                    f"{min_temp:.1f}°C"
                )

                st.caption(
                    f"Average: {avg_temp:.1f}°C"
                )

    # --------------------------------------
    # TEMPERATURE GRAPH
    # --------------------------------------

    st.markdown("## Temperature Prediction Graph")

    fig = go.Figure()

    # Actual temperature

    fig.add_trace(
        go.Scatter(
            x=history_df["time"],
            y=history_df["temperature_2m"],
            mode="lines",
            name="Actual Temperature",
            line=dict(
                color="#8AB4F8",
                width=3
            )
        )
    )

    # Predicted temperature

    fig.add_trace(
        go.Scatter(
            x=future_df["time"],
            y=future_df["predicted_temperature"],
            mode="lines",
            name="LSTM Prediction",
            line=dict(
                color="#FF8A80",
                width=3
            )
        )
    )

    fig.update_layout(
        height=430,
        plot_bgcolor="#242424",
        paper_bgcolor="#242424",

        font=dict(
            color="white",
            size=12
        ),

        hovermode="x unified",

        xaxis_title="Date and Time",
        yaxis_title="Temperature (°C)",

        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        ),

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    fig.update_xaxes(
        color="white",
        showgrid=True,
        gridcolor="#444444",
        zerolinecolor="#444444"
    )

    fig.update_yaxes(
        color="white",
        showgrid=True,
        gridcolor="#444444",
        zerolinecolor="#444444"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.caption(
        "Weather data: Open-Meteo | "
        "Temperature predictions: LSTM Deep Learning Model"
    )

else:

    st.markdown("## Bhopal 5-Day Weather Prediction")

    st.write(
        "Generate temperature predictions for the next five days "
        "using your trained LSTM model."
    )

    st.info("Click **Get Weather Forecast** to get started.")

