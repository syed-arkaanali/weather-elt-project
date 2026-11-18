# Bhopal weather predictor (LSTM)

An end-to-end Deep Learning pipeline that predicts the temperature for the next 5 days (120 hours) in Bhopal, India.

## Architecture
**Extract**: Fetches historical and live weather data from the Open-Meteo API.
**Transform (ETL)**: Cleans, Scales and creates 48-hour sequences using Pandas and Scikit-learn.
**Model**: A Multi-Step LSTM Neural Network (TensorFlow/Keras) trained to output 120 hours of temperature at once.
**App**: A streamlit UI that fetches live data, runs the model, and compares the forecast against the official Open-Meteo forecast using interactive Plotly charts.

## How to run Locally
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/weather-elt-project.git`

2. Install dependencies: `pip install -r requirements.txt`

3. Run the app: `streamlit run app/streamlit_app.py`

## Data Source
Data provided by [Open-Meteo](https://open-meteo.com/)

## Contact
**LinkedIn**: [Syed Arkaan Ali](www.linkedin.com/in/syedarkaanal)
**Phone**: +91 89820 52362
**GitHub**: [syed-arkaanali](https://github.com/syed-arkaanali)