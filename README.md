# Bhopal Weather Predictor (LSTM)

An end-to-end Deep Learning project that predicts the temperature for the next 5 days (120 hours) in Bhopal, India, using a Multi-Step LSTM Neural Network.

## Architecture

- **Extract:** Fetches historical and current weather data from the Open-Meteo API.
- **Transform (ETL):** Processes weather data, scales features, and prepares 48-hour input sequences using Pandas and Scikit-learn.
- **Model:** A Multi-Step LSTM Neural Network built with TensorFlow/Keras. It uses the previous 48 hours of weather observations to predict the next 120 hours of temperature.
- **App:** A Streamlit application that fetches weather data, runs the trained LSTM model, and displays 5-day temperature predictions with daily maximum, minimum, and average temperatures.
- **Visualization:** Interactive Plotly graph showing historical actual temperatures and future LSTM predictions.

## Features

- 5-day temperature forecasting (120 hours).
- Uses the previous 48 hours of weather data as model input.
- Daily predicted high, low, and average temperatures.
- Interactive temperature prediction graph.
- Dark-themed Streamlit interface.
- Live weather data integration through Open-Meteo.

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Open-Meteo API
- Joblib

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/syed-arkaanali/weather-elt-project.git
   ```

2. Navigate to the project directory:

   ```bash
   cd weather-elt-project
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```bash
   streamlit run app/streamlit_app.py
   ```

## Data Source

Weather data provided by [Open-Meteo](https://open-meteo.com/).

## Contact

- **LinkedIn:** [Syed Arkaan Ali](https://www.linkedin.com/in/syedarkaanal)
- **GitHub:** [syed-arkaanali](https://github.com/syed-arkaanali)