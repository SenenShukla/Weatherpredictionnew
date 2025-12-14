import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the scaler and the best model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Define the weather mapping (obtained from previous steps)
weather_mapping = {
    'drizzle': 0,
    'fog': 1,
    'rain': 2,
    'snow': 3,
    'sun': 4
}

# Inverse mapping for displaying results
inverse_weather_mapping = {v: k for k, v in weather_mapping.items()}

st.title('Seattle Weather Prediction App')
st.write('Enter the weather features below to predict the weather type and if it will rain.')

# User input for features
precipitation = st.slider('Precipitation (mm)', 0.0, 50.0, 5.0)
temp_max = st.slider('Maximum Temperature (°C)', -10.0, 40.0, 15.0)
temp_min = st.slider('Minimum Temperature (°C)', -15.0, 30.0, 5.0)
wind = st.slider('Wind (m/s)', 0.0, 10.0, 3.0)

# Create a DataFrame from user input
input_data = pd.DataFrame([[precipitation, temp_max, temp_min, wind]],
                            columns=['precipitation', 'temp_max', 'temp_min', 'wind'])

# Scale the input data
scaled_input_data = scaler.transform(input_data)

# Make prediction
prediction_encoded = best_model.predict(scaled_input_data)
predicted_weather = inverse_weather_mapping[prediction_encoded[0]]

st.subheader('Prediction Results:')
st.write(f'Predicted Weather Type: **{predicted_weather.capitalize()}**')

# Convert prediction to binary 'rain' or 'not rain'
# Based on the original mapping: rain is 2, drizzle is 0, fog is 1, snow is 3, sun is 4
# For simplicity, we can define 'rain' if the prediction is 'rain' or 'drizzle'

is_rain = 'No'
if predicted_weather == 'rain' or predicted_weather == 'drizzle':
    is_rain = 'Yes'

st.write(f'Will it Rain?: **{is_rain}**')
