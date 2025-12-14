import streamlit as st
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Weather Condition Predictor",
    page_icon="🌦️",
    layout="centered"
)

# --- HELPER: Create Model if Missing ---
def ensure_model_exists():
    """
    Checks if 'weather_model.pkl' exists. 
    If not, trains a dummy model and saves it so the app doesn't crash.
    """
    if not os.path.exists('weather_model.pkl'):
        # Generate dummy data for 4 features: [precipitation, temp_max, temp_min, wind]
        X_dummy = np.random.rand(100, 4) * 20
        # Target labels expected by the app
        y_dummy = np.random.choice(['rain', 'sun', 'snow', 'fog', 'drizzle'], 100)
        
        # Train and save a basic model
        dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy_model.fit(X_dummy, y_dummy)
        joblib.dump(dummy_model, 'weather_model.pkl')
        return True
    return False

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    # 1. Ensure model exists before trying to load it
    created_new = ensure_model_exists()
    
    try:
        # 2. Load the uploaded scaler
        # Note: 'scaler.pkl' must be in the directory. 
        # If missing, we'll return None for scaler.
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
        else:
            scaler = None
        
        # 3. Load the model
        model = joblib.load('weather_model.pkl')
        
        return scaler, model, created_new
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, False

# Load everything
scaler, model, is_new_model = load_assets()

# --- APP INTERFACE ---
st.title("Weather Binary Predictor 🌦️")
st.markdown("### Predicts: 0 (Rain/Snow/Sun) or 1 (Drizzle/Fog)")

if is_new_model:
    st.warning("⚠️ 'weather_model.pkl' was not found, so a temporary model was created automatically. Predictions will be random until you replace it with your actual trained model.")

if scaler is None:
    st.error("❌ 'scaler.pkl' is missing! Please upload the scaler file to the same folder as this script.")
    st.stop()  # Stop app execution if scaler is missing

# --- USER INPUTS ---
# Inputs match the scaler's expected features: [precipitation, temp_max, temp_min, wind]
col1, col2 = st.columns(2)

with col1:
    precipitation = st.number_input("Precipitation", min_value=0.0, value=0.0, step=0.1)
    temp_max = st.number_input("Max Temperature (°C)", value=15.0, step=0.1)

with col2:
    temp_min = st.number_input("Min Temperature (°C)", value=5.0, step=0.1)
    wind = st.number_input("Wind Speed (km/h)", min_value=0.0, value=3.0, step=0.1)

# --- PREDICTION LOGIC ---
if st.button("Predict Category", type="primary"):
    if model is not None and scaler is not None:
        try:
            # 1. Prepare and Scale Input
            # Input order: precipitation, temp_max, temp_min, wind
            input_data = np.array([[precipitation, temp_max, temp_min, wind]])
            scaled_data = scaler.transform(input_data)
            
            # 2. Predict Weather Condition
            prediction_class = model.predict(scaled_data)[0]
            
            # 3. Map to Binary Output (0 or 1)
            # Normalize string to lowercase to ensure matching works
            pred_str = str(prediction_class).lower().strip()
            
            # Define mappings
            group_0 = ['rain', 'snow', 'sun']  # Target 0
            group_1 = ['drizzle', 'fog']       # Target 1
            
            final_output = None
            
            st.divider()
            st.write(f"**Model predicted raw class:** `{pred_str}`")
            
            if pred_str in group_0:
                final_output = 0
                st.success(f"### Output: {final_output}")
                st.write(f"(Classified as **0** because it is **{pred_str}**)")
            elif pred_str in group_1:
                final_output = 1
                st.success(f"### Output: {final_output}")
                st.write(f"(Classified as **1** because it is **