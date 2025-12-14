import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Weather Condition Predictor", page_icon="Hz")

# 1. Load the Scaler and Model
@st.cache_resource
def load_assets():
    try:
        # Load the uploaded scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load the weather model (You must have this file in the folder)
        # Ensure this model predicts: 'rain', 'sun', 'snow', 'fog', 'drizzle'
        model = joblib.load('weather_model.pkl') 
        
        return scaler, model
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please make sure 'scaler.pkl' and 'weather_model.pkl' are in the same folder.")
        return None, None
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

scaler, model = load_assets()

# 2. App Interface
st.title("Weather Binary Predictor 🌦️")
st.markdown("""
This app predicts a binary weather category based on weather conditions.
* **0**: Rain, Snow, Sun
* **1**: Drizzle, Fog
""")

# 3. User Inputs (Matching scaler features: precipitation, temp_max, temp_min, wind)
col1, col2 = st.columns(2)

with col1:
    precipitation = st.number_input("Precipitation", min_value=0.0, value=0.0, step=0.1)
    temp_max = st.number_input("Max Temperature", value=15.0, step=0.1)

with col2:
    temp_min = st.number_input("Min Temperature", value=5.0, step=0.1)
    wind = st.number_input("Wind Speed", min_value=0.0, value=3.0, step=0.1)

# 4. Prediction Logic
if st.button("Predict Category"):
    if scaler is not None and model is not None:
        try:
            # Prepare data in the correct order for the scaler
            input_data = np.array([[precipitation, temp_max, temp_min, wind]])
            
            # Scale the data using your uploaded scaler
            scaled_data = scaler.transform(input_data)
            
            # Predict the weather condition (e.g., 'rain', 'sun', etc.)
            prediction_class = model.predict(scaled_data)[0]
            
            # 5. Apply Binary Mapping Logic
            # Normalize to lowercase string for comparison
            pred_str = str(prediction_class).lower().strip()
            
            # Define the groups
            group_0 = ['rain', 'snow', 'sun']
            group_1 = ['drizzle', 'fog']
            
            final_output = None
            
            if pred_str in group_0:
                final_output = 0
                description = "Rain / Snow / Sun"
                status = "info"
            elif pred_str in group_1:
                final_output = 1
                description = "Drizzle / Fog"
                status = "success"
            else:
                # Fallback if model predicts something else
                final_output = f"Unmapped ({pred_str})"
                description = "Unknown Category"
                status = "warning"

            # 6. Display Result
            st.divider()
            st.subheader(f"Prediction Output: {final_output}")
            
            if status == "success":
                st.success(f"Category 1 ({description}) detected based on '{pred_str}'.")
            elif status == "info":
                st.info(f"Category 0 ({description}) detected based on '{pred_str}'.")
            else:
                st.warning(f"Model predicted '{pred_str}', which is not in the mapping list.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.warning("Please upload 'weather_model.pkl' to run predictions.")