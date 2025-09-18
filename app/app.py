import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SAVED_MODEL_DIR = os.path.join(PROJECT_DIR, "saved_model")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Disaggregation Model Paths
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model_v2.keras")
PREPROCESSOR_PATH = os.path.join(SAVED_MODEL_DIR, "preprocessor_v2.joblib")

# Forecasting Model Paths
FORECASTING_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "forecasting_model.keras")
FORECASTING_SCALER_PATH = os.path.join(SAVED_MODEL_DIR, "forecasting_scaler.joblib")
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, "monthly_bills.csv")
LOOK_BACK = 3

# --- Load Disaggregation Model and Preprocessor ---
@st.cache_resource
def load_disaggregation_assets():
    """Loads the trained disaggregation model and preprocessor."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading disaggregation model or preprocessor: {e}")
        return None, None

# --- Load Forecasting Model and Assets ---
@st.cache_resource
def load_forecasting_assets():
    """Loads the forecasting model, scaler, and historical data."""
    try:
        forecasting_model = tf.keras.models.load_model(FORECASTING_MODEL_PATH)
        forecasting_scaler = joblib.load(FORECASTING_SCALER_PATH)
        historical_data = pd.read_csv(HISTORICAL_DATA_PATH)
        return forecasting_model, forecasting_scaler, historical_data
    except Exception as e:
        st.error(f"Error loading forecasting assets: {e}")
        st.warning("Please ensure the forecasting model is trained by running 'python -m src.train_forecaster' from the project root.")
        return None, None, None

# --- Forecasting Function ---
def run_forecasting(model, scaler, data):
    """Uses the last few months of data to predict the next month's bill."""
    try:
        # Get the last LOOK_BACK values from the historical data
        last_months_bills = data['total_bill'].values[-LOOK_BACK:]
        
        # Scale the input
        scaled_input = scaler.transform(last_months_bills.reshape(-1, 1))
        
        # Reshape for LSTM: [samples, time steps, features]
        input_for_pred = np.reshape(scaled_input, (1, LOOK_BACK, 1))
        
        # Predict
        predicted_scaled = model.predict(input_for_pred)
        
        # Inverse transform to get the actual bill amount
        predicted_bill = scaler.inverse_transform(predicted_scaled)
        
        return predicted_bill[0, 0]
    except Exception as e:
        st.error(f"Forecasting failed: {e}")
        return None

# --- Load all assets ---
model, preprocessor = load_disaggregation_assets()
forecasting_model, forecasting_scaler, historical_data = load_forecasting_assets()

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Energy Predictor")
st.title("💡 Household Electricity Bill Disaggregation & Forecasting")

st.write("""
Enter details for your household appliances to see a breakdown of your bill. 
The app will also forecast your next month's bill based on historical trends.
""")

if model is None or preprocessor is None or forecasting_model is None:
    st.stop()

# --- Get Categories from Preprocessor ---
cat_preprocessor = preprocessor.named_transformers_['cat']
appliance_names = cat_preprocessor.categories_[0].tolist()
location_types = cat_preprocessor.categories_[1].tolist()
income_levels = cat_preprocessor.categories_[2].tolist()
seasons = cat_preprocessor.categories_[3].tolist()
usage_patterns = cat_preprocessor.categories_[4].tolist()

# --- Household Information ---
st.header("Household Information")
col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    household_size = st.slider("Household Size", 1, 10, 4)
with col_h2:
    location_type = st.selectbox("Location Type", options=location_types)
with col_h3:
    income_level = st.selectbox("Income Level", options=income_levels)

col_h4, col_h5 = st.columns(2)
with col_h4:
    season = st.selectbox("Current Season", options=seasons)
with col_h5:
    monthly_bill = st.number_input("Last Monthly Electricity Bill (in your currency)", 0.0, value=1500.0, step=100.0)

st.markdown("---")
st.header("Appliance Details")

# --- Appliance Inputs ---
appliances_data = []
NUM_APPLIANCE_BLOCKS = 3
for i in range(NUM_APPLIANCE_BLOCKS):
    with st.expander(f"Appliance {i+1} Details", expanded=(i==0)):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            appliance_name = st.selectbox(f"Appliance Type {i+1}", [""] + appliance_names, key=f"app_type_{i}")
            if appliance_name:
                power_rating_watts = st.number_input(f"Power Rating (Watts) {i+1}", 0, value=100, key=f"power_{i}")
                daily_usage_hours = st.number_input(f"Usage Hours per Day {i+1}", 0.0, 24.0, 1.0, 0.5, key=f"usage_{i}")
        with col_a2:
            if appliance_name:
                star_rating = st.slider(f"Efficiency Rating (1-5 stars) {i+1}", 1, 5, 3, key=f"star_{i}")
                appliance_age_years = st.slider(f"Appliance Age (Years) {i+1}", 0, 20, 5, key=f"age_{i}")
                usage_pattern = st.selectbox(f"Usage Pattern {i+1}", [""] + usage_patterns, key=f"pattern_{i}")

        if appliance_name and usage_pattern:
            appliances_data.append({
                "appliance_name": appliance_name, "power_rating_watts": power_rating_watts,
                "daily_usage_hours": daily_usage_hours, "star_rating": star_rating,
                "appliance_age_years": appliance_age_years, "household_size": household_size,
                "location_type": location_type, "income_level": income_level, "season": season,
                "usage_pattern": usage_pattern, "monthly_bill": monthly_bill
            })

# --- Prediction and Forecasting ---
if st.button("Run Analysis", type="primary"):
    # --- Bill Forecasting ---
    st.subheader("Bill Forecasting (Next Cycle Prediction)")
    predicted_next_bill = run_forecasting(forecasting_model, forecasting_scaler, historical_data)
    if predicted_next_bill:
        st.info(f"Next month's expected bill ~₹{predicted_next_bill:.2f}")
    st.markdown("--- ")

    # --- Bill Disaggregation ---
    if not appliances_data:
        st.warning("Please enter details for at least one appliance to see the bill breakdown.")
    else:
        input_df = pd.DataFrame(appliances_data)
        try:
            processed_input = preprocessor.transform(input_df)
            predictions_kwh = model.predict(processed_input).flatten()
            
            total_predicted_daily_kwh = predictions_kwh.sum()
            total_predicted_monthly_kwh = total_predicted_daily_kwh * 30

            st.success("### Bill Allocation Results")

            if total_predicted_monthly_kwh > 0:
                dynamic_price_per_kwh = monthly_bill / total_predicted_monthly_kwh
                st.info(f"Calculated average price per kWh from your inputs: ₹{dynamic_price_per_kwh:.2f}")
            else:
                dynamic_price_per_kwh = 0
                st.warning("Total predicted consumption is zero. Cannot calculate cost allocation.")

            results_df = pd.DataFrame({
                "Appliance": input_df["appliance_name"],
                "Predicted Daily kWh": predictions_kwh,
                "Estimated Monthly Cost (₹)": predictions_kwh * 30 * dynamic_price_per_kwh
            })
            
            st.table(results_df.style.format({"Predicted Daily kWh": "{:.2f}", "Estimated Monthly Cost (₹)": "{:.2f}"}))

            st.metric(label="Total Estimated Monthly Cost", value=f"₹{results_df['Estimated Monthly Cost (₹)'].sum():.2f}")
            st.info("Note: Estimated costs are based on the proportional allocation of your entered monthly bill.")

        except Exception as e:
            st.error(f"An error occurred during disaggregation: {e}")

    st.markdown("--- ")
    st.subheader("💡 Usage Optimization Suggestions")
    st.markdown(""" 
    *   **Shift Usage:** Run high-power appliances during off-peak hours.
    *   **Unplug Devices:** Avoid phantom load from idle electronics.
    *   **Upgrade Appliances:** Newer models are more energy-efficient.
    *   **Optimize Usage:** Maintain appliances and use natural light.
    """)