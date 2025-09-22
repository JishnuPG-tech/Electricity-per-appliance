
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# --- Configuration ---
# Point to the new V2 model and preprocessor
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SAVED_MODEL_DIR = os.path.join(PROJECT_DIR, "saved_model")
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model_v2.keras")
PREPROCESSOR_PATH = os.path.join(SAVED_MODEL_DIR, "preprocessor_v2.joblib")

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_model_and_preprocessor():
    """Loads the trained V2 model and preprocessor."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        st.error("Please ensure the 'run_project.py' script has been run successfully to generate the model_v2.keras and preprocessor_v2.joblib files.")
        return None, None

model, preprocessor = load_model_and_preprocessor()

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Energy Predictor")
st.title("💡 Household Electricity Bill Disaggregation")

st.write("""
Enter details for all your household appliances below. The app will predict each appliance's consumption 
and then estimate its contribution to your total monthly electricity bill.
""")

if model is None or preprocessor is None:
    st.stop()

# --- Get Categories from Preprocessor ---
cat_preprocessor = preprocessor.named_transformers_['cat']
appliance_names = cat_preprocessor.categories_[0].tolist()
location_types = cat_preprocessor.categories_[1].tolist()
income_levels = cat_preprocessor.categories_[2].tolist()
seasons = cat_preprocessor.categories_[3].tolist()
usage_patterns = cat_preprocessor.categories_[4].tolist()

# --- Household Information (single input) ---
st.header("Household Information")
col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    household_size = st.slider("Household Size", min_value=1, max_value=10, value=4, help="Number of people in household")
with col_h2:
    location_type = st.selectbox("Location Type", options=location_types)
with col_h3:
    income_level = st.selectbox("Income Level", options=income_levels)

col_h4, col_h5 = st.columns(2)
with col_h4:
    season = st.selectbox("Current Season", options=seasons)
with col_h5:
    monthly_bill = st.number_input("Last Monthly Electricity Bill (in your currency)", min_value=0.0, value=1500.0, step=100.0)

st.markdown("---")
st.header("Appliance Details")

# List to store appliance inputs
appliances_data = []

# Create input blocks for multiple appliances
NUM_APPLIANCE_BLOCKS = 3 # Fixed number of blocks for simplicity

for i in range(NUM_APPLIANCE_BLOCKS):
    with st.expander(f"Appliance {i+1} Details", expanded=(i==0)): # Expand first one by default
        st.write(f"Enter details for Appliance {i+1}. Leave blank if not used.")
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            appliance_name = st.selectbox(f"Appliance Type {i+1}", options=[""] + appliance_names, key=f"app_type_{i}") # Add empty option
            power_rating_watts = st.number_input(f"Power Rating (Watts) {i+1}", min_value=0, value=0, key=f"power_{i}")
            daily_usage_hours = st.number_input(f"Usage Hours per Day {i+1}", min_value=0.0, max_value=24.0, value=0.0, step=0.5, key=f"usage_{i}")
        with col_a2:
            star_rating = st.slider(f"Efficiency Rating (1-5 stars) {i+1}", min_value=1, max_value=5, value=3, key=f"star_{i}")
            appliance_age_years = st.slider(f"Appliance Age (Years) {i+1}", min_value=0, max_value=20, value=0, key=f"age_{i}")
            usage_pattern = st.selectbox(f"Usage Pattern {i+1}", options=[""] + usage_patterns, key=f"pattern_{i}")

        # Only add to data if appliance name is selected
        if appliance_name != "":
            appliances_data.append({
                "appliance_name": appliance_name,
                "power_rating_watts": power_rating_watts,
                "daily_usage_hours": daily_usage_hours,
                "star_rating": star_rating,
                "appliance_age_years": appliance_age_years,
                "household_size": household_size, # Household-level features are same for all appliances
                "location_type": location_type,
                "income_level": income_level,
                "season": season,
                "usage_pattern": usage_pattern,
                "monthly_bill": monthly_bill # Monthly bill is also same for all appliances
            })

# --- Prediction ---
if st.button("Predict Consumption & Allocate Bill"):
    if not appliances_data:
        st.warning("Please enter details for at least one appliance.")
    else:
        input_df = pd.DataFrame(appliances_data)
        
        try:
            processed_input = preprocessor.transform(input_df)
            predictions_kwh = model.predict(processed_input).flatten() # Get predictions for all appliances
            
            # Calculate total predicted consumption
            total_predicted_daily_kwh = predictions_kwh.sum()
            total_predicted_monthly_kwh = total_predicted_daily_kwh * 30

            st.success("### Bill Allocation Results")

            if total_predicted_monthly_kwh > 0:
                # Calculate dynamic price per kWh based on user's bill and total predicted consumption
                # This assumes the user's bill corresponds to the sum of predicted consumption
                dynamic_price_per_kwh = monthly_bill / total_predicted_monthly_kwh
                st.info(f"Calculated average price per kWh from your inputs: ₹{dynamic_price_per_kwh:.2f}")
            else:
                dynamic_price_per_kwh = 0 # Avoid division by zero
                st.warning("Total predicted consumption is zero. Cannot calculate cost allocation.")

            results_df = pd.DataFrame()
            results_df["Appliance"] = input_df["appliance_name"]
            results_df["Predicted Daily kWh"] = predictions_kwh
            results_df["Estimated Monthly Cost (₹)"] = predictions_kwh * 30 * dynamic_price_per_kwh
            
            # Display results table
            st.table(results_df.style.format({"Predicted Daily kWh": "{:.2f}", "Estimated Monthly Cost (₹)": "{:.2f}"}))

            st.markdown("--- ")
            st.subheader("Summary")
            st.metric(label="Total Predicted Daily Consumption", value=f"{total_predicted_daily_kwh:.2f} kWh")
            st.metric(label="Total Estimated Monthly Cost", value=f"₹{results_df['Estimated Monthly Cost (₹)'].sum():.2f}")

            st.markdown("--- ")
            st.subheader("Bill Forecasting (Next Cycle Prediction)")
            st.info(f"Next month's expected bill ~₹{results_df['Estimated Monthly Cost (₹)'].sum():.2f}")

            st.markdown("--- ")
            st.subheader("💡 Usage Optimization Suggestions")
            st.write("Here are some tips to help you reduce your electricity expenses:")
            st.markdown("""
            *   **Shift Usage:** Consider running high-power appliances (like washing machines, dishwashers, or water heaters) during off-peak hours if your electricity provider offers time-of-use tariffs. (Check with your local provider for specific off-peak times).
            *   **Unplug Devices:** Many electronics consume power even when turned off (phantom load). Unplug chargers, TVs, and other devices when not in use.
            *   **Upgrade Appliances:** Older appliances, especially refrigerators and air conditioners, can be significantly less energy-efficient. Consider upgrading to newer, higher star-rated models.
            *   **Optimize Usage:** Ensure your refrigerator door seals are tight, clean AC filters regularly, and use natural light when possible.
            """)

            st.info("Note: Estimated costs are based on the proportional allocation of your entered monthly bill across the predicted consumption of your appliances.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
