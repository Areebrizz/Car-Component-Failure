import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Car Component Failure Prediction", layout="centered")

# --- Load model and expected feature columns ---
model = joblib.load("car_component_failure_balanced.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# --- App UI ---
st.title("🚗 Car Component Failure Prediction")
st.markdown("Predict potential failure based on sensor inputs and vehicle conditions.")

st.subheader("🔧 Input Vehicle Data")

# Get user input
engine_temp = st.slider("Engine Temperature (°C)", 40, 120, 75)
brake_pressure = st.slider("Brake Pressure (bar)", 0, 100, 50)
oil_level_low = st.selectbox("Is Oil Level Low?", ["No", "Yes"])
tire_condition = st.selectbox("Tire Condition", ["Bad", "Good"])

# Convert categorical selections
oil_level_low_val = 1 if oil_level_low == "Yes" else 0
tire_condition_val = 1 if tire_condition == "Good" else 0

# --- Prepare input for model ---
input_dict = dict.fromkeys(feature_columns, 0)  # initialize all features to 0

# Set numeric values
input_dict['Engine_Temperature'] = engine_temp
input_dict['Brake_Pressure'] = brake_pressure

# Set one-hot encoded values
input_dict[f'Oil_Level_Low_{oil_level_low_val}'] = 1
input_dict[f'Tire_Condition_Good_{tire_condition_val}'] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# --- Prediction ---
if st.button("🔍 Predict Failure"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ Component is likely to FAIL! (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ Component is likely to be OK. (Confidence: {proba:.2f})")

# Optional: Expandable section to view input
with st.expander("🔎 View Input Data Used"):
    st.dataframe(input_df)
