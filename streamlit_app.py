import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and features
model = joblib.load("car_component_failure_balanced.pkl11111")
feature_columns = joblib.load("feature_columns.pkl")

# Page config
st.set_page_config(page_title="Car Component Failure Prediction", page_icon="🚗", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🚗 Car Component Failure Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Enter component readings to assess risk of failure.")

# UI Layout with columns
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Temperature (°C)", 40.0, 160.0, 90.0)
    vibration = st.slider("Vibration (units)", 5.0, 100.0, 50.0)
    usage = st.slider("Usage (hours)", 50.0, 2200.0, 1000.0)
    st.markdown("### ⚙️ Component Wear (%)")
    component_engine = st.slider("Engine Wear (%)", 0, 100, 5)
    component_transmission = st.slider("Transmission Wear (%)", 0, 100, 5)

with col2:
    component_brakepad = st.slider("Brake Pad Wear (%)", 0, 100, 5)
    component_suspension = st.slider("Suspension Wear (%)", 0, 100, 5)
    component_exhaust = st.slider("Exhaust Wear (%)", 0, 100, 5)
    maintenance_due_yes = st.radio("Maintenance Due?", ["Yes", "No"]) == "Yes"

maintenance_due_no = not maintenance_due_yes

# Prepare input for prediction
input_data = {
    "Temperature": temperature,
    "Vibration": vibration,
    "Usage": usage,
    "Component_Engine": component_engine / 100,
    "Component_Transmission": component_transmission / 100,
    "Component_Brake Pad": component_brakepad / 100,
    "Component_Suspension": component_suspension / 100,
    "Component_Exhaust": component_exhaust / 100,
    "Maintenance_Due_Yes": 1 if maintenance_due_yes else 0,
    "Maintenance_Due_No": 1 if maintenance_due_no else 0,
}

input_df = pd.DataFrame([input_data])[feature_columns]

# Prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][prediction]

# Result Display
st.markdown("---")
st.markdown("## 🧠 Prediction Result")

if prediction == 1:
    st.error(f"❌ Failure Likely — Confidence: {probability*100:.2f}%")
else:
    st.success(f"✅ No Failure Expected — Confidence: {probability*100:.2f}%")

# Feature Importance
st.markdown("---")
st.markdown("### 🔍 Feature Importances")

importances = model.feature_importances_
features = feature_columns
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(features)), importances[indices], color="#4A90E2")
ax.set_xticks(range(len(features)))
ax.set_xticklabels([features[i] for i in indices], rotation=45, ha="right")
ax.set_title("Top Feature Importances")
st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Developed by Muhammad Areeb Rizwan 🚀</p>", unsafe_allow_html=True)
