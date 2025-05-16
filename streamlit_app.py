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
    Temperature = st.slider("Temperature", 0.0, 150.0, 20.0)
    Vibration = st.slider("Vibration", 0.0, 1.0, 0.1)
    Usage = st.slider("Usage (Hours)", 0.0, 1.0, 0.05)
    Component_Engine = st.slider("Engine Wear", 0.0, 1.0, 0.05)
    Component_Transmission = st.slider("Transmission Wear", 0.0, 1.0, 0.05)

with col2:
    Component_BrakePad = st.slider("Brake Pad Wear", 0.0, 1.0, 0.05)
    Component_Suspension = st.slider("Suspension Wear", 0.0, 1.0, 0.05)
    Component_Exhaust = st.slider("Exhaust Wear", 0.0, 1.0, 0.05)
    Maintenance_Due_Yes = st.radio("Maintenance Due?", ["Yes", "No"]) == "Yes"
    
# Prepare input for prediction
input_data = {
    "Temperature": Temperature,
    "Vibration": Vibration,
    "Usage": Usage,
    "Component_Engine": Component_Engine,
    "Component_Transmission": Component_Transmission,
    "Component_Brake Pad": Component_BrakePad,
    "Component_Suspension": Component_Suspension,
    "Component_Exhaust": Component_Exhaust,
    "Maintenance_Due_Yes": 1 if Maintenance_Due_Yes else 0,
    "Maintenance_Due_No": 0 if Maintenance_Due_Yes else 1,
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
