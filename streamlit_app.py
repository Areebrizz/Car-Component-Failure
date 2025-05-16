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

# Title with style
st.markdown("<h1 style='text-align: center; color: #4A90E2; font-weight: bold;'>🚗 Car Component Failure Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Enter component readings to assess risk of failure.</p>", unsafe_allow_html=True)

# UI Layout with columns and card-like boxes
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔥 Sensor Readings")
    temperature = st.slider("Temperature (°C)", 40.0, 160.0, 90.0)
    vibration = st.slider("Vibration (units)", 5.0, 100.0, 50.0)
    usage = st.slider("Usage (hours)", 50.0, 2200.0, 1000.0)
    
    st.markdown("### ⚙️ Component Wear")
    Component_Engine = st.slider("Engine Wear", 0.0, 1.0, 0.05, format="%.2f")
    Component_Transmission = st.slider("Transmission Wear", 0.0, 1.0, 0.05, format="%.2f")

with col2:
    st.markdown("### ⚙️ Component Wear (cont.)")
    Component_BrakePad = st.slider("Brake Pad Wear", 0.0, 1.0, 0.05, format="%.2f")
    Component_Suspension = st.slider("Suspension Wear", 0.0, 1.0, 0.05, format="%.2f")
    Component_Exhaust = st.slider("Exhaust Wear", 0.0, 1.0, 0.05, format="%.2f")
    
    maintenance_due = st.radio("Maintenance Due?", ["Yes", "No"])
    Maintenance_Due_Yes = 1 if maintenance_due == "Yes" else 0
    Maintenance_Due_No = 0 if maintenance_due == "Yes" else 1

# Prepare input for prediction
input_data = {
    "Temperature": temperature,
    "Vibration": vibration,
    "Usage": usage,
    "Component_Engine": Component_Engine,
    "Component_Transmission": Component_Transmission,
    "Component_Brake Pad": Component_BrakePad,
    "Component_Suspension": Component_Suspension,
    "Component_Exhaust": Component_Exhaust,
    "Maintenance_Due_Yes": Maintenance_Due_Yes,
    "Maintenance_Due_No": Maintenance_Due_No,
}

input_df = pd.DataFrame([input_data])[feature_columns]

# Prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][prediction]

# Result Display with color-coded cards
st.markdown("---")
st.markdown("## 🧠 Prediction Result")

if prediction == 1:
    st.markdown(f"<div style='background-color:#FF6B6B; padding:15px; border-radius:8px; color:white; font-weight:bold; font-size:18px; text-align:center;'>❌ Failure Likely — Confidence: {probability*100:.2f}%</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div style='background-color:#4CAF50; padding:15px; border-radius:8px; color:white; font-weight:bold; font-size:18px; text-align:center;'>✅ No Failure Expected — Confidence: {probability*100:.2f}%</div>", unsafe_allow_html=True)

# Feature Importance Visualization
st.markdown("---")
st.markdown("### 🔍 Feature Importances")

importances = model.feature_importances_
features = feature_columns
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(features)), importances[indices], color="#4A90E2", alpha=0.8)
ax.set_xticks(range(len(features)))
ax.set_xticklabels([features[i] for i in indices], rotation=45, ha="right")
ax.set_title("Feature Importances in Predicting Failure", fontsize=16)
ax.set_ylabel("Importance", fontsize=14)
ax.set_xlabel("Features", fontsize=14)
ax.grid(axis="y", linestyle='--', alpha=0.7)

# Adding values on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>Developed by Muhammad Areeb Rizwan 🚀</p>", unsafe_allow_html=True)
