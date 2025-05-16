import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and features
model = joblib.load("car_component_failure_balanced.pkl11111")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Car Component Failure Prediction", page_icon="🚗", layout="centered")

# Dark mode toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

# CSS for light and dark modes
light_css = """
<style>
    .main {
        background-color: #f9fbfc;
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        font-size: 3rem;
        font-weight: 700;
        color: #2e86de;
        text-align: center;
        margin-bottom: 0.1em;
        font-family: 'Segoe UI Black', sans-serif;
    }
    .subheader {
        font-size: 1.3rem;
        text-align: center;
        color: #34495e;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .input-container {
        background: white;
        border-radius: 15px;
        padding: 20px 25px;
        box-shadow: 0 8px 20px rgb(0 0 0 / 0.05);
        margin-bottom: 2rem;
    }
    .slider-label {
        font-weight: 600;
        color: #34495e;
        margin-bottom: 0.3rem;
        font-size: 1.1rem;
    }
    .slider-value {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: -10px;
        margin-bottom: 10px;
        font-family: monospace;
    }
    .radio-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #34495e;
        margin-bottom: 0.5rem;
    }
    .result-success {
        background-color: #d4efdf;
        color: #27ae60;
        border-radius: 12px;
        padding: 20px;
        font-weight: 700;
        font-size: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 15px rgb(39 174 96 / 0.4);
    }
    .result-fail {
        background-color: #f9d6d5;
        color: #c0392b;
        border-radius: 12px;
        padding: 20px;
        font-weight: 700;
        font-size: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 15px rgb(192 57 43 / 0.4);
    }
    .chart-title {
        font-weight: 700;
        font-size: 1.3rem;
        color: #2e86de;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    footer {
        text-align: center;
        font-size: 0.9rem;
        color: #95a5a6;
        margin-top: 3rem;
        padding-top: 10px;
        border-top: 1px solid #ecf0f1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
"""

dark_css = """
<style>
    .main {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        font-size: 3rem;
        font-weight: 700;
        color: #4ab3f4;
        text-align: center;
        margin-bottom: 0.1em;
        font-family: 'Segoe UI Black', sans-serif;
    }
    .subheader {
        font-size: 1.3rem;
        text-align: center;
        color: #b0bec5;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .input-container {
        background: #1e1e1e;
        border-radius: 15px;
        padding: 20px 25px;
        box-shadow: 0 8px 20px rgb(0 0 0 / 0.9);
        margin-bottom: 2rem;
    }
    .slider-label {
        font-weight: 600;
        color: #b0bec5;
        margin-bottom: 0.3rem;
        font-size: 1.1rem;
    }
    .slider-value {
        color: #90a4ae;
        font-size: 0.9rem;
        margin-top: -10px;
        margin-bottom: 10px;
        font-family: monospace;
    }
    .radio-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #b0bec5;
        margin-bottom: 0.5rem;
    }
    .result-success {
        background-color: #2e7d32;
        color: #a5d6a7;
        border-radius: 12px;
        padding: 20px;
        font-weight: 700;
        font-size: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 15px rgb(165 214 167 / 0.7);
    }
    .result-fail {
        background-color: #c62828;
        color: #ef9a9a;
        border-radius: 12px;
        padding: 20px;
        font-weight: 700;
        font-size: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 15px rgb(239 154 154 / 0.7);
    }
    .chart-title {
        font-weight: 700;
        font-size: 1.3rem;
        color: #4ab3f4;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    footer {
        text-align: center;
        font-size: 0.9rem;
        color: #78909c;
        margin-top: 3rem;
        padding-top: 10px;
        border-top: 1px solid #263238;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
"""

# Inject CSS based on toggle
st.markdown(dark_css if dark_mode else light_css, unsafe_allow_html=True)

# --- Title and subtitle ---
st.markdown('<div class="header">🚗 Car Component Failure Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter component readings to assess risk of failure.</div>', unsafe_allow_html=True)

# --- Input sliders container ---
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="slider-label">Temperature (°C)</div>', unsafe_allow_html=True)
        temperature = st.slider("", 40.0, 160.0, 90.0, key="temp")
        st.markdown(f'<div class="slider-value">{temperature:.1f}</div>', unsafe_allow_html=True)

        st.markdown('<div class="slider-label">Vibration (units)</div>', unsafe_allow_html=True)
        vibration = st.slider("", 5.0, 100.0, 50.0, key="vib")
        st.markdown(f'<div class="slider-value">{vibration:.1f}</div>', unsafe_allow_html=True)

        st.markdown('<div class="slider-label">Usage (hours)</div>', unsafe_allow_html=True)
        usage = st.slider("", 50.0, 2200.0, 1000.0, key="usage")
        st.markdown(f'<div class="slider-value">{usage:.1f}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="slider-label">⚙️ Component Wear (%)</div>', unsafe_allow_html=True)

        st.markdown('<div class="slider-label">Engine Wear</div>', unsafe_allow_html=True)
        component_engine = st.slider("", 0, 100, 5, key="engine_wear")
        st.markdown(f'<div class="slider-value">{component_engine}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="slider-label">Transmission Wear</div>', unsafe_allow_html=True)
        component_transmission = st.slider("", 0, 100, 5, key="trans_wear")
        st.markdown(f'<div class="slider-value">{component_transmission}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="slider-label">Brake Pad Wear</div>', unsafe_allow_html=True)
        component_brakepad = st.slider("", 0, 100, 5, key="brake_wear")
        st.markdown(f'<div class="slider-value">{component_brakepad}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="slider-label">Suspension Wear</div>', unsafe_allow_html=True)
        component_suspension = st.slider("", 0, 100, 5, key="susp_wear")
        st.markdown(f'<div class="slider-value">{component_suspension}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="slider-label">Exhaust Wear</div>', unsafe_allow_html=True)
        component_exhaust = st.slider("", 0, 100, 5, key="exhaust_wear")
        st.markdown(f'<div class="slider-value">{component_exhaust}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="radio-label">Maintenance Due?</div>', unsafe_allow_html=True)
        maintenance_due_yes = st.radio("", ["Yes", "No"], key="maint_due") == "Yes"

    st.markdown('</div>', unsafe_allow_html=True)

# Prepare input for prediction (convert % to 0-1)
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
    "Maintenance_Due_No": 0 if maintenance_due_yes else 1,
}

input_df = pd.DataFrame([input_data])[feature_columns]

# Prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][prediction]

# Result Display with styled boxes
st.markdown("---")
if prediction == 1:
    st.markdown(f'<div class="result-fail">❌ Failure Likely — Confidence: {probability*100:.2f}%</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="result-success">✅ No Failure Expected — Confidence: {probability*100:.2f}%</div>', unsafe_allow_html=True)

# Feature Importance
st.markdown("---")
st.markdown('<div class="chart-title">🔍 Feature Importances</div>', unsafe_allow_html=True)

importances = model.feature_importances_
features = feature_columns
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
color = "#4ab3f4" if dark_mode else "#2e86de"
ax.bar(range(len(features)), importances[indices], color=color, alpha=0.8)
ax.set_xticks(range(len(features)))
ax.set_xticklabels([features[i] for i in indices], rotation=45, ha="right", fontsize=10,
                   color="#b0bec5" if dark_mode else "#34495e")
ax.set_title("Top Feature Importances", fontsize=14, color=color, pad=15)
ax.grid(axis="y", linestyle="--", alpha=0.5)
st.pyplot(fig)

# Footer
footer_color = "#78909c" if dark_mode else "#95a5a6"
st.markdown(
    f"""
    <footer style="color:{footer_color};">
        Developed by Muhammad Areeb Rizwan 🚀
    </footer>
    """,
    unsafe_allow_html=True,
)
