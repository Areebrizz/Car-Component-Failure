import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Car Component Failure Prediction", page_icon="🚗", layout="wide")

# --- Dark Mode Toggle ---
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

# --- CSS Themes ---
light_css = """
<style>
body, .main {
    background-color: #f9fbfc;
    color: #2c3e50;
}
.header {
    font-size: 2.8rem;
    font-weight: 800;
    color: #2e86de;
    text-align: center;
    margin-top: 1.5rem;
    margin-bottom: 0.2rem;
    letter-spacing: 1px;
}
.subheader {
    font-size: 1.2rem;
    text-align: center;
    color: #34495e;
    margin-bottom: 2rem;
    font-weight: 500;
    line-height: 1.4;
}
.input-column {
    padding: 1rem;
}
.slider-label, .radio-label {
    margin-top: 1rem;
    font-weight: 600;
}
.slider-value {
    margin-bottom: 1rem;
    color: #2980b9;
}
.result-fail {
    font-size: 1.5rem;
    font-weight: bold;
    color: #e74c3c;
    text-align: center;
    margin-top: 2rem;
}
.result-success {
    font-size: 1.5rem;
    font-weight: bold;
    color: #27ae60;
    text-align: center;
    margin-top: 2rem;
}
.chart-title {
    text-align: center;
    font-weight: bold;
    margin-top: 2rem;
    font-size: 1.3rem;
}
footer {
    margin-top: 2rem;
    text-align: center;
    font-size: 0.9rem;
    color: #95a5a6;
}
</style>
"""

dark_css = """
<style>
body, .main {
    background-color: #121212;
    color: #e0e0e0;
}
.header {
    font-size: 2.8rem;
    font-weight: 800;
    color: #4ab3f4;
    text-align: center;
    margin-top: 1.5rem;
    margin-bottom: 0.2rem;
    letter-spacing: 1px;
}
.subheader {
    font-size: 1.2rem;
    text-align: center;
    color: #b0bec5;
    margin-bottom: 2rem;
    font-weight: 500;
    line-height: 1.4;
}
.input-column {
    padding: 1rem;
}
.slider-label, .radio-label {
    margin-top: 1rem;
    font-weight: 600;
}
.slider-value {
    margin-bottom: 1rem;
    color: #4ab3f4;
}
.result-fail {
    font-size: 1.5rem;
    font-weight: bold;
    color: #ef5350;
    text-align: center;
    margin-top: 2rem;
}
.result-success {
    font-size: 1.5rem;
    font-weight: bold;
    color: #66bb6a;
    text-align: center;
    margin-top: 2rem;
}
.chart-title {
    text-align: center;
    font-weight: bold;
    margin-top: 2rem;
    font-size: 1.3rem;
}
footer {
    margin-top: 2rem;
    text-align: center;
    font-size: 0.9rem;
    color: #90a4ae;
}
</style>
"""

# --- Apply Theme ---
st.markdown(dark_css if dark_mode else light_css, unsafe_allow_html=True)

# --- Load Model ---
model = joblib.load("car_component_failure_balanced.pkl11111")  # <- update if needed
feature_columns = joblib.load("feature_columns.pkl")

# --- Sidebar ---
st.sidebar.markdown("""
---
### Credits & Links
- 👨‍💻 Made By:- [Muhammad Areeb Rizwan](https://www.linkedin.com/in/areebrizwan)  
- 🌐 [Portfolio](https://sites.google.com/view/m-areeb-rizwan/home)  
- 💻 [GitHub](https://github.com/Areebrizz)  
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header">🚗 Car Component Failure Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter component readings to assess risk of failure.</div>', unsafe_allow_html=True)

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-column">', unsafe_allow_html=True)
    temperature = st.slider("Temperature (°C)", 40.0, 160.0, 112.0)
    vibration = st.slider("Vibration (units)", 5.0, 100.0, 44.4)
    pressure = st.slider("Pressure (PSI)", 10.0, 50.0, 25.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-column">', unsafe_allow_html=True)
    load = st.slider("Load (kg)", 100.0, 1500.0, 800.0)
    age = st.slider("Age (months)", 1, 120, 60)
    op_mode = st.radio("Operational Mode", ["Normal", "High Load", "Low Load"])
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prepare input for prediction ---
op_mode_map = {"Normal": 0, "High Load": 1, "Low Load": 2}
input_dict = {
    "Temperature": temperature,
    "Vibration": vibration,
    "Pressure": pressure,
    "Load": load,
    "Age": age,
    "Operational_Mode": op_mode_map[op_mode]
}
input_df = pd.DataFrame([input_dict], columns=feature_columns)

# --- Prediction Button ---
if st.button("Predict Failure Risk 🚦"):
    prediction = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.markdown(
            f'<div class="result-fail">⚠️ High Risk of Failure (Probability: {pred_prob:.2%})</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-success">✅ Low Risk of Failure (Probability: {1 - pred_prob:.2%})</div>',
            unsafe_allow_html=True
        )

    # --- Feature Importance Chart ---
    st.markdown('<div class="chart-title">Feature Importance</div>', unsafe_allow_html=True)
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    plt.figure(figsize=(8, 4))
    plt.barh(range(len(importance)), importance[sorted_idx],
             color="#2e86de" if not dark_mode else "#4ab3f4")
    plt.yticks(range(len(importance)), np.array(feature_columns)[sorted_idx])
    plt.xlabel("Importance")
    plt.tight_layout()
    st.pyplot(plt)

else:
    st.markdown(
        '<div style="text-align:center; font-style:italic; color:#7f8c8d; margin-top:1.5rem;">Click "Predict Failure Risk 🚦" to see the result.</div>',
        unsafe_allow_html=True
    )

# --- Footer ---
st.markdown("""
<footer>
    Made with ❤️ by Muhammad Areeb Rizwan • Mechanical Engineer & AI Enthusiast
</footer>
""", unsafe_allow_html=True)
