import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Car Component Failure Prediction", page_icon="🚗", layout="wide")

# --- Dark Mode Toggle ---
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

# --- CSS Styles ---
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
</style>
"""

st.markdown(dark_css if dark_mode else light_css, unsafe_allow_html=True)

# --- Load Model and Features ---
model = joblib.load("car_component_failure_balanced.pkl11111")
feature_columns = joblib.load("feature_columns.pkl")

# --- Sidebar Credits ---
st.sidebar.markdown("""
---
### 👨‍💻 Made By: M Areeb Rizwan  
- 🌐 [Portfolio](https://sites.google.com/view/m-areeb-rizwan/home)  
- 💻 [GitHub](https://github.com/Areebrizz)  
""", unsafe_allow_html=True)

# --- Title and Subheader ---
st.markdown('<div class="header">🚗 Car Component Failure Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter component readings to assess risk of failure.</div>', unsafe_allow_html=True)

# --- Input Sliders ---
st.subheader("🔧 Component Readings")

temperature = st.slider("Temperature (°C)", 40.0, 160.0, 112.0, format="%.1f")
vibration = st.slider("Vibration (units)", 5.0, 100.0, 44.4, format="%.2f")
pressure = st.slider("Pressure (PSI)", 10.0, 50.0, 25.0, format="%.1f")
load = st.slider("Load (kg)", 100.0, 1500.0, 800.0, format="%.1f")
age = st.slider("Age (months)", 1, 120, 60)
op_mode = st.radio("Operational Mode", ["Normal", "High Load", "Low Load"])

# --- Prepare Input ---
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

# --- Input Summary ---
st.markdown("### 📋 Input Summary")
st.dataframe(input_df)

# --- Prediction ---
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

    # --- Feature Importance ---
    st.markdown("### 📊 Feature Importance")
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(len(importance)), importance[sorted_idx],
                   color="#2e86de" if not dark_mode else "#4ab3f4")
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(np.array(feature_columns)[sorted_idx])
    ax.set_xlabel("Importance")

    for i, v in enumerate(importance[sorted_idx]):
        ax.text(v + 0.001, i, f"{v:.3f}", va='center', fontsize=8)

    st.pyplot(fig)
else:
    st.markdown(
        '<div style="text-align:center; font-style:italic; color:#7f8c8d; margin-top:1.5rem;">Click "Predict Failure Risk 🚦" to see the result.</div>',
        unsafe_allow_html=True
    )

# --- Footer ---
st.markdown("""
<footer style="text-align: center; margin-top: 3rem; font-size: 0.9rem; color: #888;">
Made with ❤️ by Muhammad Areeb Rizwan • Mechanical Engineer & AI Enthusiast
</footer>
""", unsafe_allow_html=True)
