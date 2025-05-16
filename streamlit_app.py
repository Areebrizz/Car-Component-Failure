import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and features
model = joblib.load("car_component_failure_balanced.pkl11111")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Car Component Failure Prediction", page_icon="🚗", layout="wide")

# Dark mode toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

# Inject CSS based on dark mode toggle
st.markdown(dark_css if dark_mode else light_css, unsafe_allow_html=True)


# --- Sidebar credits ---
st.sidebar.markdown("""
---
### Credits & Links

- 👨‍💻 [Muhammad Areeb Rizwan](https://www.linkedin.com/in/areebrizwan)  
- 🌐 [Portfolio](https://sites.google.com/view/m-areeb-rizwan/home)  
- 💻 [GitHub](https://github.com/Areebrizz)  
""", unsafe_allow_html=True)

# --- Title and subtitle ---
st.markdown('<div class="header">🚗 Car Component Failure Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter component readings to assess risk of failure.</div>', unsafe_allow_html=True)

# --- Inputs layout ---
st.markdown('<div class="inputs-wrapper">', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-column">', unsafe_allow_html=True)
    st.markdown('<div class="slider-label">Temperature (°C)</div>', unsafe_allow_html=True)
    temperature = st.slider("", 40.0, 160.0, 112.0, key="temp")
    st.markdown(f'<div class="slider-value">{temperature:.1f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="slider-label">Vibration (units)</div>', unsafe_allow_html=True)
    vibration = st.slider("", 5.0, 100.0, 44.4, key="vib")
    st.markdown(f'<div class="slider-value">{vibration:.2f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="slider-label">Pressure (PSI)</div>', unsafe_allow_html=True)
    pressure = st.slider("", 10.0, 50.0, 25.0, key="pressure")
    st.markdown(f'<div class="slider-value">{pressure:.1f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-column">', unsafe_allow_html=True)
    st.markdown('<div class="slider-label">Load (kg)</div>', unsafe_allow_html=True)
    load = st.slider("", 100.0, 1500.0, 800.0, key="load")
    st.markdown(f'<div class="slider-value">{load:.1f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="slider-label">Age (months)</div>', unsafe_allow_html=True)
    age = st.slider("", 1, 120, 60, key="age")
    st.markdown(f'<div class="slider-value">{age}</div>', unsafe_allow_html=True)

    st.markdown('<div class="radio-label">Operational Mode</div>', unsafe_allow_html=True)
    op_mode = st.radio("", ["Normal", "High Load", "Low Load"], key="opmode")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close inputs-wrapper

# Prepare input dataframe for model prediction
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

    # --- Feature importance plot ---
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
