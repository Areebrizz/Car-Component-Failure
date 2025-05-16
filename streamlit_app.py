import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and features
model = joblib.load("car_component_failure_balanced.pkl11111")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Car Component Failure Prediction", page_icon="🚗", layout="wide")

# Add credits and links to sidebar
st.sidebar.markdown("""
---
### Credits & Links

- 👨‍💻 Made by [Muhammad Areeb Rizwan](https://www.linkedin.com/in/areebrizwan)  
- 🌐 [Portfolio](https://sites.google.com/view/m-areeb-rizwan/home)  
- 💻 [GitHub](https://github.com/Areebrizz)  
""", unsafe_allow_html=True)

# Dark mode toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

# CSS for light and dark modes with better typography and layout
light_css = """
<style>
    body, .main {
        background-color: #f9fbfc;
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 0 2rem;
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
    .inputs-wrapper {
        display: flex;
        justify-content: space-between;
        gap: 2rem;
        max-width: 900px;
        margin: 0 auto 3rem auto;
    }
    .input-column {
        background: white;
        border-radius: 16px;
        padding: 25px 30px;
        box-shadow: 0 8px 20px rgb(0 0 0 / 0.07);
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 20px;
        max-height: 520px;
        overflow-y: auto;
    }
    .slider-label {
        font-weight: 700;
        color: #34495e;
        font-size: 1.1rem;
        margin-bottom: 6px;
        user-select: none;
    }
    .slider-value {
        color: #7f8c8d;
        font-size: 0.9rem;
        font-family: monospace;
        margin-top: -12px;
        margin-bottom: 12px;
        user-select: none;
    }
    .radio-label {
        font-size: 1.15rem;
        font-weight: 700;
        color: #34495e;
        margin-top: 12px;
        margin-bottom: 8px;
        user-select: none;
    }
    /* Streamlit default tweaks */
    .stSlider > div > div {
        padding-bottom: 8px !important;
    }
    .stRadio > div {
        margin-bottom: 8px;
    }
    /* Prediction result boxes */
    .result-success {
        background-color: #d4efdf;
        color: #27ae60;
        border-radius: 16px;
        padding: 18px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 4px 16px rgb(39 174 96 / 0.3);
        max-width: 460px;
        margin: 1rem auto 2rem auto;
        user-select: none;
    }
    .result-fail {
        background-color: #f9d6d5;
        color: #c0392b;
        border-radius: 16px;
        padding: 18px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 4px 16px rgb(192 57 43 / 0.3);
        max-width: 460px;
        margin: 1rem auto 2rem auto;
        user-select: none;
    }
    /* Feature importance chart title */
    .chart-title {
        font-weight: 700;
        font-size: 1.3rem;
        color: #2e86de;
        margin-bottom: 1rem;
        text-align: center;
        user-select: none;
    }
    footer {
        text-align: center;
        font-size: 0.9rem;
        color: #95a5a6;
        margin-top: 3rem;
        padding-top: 12px;
        border-top: 1px solid #ecf0f1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        user-select: none;
    }
    /* Scrollbar for input columns */
    .input-column::-webkit-scrollbar {
        width: 7px;
    }
    .input-column::-webkit-scrollbar-thumb {
        background-color: #b0bec5;
        border-radius: 10px;
    }
</style>
"""

dark_css = """
<style>
    body, .main {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 0 2rem;
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
    .inputs-wrapper {
        display: flex;
        justify-content: space-between;
        gap: 2rem;
        max-width: 900px;
        margin: 0 auto 3rem auto;
    }
    .input-column {
        background: #1e1e1e;
        border-radius: 16px;
        padding: 25px 30px;
        box-shadow: 0 8px 20px rgb(0 0 0 / 0.9);
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 20px;
        max-height: 520px;
        overflow-y: auto;
    }
    .slider-label {
        font-weight: 700;
        color: #b0bec5;
        font-size: 1.1rem;
        margin-bottom: 6px;
        user-select: none;
    }
    .slider-value {
        color: #90a4ae;
        font-size: 0.9rem;
        font-family: monospace;
        margin-top: -12px;
        margin-bottom: 12px;
        user-select: none;
    }
    .radio-label {
        font-size: 1.15rem;
        font-weight: 700;
        color: #b0bec5;
        margin-top: 12px;
        margin-bottom: 8px;
        user-select: none;
    }
    /* Streamlit default tweaks */
    .stSlider > div > div {
        padding-bottom: 8px !important;
    }
    .stRadio > div {
        margin-bottom: 8px;
    }
    /* Prediction result boxes */
    .result-success {
        background-color: #2e7d32;
        color: #a5d6a7;
        border-radius: 16px;
        padding: 18px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 4px 16px rgb(165 214 167 / 0.7);
        max-width: 460px;
        margin: 1rem auto 2rem auto;
        user-select: none;
    }
    .result-fail {
        background-color: #c62828;
        color: #ef9a9a;
        border-radius: 16px;
        padding: 18px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 4px 16px rgb(239 154 154 / 0.7);
        max-width: 460px;
        margin: 1rem auto 2rem auto;
        user-select: none;
    }
    /* Feature importance chart title */
    .chart-title {
        font-weight: 700;
        font-size: 1.3rem;
        color: #4ab3f4;
        margin-bottom: 1rem;
        text-align: center;
        user-select: none;
    }
    footer {
        text-align: center;
        font-size: 0.9rem;
        color: #78909c;
        margin-top: 3rem;
        padding-top: 12px;
        border-top: 1px solid #263238;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        user-select: none;
    }
    /* Scrollbar for input columns */
    .input-column::-webkit-scrollbar {
        width: 7px;
    }
    .input-column::-webkit-scrollbar-thumb {
        background-color: #546e7a;
        border-radius: 10px;
    }
</style>
"""

# Inject CSS based on toggle
st.markdown(dark_css if dark_mode else light_css, unsafe_allow_html=True)

# --- Title and subtitle ---
st.markdown('<div class="header">🚗 Car Component Failure Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter component readings to assess risk of failure.</div>', unsafe_allow_html=True)

# --- Inputs layout ---
st.markdown('<div class="inputs-wrapper">', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-column">', unsafe_allow_html=True)
    st.markdown('<div class="slider-label">Temperature (°C)</div>', unsafe_allow_html=True)
    temperature = st.slider("", 40
