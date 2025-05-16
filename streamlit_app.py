import streamlit as st
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load("car_component_failure_balanced.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# App title and intro
st.set_page_config(page_title="Car Component Failure Prediction", layout="centered")
st.title("🚗 Car Component Failure Prediction")
st.markdown("### Input sensor readings or component values to predict possible failure.")

# User input form
with st.form("prediction_form"):
    user_input = []
    for col in feature_columns:
        value = st.number_input(f"{col}", min_value=0.0, max_value=1000.0, value=0.0, format="%.2f")
        user_input.append(value)
    
    submitted = st.form_submit_button("🔍 Predict")

# Prediction
if submitted:
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]  # probability of class 1 (failure)

    st.markdown("---")
    st.markdown("### 🧠 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ Failure Predicted with {round(prediction_proba * 100, 2)}% confidence.")
    else:
        st.success(f"✅ No Failure Expected — Confidence: {round((1 - prediction_proba) * 100, 2)}%")
