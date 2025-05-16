import streamlit as st
import joblib
import numpy as np

# Load model once
model = joblib.load("car_component_failure_model_balanced.pkl")

st.title("🚗 Car Component Failure Prediction")
st.markdown("""
Predict the likelihood of failure based on input sensor values.
""")

with st.form("input_form"):
    st.subheader("Input Feature Values")

    col1, col2 = st.columns(2)
    with col1:
        engine_temp = st.slider("Engine Temperature (°C)", min_value=0.0, max_value=150.0, step=0.1, value=70.0, help="Temperature of the engine")
        brake_pressure = st.slider("Brake Pressure (psi)", min_value=0.0, max_value=200.0, step=0.1, value=50.0)
    with col2:
        oil_level_low = st.selectbox("Oil Level Low", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Is the oil level low?")
        tire_condition_good = st.selectbox("Tire Condition Good", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Are the tires in good condition?")

    submitted = st.form_submit_button("Predict Failure")

if submitted:
    # Prepare feature vector as DataFrame or numpy array matching training data order
    features = np.array([[engine_temp, brake_pressure, oil_level_low, tire_condition_good]])
    
    # Predict
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ Predicted Failure! Confidence: {proba:.2%}")
    else:
        st.success(f"✅ No Failure predicted. Confidence: {proba:.2%}")
