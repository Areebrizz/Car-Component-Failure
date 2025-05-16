import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("car_component_failure_balanced.pkl")

st.title("Car Component Failure Prediction")

st.write("Input the feature values to predict failure:")

# Define the features exactly as your model expects (modify as per your dataset)
# Example assuming features: 'Feature1', 'Feature2', 'Feature3', etc.
# Replace these with your actual feature column names

feature_names = [
    # List your encoded feature names here exactly as in training
    # If you used one-hot encoding, include the columns accordingly
    # Example:
    'Engine_Temperature', 'Brake_Pressure', 'Oil_Level_Low', 'Tire_Condition_Good', # etc.
]

# For simplicity, let’s generate input fields dynamically
input_data = {}
for feature in feature_names:
    # If your features are numeric
    input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Predict Failure"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 Component likely to FAIL! Probability: {proba:.2f}")
    else:
        st.success(f"✅ Component likely to be OK. Probability of failure: {proba:.2f}")
