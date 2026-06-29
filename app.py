import streamlit as st
import numpy as np
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details:")

# Example inputs (you will adjust based on your dataset)
tenure = st.number_input("Tenure", 0, 100, 1)
monthly_charges = st.number_input("Monthly Charges", 0, 200, 50)

# Example categorical inputs
internet_fiber = st.selectbox("Fiber Optic Internet", [0, 1])

# Create input array (must match training columns!)
input_data = np.zeros(len(features))

# map features manually (example)
if "tenure" in features:
    input_data[list(features).index("tenure")] = tenure

if "MonthlyCharges" in features:
    input_data[list(features).index("MonthlyCharges")] = monthly_charges

if "InternetService_Fiber optic" in features:
    input_data[list(features).index("InternetService_Fiber optic")] = internet_fiber

if st.button("Predict"):
    prob = model.predict_proba(np.array(input_data).reshape(1, -1))[0][1]
    prediction = model.predict([input_data])[0]

    st.write(f"Churn Probability: {prob:.2f}")
    st.write("Prediction:", "Yes" if prediction == 1 else "No")