import streamlit as st
import numpy as np
import joblib
import pandas as pd
import sklearn

model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details:")

# Customer Details

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=1
)

PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=0.0
)

input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=features, fill_value=0)

# map features manually (example)
if st.button("Predict"):

    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction")

    st.write(f"**Churn Probability:** {prob:.2%}")

    st.progress(float(prob))

    if prob < 0.30:
        st.success("🟢 Low Churn Risk")

    elif prob < 0.60:
        st.warning("🟡 Medium Churn Risk")

    else:
        st.error("🔴 High Churn Risk")

    st.subheader("💡 Recommended Retention Strategies")

    recommendations = []

    if MonthlyCharges > 80:
        recommendations.append(
            "💰 Offer a discount or a lower-cost plan to reduce monthly expenses."
        )
    if Contract == "Month-to-month":
        recommendations.append(
            "📅 Encourage the customer to switch to a yearly contract with exclusive benefits."
        )
    if TechSupport == "No":
        recommendations.append(
            "🛠️ Offer free or discounted Tech Support."
        )
    if OnlineSecurity == "No":
        recommendations.append(
            "🔒 Recommend adding the Online Security package."
        )
    if tenure < 12:
        recommendations.append(
            "🎁 Provide a loyalty bonus or welcome offer to increase engagement."
        )
    if InternetService == "Fiber optic":
        recommendations.append(
            "🌐 Contact the customer to understand any issues with the Fiber Optic service."
        )
    if PaperlessBilling == "Yes":
        recommendations.append(
            "📧 Send personalized offers through email."
        )
    if recommendations:

        for rec in recommendations:
            st.write(rec)

    else:

        st.success(
            "🎉 No major churn indicators detected. Continue maintaining customer satisfaction."
        )
