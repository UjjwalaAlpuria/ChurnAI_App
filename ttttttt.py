import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will churn or not.")

# -----------------------------
# USER INPUTS
# -----------------------------
credit_score = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
products = st.number_input("Number of Products", 1, 5, 1)
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Churn"):

    input_data = np.array([[credit_score,
                            age,
                            tenure,
                            balance,
                            products,
                            salary]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is not likely to churn.")