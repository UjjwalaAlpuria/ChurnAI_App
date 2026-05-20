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
# Make sure these files are in the same folder:
# 1. churn_model.pkl
# 2. scaler.pkl

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn or not using Machine Learning.")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Customer Input Features")

def user_input_features():

    account_length = st.sidebar.slider("Account Length", 1, 250, 100)

    voice_messages = st.sidebar.slider("Voice Messages", 0, 60, 10)

    intl_mins = st.sidebar.slider("International Minutes", 0.0, 25.0, 10.0)

    intl_calls = st.sidebar.slider("International Calls", 0, 20, 5)

    intl_charge = st.sidebar.slider("International Charge", 0.0, 10.0, 2.5)

    day_mins = st.sidebar.slider("Day Minutes", 0.0, 400.0, 180.0)

    day_calls = st.sidebar.slider("Day Calls", 0, 200, 100)

    eve_calls = st.sidebar.slider("Evening Calls", 0, 200, 100)

    eve_charge = st.sidebar.slider("Evening Charge", 0.0, 40.0, 15.0)

    night_mins = st.sidebar.slider("Night Minutes", 0.0, 400.0, 200.0)

    night_calls = st.sidebar.slider("Night Calls", 0, 200, 100)

    night_charge = st.sidebar.slider("Night Charge", 0.0, 20.0, 8.0)

    customer_calls = st.sidebar.slider("Customer Service Calls", 0, 10, 1)

    voice_plan = st.sidebar.selectbox(
        "Voice Plan",
        ("yes", "no")
    )

    intl_plan = st.sidebar.selectbox(
        "International Plan",
        ("yes", "no")
    )

    area_code = st.sidebar.selectbox(
        "Area Code",
        ("area_code_408", "area_code_415", "area_code_510")
    )

    # Convert categorical values
    voice_plan_yes = 1 if voice_plan == "yes" else 0
    intl_plan_yes = 1 if intl_plan == "yes" else 0

    area_code_415 = 1 if area_code == "area_code_415" else 0
    area_code_510 = 1 if area_code == "area_code_510" else 0

    # Data dictionary
    data = {
        'account.length': account_length,
        'voice.messages': voice_messages,
        'intl.mins': intl_mins,
        'intl.calls': intl_calls,
        'intl.charge': intl_charge,
        'day.mins': day_mins,
        'day.calls': day_calls,
        'eve.calls': eve_calls,
        'eve.charge': eve_charge,
        'night.mins': night_mins,
        'night.calls': night_calls,
        'night.charge': night_charge,
        'customer.calls': customer_calls,
        'voice.plan_yes': voice_plan_yes,
        'intl.plan_yes': intl_plan_yes,
        'area.code_area_code_415': area_code_415,
        'area.code_area_code_510': area_code_510
    }

    features = pd.DataFrame(data, index=[0])

    return features

input_df = user_input_features()

# -----------------------------
# DISPLAY INPUT
# -----------------------------
st.subheader("📌 Customer Input Data")
st.write(input_df)

# -----------------------------
# SCALE DATA
# -----------------------------
scaled_data = scaler.transform(input_df)

# -----------------------------
# PREDICTION
# -----------------------------
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

# -----------------------------
# OUTPUT
# -----------------------------
st.subheader("📈 Prediction Result")

if prediction[0] == 1:
    st.error("⚠️ Customer is likely to CHURN")
else:
    st.success("✅ Customer is NOT likely to churn")

# -----------------------------
# PROBABILITY
# -----------------------------
st.subheader("📊 Prediction Probability")

prob_df = pd.DataFrame({
    'No Churn Probability': [prediction_proba[0][0]],
    'Churn Probability': [prediction_proba[0][1]]
})

st.write(prob_df)

st.bar_chart(prob_df.T)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Developed using Streamlit & Machine Learning")