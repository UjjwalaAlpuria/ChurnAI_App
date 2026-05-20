import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model   = joblib.load("churn_model.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, scaler, columns

try:
    model, scaler, FEATURE_COLS = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning(
        "⚠️ Model files not found. Run `python train_model.py` first to generate "
        "`churn_model.pkl`, `scaler.pkl`, and `feature_columns.pkl`."
    )

# --------------------------------------------------
# ALL US STATES (matching training data)
# --------------------------------------------------
ALL_STATES = [
    "AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI",
    "IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN",
    "MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH",
    "OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA",
    "WI","WV","WY"
]

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn using a **Random Forest** model trained on real telecom data.")

# --------------------------------------------------
# SIDEBAR — INPUT
# --------------------------------------------------
st.sidebar.header("Customer Input Features")

def user_input_features():
    st.sidebar.subheader("Account Info")
    state        = st.sidebar.selectbox("State", ALL_STATES, index=ALL_STATES.index("KS"))
    area_code    = st.sidebar.selectbox("Area Code", ["area_code_408", "area_code_415", "area_code_510"])
    account_len  = st.sidebar.slider("Account Length (days)", 1, 250, 100)

    st.sidebar.subheader("Plans")
    voice_plan   = st.sidebar.selectbox("Voice Plan", ["yes", "no"])
    intl_plan    = st.sidebar.selectbox("International Plan", ["yes", "no"])
    voice_msgs   = st.sidebar.slider("Voice Messages", 0, 60, 10)

    st.sidebar.subheader("International Usage")
    intl_mins    = st.sidebar.slider("International Minutes", 0.0, 25.0, 10.0)
    intl_calls   = st.sidebar.slider("International Calls", 0, 20, 5)
    intl_charge  = st.sidebar.slider("International Charge ($)", 0.0, 10.0, 2.5)

    st.sidebar.subheader("Day Usage")
    day_mins     = st.sidebar.slider("Day Minutes", 0.0, 400.0, 180.0)
    day_calls    = st.sidebar.slider("Day Calls", 0, 200, 100)
    day_charge   = st.sidebar.slider("Day Charge ($)", 0.0, 70.0, 30.0)

    st.sidebar.subheader("Evening Usage")
    eve_mins     = st.sidebar.slider("Evening Minutes", 0.0, 400.0, 200.0)
    eve_calls    = st.sidebar.slider("Evening Calls", 0, 200, 100)
    eve_charge   = st.sidebar.slider("Evening Charge ($)", 0.0, 40.0, 15.0)

    st.sidebar.subheader("Night Usage")
    night_mins   = st.sidebar.slider("Night Minutes", 0.0, 400.0, 200.0)
    night_calls  = st.sidebar.slider("Night Calls", 0, 200, 100)
    night_charge = st.sidebar.slider("Night Charge ($)", 0.0, 20.0, 8.0)

    st.sidebar.subheader("Support")
    cust_calls   = st.sidebar.slider("Customer Service Calls", 0, 10, 1)

    # --------------------------------------------------
    # BUILD RAW ROW  (same structure as training data before encoding)
    # --------------------------------------------------
    raw = {
        "account.length": account_len,
        "voice.messages": voice_msgs,
        "intl.mins":      intl_mins,
        "intl.calls":     intl_calls,
        "intl.charge":    intl_charge,
        "day.mins":       day_mins,
        "day.calls":      day_calls,
        "day.charge":     day_charge,
        "eve.mins":       eve_mins,
        "eve.calls":      eve_calls,
        "eve.charge":     eve_charge,
        "night.mins":     night_mins,
        "night.calls":    night_calls,
        "night.charge":   night_charge,
        "customer.calls": cust_calls,
        "state":          state,
        "area.code":      area_code,
        "voice.plan":     voice_plan,
        "intl.plan":      intl_plan,
    }
    return pd.DataFrame([raw])

raw_df = user_input_features()

# --------------------------------------------------
# DISPLAY RAW INPUT
# --------------------------------------------------
st.subheader("📌 Customer Input Data")
st.dataframe(raw_df, use_container_width=True)

# --------------------------------------------------
# PREPROCESS  (mirror train_model.py pipeline)
# --------------------------------------------------
if model_loaded:
    # One-hot encode categoricals
    df_enc = pd.get_dummies(raw_df, columns=["state", "area.code", "voice.plan", "intl.plan"])

    # Feature engineering
    df_enc["total_call_minutes"] = df_enc["day.mins"]   + df_enc["night.mins"] + df_enc["intl.mins"]
    df_enc["total_calls_made"]   = df_enc["day.calls"]  + df_enc["eve.calls"]  + df_enc["night.calls"] \
                                  + df_enc["intl.calls"] + df_enc["customer.calls"]
    df_enc["total_charge_cost"]  = df_enc["intl.charge"] + df_enc["eve.charge"] + df_enc["night.charge"]

    # Align columns to training feature set (fill missing one-hot cols with 0)
    df_aligned = df_enc.reindex(columns=FEATURE_COLS, fill_value=0)

    # Scale
    scaled = scaler.transform(df_aligned)

    # Predict
    prediction       = model.predict(scaled)
    prediction_proba = model.predict_proba(scaled)

    churn_prob    = float(prediction_proba[0][1])
    no_churn_prob = float(prediction_proba[0][0])

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    st.subheader("📈 Prediction Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability",    f"{churn_prob:.1%}")
    col2.metric("No-Churn Probability", f"{no_churn_prob:.1%}")

    # Determine churn using 0.5 threshold on the scaled prediction value
    is_churn = churn_prob > 0.5

    if is_churn:
        st.error("⚠️ This customer is likely to **CHURN**")
    else:
        st.success("✅ This customer is **NOT** likely to churn")

    # --------------------------------------------------
    # PROBABILITY BAR CHART
    # --------------------------------------------------
    st.subheader("📊 Prediction Probability")
    prob_df = pd.DataFrame({
        "Category":    ["No Churn", "Churn"],
        "Probability": [no_churn_prob, churn_prob]
    }).set_index("Category")

    st.bar_chart(prob_df)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("Developed using **Streamlit** & **Scikit-learn** | Trained on P670 Telecom Dataset")
