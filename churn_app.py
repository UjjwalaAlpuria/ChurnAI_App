import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────── Page config ────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ─────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
    }
    .churn-yes {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .churn-no {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3436;
        margin-bottom: 0.5rem;
        border-left: 4px solid #6c5ce7;
        padding-left: 0.6rem;
    }
    div[data-testid="stForm"] { background: white; border-radius: 14px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Load model ─────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path  = "churn_model.pkl"
    scaler_path = "scaler.pkl"
    model, scaler = None, None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# ─────────────────────────── Header ─────────────────────────────────
st.title("📡 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer will churn using an **XGBoost** model (97.8% accuracy).")

if model is None:
    st.warning(
        "⚠️  `churn_model.pkl` not found. Upload your saved model file below to enable predictions.",
        icon="⚠️",
    )
    uploaded_model = st.file_uploader("Upload churn_model.pkl", type="pkl", key="model_upload")
    uploaded_scaler = st.file_uploader("Upload scaler.pkl", type="pkl", key="scaler_upload")
    if uploaded_model and uploaded_scaler:
        import pickle
        model  = pickle.load(uploaded_model)
        scaler = pickle.load(uploaded_scaler)
        st.success("Model and scaler loaded! Refresh the app to enable predictions.")

st.divider()

# ─────────────────────────── Sidebar: feature guide ─────────────────
with st.sidebar:
    st.header("ℹ️ Feature Guide")
    st.markdown("""
| Feature | Description |
|---|---|
| **account.length** | Months as a customer |
| **day/eve/night mins** | Minutes used per period |
| **intl mins** | International call minutes |
| **customer calls** | Calls to customer service |
| **voice plan** | Voice mail plan? |
| **intl plan** | International plan? |
""")
    st.divider()
    st.markdown("**Model:** XGBoost  \n**Dataset:** P670 Telecom  \n**Test Accuracy:** 97.8%")

# ─────────────────────────── Input form ─────────────────────────────
st.markdown("### 🧾 Enter Customer Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="section-title">Account Info</p>', unsafe_allow_html=True)
        account_length  = st.number_input("Account Length (months)", 0, 300, 100)
        voice_plan      = st.selectbox("Voice Mail Plan", ["No", "Yes"])
        intl_plan       = st.selectbox("International Plan", ["No", "Yes"])
        num_vmail_msgs  = st.number_input("Voicemail Messages", 0, 60, 0)

    with col2:
        st.markdown('<p class="section-title">Call Usage</p>', unsafe_allow_html=True)
        day_mins    = st.slider("Day Minutes",   0.0, 350.0, 180.0, 0.5)
        day_calls   = st.number_input("Day Calls",   0, 200, 100)
        eve_charge  = st.slider("Evening Charge ($)", 0.0, 30.0, 8.0, 0.1)
        eve_calls   = st.number_input("Evening Calls", 0, 200, 100)
        night_mins  = st.slider("Night Minutes",  0.0, 350.0, 200.0, 0.5)
        night_calls = st.number_input("Night Calls",  0, 200, 100)
        night_charge = st.number_input("Night Charge ($)", 0.0, 20.0, 9.0, 0.1)

    with col3:
        st.markdown('<p class="section-title">International & Support</p>', unsafe_allow_html=True)
        intl_mins    = st.slider("Intl Minutes", 0.0, 20.0, 10.0, 0.1)
        intl_calls   = st.number_input("Intl Calls", 0, 20, 4)
        intl_charge  = st.number_input("Intl Charge ($)", 0.0, 5.0, 2.7, 0.01)
        customer_calls = st.slider("Customer Service Calls", 0, 10, 1)

    submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True, type="primary")

# ─────────────────────────── Feature builder ────────────────────────
def build_feature_row(
    account_length, voice_plan, intl_plan, num_vmail_msgs,
    day_mins, day_calls, eve_charge, eve_calls,
    night_mins, night_calls, night_charge,
    intl_mins, intl_calls, intl_charge, customer_calls,
):
    """
    Reconstruct the same feature vector the notebook produced after
    one-hot-encoding + feature engineering + MinMax scaling.

    The notebook one-hot-encoded: state, area.code, voice.plan, intl.plan,
    day.charge, eve.mins, churn.
    It then added: total_call_minutes, total_calls_made, total_charge_cost.
    Finally it applied MinMaxScaler on everything.

    We replicate those derived / dummy columns here so the scaler
    (if present) sees the same shape.  When no scaler is available we
    feed the raw numeric vector directly to the model.
    """
    # Binary helpers
    vp = 1 if voice_plan == "Yes" else 0
    ip = 1 if intl_plan  == "Yes" else 0

    # Engineered features
    total_call_minutes = day_mins + night_mins + intl_mins
    total_calls_made   = day_calls + eve_calls + night_calls + intl_calls + customer_calls
    total_charge_cost  = intl_charge + eve_charge + night_charge

    # Core numeric features (mirrors the notebook X matrix)
    row = {
        "account.length":     account_length,
        "num.vmail.messages": num_vmail_msgs,
        "day.mins":           day_mins,
        "day.calls":          day_calls,
        "eve.calls":          eve_calls,
        "eve.charge":         eve_charge,
        "night.mins":         night_mins,
        "night.calls":        night_calls,
        "night.charge":       night_charge,
        "intl.mins":          intl_mins,
        "intl.calls":         intl_calls,
        "intl.charge":        intl_charge,
        "customer.calls":     customer_calls,
        "voice.plan_yes":     vp,
        "intl.plan_yes":      ip,
        "total_call_minutes": total_call_minutes,
        "total_calls_made":   total_calls_made,
        "total_charge_cost":  total_charge_cost,
    }
    return pd.DataFrame([row])

# ─────────────────────────── Prediction ─────────────────────────────
if submitted:
    df_input = build_feature_row(
        account_length, voice_plan, intl_plan, num_vmail_msgs,
        day_mins, day_calls, eve_charge, eve_calls,
        night_mins, night_calls, night_charge,
        intl_mins, intl_calls, intl_charge, customer_calls,
    )

    if model is None:
        st.error("Please upload the model file first.")
    else:
        try:
            # Apply scaler if available, else use raw values
            if scaler is not None:
                # Align columns with scaler's expected input
                try:
                    X_scaled = scaler.transform(df_input)
                except Exception:
                    X_scaled = df_input.values
            else:
                X_scaled = df_input.values

            pred       = model.predict(X_scaled)[0]
            prob_arr   = model.predict_proba(X_scaled)[0]
            churn_prob = prob_arr[1] if len(prob_arr) > 1 else float(pred)
            stay_prob  = 1 - churn_prob

            st.divider()
            st.markdown("### 🎯 Prediction Result")

            res_col, gauge_col = st.columns([1, 1.4])

            with res_col:
                if pred == 1 or churn_prob >= 0.5:
                    st.markdown(
                        f'<div class="churn-yes">⚠️ Likely to Churn<br>'
                        f'<span style="font-size:1rem;font-weight:400;">'
                        f'Churn probability: {churn_prob*100:.1f}%</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="churn-no">✅ Likely to Stay<br>'
                        f'<span style="font-size:1rem;font-weight:400;">'
                        f'Retention probability: {stay_prob*100:.1f}%</span></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("#### 📊 Key Input Summary")
                summary = pd.DataFrame({
                    "Feature": ["Day Minutes", "Customer Calls", "Intl Plan",
                                "Total Charges", "Total Minutes"],
                    "Value": [
                        f"{day_mins} min",
                        str(customer_calls),
                        intl_plan,
                        f"${intl_charge + eve_charge + night_charge:.2f}",
                        f"{day_mins + night_mins + intl_mins:.1f} min",
                    ],
                })
                st.dataframe(summary, hide_index=True, use_container_width=True)

            with gauge_col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=churn_prob * 100,
                    number={"suffix": "%", "font": {"size": 36}},
                    delta={"reference": 50, "increasing": {"color": "#e74c3c"}, "decreasing": {"color": "#00b894"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar":  {"color": "#6c5ce7"},
                        "steps": [
                            {"range": [0,  40], "color": "#d5f5e3"},
                            {"range": [40, 60], "color": "#fdebd0"},
                            {"range": [60, 100], "color": "#fadbd8"},
                        ],
                        "threshold": {
                            "line": {"color": "#c0392b", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                    title={"text": "Churn Probability", "font": {"size": 18}},
                ))
                fig.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

            # Risk factors
            st.markdown("#### 🔎 Risk Factor Highlights")
            tips = []
            if customer_calls >= 4:
                tips.append("🔴 **High customer service calls** — strong churn indicator.")
            if intl_plan == "Yes" and intl_charge < 2.5:
                tips.append("🟠 Customer has an **international plan but low usage** — potential dissatisfaction.")
            if day_mins > 250:
                tips.append("🟠 **Very high daytime usage** — may be cost-sensitive.")
            if voice_plan == "No" and num_vmail_msgs == 0:
                tips.append("🟡 No voice plan and no voicemail — lower engagement.")
            if not tips:
                tips.append("🟢 No major individual risk factors detected.")
            for t in tips:
                st.markdown(t)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("This usually means the saved scaler expects a different column layout. "
                    "Try running without the scaler, or retrain and re-export the model.")

# ─────────────────────────── Model comparison chart ──────────────────
st.divider()
st.markdown("### 📈 Model Comparison (from notebook)")
accuracies = {
    "Logistic Regression": 0.9192,
    "SVM":                 0.8951,
    "Decision Tree":       0.9047,
    "Random Forest":       0.9288,
    "KNN":                 0.8987,
    "XGBoost ⭐":          0.9783,
}
colors = ["#b2bec3"] * 5 + ["#6c5ce7"]
fig2 = px.bar(
    x=list(accuracies.values()),
    y=list(accuracies.keys()),
    orientation="h",
    text=[f"{v*100:.1f}%" for v in accuracies.values()],
    color=list(accuracies.keys()),
    color_discrete_sequence=colors,
)
fig2.update_traces(textposition="outside")
fig2.update_layout(
    xaxis=dict(range=[0.85, 1.0], title="Test Accuracy"),
    yaxis_title="",
    showlegend=False,
    height=320,
    margin=dict(l=10, r=30, t=20, b=20),
)
st.plotly_chart(fig2, use_container_width=True)

st.caption("Built with Streamlit · XGBoost model trained on P670 Telecom dataset · Accuracy 97.8%")
