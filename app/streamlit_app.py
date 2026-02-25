"""
Streamlit Demo — Customer Churn Predictor
Run: streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "data" / "06_models"
CHURN_THRESHOLD = 0.5


# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts() -> tuple[Any, Any, dict[str, Any]]:
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    with open(MODELS_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    return model, preprocessor, meta


model, preprocessor, meta = load_artefacts()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="centered")
st.title("📡 Customer Churn Predictor")
st.caption(
    f"Model: **{meta['model_name']}**  |  Test AUC-ROC: **{meta['test_metrics']['auc_roc']}**"
)

st.header("Customer Features")

col1, col2, col3 = st.columns(3)

with col1:
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet_svc = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col3:
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 79.85, step=0.5)
total_charges = st.number_input(
    "Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), step=1.0
)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [
            {
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "MultipleLines": multiple_lines,
                "InternetService": internet_svc,
                "OnlineSecurity": online_sec,
                "OnlineBackup": online_bkp,
                "DeviceProtection": device_prot,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_mv,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    transformed = preprocessor.transform(input_df)
    proba = model.predict_proba(transformed)[0, 1]

    st.divider()
    if proba >= CHURN_THRESHOLD:
        st.error(f"⚠️ **HIGH CHURN RISK** — Probability: **{proba:.1%}**")
    else:
        st.success(f"✅ **LOW CHURN RISK** — Probability: **{proba:.1%}**")

    st.progress(float(proba), text=f"Churn probability: {proba:.1%}")
