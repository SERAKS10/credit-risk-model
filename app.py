
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Risk Demo",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Credit Risk Prediction")
st.write("An interactive machine-learning portfolio demonstration.")
st.warning(
    "This model uses synthetic data with randomly assigned default labels. "
    "Its predictions are not suitable for real lending decisions."
)

MODEL_PATH = Path(__file__).parent / "credit_risk_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

st.subheader("Applicant details")

with st.form("credit_application"):
    age = st.number_input("Age", 18, 100, 35)
    job = st.selectbox(
        "Job type",
        ["unskilled", "skilled", "highly skilled", "management"]
    )
    credit_amount = st.number_input(
        "Credit amount", 500, 20000, 5000, step=500
    )
    duration = st.number_input(
        "Loan duration (months)", 6, 72, 24
    )
    purpose = st.selectbox(
        "Loan purpose",
        ["car", "furniture", "education", "business", "repairs"]
    )
    savings = st.selectbox(
        "Savings",
        ["little", "moderate", "quite rich", "rich"]
    )
    employment = st.selectbox(
        "Employment duration",
        ["unemployed", "<1 year", "1-4 years", "4-7 years", ">7 years"]
    )
    housing = st.selectbox("Housing", ["own", "rent", "free"])

    submitted = st.form_submit_button("Predict credit risk")

if submitted:
    applicant = pd.DataFrame([{
        "age": age,
        "job": job,
        "credit_amount": credit_amount,
        "duration": duration,
        "purpose": purpose,
        "savings": savings,
        "employment": employment,
        "housing": housing
    }])

    prediction = int(model.predict(applicant)[0])
    probabilities = model.predict_proba(applicant)[0]
    class_labels = list(model.classes_)
    risk_probability = float(
        probabilities[class_labels.index(1)]
    )

    st.subheader("Prediction result")
    st.metric("Class 1 probability", f"{risk_probability:.1%}")
    st.write(f"Predicted class: **{prediction}**")
    st.caption(
        "Class 1 represents the dataset's default label; "
        "the probability is not a validated real-world risk estimate."
    )
