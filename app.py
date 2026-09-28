import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join("model", "decision_tree_pipeline.joblib")
METRICS_PATH = os.path.join("model", "metrics.json")

CATEGORICAL_OPTIONS = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                  "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["failure", "nonexistent", "success"],
}

st.set_page_config(page_title="Bank Marketing — Subscription Predictor", page_icon="🏦", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["target_encoder"]


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


pipeline, target_encoder = load_model()
metrics = load_metrics()

st.title("🏦 Bank Marketing — Term Deposit Subscription Predictor")
st.caption(
    "Decision tree classifier trained on the UCI Bank Marketing dataset. "
    "Predicts whether a customer will subscribe to a term deposit."
)

tab_predict, tab_performance = st.tabs(["🔮 Predict", "📊 Model Performance"])

with tab_predict:
    st.subheader("Customer details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=17, max_value=98, value=40)
        job = st.selectbox("Job", CATEGORICAL_OPTIONS["job"])
        marital = st.selectbox("Marital status", CATEGORICAL_OPTIONS["marital"])
        education = st.selectbox("Education", CATEGORICAL_OPTIONS["education"])

    with col2:
        default = st.selectbox("Has credit in default?", CATEGORICAL_OPTIONS["default"])
        housing = st.selectbox("Has housing loan?", CATEGORICAL_OPTIONS["housing"])
        loan = st.selectbox("Has personal loan?", CATEGORICAL_OPTIONS["loan"])
        contact = st.selectbox("Contact type", CATEGORICAL_OPTIONS["contact"])

    with col3:
        month = st.selectbox("Last contact month", CATEGORICAL_OPTIONS["month"], index=4)
        day_of_week = st.selectbox("Last contact day of week", CATEGORICAL_OPTIONS["day_of_week"])
        poutcome = st.selectbox("Outcome of previous campaign", CATEGORICAL_OPTIONS["poutcome"], index=1)

    st.subheader("Campaign & economic context")
    col4, col5, col6 = st.columns(3)
    with col4:
        campaign = st.number_input("Contacts this campaign", min_value=1, max_value=56, value=2)
        pdays = st.number_input("Days since last contact (999 = never)", min_value=0, max_value=999, value=999)
        previous = st.number_input("Contacts before this campaign", min_value=0, max_value=7, value=0)
    with col5:
        emp_var_rate = st.slider("Employment variation rate", -3.5, 1.5, 1.1, 0.1)
        cons_price_idx = st.slider("Consumer price index", 92.0, 95.0, 93.9, 0.01)
        cons_conf_idx = st.slider("Consumer confidence index", -51.0, -26.0, -40.5, 0.1)
    with col6:
        euribor3m = st.slider("Euribor 3-month rate", 0.6, 5.1, 3.6, 0.01)
        nr_employed = st.slider("Number of employees (national indicator)", 4960.0, 5230.0, 5167.0, 0.1)

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([{
            "job": job, "marital": marital, "education": education, "default": default,
            "housing": housing, "loan": loan, "contact": contact, "month": month,
            "day_of_week": day_of_week, "poutcome": poutcome,
            "age": age, "campaign": campaign, "pdays": pdays, "previous": previous,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m, "nr.employed": nr_employed,
        }])

        pred = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0]
        label = target_encoder.inverse_transform([pred])[0]
        yes_idx = list(target_encoder.classes_).index("yes")

        if label == "yes":
            st.success(f"**Prediction: Likely to subscribe** (probability: {proba[yes_idx]:.1%})")
        else:
            st.info(f"**Prediction: Not likely to subscribe** (probability of yes: {proba[yes_idx]:.1%})")

with tab_performance:
    st.subheader("Test set metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Precision (yes)", f"{metrics['precision']:.1%}")
    c3.metric("Recall (yes)", f"{metrics['recall']:.1%}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Confusion Matrix**")
        if os.path.exists("outputs/confusion_matrix.png"):
            st.image("outputs/confusion_matrix.png")
    with col_b:
        st.markdown("**Top 5 Feature Importances**")
        if os.path.exists("outputs/feature_importance.png"):
            st.image("outputs/feature_importance.png")

    st.markdown("**Full classification report**")
    st.code(metrics["classification_report"])

    st.caption(
        "Note: `duration` (call length) is excluded from the model because it is only known "
        "after a call takes place and leaks the outcome — this keeps the model realistic for "
        "pre-call prediction, per the dataset documentation."
    )
