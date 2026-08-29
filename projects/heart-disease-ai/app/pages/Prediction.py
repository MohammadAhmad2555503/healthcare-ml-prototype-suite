from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import configure_page, data, probability_gauge, predictor, risk_badge, title_block
from src.data import feature_columns

configure_page("Prediction")
title_block("Prediction", "Enter patient-level structured clinical values for heart disease risk prediction.")

df = data()
features = feature_columns()
payload = {}
with st.form("prediction_form"):
    cols = st.columns(2)
    for idx, feature in enumerate(features):
        values = df[feature]
        with cols[idx % 2]:
            payload[feature] = st.number_input(
                feature,
                min_value=float(values.min()),
                max_value=float(values.max()),
                value=float(values.median()),
                step=float(max((values.max() - values.min()) / 100, 0.1)),
            )
    submitted = st.form_submit_button("Generate Prediction", type="primary", use_container_width=True)

if submitted:
    result = predictor().predict(payload)
    left, right = st.columns(2, gap="large")
    with left:
        st.metric("Prediction", result["prediction"])
        st.metric("Confidence", f"{result['confidence_score'] * 100:.2f}%")
        st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
        st.write(result["clinical_recommendation"])
    with right:
        st.plotly_chart(probability_gauge(result["heart_disease_probability"]), use_container_width=True)
    st.subheader("Input Values")
    st.dataframe([{"Feature": k, "Value": v} for k, v in payload.items()], use_container_width=True, hide_index=True)
else:
    st.info("Submit the form to generate a prediction.")

