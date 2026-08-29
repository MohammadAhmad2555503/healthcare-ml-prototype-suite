from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import configure_page, probability_chart, predictor, risk_badge, title_block
from src.explainability import explain_image
from src.utils import load_config

config = load_config()
configure_page("Prediction")
title_block("Prediction", f"Upload a {config['clinical']['image_modality']} for triage-oriented prediction.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("Upload an image to generate a prediction.")
    st.stop()

image_bytes = uploaded.getvalue()
left, right = st.columns([0.85, 1.15], gap="large")
with left:
    st.image(image_bytes, caption=uploaded.name, use_container_width=True)

result = predictor().predict_image(image_bytes)
with right:
    st.subheader("Prediction Result")
    st.metric("Predicted Class", result["prediction"])
    st.metric("Confidence", f"{result['confidence_score'] * 100:.2f}%")
    st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
    st.write(result["clinical_recommendation"])
    st.plotly_chart(probability_chart(result["probabilities"]), use_container_width=True)

st.subheader("Top Influential Engineered Features")
st.dataframe(explain_image(image_bytes).head(20), use_container_width=True, hide_index=True)

