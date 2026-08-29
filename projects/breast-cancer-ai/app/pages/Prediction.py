from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import (
    artifact_ready,
    configure_page,
    feature_defaults,
    feature_groups,
    load_app_data,
    load_feature_names,
    load_predictor,
    probability_gauge,
    render_title,
    risk_badge,
)


configure_page("Prediction")
render_title(
    "Prediction",
    "Enter tumor morphology measurements to estimate benign or malignant diagnosis probability.",
)

if not artifact_ready():
    st.error("Model artifacts are unavailable. Run `python -m src.train` from the project root.")
    st.stop()

data = load_app_data()
feature_names = load_feature_names()
defaults = feature_defaults(data, feature_names)
groups = feature_groups(feature_names)

with st.form("prediction_form"):
    payload: dict[str, float] = {}
    tabs = st.tabs([name for name, features in groups.items() if features])
    active_groups = [(name, features) for name, features in groups.items() if features]
    for tab, (_, features) in zip(tabs, active_groups):
        with tab:
            columns = st.columns(2)
            for idx, feature in enumerate(features):
                stats = defaults[feature]
                with columns[idx % 2]:
                    payload[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=stats["min"],
                        max_value=stats["max"],
                        value=stats["median"],
                        step=stats["step"],
                        format="%.5f",
                    )
    submitted = st.form_submit_button("Generate Clinical Prediction", type="primary", use_container_width=True)

if submitted:
    predictor = load_predictor()
    result = predictor.predict(payload)
    left, middle, right = st.columns([0.95, 1.1, 0.95], gap="large")

    with left:
        st.subheader("Prediction")
        st.metric("Predicted Diagnosis", result["prediction"])
        st.metric("Confidence Score", f"{result['confidence_score'] * 100:.2f}%")
        st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
        st.write(result["clinical_recommendation"])

    with middle:
        st.plotly_chart(probability_gauge(result["malignant_probability"]), use_container_width=True)

    with right:
        st.subheader("Probability")
        st.metric("Benign", f"{result['benign_probability'] * 100:.2f}%")
        st.metric("Malignant", f"{result['malignant_probability'] * 100:.2f}%")
        st.caption(f"Model: {result['selected_model']}")

    st.subheader("Submitted Feature Values")
    st.dataframe(
        [{"Feature": feature, "Value": value} for feature, value in payload.items()],
        use_container_width=True,
        hide_index=True,
        height=420,
    )
else:
    st.info("Complete the feature form and submit to generate a prediction.")

