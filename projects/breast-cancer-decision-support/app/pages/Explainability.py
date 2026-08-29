from __future__ import annotations

from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import (
    REPORTS_DIR,
    artifact_ready,
    configure_page,
    feature_defaults,
    image_if_exists,
    load_app_data,
    load_feature_importance,
    load_feature_names,
    render_title,
)
from src.explainability import explain_prediction, generate_shap_artifacts


configure_page("Explainability")
render_title(
    "Explainability",
    "Review global and patient-level explanations for the selected model using SHAP artifacts.",
)

if not artifact_ready():
    st.error("Model artifacts are unavailable. Run `python -m src.train` from the project root.")
    st.stop()

if st.button("Generate or Refresh SHAP Artifacts", type="primary"):
    try:
        with st.spinner("Generating SHAP plots..."):
            generate_shap_artifacts()
        st.success("SHAP artifacts generated.")
        st.cache_data.clear()
    except ImportError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Unable to generate SHAP artifacts: {exc}")

summary = image_if_exists(REPORTS_DIR / "shap_summary.png")
waterfall = image_if_exists(REPORTS_DIR / "shap_waterfall.png")
importance_plot = image_if_exists(REPORTS_DIR / "shap_feature_importance.png")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("SHAP Summary Plot")
    if summary:
        st.image(str(summary), use_container_width=True)
    else:
        st.info("Generate SHAP artifacts to display the summary plot.")

with right:
    st.subheader("SHAP Feature Importance Plot")
    if importance_plot:
        st.image(str(importance_plot), use_container_width=True)
    else:
        fi = load_feature_importance().head(20)
        if not fi.empty:
            fig = px.bar(
                fi.sort_values("normalized_importance"),
                x="normalized_importance",
                y="feature",
                orientation="h",
                labels={"normalized_importance": "Importance", "feature": "Feature"},
                color="normalized_importance",
                color_continuous_scale="Teal",
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

st.subheader("SHAP Waterfall Plot")
if waterfall:
    st.image(str(waterfall), use_container_width=True)
else:
    st.info("Generate SHAP artifacts to display the waterfall plot.")

st.subheader("Top Influential Features for a Custom Case")
data = load_app_data()
feature_names = load_feature_names()
defaults = feature_defaults(data, feature_names)
with st.expander("Enter a case for patient-level SHAP explanation"):
    payload: dict[str, float] = {}
    cols = st.columns(3)
    for idx, feature in enumerate(feature_names):
        stats = defaults[feature]
        with cols[idx % 3]:
            payload[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=stats["min"],
                max_value=stats["max"],
                value=stats["median"],
                step=stats["step"],
                format="%.5f",
                key=f"explain_{feature}",
            )
    if st.button("Explain Case"):
        try:
            impacts = explain_prediction(payload)
            st.dataframe(impacts.head(15), use_container_width=True, hide_index=True)
        except ImportError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Unable to explain this case: {exc}")

