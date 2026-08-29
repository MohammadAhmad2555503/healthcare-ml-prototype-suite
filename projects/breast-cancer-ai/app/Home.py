from __future__ import annotations

import plotly.express as px
import streamlit as st

from app_utils import (
    artifact_ready,
    class_distribution_chart,
    configure_page,
    imbalance_summary,
    kpi_card,
    load_app_data,
    load_app_metrics,
    load_model_metadata,
    render_title,
)


configure_page("Home")
render_title(
    "Breast Cancer Prediction and Clinical Decision Support System",
    "A healthcare AI dashboard for malignancy risk prediction, model monitoring, and explainable clinical support.",
)

data = load_app_data()
metrics = load_app_metrics()
metadata = load_model_metadata()
imbalance = imbalance_summary(data)
test_metrics = metrics.get("test_metrics", {})

if not artifact_ready():
    st.warning("Model artifacts are missing. Run `python -m src.train` from the project root before using prediction pages.")

total_patients = len(data)
malignant_rate = imbalance["malignant_percentage"]
benign_rate = imbalance["benign_percentage"]
selected_model = metrics.get("selected_model", metadata.get("selected_model", "Not trained"))
roc_auc = test_metrics.get("roc_auc")
recall = test_metrics.get("recall")

cols = st.columns(4)
cols[0].markdown(kpi_card("Patient Records", f"{total_patients:,}", "Validated rows in the dataset"), unsafe_allow_html=True)
cols[1].markdown(kpi_card("Malignant Rate", f"{malignant_rate:.1f}%", "Target-positive prevalence"), unsafe_allow_html=True)
cols[2].markdown(kpi_card("Best Model", selected_model, "Automatically selected by ROC-AUC"), unsafe_allow_html=True)
cols[3].markdown(
    kpi_card("ROC-AUC / Recall", f"{roc_auc:.3f} / {recall:.3f}" if roc_auc is not None else "N/A", "Held-out evaluation"),
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Clinical Use Case")
    st.write(
        "This MVP supports breast cancer diagnostic workflows by combining structured tumor measurements with a validated machine learning pipeline. "
        "It provides malignancy probability, risk stratification, model performance evidence, and patient-level explanation artifacts for clinical review."
    )
    st.subheader("Dataset Statistics")
    summary = data.drop(columns=["diagnosis_label"], errors="ignore").describe().T
    st.dataframe(summary[["mean", "std", "min", "50%", "max"]], use_container_width=True, height=360)

with right:
    st.subheader("Class Distribution")
    st.plotly_chart(class_distribution_chart(data), use_container_width=True)
    st.subheader("Model Statistics")
    model_rows = [
        {"Metric": "Accuracy", "Value": test_metrics.get("accuracy")},
        {"Metric": "Precision", "Value": test_metrics.get("precision")},
        {"Metric": "Recall", "Value": test_metrics.get("recall")},
        {"Metric": "F1 Score", "Value": test_metrics.get("f1")},
        {"Metric": "ROC-AUC", "Value": test_metrics.get("roc_auc")},
    ]
    st.dataframe(model_rows, use_container_width=True, hide_index=True)

st.subheader("KPI Dashboard")
comparison = metrics.get("comparison", [])
if comparison:
    comparison_fig = px.bar(
        comparison,
        x="model",
        y="roc_auc",
        color="recall",
        color_continuous_scale="Teal",
        text_auto=".3f",
        labels={"model": "Model", "roc_auc": "ROC-AUC", "recall": "Recall"},
    )
    comparison_fig.update_layout(height=420, xaxis_tickangle=-20)
    st.plotly_chart(comparison_fig, use_container_width=True)
else:
    st.info("Model comparison results will appear after training.")

