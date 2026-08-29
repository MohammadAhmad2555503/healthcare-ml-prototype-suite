from __future__ import annotations

from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import configure_page, kpi_card, load_app_metrics, render_title


configure_page("Model Performance")
render_title(
    "Model Performance",
    "Review held-out classification metrics, confusion matrix, ROC curve, precision-recall curve, and model comparison.",
)

metrics = load_app_metrics()
test_metrics = metrics.get("test_metrics", {})
selected_model = metrics.get("selected_model", "Not trained")

cols = st.columns(5)
for col, label, key in zip(
    cols,
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    ["accuracy", "precision", "recall", "f1", "roc_auc"],
):
    value = test_metrics.get(key)
    col.markdown(kpi_card(label, f"{value:.3f}" if value is not None else "N/A", selected_model), unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Confusion Matrix")
    matrix = metrics.get("confusion_matrix")
    if matrix:
        cm = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted Benign", "Predicted Malignant"],
                y=["Actual Benign", "Actual Malignant"],
                colorscale="Blues",
                text=matrix,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        cm.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(cm, use_container_width=True)
    else:
        st.info("Confusion matrix will appear after training.")

with right:
    st.subheader("Model Comparison")
    comparison = metrics.get("comparison", [])
    if comparison:
        comparison_fig = go.Figure()
        comparison_fig.add_bar(
            x=[row["model"] for row in comparison],
            y=[row["roc_auc"] for row in comparison],
            name="ROC-AUC",
            marker_color="#1167b1",
        )
        comparison_fig.add_scatter(
            x=[row["model"] for row in comparison],
            y=[row["recall"] for row in comparison],
            name="Recall",
            mode="lines+markers",
            line=dict(color="#b42318", width=3),
        )
        comparison_fig.update_layout(height=430, yaxis=dict(range=[0, 1]), xaxis_tickangle=-20)
        st.plotly_chart(comparison_fig, use_container_width=True)
    else:
        st.info("Model comparison will appear after training.")

curve_cols = st.columns(2, gap="large")
curves = metrics.get("curves", {})
with curve_cols[0]:
    st.subheader("ROC Curve")
    roc = curves.get("roc", {})
    if roc:
        fig = go.Figure()
        fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC", line=dict(color="#1167b1", width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#98a2b3", dash="dash"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=430)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ROC curve will appear after training.")

with curve_cols[1]:
    st.subheader("Precision-Recall Curve")
    pr = curves.get("precision_recall", {})
    if pr:
        fig = go.Figure()
        fig.add_scatter(
            x=pr["recall"],
            y=pr["precision"],
            mode="lines",
            name="Precision-Recall",
            line=dict(color="#0f766e", width=3),
        )
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=430)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Precision-recall curve will appear after training.")

st.subheader("Detailed Metrics")
if metrics:
    st.json(metrics.get("classification_report", {}), expanded=False)
else:
    st.info("Metrics will appear after training.")

