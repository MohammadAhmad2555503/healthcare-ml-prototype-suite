from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import card, configure_page, metrics, title_block

configure_page("Model Performance")
title_block("Model Performance", "Held-out validation metrics for the selected heart disease model.")

met = metrics()
test = met.get("test_metrics", {})
cols = st.columns(5)
for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], ["accuracy", "precision", "recall", "f1", "roc_auc"]):
    col.markdown(card(label, f"{test.get(key, 0):.3f}" if test else "N/A"), unsafe_allow_html=True)

matrix = met.get("confusion_matrix")
classes = met.get("classes", [])
if matrix:
    fig = go.Figure(data=go.Heatmap(z=matrix, x=classes, y=classes, text=matrix, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(height=480, xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Classification Report")
st.json(met.get("classification_report", {}), expanded=False)

