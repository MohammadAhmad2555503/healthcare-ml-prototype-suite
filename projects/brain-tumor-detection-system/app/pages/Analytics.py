from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import class_distribution_chart, configure_page, feature_importance_chart, manifest, metrics, title_block

configure_page("Analytics")
title_block("Analytics", "Dataset distribution, source labels, model comparison, and feature importance.")

df = manifest()
met = metrics()
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Class Distribution")
    st.plotly_chart(class_distribution_chart(), use_container_width=True)
with right:
    st.subheader("Source Label Distribution")
    source_counts = df["source_label"].value_counts().reset_index()
    source_counts.columns = ["source_label", "count"]
    fig = px.bar(source_counts, x="source_label", y="count", color="source_label", text="count")
    fig.update_layout(showlegend=False, height=360)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Model Comparison")
comparison = met.get("comparison", [])
if comparison:
    st.dataframe(comparison, use_container_width=True, hide_index=True)
else:
    st.info("Run training to generate comparison metrics.")

st.subheader("Feature Importance")
fig = feature_importance_chart()
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance is unavailable for the selected model.")

