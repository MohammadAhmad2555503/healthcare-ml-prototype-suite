from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import (
    COLOR_MAP,
    class_distribution_chart,
    configure_page,
    load_app_data,
    load_feature_importance,
    render_title,
)
from src.feature_engineering import class_imbalance_report, correlation_analysis, detect_outliers_iqr
from src.utils import get_feature_columns


configure_page("Analytics")
render_title(
    "Analytics",
    "Explore target distribution, tumor measurement patterns, correlation structure, outliers, and global feature importance.",
)

data = load_app_data()
feature_names = get_feature_columns(data)

overview_cols = st.columns(3)
imbalance = class_imbalance_report(data)
overview_cols[0].metric("Benign", f"{imbalance['benign']:,}", f"{imbalance['benign_percentage']:.1f}%")
overview_cols[1].metric("Malignant", f"{imbalance['malignant']:,}", f"{imbalance['malignant_percentage']:.1f}%")
overview_cols[2].metric("Imbalance Ratio", f"{imbalance['imbalance_ratio']:.2f}:1", imbalance["assessment"])

left, right = st.columns([0.95, 1.05], gap="large")
with left:
    st.subheader("Class Distribution")
    st.plotly_chart(class_distribution_chart(data), use_container_width=True)

with right:
    st.subheader("Feature Distribution")
    selected_feature = st.selectbox("Feature", feature_names, index=0)
    hist = px.histogram(
        data,
        x=selected_feature,
        color="diagnosis_label",
        marginal="box",
        nbins=35,
        color_discrete_map=COLOR_MAP,
        labels={"diagnosis_label": "Diagnosis"},
    )
    hist.update_layout(height=360)
    st.plotly_chart(hist, use_container_width=True)

st.subheader("Correlation Heatmap")
corr, high_corr = correlation_analysis(data)
heatmap = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        colorbar={"title": "r"},
    )
)
heatmap.update_layout(height=760, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(heatmap, use_container_width=True)

fi = load_feature_importance()
outliers = detect_outliers_iqr(data)

left2, right2 = st.columns(2, gap="large")
with left2:
    st.subheader("Feature Importance")
    if not fi.empty and "normalized_importance" in fi:
        top = fi.head(20).sort_values("normalized_importance", ascending=True)
        fig = px.bar(
            top,
            x="normalized_importance",
            y="feature",
            orientation="h",
            labels={"normalized_importance": "Normalized Importance", "feature": "Feature"},
            color="normalized_importance",
            color_continuous_scale="Teal",
        )
        fig.update_layout(height=560, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance will appear after training.")

with right2:
    st.subheader("Dataset Insights")
    st.write("Highest outlier counts by IQR method")
    st.dataframe(outliers.head(12), use_container_width=True, hide_index=True, height=260)
    st.write("Highly correlated feature pairs")
    if high_corr.empty:
        st.success("No feature pairs exceeded the configured high-correlation threshold.")
    else:
        st.dataframe(high_corr.head(12), use_container_width=True, hide_index=True, height=260)

