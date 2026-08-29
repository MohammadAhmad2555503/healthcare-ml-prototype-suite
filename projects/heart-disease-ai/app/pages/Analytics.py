from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import class_distribution_chart, configure_page, data, feature_importance_chart, metrics, title_block
from src.data import feature_columns

configure_page("Analytics")
title_block("Analytics", "Explore clinical feature distributions, target balance, and model feature importance.")

df = data()
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Class Distribution")
    st.plotly_chart(class_distribution_chart(), use_container_width=True)
with right:
    st.subheader("Feature Distribution")
    feature = st.selectbox("Feature", feature_columns())
    fig = px.histogram(df, x=feature, color=df["target"].astype(str), marginal="box")
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Correlation")
fig = px.imshow(df[feature_columns()].corr(), color_continuous_scale="RdBu", zmin=-1, zmax=1)
fig.update_layout(height=680)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Importance")
fi = feature_importance_chart()
if fi:
    st.plotly_chart(fi, use_container_width=True)

st.subheader("Model Comparison")
st.dataframe(metrics().get("comparison", []), use_container_width=True, hide_index=True)

