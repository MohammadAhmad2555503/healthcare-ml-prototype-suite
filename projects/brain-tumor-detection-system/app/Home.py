from __future__ import annotations

import streamlit as st

from app_utils import card, class_distribution_chart, configure_page, distribution, manifest, metadata, metrics, title_block
from src.utils import load_config

config = load_config()
configure_page("Home")
title_block(config["project"]["name"], config["clinical"]["use_case"])

df = manifest()
dist = distribution()
met = metrics()
meta = metadata()
test_metrics = met.get("test_metrics", {})

cols = st.columns(4)
cols[0].markdown(card("Images", f"{len(df):,}"), unsafe_allow_html=True)
cols[1].markdown(card("Classes", f"{dist.shape[0]}"), unsafe_allow_html=True)
cols[2].markdown(card("Best Model", str(met.get("selected_model", meta.get("selected_model", "Not trained")))), unsafe_allow_html=True)
cols[3].markdown(card("F1 Macro", f"{test_metrics.get('f1_macro', 0):.3f}" if test_metrics else "N/A"), unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("Clinical Overview")
    st.write(config["clinical"]["use_case"])
    st.write(
        "This prototype uses deterministic image quality, color, intensity, edge, and thumbnail features with tuned classical predictive modeling models. "
        "It is intended for MVP demonstration, workflow exploration, and model governance discussion."
    )
    st.subheader("Dataset Sample")
    st.dataframe(df.head(25), use_container_width=True, hide_index=True, height=360)
with right:
    st.subheader("Class Distribution")
    st.plotly_chart(class_distribution_chart(), use_container_width=True)

st.subheader("Clinical Safety Note")
st.info(
    "This application is a decision support prototype, not a standalone diagnostic device. Results require qualified clinical review and external validation."
)

