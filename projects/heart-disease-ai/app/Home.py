from __future__ import annotations

import streamlit as st

from app_utils import card, class_distribution_chart, configure_page, data, metrics, title_block
from src.utils import load_config

config = load_config()
configure_page("Home")
title_block(config["project"]["name"], config["clinical"]["use_case"])

df = data()
met = metrics()
test = met.get("test_metrics", {})

cols = st.columns(4)
cols[0].markdown(card("Patient Records", f"{len(df):,}"), unsafe_allow_html=True)
cols[1].markdown(card("Features", f"{df.shape[1] - 1}"), unsafe_allow_html=True)
cols[2].markdown(card("Best Model", str(met.get("selected_model", "Not trained"))), unsafe_allow_html=True)
cols[3].markdown(card("ROC-AUC", f"{test.get('roc_auc', 0):.3f}" if test else "N/A"), unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("Clinical Overview")
    st.write(config["clinical"]["use_case"])
    st.subheader("Dataset Preview")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True, height=360)
with right:
    st.subheader("Class Distribution")
    st.plotly_chart(class_distribution_chart(), use_container_width=True)

st.info("This is a clinical decision support prototype and requires qualified clinical review before any real-world interpretation.")

