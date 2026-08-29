from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import configure_page, metrics, title_block
from src.utils import load_config

config = load_config()
configure_page("About")
title_block("About", config["project"]["name"])

st.subheader("Project Details")
st.write(config["clinical"]["use_case"])
st.write(f"Clinical domain: {config['clinical']['domain']}")

st.subheader("Model Information")
st.json(metrics(), expanded=False)

st.subheader("Limitations")
st.write(
    "This MVP is trained on a public tabular dataset and is not a diagnostic medical device. "
    "Production use requires external validation, calibration, fairness review, security controls, privacy assessment, monitoring, and clinical governance."
)

st.subheader("Future Work")
st.write("Add EHR/FHIR integration, calibration monitoring, explainability reports, audit logging, and prospective validation.")

