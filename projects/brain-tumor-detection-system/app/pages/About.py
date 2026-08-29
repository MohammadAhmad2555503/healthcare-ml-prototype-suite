from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import configure_page, metadata, title_block
from src.utils import load_config

config = load_config()
configure_page("About")
title_block("About", config["project"]["name"])

st.subheader("Project Details")
st.write(config["clinical"]["use_case"])
st.write(f"Clinical domain: {config['clinical']['domain']}")
st.write(f"Image modality: {config['clinical']['image_modality']}")

st.subheader("Model Information")
st.json(metadata(), expanded=False)

st.subheader("Limitations")
st.write(
    "This MVP uses engineered image features and classical predictive modeling for fast, transparent prototyping. "
    "It is not validated for clinical diagnosis. Production use requires clinically governed datasets, external validation, monitoring, privacy review, and regulatory assessment."
)

st.subheader("Future Work")
st.write(
    "Add deep learning backbones, DICOM handling, segmentation/lesion localization, calibration, uncertainty estimation, role-based access control, audit logging, and model registry integration."
)

