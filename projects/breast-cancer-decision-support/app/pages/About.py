from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_utils import configure_page, load_app_data, load_model_metadata, render_title


configure_page("About")
render_title(
    "About",
    "Project details, dataset lineage, model information, clinical limitations, and deployment roadmap.",
)

data = load_app_data()
metadata = load_model_metadata()

st.subheader("Project Details")
st.write(
    "Breast Cancer Prediction and Clinical Decision Support System is a production-oriented healthcare MVP. "
    "It combines reproducible model training, automated model selection, explainability, test coverage, deployment assets, and a Streamlit clinical dashboard."
)

st.subheader("Dataset Details")
st.write(
    "The dataset contains digitized breast mass characteristics from fine needle aspirate images. "
    "The target column is `diagnosis`, where `M` maps to malignant and `B` maps to benign."
)
st.dataframe(
    {
        "Field": ["Rows", "Columns", "Target", "Feature Count"],
        "Value": [len(data), data.shape[1], "diagnosis", metadata.get("feature_count", "N/A")],
    },
    use_container_width=True,
    hide_index=True,
)

st.subheader("Model Information")
st.write(
    f"Selected model: `{metadata.get('selected_model', 'Not trained')}`. "
    "The training pipeline compares Logistic Regression, Random Forest, Support Vector Machine, XGBoost, LightGBM, and CatBoost when their libraries are installed. "
    "The best model is selected automatically by held-out ROC-AUC with recall and F1 score as secondary signals."
)
if metadata:
    st.json(
        {
            "generated_at": metadata.get("generated_at"),
            "best_params": metadata.get("best_params"),
            "test_metrics": metadata.get("test_metrics"),
            "skipped_models": metadata.get("skipped_models"),
        },
        expanded=False,
    )

st.subheader("Clinical Limitations")
st.write(
    "This software is a decision support demonstration and is not a standalone diagnostic device. "
    "Predictions must be interpreted by qualified clinicians and correlated with imaging, pathology, patient history, exam findings, local protocols, and regulatory requirements. "
    "The model is trained on a small public dataset and requires external validation, monitoring, fairness assessment, privacy review, and clinical governance before real-world use."
)

st.subheader("Future Work")
st.write(
    "Recommended next steps include prospective validation, DICOM/image model integration, EHR/FHIR interoperability, drift monitoring, model registry integration, audit logging, role-based access control, and a clinician feedback loop."
)

