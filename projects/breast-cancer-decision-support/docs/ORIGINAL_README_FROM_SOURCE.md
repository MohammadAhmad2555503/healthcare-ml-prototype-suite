# Breast Cancer Prediction and Clinical Decision Support System

## Overview

Breast Cancer Prediction and Clinical Decision Support System is a clinical decision support prototype for Breast oncology. It provides a complete workflow for data review, reproducible model training, patient or image prediction, performance reporting, and an interactive Streamlit dashboard.

## Clinical Use Case

Uses tumor morphology measurements to support malignancy risk review and referral discussion.

This project is designed for portfolio submission, technical demonstration, and discussion of screening or triage workflows. It is not intended for real patient care without external validation and clinical governance.

## Dataset

| Item | Detail |
| --- | --- |
| Dataset | Wisconsin breast cancer diagnostic dataset |
| Input Type | Structured tumor measurements |
| Target | diagnosis |
| Classes | Malignant and Benign |

The project includes the CSV used for training under the `data/` directory.

## Features

- Data loading and validation
- Missing value and duplicate checks
- Class distribution reporting
- Tuned classification model comparison
- Automatic best-model selection
- Saved prediction artifacts
- Interactive prediction page
- Analytics and performance dashboards
- Feature-importance reporting
- Docker and Compose deployment files
- Unit tests

## Project Structure

```text
BreastCancerDecisionSupport/
├── app/
│   ├── Home.py
│   └── pages/
├── data/
├── models/
├── reports/
├── src/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Current Model Results

Selected model: `Random Forest`

| Metric | Value |
| --- | --- |
| Accuracy | 0.974 |
| Precision | 1.000 |
| Recall | 0.929 |
| F1 Score | 0.963 |
| ROC-AUC | 0.998 |

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train

```powershell
python -m src.train --data data/breast_cancer.csv
```

## Run The Dashboard

```powershell
streamlit run app/Home.py
```

## Test

```powershell
python -m unittest discover -s tests
```

## Docker

```powershell
docker build -t breast-cancer-decision-support .
docker run -p 8501:8501 breast-cancer-decision-support
```

## Deployment

The project can be deployed on Streamlit Cloud, Docker, AWS EC2, Azure App Service, or any Python-compatible application host. Use `app/Home.py` as the Streamlit entry point and keep `models/`, `reports/`, and `data/` available at runtime.

## Limitations

- This is a demonstration system, not a regulated diagnostic product.
- Public datasets may contain bias, duplicates, acquisition artifacts, and limited clinical coverage.
- Reported metrics are based on the current dataset split and require independent validation.
- Real use requires privacy review, security hardening, monitoring, audit logging, clinical oversight, and regulatory assessment.

## Future Improvements

- External validation on multi-site datasets
- Calibration and uncertainty reporting
- Role-based access control
- Audit logging
- Drift monitoring
- Model registry integration
- Clinical feedback workflow

