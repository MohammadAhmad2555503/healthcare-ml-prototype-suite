# Eye Disease Screening and Referral Support System

## Overview

Eye Disease Screening and Referral Support System is a clinical decision support prototype for Ophthalmology imaging. It provides a complete workflow for data review, reproducible model training, patient or image prediction, performance reporting, and an interactive Streamlit dashboard.

## Clinical Use Case

Supports screening workflow demonstrations across common ophthalmic disease categories.

This project is designed for portfolio submission, technical demonstration, and discussion of screening or triage workflows. It is not intended for real patient care without external validation and clinical governance.

## Dataset

| Item | Detail |
| --- | --- |
| Dataset | Eye disease image dataset |
| Input Type | Fundus or eye image |
| Target | Image folder label |
| Classes | Cataract, Diabetic Retinopathy, Glaucoma, and Normal |

The image projects keep a compact `data/manifest.csv` and trained artifacts in the repository. For retraining, place the original `datasets.zip` file at `data/datasets.zip`, or set the `HEALTH_DATASET_ZIP` environment variable to the archive path.

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
EyeDiseaseScreeningSystem/
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
| Accuracy | 0.827 |
| Precision Macro | 0.828 |
| Recall Macro | 0.827 |
| F1 Macro | 0.826 |
| ROC-AUC OVR | 0.944 |

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train

```powershell
python -m src.train --max-per-class 260
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
docker build -t eye-disease-screening-system .
docker run -p 8501:8501 eye-disease-screening-system
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

