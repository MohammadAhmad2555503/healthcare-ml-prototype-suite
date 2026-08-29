# Heart Disease Risk Prediction AI

Production-style healthcare AI prototype for Cardiology risk prediction.

## Use Case

Structured clinical risk prediction for heart disease screening and care pathway support.

## What Is Included

- Data manifest/loading and validation
- Reproducible training pipeline
- Logistic Regression, Random Forest, and Support Vector Machine model comparison
- Automatic best-model selection
- Model artifacts under `models/`
- Metrics, confusion matrix, class distribution, feature importance, and comparison reports under `reports/`
- Multi-page Streamlit app
- Unit tests
- Dockerfile and docker-compose.yml

## Data

The tabular CSV is copied into `data/heart.csv`.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

## Run App

```bash
streamlit run app/Home.py
```

## Test

```bash
python -m unittest discover -s tests
```

## Docker

```bash
docker build -t heartdiseaseai .
docker run -p 8501:8501 heartdiseaseai
```

## Clinical Safety

This is an MVP decision support prototype. It is not a standalone diagnostic device and must not be used for clinical care without external validation, clinician review, quality management, privacy/security review, monitoring, and regulatory assessment.

